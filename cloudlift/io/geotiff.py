"""GeoTIFF read/write that preserves CRS, transform, nodata, band descriptions and tags."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import Window


@dataclass
class RasterMeta:
    crs: CRS | None
    transform: Affine
    width: int
    height: int
    nodata: float | None = None
    descriptions: tuple[str | None, ...] = ()
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        from rasterio.transform import array_bounds

        west, south, east, north = array_bounds(self.height, self.width, self.transform)
        return (min(west, east), min(south, north), max(west, east), max(south, north))


def read(path: str | Path, window: Window | None = None) -> tuple[np.ndarray, RasterMeta]:
    """Read all bands as ``[C, H, W]``. With ``window``, only that window is read."""
    with rasterio.open(path) as ds:
        arr = ds.read(window=window)
        transform = ds.window_transform(window) if window is not None else ds.transform
        meta = RasterMeta(
            crs=ds.crs,
            transform=transform,
            width=arr.shape[2],
            height=arr.shape[1],
            nodata=ds.nodata,
            descriptions=tuple(ds.descriptions),
            tags=dict(ds.tags()),
        )
    return arr, meta


def read_meta(path: str | Path) -> RasterMeta:
    with rasterio.open(path) as ds:
        return RasterMeta(
            crs=ds.crs,
            transform=ds.transform,
            width=ds.width,
            height=ds.height,
            nodata=ds.nodata,
            descriptions=tuple(ds.descriptions),
            tags=dict(ds.tags()),
        )


def write(
    path: str | Path,
    array: np.ndarray,
    *,
    crs: CRS | str | None,
    transform: Affine,
    nodata: float | None = None,
    descriptions: tuple[str, ...] | list[str] | None = None,
    tags: dict[str, Any] | None = None,
    analysis_ready: bool = False,
) -> Path:
    """Write ``[C, H, W]`` (or ``[H, W]``) to GeoTIFF.

    ``analysis_ready=True`` adds DEFLATE compression with a predictor, 256 px internal tiling
    and overviews, which is the CloudLift export profile.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if array.ndim == 2:
        array = array[None]
    count, height, width = array.shape
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "count": count,
        "height": height,
        "width": width,
        "dtype": array.dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
    }
    if analysis_ready:
        tiled = height >= 256 and width >= 256
        predictor = 3 if np.issubdtype(array.dtype, np.floating) else 2
        profile.update(compress="deflate", predictor=predictor, zlevel=6)
        if tiled:
            profile.update(tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(array)
        if descriptions:
            for i, d in enumerate(descriptions, start=1):
                ds.set_band_description(i, d)
        if tags:
            ds.update_tags(**{k: str(v) for k, v in tags.items()})
        if analysis_ready:
            factors = [f for f in (2, 4, 8, 16) if min(height, width) // f >= 64]
            if factors:
                ds.build_overviews(factors, Resampling.average)
                ds.update_tags(ns="rio_overview", resampling="average")
    return path
