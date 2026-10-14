"""Co-registration onto the LISS-IV grid.

1. Reproject SAR / reference / DEM onto the optical grid (bilinear for continuous data,
   nearest for masks) with Rasterio.
2. Sub-pixel refinement of the temporal reference (optical-to-optical) by per-tile phase
   correlation over clear pixels; the median shift is applied and the residual reported.
   SAR is geocoded by the provider and only reprojected: phase correlation between SAR and
   optical is cross-modal and unreliable, so it is deliberately not applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import shift as nd_shift
from skimage.registration import phase_cross_correlation

from cloudlift.io.geotiff import RasterMeta


def reproject_to_grid(
    src_path: str | Path,
    dst: RasterMeta,
    *,
    categorical: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject ``src_path`` onto ``dst``'s grid.

    Returns ``(array [C,H,W] float32, valid [H,W] bool)``. Pixels outside the source footprint
    or at source nodata are invalid (filled with 0).
    """
    with rasterio.open(src_path) as ds:
        out = np.full((ds.count, dst.height, dst.width), np.nan, np.float32)
        for b in range(ds.count):
            reproject(
                source=rasterio.band(ds, b + 1),
                destination=out[b],
                src_nodata=ds.nodata,
                dst_transform=dst.transform,
                dst_crs=dst.crs,
                dst_nodata=np.nan,
                resampling=Resampling.nearest if categorical else Resampling.bilinear,
            )
    valid = np.all(np.isfinite(out), axis=0)
    return np.nan_to_num(out, nan=0.0), valid


@dataclass
class ShiftEstimate:
    shift: tuple[float, float]  # (dy, dx) applied to the moving image
    residual_px: float  # magnitude of the shift remaining after correction
    n_tiles: int
    spread_px: float  # median absolute deviation of per-tile shifts

    def as_dict(self) -> dict[str, float | int | list[float]]:
        return {
            "shift_dy_dx": [round(self.shift[0], 4), round(self.shift[1], 4)],
            "residual_px": round(self.residual_px, 4),
            "n_tiles": self.n_tiles,
            "spread_px": round(self.spread_px, 4),
        }


def _tile_shifts(
    ref: np.ndarray, mov: np.ndarray, valid: np.ndarray, tile: int, min_valid: float, upsample: int
) -> list[tuple[float, float]]:
    h, w = ref.shape
    out: list[tuple[float, float]] = []
    step = tile
    for y in range(0, max(1, h - tile + 1), step):
        for x in range(0, max(1, w - tile + 1), step):
            v = valid[y : y + tile, x : x + tile]
            if v.size == 0 or v.mean() < min_valid:
                continue
            r = ref[y : y + tile, x : x + tile].astype(np.float64)
            m = mov[y : y + tile, x : x + tile].astype(np.float64)
            r = np.where(v, r, r[v].mean())
            m = np.where(v, m, m[v].mean())
            if r.std() < 1e-6 or m.std() < 1e-6:
                continue
            s, _, _ = phase_cross_correlation(r, m, upsample_factor=upsample)
            out.append((float(s[0]), float(s[1])))
    return out


def estimate_shift(
    ref: np.ndarray,
    mov: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    tile: int = 256,
    min_valid: float = 0.9,
    upsample: int = 20,
    max_shift: float = 10.0,
) -> tuple[tuple[float, float], int, float]:
    """Median per-tile phase-correlation shift that registers ``mov`` onto ``ref``."""
    if valid is None:
        valid = np.ones(ref.shape, bool)
    tile = min(tile, ref.shape[0], ref.shape[1])
    shifts: list[tuple[float, float]] = []
    while True:  # under heavy cloud few large tiles are clear enough: halve the tile size
        shifts = _tile_shifts(ref, mov, valid, tile, min_valid, upsample)
        shifts = [s for s in shifts if abs(s[0]) <= max_shift and abs(s[1]) <= max_shift]
        if len(shifts) >= 3 or tile <= 64:
            break
        tile //= 2
    if not shifts:
        return (0.0, 0.0), 0, 0.0
    arr = np.asarray(shifts)
    med = np.median(arr, axis=0)
    spread = float(np.median(np.abs(arr - med)))
    return (float(med[0]), float(med[1])), len(shifts), spread


def apply_shift(arr: np.ndarray, shift: tuple[float, float]) -> np.ndarray:
    """Sub-pixel shift of ``[C,H,W]`` (or ``[H,W]``) with bilinear interpolation."""
    if shift == (0.0, 0.0):
        return arr
    if arr.ndim == 2:
        return nd_shift(arr, shift, order=1, mode="nearest").astype(arr.dtype)
    return np.stack([nd_shift(b, shift, order=1, mode="nearest") for b in arr]).astype(arr.dtype)


def refine_reference(
    optical: np.ndarray,
    reference: np.ndarray,
    clear: np.ndarray,
    *,
    band: int = 2,
    tile: int = 256,
) -> tuple[np.ndarray, ShiftEstimate]:
    """Align the temporal reference to the current optical pass using clear pixels (NIR band)."""
    s, n, spread = estimate_shift(optical[band], reference[band], clear, tile=tile)
    aligned = apply_shift(reference, s)
    (rdy, rdx), _, _ = estimate_shift(optical[band], aligned[band], clear, tile=tile)
    return aligned, ShiftEstimate(s, float(np.hypot(rdy, rdx)), n, spread)
