"""Data sources. Default: ``LocalFolderSource`` over files the user downloaded from Bhoonidhi.

Folder convention (one folder per scene, under ``data/raw/``)::

    data/raw/<scene_id>/
        liss4.tif        required  LISS-IV (ResourceSat-2/2A) bands in order Green, Red, NIR
        sar.tif          required  EOS-04 / RISAT-1A C-band, 2 polarisations (any grid; reprojected)
        reference.tif    required  prior cloud-free LISS-IV pass of the same AOI (T-N days)
        dem.tif          optional  CartoDEM
        clean.tif        optional  cloud-free truth for full-reference evaluation (synthetic only)
        mask_gt.tif      optional  ground-truth cloud mask (synthetic only)
        meta.json        optional  {"acquired": "...", "reference_acquired": "...", "scale": 1023, ...}

CloudLift never scrapes Bhoonidhi or bypasses its login: download scenes manually
(Indian Space Policy 2023, data stays in-country) and drop them in this layout.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from pyproj import CRS, Transformer
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shp_transform

from cloudlift.io import geotiff
from cloudlift.io.aoi import AOI


@dataclass
class Scene:
    scene_id: str
    optical: Path
    sar: Path | None
    reference: Path | None
    dem: Path | None = None
    clean: Path | None = None
    mask_gt: Path | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def synthetic(self) -> bool:
        return bool(self.meta.get("synthetic", False))

    @property
    def acquired(self) -> str:
        return str(self.meta.get("acquired", "unknown"))

    def reference_gap_days(self) -> int:
        """Days between reference and current pass (used as the temporal position)."""
        try:
            a = date.fromisoformat(str(self.meta["acquired"]))
            r = date.fromisoformat(str(self.meta["reference_acquired"]))
            return max(1, (a - r).days)
        except (KeyError, ValueError):
            return 30

    def footprint(self) -> tuple[BaseGeometry, str]:
        m = geotiff.read_meta(self.optical)
        crs = m.crs.to_string() if m.crs else "EPSG:4326"
        return box(*m.bounds), crs

    def input_paths(self) -> list[Path]:
        return [p for p in (self.optical, self.sar, self.reference, self.dem) if p is not None]


class DataSource(ABC):
    @abstractmethod
    def list_scenes(self, aoi: AOI | None = None) -> list[Scene]: ...

    def get_scene(self, scene_id: str) -> Scene:
        for s in self.list_scenes():
            if s.scene_id == scene_id:
                return s
        raise KeyError(f"scene {scene_id!r} not found in {self!r}")

    def resolve_tile(self, tile_id: str) -> tuple[BaseGeometry, str] | None:
        """AOI tile-ID resolver: a tile ID is the scene folder name; its footprint is the AOI."""
        try:
            return self.get_scene(tile_id).footprint()
        except KeyError:
            return None


def _intersects(scene: Scene, aoi: AOI) -> bool:
    if aoi.kind == "tile":
        return aoi.tile_id == scene.scene_id
    if aoi.geometry is None:
        return False
    fp, crs = scene.footprint()
    tr = Transformer.from_crs(CRS.from_user_input(crs), CRS.from_user_input(aoi.crs), always_xy=True)
    return bool(shp_transform(tr.transform, fp).intersects(aoi.geometry))


class LocalFolderSource(DataSource):
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def __repr__(self) -> str:
        return f"LocalFolderSource({str(self.root)!r})"

    def _scene(self, d: Path) -> Scene | None:
        optical = d / "liss4.tif"
        if not optical.exists():
            return None

        def opt(name: str) -> Path | None:
            p = d / name
            return p if p.exists() else None

        meta: dict[str, Any] = {}
        if (d / "meta.json").exists():
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        return Scene(
            scene_id=d.name,
            optical=optical,
            sar=opt("sar.tif"),
            reference=opt("reference.tif"),
            dem=opt("dem.tif"),
            clean=opt("clean.tif"),
            mask_gt=opt("mask_gt.tif"),
            meta=meta,
        )

    def list_scenes(self, aoi: AOI | None = None) -> list[Scene]:
        if not self.root.exists():
            return []
        scenes = [s for d in sorted(self.root.iterdir()) if d.is_dir() and (s := self._scene(d))]
        if aoi is not None:
            scenes = [s for s in scenes if _intersects(s, aoi)]
        return scenes


class SyntheticSource(LocalFolderSource):
    """Generates labelled SYNTHETIC scenes on demand, then behaves like a local folder."""

    def __init__(self, root: str | Path, n_scenes: int = 1, size: int = 1024, seed: int = 0) -> None:
        super().__init__(root)
        self.n_scenes, self.size, self.seed = n_scenes, size, seed

    def __repr__(self) -> str:
        return f"SyntheticSource({str(self.root)!r}, n={self.n_scenes}, size={self.size}, seed={self.seed})"

    def ensure(self) -> None:
        from cloudlift.io.synthetic import generate_scene

        for i in range(self.n_scenes):
            sid = f"SYN_NER_{i + 1:04d}"
            if not (self.root / sid / "meta.json").exists():
                generate_scene(
                    self.root, sid, size=self.size, seed=self.seed + i, cloud_fraction=0.25 + 0.1 * (i % 4)
                )

    def list_scenes(self, aoi: AOI | None = None) -> list[Scene]:
        self.ensure()
        return super().list_scenes(aoi)
