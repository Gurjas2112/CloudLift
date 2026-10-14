"""Data sources: scene folders on disk (default), synthetic fixtures, and user uploads.

Folder convention (one folder per scene, under ``data/raw/``; uploads go to ``data/uploads/``)::

    data/raw/<scene_id>/
        liss4.tif        required  optical pass: LISS-IV or any GeoTIFF with Green, Red, NIR bands
        sar.tif          optional  C-band SAR, 2 polarisations (VV, VH), any grid; reprojected
        reference.tif    optional  earlier cloud-free optical pass of the same area
        dem.tif          optional  DEM
        clean.tif        optional  cloud-free truth for full-reference evaluation
        mask_gt.tif      optional  ground-truth cloud mask
        meta.json        optional  {"acquired": "2026-07-15", "reference_acquired": "2026-02-10",
                                    "bands": [1, 2, 3], "scale": "auto", "sar_units": "db"}

``bands`` are the 1-based (Green, Red, NIR) indexes in the optical file (e.g. ``[2, 3, 4]`` for a
Blue/Green/Red/NIR product). Without SAR or a reference the pipeline still runs, with higher
uncertainty. For Bhoonidhi data, download scenes manually (Indian Space Policy 2023: data stays
in-country); CloudLift never scrapes Bhoonidhi or bypasses its login.
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
from cloudlift.io.radiometry import DEFAULT_BANDS, band_indexes


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


UPLOAD_ROLES = {"liss4": 3, "sar": 2, "reference": 3, "dem": 1}  # role -> expected band count


class UploadError(ValueError):
    pass


def save_scene(root: str | Path, blobs: dict[str, bytes], meta: dict[str, Any] | None = None) -> Scene:
    """Store user-supplied GeoTIFFs as a ``LocalFolderSource`` scene named by content hash.

    ``blobs`` maps a role (``liss4`` required; ``sar``, ``reference``, ``dem`` optional) to file
    bytes. Each file is checked to be a georeferenced raster with the expected band count, so a
    bad upload fails here with a clear message instead of deep inside the pipeline.
    """
    import hashlib

    from rasterio.io import MemoryFile

    blobs = {k: v for k, v in blobs.items() if v}
    if "liss4" not in blobs:
        raise UploadError("an optical GeoTIFF with Green, Red and NIR bands is required")
    for role, data in blobs.items():
        if role not in UPLOAD_ROLES:
            raise UploadError(f"unknown file role {role!r}; expected one of {sorted(UPLOAD_ROLES)}")
        try:
            with MemoryFile(data) as mf, mf.open() as ds:
                count, crs = ds.count, ds.crs
        except Exception as e:
            raise UploadError(f"{role}: not a readable GeoTIFF ({e})") from e
        if crs is None:
            raise UploadError(f"{role}: file has no coordinate reference system")
        if count < UPLOAD_ROLES[role]:
            raise UploadError(f"{role}: expected at least {UPLOAD_ROLES[role]} band(s), found {count}")
        if role in ("liss4", "reference"):
            key = "bands" if role == "liss4" or "reference_bands" not in (meta or {}) else "reference_bands"
            try:
                band_indexes({"bands": (meta or {}).get(key, DEFAULT_BANDS)}, count)
            except ValueError as e:
                raise UploadError(f"{role}: {e}") from e
    scale = (meta or {}).get("scale", "auto")
    if scale not in (None, "", "auto"):
        try:
            if float(scale) <= 0:
                raise ValueError
        except (TypeError, ValueError) as e:
            raise UploadError(f"scale must be 'auto' or a positive number, got {scale!r}") from e
    h = hashlib.sha256()
    for k in sorted(blobs):
        h.update(k.encode())
        h.update(hashlib.sha256(blobs[k]).digest())
    meta = {k: v for k, v in (meta or {}).items() if v not in (None, "")}
    h.update(json.dumps(meta, sort_keys=True).encode())
    sid = f"UPLOAD_{h.hexdigest()[:12]}"
    d = Path(root) / sid
    d.mkdir(parents=True, exist_ok=True)
    for role, data in blobs.items():
        (d / f"{role}.tif").write_bytes(data)
    if meta:
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return LocalFolderSource(root).get_scene(sid)
