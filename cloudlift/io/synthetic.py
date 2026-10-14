"""SYNTHETIC fixture scene: procedural terrain + river + fields + clouds, LISS-IV-like 3 bands + 2-pol SAR.

Everything written here is tagged ``SYNTHETIC=true``. It exists so that every phase is
testable before real Bhoonidhi data is supplied; it is not, and must never be presented as,
real LISS-IV / EOS-04 data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rasterio.transform import from_origin
from scipy.ndimage import shift as nd_shift

from cloudlift.io import geotiff
from cloudlift.preprocess.synthetic_clouds import fbm_noise, make_clouds

SYN_CRS = "EPSG:32646"  # UTM 46N, covers the North-Eastern Region (Guwahati ~ 91.7E)
SYN_ORIGIN = (360000.0, 2895000.0)  # upper-left, near Guwahati / Brahmaputra
PIXEL = 5.8
SYN_TAGS = {"SYNTHETIC": "true", "DESCRIPTION": "SYNTHETIC FIXTURE - not real LISS-IV / EOS-04 data"}

# (Green, Red, NIR) surface reflectance per class
_CLASS_REFL = {
    "water": (0.07, 0.05, 0.03),
    "forest": (0.06, 0.04, 0.36),
    "crop": (0.09, 0.07, 0.30),
    "bare": (0.14, 0.18, 0.24),
    "urban": (0.16, 0.17, 0.19),
}
# (VV, VH) backscatter in dB per class
_CLASS_SAR = {
    "water": (-22.0, -28.0),
    "forest": (-8.0, -13.0),
    "crop": (-10.0, -17.0),
    "bare": (-12.0, -20.0),
    "urban": (-3.0, -11.0),
}


def _surface(size: int, seed: int, season: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (reflectance [3,H,W], class map [H,W] int, DEM metres [H,W])."""
    rng = np.random.default_rng(seed)
    elev = fbm_noise((size, size), seed=seed, base_cells=3.0, octaves=6)
    dem = (50.0 + 900.0 * elev**1.6).astype(np.float32)

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    meander = 0.45 * size + 0.12 * size * np.sin(2 * np.pi * yy / (0.9 * size) + 0.7)
    meander += 0.03 * size * (fbm_noise((size, size), seed=seed + 1, base_cells=2.0, octaves=2) - 0.5)
    river = np.abs(xx - meander) < (0.012 * size + 4)

    classes = np.full((size, size), 3, np.int32)  # bare
    classes[elev > 0.55] = 1  # forest on hills
    fields = (elev <= 0.55) & (elev > 0.2)
    cell = max(16, size // 24)
    field_id = (yy // cell).astype(np.int64) * 10007 + (xx // cell).astype(np.int64)
    crop_on = (np.sin(field_id * 12.9898 + seed) * 43758.5453) % 1.0 > 0.35
    classes[fields & crop_on] = 2  # crop
    urban = ((yy - 0.7 * size) ** 2 + (xx - meander - 0.08 * size) ** 2) < (0.06 * size) ** 2
    classes[urban] = 4
    classes[river] = 0

    names = ["water", "forest", "crop", "bare", "urban"]
    refl = np.zeros((3, size, size), np.float32)
    for i, n in enumerate(names):
        refl[:, classes == i] = np.asarray(_CLASS_REFL[n], np.float32)[:, None]

    # per-field crop vigour; ``season`` shifts it to emulate the temporal reference
    vigour = ((np.cos(field_id * 78.233 + seed) * 12345.678) % 1.0).astype(np.float32)
    vig = np.clip(vigour + season, 0, 1)
    crop = classes == 2
    refl[2][crop] *= 0.7 + 0.6 * vig[crop]
    refl[1][crop] *= 1.3 - 0.6 * vig[crop]

    gy, gx = np.gradient(dem)
    shade = np.clip(1.0 + 0.004 * (-gx - gy), 0.75, 1.2).astype(np.float32)
    texture = 1.0 + 0.08 * (fbm_noise((size, size), seed=seed + 2, base_cells=40.0, octaves=3) - 0.5)
    refl = refl * (shade * texture)[None]
    refl += rng.normal(0, 0.003, refl.shape).astype(np.float32)
    return np.clip(refl, 0.0, 1.0).astype(np.float32), classes, dem


def _sar(classes: np.ndarray, dem: np.ndarray, seed: int) -> np.ndarray:
    """Two-pol SAR normalised to [0, 1] from dB via (dB + 25) / 25, with 4-look speckle."""
    rng = np.random.default_rng(seed + 99)
    names = ["water", "forest", "crop", "bare", "urban"]
    db = np.zeros((2, *classes.shape), np.float32)
    for i, n in enumerate(names):
        db[:, classes == i] = np.asarray(_CLASS_SAR[n], np.float32)[:, None]
    gy, gx = np.gradient(dem)
    db += (0.05 * (gx + gy)).astype(np.float32)[None]  # foreshortening-like slope term
    lin = 10 ** (db / 10.0)
    lin *= rng.gamma(4.0, 1.0 / 4.0, lin.shape).astype(np.float32)
    db = 10 * np.log10(np.maximum(lin, 1e-6))
    return np.clip((db + 25.0) / 25.0, 0.0, 1.0).astype(np.float32)


def generate_scene(
    out_dir: str | Path,
    scene_id: str = "SYN_NER_0001",
    size: int = 1024,
    seed: int = 0,
    cloud_fraction: float = 0.35,
    ref_shift: tuple[float, float] = (1.6, -0.8),
) -> Path:
    """Write a deterministic synthetic scene folder in the ``LocalFolderSource`` convention."""
    out = Path(out_dir) / scene_id
    out.mkdir(parents=True, exist_ok=True)
    clean, classes, dem = _surface(size, seed, season=0.0)
    ref, _, _ = _surface(size, seed, season=-0.12)  # earlier pass, crops less developed
    ref += np.random.default_rng(seed + 5).normal(0, 0.004, ref.shape).astype(np.float32)
    ref = np.stack([nd_shift(b, ref_shift, order=3, mode="nearest") for b in ref]).astype(np.float32)
    ref = np.clip(ref, 0, 1)
    sar = _sar(classes, dem, seed)
    clouds = make_clouds(clean, cloud_fraction, seed=seed + 11)

    transform = from_origin(SYN_ORIGIN[0], SYN_ORIGIN[1], PIXEL, PIXEL)
    common = {"crs": SYN_CRS, "transform": transform}
    bands = ("Green", "Red", "NIR")
    geotiff.write(out / "liss4.tif", clouds.cloudy, descriptions=bands, tags=SYN_TAGS, **common)
    geotiff.write(out / "reference.tif", ref, descriptions=bands, tags=SYN_TAGS, **common)
    geotiff.write(out / "sar.tif", sar, descriptions=("VV", "VH"), tags=SYN_TAGS, **common)
    geotiff.write(out / "dem.tif", dem, descriptions=("elevation_m",), tags=SYN_TAGS, **common)
    geotiff.write(out / "clean.tif", clean, descriptions=bands, tags=SYN_TAGS, **common)
    geotiff.write(out / "mask_gt.tif", clouds.mask, descriptions=("cloud_mask",), tags=SYN_TAGS, **common)
    meta = {
        "scene_id": scene_id,
        "synthetic": True,
        "acquired": "2026-07-15",
        "reference_acquired": "2026-02-10",
        "sar_acquired": "2026-07-14",
        "seed": seed,
        "size": size,
        "cloud_fraction_requested": cloud_fraction,
        "cloud_fraction_actual": round(clouds.cloud_fraction, 6),
        "reference_shift_px": list(ref_shift),
        "note": "SYNTHETIC FIXTURE - procedural data, not real LISS-IV / EOS-04",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out
