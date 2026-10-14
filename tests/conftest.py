"""Shared fixtures. Everything here is SYNTHETIC: procedural scenes, never real LISS-IV."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from cloudlift.config import Config, load_config
from cloudlift.io.sources import LocalFolderSource, Scene
from cloudlift.io.synthetic import generate_scene

logging.getLogger("cloudlift.model.uncrtaints_wrapper").setLevel(logging.ERROR)

SCENE_SIZE = 512


@pytest.fixture(scope="session")
def scene_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("scenes")
    generate_scene(root, "SYN_TEST_0001", size=SCENE_SIZE, seed=3, cloud_fraction=0.3)
    return root


@pytest.fixture(scope="session")
def scene(scene_root: Path) -> Scene:
    return LocalFolderSource(scene_root).get_scene("SYN_TEST_0001")


@pytest.fixture()
def cfg(tmp_path: Path) -> Config:
    return load_config(
        overrides={
            "device": "cpu",
            "paths": {"results_root": str(tmp_path / "results"), "cache_root": str(tmp_path / "cache")},
            "model": {"size": "tiny"},
        }
    )


@pytest.fixture()
def has_cuda() -> bool:
    return torch.cuda.is_available()


def to_four_band_uint16(src: Path, dst: Path) -> Path:
    """Re-encode a 3-band float G,R,NIR raster as a 4-band uint16 B,G,R,NIR raster scaled x10000,
    the layout of many common optical products (SYNTHETIC content)."""
    from cloudlift.io import geotiff

    arr, meta = geotiff.read(src)
    blue = np.clip(arr[0] * 0.8, 0, 1)[None]
    four = np.round(np.concatenate([blue, arr], axis=0) * 10000).astype(np.uint16)
    return geotiff.write(
        dst, four, crs=meta.crs, transform=meta.transform, descriptions=("Blue", "Green", "Red", "NIR")
    )


@pytest.fixture(scope="session")
def four_band_scene(scene_root: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    d = scene_root / "SYN_TEST_0001"
    out = tmp_path_factory.mktemp("fourband")
    return to_four_band_uint16(d / "liss4.tif", out / "opt.tif"), to_four_band_uint16(
        d / "reference.tif", out / "ref.tif"
    )
