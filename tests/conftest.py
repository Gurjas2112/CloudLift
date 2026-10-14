"""Shared fixtures. Everything here is SYNTHETIC: procedural scenes, never real LISS-IV."""

from __future__ import annotations

import logging
from pathlib import Path

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
