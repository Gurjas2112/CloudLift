"""2-epoch training smoke on SYNTHETIC tiles (CPU; also runs on CUDA when available)."""

import json

import lightning as L
import pytest
import torch

from cloudlift.config import load_config
from cloudlift.io.sources import SyntheticSource
from cloudlift.model.datamodule import CloudPairDataset, TileDataModule, build_tile_cache
from cloudlift.model.lit_module import CloudLiftLit, fit
from cloudlift.model.uncrtaints_wrapper import load


def _cfg(tmp_path, device="cpu", precision="32-true"):
    return load_config(
        overrides={
            "device": device,
            "paths": {"results_root": str(tmp_path / "results"), "cache_root": str(tmp_path / "cache")},
            "data": {"tile": 128, "preproc_version": "test"},
            "model": {"size": "tiny"},
            "train": {"max_epochs": 2, "num_workers": 0, "precision": precision, "val_fraction": 0.25},
        }
    )


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return SyntheticSource(tmp_path_factory.mktemp("syn"), n_scenes=2, size=256, seed=0)


def test_dataset_reproducible_and_geometric_only(tmp_path, source):
    cfg = _cfg(tmp_path)
    root = build_tile_cache(source, cfg)
    names = json.loads((root / "index.json").read_text(encoding="utf-8"))["tiles"]
    assert len(names) == 8
    ds = CloudPairDataset(root, names, cfg, augment=True, seed_offset=0)
    a, b = ds[3], ds[3]
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert a["x"].shape == (2, 5, 128, 128) and a["target"].shape == (3, 128, 128)
    # flips/rot90 permute pixels but never change the spectrum histogram of the target
    plain = CloudPairDataset(root, names, cfg, augment=False, seed_offset=0)[3]
    assert torch.allclose(a["target"].sum(), plain["target"].sum(), rtol=1e-5)
    # cache is preprocess-once: a second build adds nothing
    build_tile_cache(source, cfg)
    assert json.loads((root / "index.json").read_text(encoding="utf-8"))["tiles"] == names


def _smoke(tmp_path, source, device, precision):
    cfg = _cfg(tmp_path, device, precision)
    summary = fit(cfg, source, run_name="smoke")
    hist = summary["history"]
    assert len(hist) == 2
    assert hist[1]["train_loss_epoch"] < hist[0]["train_loss_epoch"]

    run = tmp_path / "results" / "smoke"
    last = run / "checkpoints" / "last.ckpt"
    assert last.exists() and (run / "cloudlift_weights.pt").exists()
    dm = TileDataModule(cfg, tmp_path / "cache" / "test")
    trainer = L.Trainer(
        accelerator="cpu", devices=1, logger=False, enable_progress_bar=False, enable_model_summary=False
    )
    v1 = trainer.validate(CloudLiftLit.load_from_checkpoint(last, map_location="cpu"), datamodule=dm)[0]
    v2 = trainer.validate(CloudLiftLit.load_from_checkpoint(last, map_location="cpu"), datamodule=dm)[0]
    assert v1 == v2
    # the portable weights and the Lightning checkpoint load into the same network
    a, b = load(run / "cloudlift_weights.pt").eval(), load(last).eval()
    x, d = torch.rand(1, 2, 5, 64, 64), torch.tensor([[0.0, 30.0]])
    assert torch.equal(a(x, d)[0], b(x, d)[0])


def test_two_epoch_smoke_cpu(tmp_path, source):
    _smoke(tmp_path, source, "cpu", "32-true")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_two_epoch_smoke_cuda_amp(tmp_path, source):
    _smoke(tmp_path, source, "cuda", "16-mixed")
