"""Same input -> byte-identical reconstruction (golden SHA-256)."""

import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np
import pytest
import torch

from cloudlift import pipeline
from cloudlift.config import load_config
from cloudlift.determinism import set_determinism

GOLDEN = Path(__file__).parent / "golden_hashes.json"


def _env_key(device: str) -> str:
    return f"{platform.system()}-torch{torch.__version__}-numpy{np.__version__}-{device}"


def _run(scene, tmp_path, device, name):
    cfg = load_config(
        overrides={
            "device": device,
            "model": {"size": "tiny"},
            "seed": 0,
            "paths": {"results_root": str(tmp_path / name)},
        }
    )
    pipeline._NET_CACHE.clear()  # fresh weights init each time
    r = pipeline.run(scene, None, cfg)
    recon = np.load(r.out_dir / "recon.npy")
    return hashlib.sha256(recon.tobytes()).hexdigest(), r


def test_set_determinism_env():
    set_determinism(5)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
    a = torch.rand(3)
    set_determinism(5)
    assert torch.equal(a, torch.rand(3))


def test_config_hash_stable():
    a = load_config(overrides={"device": "cpu"})
    b = load_config(overrides={"device": "cpu", "paths": {"results_root": "elsewhere"}})
    c = load_config(overrides={"device": "cpu", "seed": 1})
    assert a.config_hash() == b.config_hash() != c.config_hash()


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
        ),
    ],
)
def test_byte_identical_runs(scene, tmp_path, device):
    h1, r1 = _run(scene, tmp_path, device, "a")
    h2, r2 = _run(scene, tmp_path, device, "b")
    assert h1 == h2 == r1.extra["recon_sha256"] == r2.extra["recon_sha256"]
    golden = json.loads(GOLDEN.read_text(encoding="utf-8")) if GOLDEN.exists() else {}
    key = _env_key(device)
    if os.environ.get("CLOUDLIFT_UPDATE_GOLDEN") == "1":
        golden[key] = h1
        GOLDEN.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif key in golden:
        assert h1 == golden[key], "reconstruction changed vs golden hash for this environment"
    else:
        pytest.skip(f"no golden hash recorded for {key} (run-to-run identity verified)")
