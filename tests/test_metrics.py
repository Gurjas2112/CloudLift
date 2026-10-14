import json
import math

import numpy as np
import pytest

from cloudlift.eval import metrics as M
from cloudlift.eval.report import write_panels, write_report


def test_identical():
    x = np.random.default_rng(0).random((3, 32, 32))
    assert M.psnr(x, x) == float("inf")
    assert M.ssim(x, x) == pytest.approx(1.0)
    assert M.sam_deg(x, x) == 0.0
    assert M.rmse(x, x) == 0.0
    assert M.ndvi_agreement(x, x)["ndvi_r2"] == pytest.approx(1.0)


def test_hand_computed():
    a = np.zeros((3, 2, 2))
    b = np.full((3, 2, 2), 0.1)
    assert M.mse(a, b) == pytest.approx(0.01)
    assert M.psnr(a, b) == pytest.approx(20.0)
    assert M.rmse(a, b) == pytest.approx(0.1)
    p = np.zeros((3, 1, 1))
    q = np.zeros((3, 1, 1))
    p[0], q[1] = 1.0, 1.0
    assert M.sam_deg(p, q) == pytest.approx(90.0)
    p2 = np.array([1.0, 1.0, 0.0]).reshape(3, 1, 1)
    q2 = np.array([1.0, 0.0, 0.0]).reshape(3, 1, 1)
    assert M.sam_deg(p2, q2) == pytest.approx(45.0)
    arr = np.array([0.0, 0.1, 0.3]).reshape(3, 1, 1)  # R=0.1, NIR=0.3 -> NDVI 0.5
    assert M.ndvi(arr)[0, 0] == pytest.approx(0.5, abs=1e-5)


def test_masked_and_empty_mask():
    a = np.zeros((3, 4, 4))
    b = a.copy()
    b[:, :2] = 0.2
    m = np.zeros((4, 4), bool)
    m[2:] = True
    assert M.psnr(a, b, m) == float("inf")
    assert M.rmse(a, b, ~m) == pytest.approx(0.2)
    assert math.isnan(M.psnr(a, b, np.zeros((4, 4), bool)))


def test_calibration():
    rng = np.random.default_rng(1)
    t = rng.random((3, 64, 64))
    s = np.full_like(t, 0.05)
    good = t + rng.normal(0, 0.05, t.shape)
    bad = t + rng.normal(0, 0.25, t.shape)
    assert M.calibration(good, s, t)["ece"] < 0.02
    assert M.calibration(bad, s, t)["ece"] > 0.2


def _strict(c: str) -> None:
    raise AssertionError(f"non-strict JSON constant {c}")


def test_report_files(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.random((3, 64, 64)).astype(np.float32)
    mask = np.zeros((64, 64), bool)
    mask[10:30] = True
    m = M.full_reference(x, x, mask, np.full_like(x, 0.01))
    panel = write_panels(tmp_path / "p.png", x, x, x, np.zeros((64, 64)), mask)
    flat = {k: v for k, v in m.items() if not isinstance(v, dict)}
    j, md = write_report(tmp_path, {"run_id": "t", "scenes": {"s": flat}}, [panel])
    data = json.loads(j.read_text(encoding="utf-8"), parse_constant=_strict)
    assert data["scenes"]["s"]["psnr"] == "inf"
    assert "| s | sam_deg | 0.0000 |" in md.read_text(encoding="utf-8")
    assert panel.stat().st_size > 1000


def test_reference_consistency_mode():
    x = np.random.default_rng(0).random((3, 16, 16))
    m = np.ones((16, 16), bool)
    out = M.reference_consistency(x, x, m)
    assert out["ndvi_r2_vs_reference"] == pytest.approx(1.0)
