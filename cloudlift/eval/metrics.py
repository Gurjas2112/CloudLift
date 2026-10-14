"""Evaluation metrics over ``[C,H,W]`` reflectance arrays in [0, 1] (band order Green, Red, NIR).

Region-restricted metrics take a boolean ``mask`` ``[H,W]``; CloudLift reports them over the
cloud/shadow pixels, which is where reconstruction actually happens.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import norm
from skimage.metrics import structural_similarity

RED, NIR = 1, 2


def _sel(arr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """``[C,H,W]`` -> ``[C,N]`` of masked pixels."""
    a = arr.reshape(arr.shape[0], -1).astype(np.float64)
    return a if mask is None else a[:, mask.reshape(-1)]


def _empty(mask: np.ndarray | None) -> bool:
    return mask is not None and not mask.any()


def mse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    if _empty(mask):
        return float("nan")
    d = _sel(pred, mask) - _sel(target, mask)
    return float(np.mean(d * d))


def rmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    return math.sqrt(mse(pred, target, mask)) if not _empty(mask) else float("nan")


def psnr(
    pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None, data_range: float = 1.0
) -> float:
    """PSNR in dB; identical inputs return ``inf``."""
    m = mse(pred, target, mask)
    if math.isnan(m):
        return m
    if m == 0.0:
        return float("inf")
    return float(10.0 * math.log10(data_range**2 / m))


def ssim_bands(
    pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None, data_range: float = 1.0
) -> list[float]:
    """Per-band SSIM (Gaussian window); with ``mask`` the SSIM map is averaged over masked pixels."""
    if _empty(mask):
        return [float("nan")] * pred.shape[0]
    out = []
    for p, t in zip(pred.astype(np.float64), target.astype(np.float64), strict=True):
        if mask is None:
            out.append(float(structural_similarity(p, t, data_range=data_range, gaussian_weights=True)))
        else:
            _, smap = structural_similarity(p, t, data_range=data_range, gaussian_weights=True, full=True)
            out.append(float(smap[mask].mean()))
    return out


def ssim(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    return float(np.mean(ssim_bands(pred, target, mask)))


def sam_map_deg(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-pixel spectral angle in degrees (exactly 0 for identical spectra)."""
    p, t = pred.astype(np.float64), target.astype(np.float64)
    dot = (p * t).sum(0)
    cross2 = np.clip((p * p).sum(0) * (t * t).sum(0) - dot**2, 0.0, None)
    return np.degrees(np.arctan2(np.sqrt(cross2), dot))


def sam_deg(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    if _empty(mask):
        return float("nan")
    m = sam_map_deg(pred, target)
    return float(m.mean() if mask is None else m[mask].mean())


def ndvi(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    r, n = arr[RED].astype(np.float64), arr[NIR].astype(np.float64)
    return (n - r) / (n + r + eps)


def ndvi_agreement(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    """R² and RMSE between reconstructed and reference NDVI over ``mask``."""
    if _empty(mask):
        return {"ndvi_r2": float("nan"), "ndvi_rmse": float("nan")}
    a, b = ndvi(pred), ndvi(target)
    if mask is not None:
        a, b = a[mask], b[mask]
    a, b = a.ravel(), b.ravel()
    ss_res = float(np.sum((b - a) ** 2))
    ss_tot = float(np.sum((b - b.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else (1.0 if ss_res == 0 else float("nan"))
    return {"ndvi_r2": r2, "ndvi_rmse": math.sqrt(ss_res / a.size)}


def calibration(
    pred: np.ndarray,
    sigma: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
    n_bins: int = 10,
) -> dict[str, float]:
    """Uncertainty calibration of a Gaussian predictive distribution.

    * ``ece``: coverage expected-calibration-error, mean over p in {0.1..0.9} of
      |observed fraction with |err| <= z_p·σ  -  p|.
    * ``uce``: binned |RMSE - mean σ| weighted by bin size (UnCRtainTS paper style).
    """
    if _empty(mask):
        return {"ece": float("nan"), "uce": float("nan")}
    s = sigma if sigma.shape[0] == pred.shape[0] else np.broadcast_to(sigma, pred.shape)
    err = np.abs(_sel(pred, mask) - _sel(target, mask)).ravel()
    sg = np.maximum(_sel(np.asarray(s), mask).ravel(), 1e-12)
    ps = np.linspace(0.1, 0.9, 9)
    obs = np.array([np.mean(err <= norm.ppf(0.5 + p / 2) * sg) for p in ps])
    ece = float(np.mean(np.abs(obs - ps)))
    edges = np.quantile(sg, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(edges, sg, side="right") - 1, 0, n_bins - 1)
    uce = 0.0
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            uce += sel.mean() * abs(math.sqrt(np.mean(err[sel] ** 2)) - float(np.mean(sg[sel])))
    return {"ece": ece, "uce": float(uce)}


def full_reference(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    sigma: np.ndarray | None = None,
) -> dict[str, Any]:
    """All full-reference metrics, over the whole tile and over the cloud/shadow mask."""
    out: dict[str, Any] = {
        "psnr": psnr(pred, target),
        "ssim": ssim(pred, target),
        "sam_deg": sam_deg(pred, target),
        "rmse": rmse(pred, target),
        "psnr_masked": psnr(pred, target, mask),
        "ssim_masked": ssim(pred, target, mask),
        "ssim_bands_masked": dict(zip(("Green", "Red", "NIR"), ssim_bands(pred, target, mask), strict=True)),
        "sam_deg_masked": sam_deg(pred, target, mask),
        "rmse_masked": rmse(pred, target, mask),
        **ndvi_agreement(pred, target, mask),
        "masked_fraction": float(mask.mean()),
    }
    if sigma is not None:
        out.update(calibration(pred, sigma, target, mask))
    return out


def reference_consistency(pred: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """Real-scene mode (no cloud-free truth): agreement with the temporal reference only.

    This is NOT a ground-truth score: land cover may have changed between passes.
    """
    nd = ndvi_agreement(pred, reference, mask)
    return {
        "ndvi_r2_vs_reference": nd["ndvi_r2"],
        "ndvi_rmse_vs_reference": nd["ndvi_rmse"],
        "sam_deg_vs_reference_masked": sam_deg(pred, reference, mask),
        "masked_fraction": float(mask.mean()),
    }
