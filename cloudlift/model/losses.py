"""Differentiable losses: L1, SSIM, spectral angle (SAM), heteroscedastic Gaussian NLL."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from cloudlift.config import LossCfg


def _gauss_kernel(size: int = 11, sigma: float = 1.5, device: torch.device | None = None) -> torch.Tensor:
    ax = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-(ax**2) / (2 * sigma**2))
    g = g / g.sum()
    return torch.outer(g, g)


def ssim_map(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0, size: int = 11) -> torch.Tensor:
    """Per-pixel SSIM ``[B,C,H,W]`` (Gaussian window, Wang et al. 2004 constants)."""
    c = x.shape[1]
    k = _gauss_kernel(size, device=x.device).to(x.dtype)[None, None].repeat(c, 1, 1, 1)
    pad = size // 2

    def filt(t: torch.Tensor) -> torch.Tensor:
        return F.conv2d(F.pad(t, (pad, pad, pad, pad), mode="reflect"), k, groups=c)

    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mx, my = filt(x), filt(y)
    sxx = filt(x * x) - mx**2
    syy = filt(y * y) - my**2
    sxy = filt(x * y) - mx * my
    return ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx**2 + my**2 + c1) * (sxx + syy + c2))


def ssim_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 1.0 - ssim_map(x.float(), y.float()).mean()


def spectral_angle(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-pixel spectral angle in radians, ``[B,H,W]``.

    Uses ``atan2(||x||·||y||·sin, x·y)`` computed via the Lagrange identity, which is stable
    near 0 (unlike ``acos``) and exactly invariant to positive scaling of either input.
    """
    x, y = x.float(), y.float()
    dot = (x * y).sum(1)
    nx2, ny2 = (x * x).sum(1), (y * y).sum(1)
    cross2 = (nx2 * ny2 - dot**2).clamp_min(0.0)
    return torch.atan2(torch.sqrt(cross2 + eps), dot)


def sam_loss(x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    a = spectral_angle(x, y)
    if weight is None:
        return a.mean()
    return (a * weight).sum() / weight.sum().clamp_min(1e-8)


def gaussian_nll(mean: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Heteroscedastic NLL ``0.5·(log σ² + (y-μ)²/σ²)`` (constant dropped)."""
    var = sigma.float().pow(2).clamp_min(1e-6)
    return 0.5 * (torch.log(var) + (target.float() - mean.float()) ** 2 / var).mean()


class CombinedLoss(nn.Module):
    """``w_l1·L1 + w_ssim·(1-SSIM) + w_sam·SAM + w_nll·NLL`` with cloud-pixel up-weighting in L1/SAM."""

    def __init__(self, cfg: LossCfg) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        recon: torch.Tensor,
        sigma: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cfg = self.cfg
        recon32, target32 = recon.float(), target.float()
        w = torch.ones_like(target32[:, :1])
        if mask is not None:
            w = 1.0 + (cfg.cloud_weight - 1.0) * mask.float().reshape(w.shape)
        l1 = ((recon32 - target32).abs() * w).sum() / (w.sum() * target32.shape[1])
        parts = {"l1": l1}
        total = cfg.l1 * l1
        if cfg.ssim:
            parts["ssim"] = ssim_loss(recon32, target32)
            total = total + cfg.ssim * parts["ssim"]
        if cfg.sam:
            parts["sam"] = sam_loss(recon32, target32, w[:, 0])
            total = total + cfg.sam * parts["sam"]
        if cfg.nll:
            parts["nll"] = gaussian_nll(recon32, sigma, target32)
            total = total + cfg.nll * parts["nll"]
        parts["total"] = total
        return total, parts
