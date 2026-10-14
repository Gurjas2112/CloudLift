"""Load UnCRtainTS (vendored, unmodified) and expose ``forward(x, dates) -> (recon, sigma)``.

Weights resolution:

* upstream UnCRtainTS checkpoint (``{"state_dict_G": ...}``, SEN12MS-CR-TS) -> full config,
  strict load, then 3-band adapter surgery;
* CloudLift checkpoint (``{"cloudlift": 1, "size": ..., "state_dict": ...}``) or a Lightning
  ``.ckpt`` produced by ``cloudlift train`` -> loaded as-is;
* ``None`` -> tiny random-init network for tests, with a loud NOT TRAINED warning. Every
  output produced with it carries ``trained=false`` in its metadata.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from cloudlift.config import REPO_ROOT
from cloudlift.model.adapter import N_IN, N_OPT, adapt_input_conv, adapt_output_head, set_band_layout

log = logging.getLogger(__name__)
UNCRTAINTS_ROOT = REPO_ROOT / "third_party" / "UnCRtainTS" / "model"

# Paper configuration (UnCRtainTS, Ebel et al. 2023) and a small one for tests / CPU demos.
FULL_CFG: dict[str, Any] = dict(
    encoder_widths=[128], decoder_widths=[128, 128, 128, 128, 128], n_head=16, d_model=256, d_k=4
)
TINY_CFG: dict[str, Any] = dict(encoder_widths=[32], decoder_widths=[32, 32], n_head=4, d_model=64, d_k=4)
COMMON_CFG: dict[str, Any] = dict(
    out_nonlin_mean=True,
    out_nonlin_var="softplus",
    agg_mode="att_group",
    encoder_norm="group",
    decoder_norm="batch",
    pad_value=0,
    padding_mode="reflect",
    positional_encoding=True,
    scale_by=1.0,
    separate_out=False,
    use_v=False,
    block_type="mbconv",
    is_mono=False,
)


def _import_uncrtaints() -> ModuleType:
    """Import the vendored backbone without installing the upstream repo's pinned requirements."""
    if not (UNCRTAINTS_ROOT / "src" / "backbones" / "uncrtaints.py").exists():
        raise RuntimeError(
            f"UnCRtainTS submodule missing at {UNCRTAINTS_ROOT}; run `git submodule update --init`"
        )
    root = str(UNCRTAINTS_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module("src.backbones.uncrtaints")


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class CloudLiftNet(nn.Module):
    """UnCRtainTS adapted to LISS-IV G/R/NIR + EOS-04 2-pol SAR."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        size: str,
        covmode: str,
        trained: bool,
        weights_hash: str,
        weights_source: str,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.size = size
        self.covmode = covmode
        self.trained = trained
        self.weights_hash = weights_hash
        self.weights_source = weights_source
        self._ckpt_enabled = False

    def enable_gradient_checkpointing(self) -> None:
        """Recompute MBConv blocks in backward to fit 256-px tiles in 6 GB."""
        if self._ckpt_enabled:
            return
        blocks = [*cast(nn.ModuleList, self.backbone.in_block), *cast(nn.ModuleList, self.backbone.out_block)]
        for block in blocks:
            fwd = block.forward

            def wrapped(x: torch.Tensor, _f: Any = fwd) -> torch.Tensor:
                if torch.is_grad_enabled() and x.requires_grad:
                    return checkpoint(_f, x, use_reentrant=False)
                return _f(x)

            block.forward = wrapped
        self._ckpt_enabled = True

    def forward(self, x: torch.Tensor, dates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``x [B,T,5,H,W]``, ``dates [B,T]`` -> ``recon [B,3,H,W]``, ``sigma [B,3|1,H,W]``."""
        if x.ndim != 5 or x.shape[2] != N_IN:
            raise ValueError(f"expected x [B,T,{N_IN},H,W], got {tuple(x.shape)}")
        # upstream PositionalEncoder keeps ``denom`` as a plain attribute moved only on first call
        for m in self.backbone.modules():
            denom = getattr(m, "denom", None)
            if isinstance(denom, torch.Tensor) and denom.device != x.device:
                m.denom = denom.to(x.device)
        out = self.backbone(x, batch_positions=dates)[:, 0]  # [B, 3 + n_var, H, W]
        mean = out[:, :N_OPT]
        var = out[:, N_OPT:].float().clamp_min(1e-8)
        return mean, torch.sqrt(var).to(mean.dtype)

    def card(self) -> dict[str, Any]:
        return {
            "architecture": "UnCRtainTS",
            "size": self.size,
            "covmode": self.covmode,
            "trained": self.trained,
            "weights_hash": self.weights_hash,
            "weights_source": self.weights_source,
            "params": sum(p.numel() for p in self.parameters()),
        }


def build_backbone(size: str, input_dim: int, out_dim: int, covmode: str = "diag") -> nn.Module:
    U = _import_uncrtaints()
    cfg = {**COMMON_CFG, **(FULL_CFG if size == "full" else TINY_CFG)}
    return U.UNCRTAINTS(input_dim=input_dim, out_conv=[out_dim], covmode=covmode, **cfg)


def _n_out(covmode: str) -> int:
    return N_OPT + (N_OPT if covmode == "diag" else 1)


def _strip_prefix(sd: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in sd.items()}


def load(
    weights_path: str | Path | None = None,
    *,
    size: str = "tiny",
    covmode: str = "diag",
    gradient_checkpointing: bool = False,
    seed: int = 0,
) -> CloudLiftNet:
    """Build the network; see module docstring for how ``weights_path`` is interpreted."""
    if weights_path is None:
        torch.manual_seed(seed)
        bb = build_backbone(size, N_IN, _n_out(covmode), covmode)
        set_band_layout(bb, covmode)
        log.warning(
            "=" * 72 + "\n  CloudLift model is NOT TRAINED (random-init %s UnCRtainTS).\n"
            "  Outputs are for pipeline testing only. Supply weights via paths.weights.\n" + "=" * 72,
            size,
        )
        net = CloudLiftNet(
            bb,
            size=size,
            covmode=covmode,
            trained=False,
            weights_hash=f"untrained-{size}-seed{seed}",
            weights_source="random-init",
        )
    else:
        p = Path(weights_path)
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        whash = file_hash(p)
        if isinstance(ckpt, dict) and "state_dict_G" in ckpt:  # upstream UnCRtainTS
            sd = _strip_prefix(ckpt["state_dict_G"], "module.")
            in_dim = (
                int(sd["in_conv.conv.conv.0.weight"].shape[1]) if "in_conv.conv.conv.0.weight" in sd else 15
            )
            out_key = next(k for k in sd if k.startswith("out_conv") and k.endswith("weight"))
            out_dim = int(sd[out_key].shape[0])
            bb = build_backbone("full", in_dim, out_dim, covmode)
            bb.load_state_dict(sd, strict=True)
            adapt_input_conv(bb)
            adapt_output_head(bb, covmode)
            net = CloudLiftNet(
                bb,
                size="full",
                covmode=covmode,
                trained=True,
                weights_hash=whash,
                weights_source=f"uncrtaints-pretrained:{p.name}",
            )
        else:
            if isinstance(ckpt, dict) and "cloudlift" in ckpt:
                size = str(ckpt.get("size", size))
                covmode = str(ckpt.get("covmode", covmode))
                sd = ckpt["state_dict"]
                source = str(ckpt.get("trained_on", "cloudlift"))
            elif isinstance(ckpt, dict) and "state_dict" in ckpt:  # Lightning checkpoint
                hp = ckpt.get("hyper_parameters", {})
                size = str(hp.get("size", size))
                covmode = str(hp.get("covmode", covmode))
                sd = {
                    k[len("net.backbone.") :]: v
                    for k, v in ckpt["state_dict"].items()
                    if k.startswith("net.backbone.")
                }
                source = "lightning-ckpt"
            else:
                raise ValueError(f"unrecognised checkpoint format: {p}")
            sd = _strip_prefix(sd, "backbone.")
            bb = build_backbone(size, N_IN, _n_out(covmode), covmode)
            set_band_layout(bb, covmode)
            bb.load_state_dict(sd, strict=True)
            net = CloudLiftNet(
                bb,
                size=size,
                covmode=covmode,
                trained=True,
                weights_hash=whash,
                weights_source=f"{source}:{p.name}",
            )
    if gradient_checkpointing:
        net.enable_gradient_checkpointing()
    return net


def save_cloudlift_weights(net: CloudLiftNet, path: str | Path, trained_on: str) -> Path:
    """Save a portable CloudLift checkpoint (backbone weights + layout)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cloudlift": 1,
            "size": net.size,
            "covmode": net.covmode,
            "trained_on": trained_on,
            "state_dict": net.backbone.state_dict(),
        },
        path,
    )
    return path
