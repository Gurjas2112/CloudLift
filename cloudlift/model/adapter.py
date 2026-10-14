"""3-band LISS-IV + 2-pol SAR input adapter for UnCRtainTS.

Tensor contract (CloudLift)::

    x     [B, T, 5, H, W]   per time step: SAR_VV, SAR_VH, Green, Red, NIR   (all in [0, 1])
    dates [B, T]            days since the oldest observation (temporal positional encoding)
    out   mean [B, 3, H, W] and variance [B, 3, H, W]  (Green, Red, NIR)

Upstream UnCRtainTS (SEN12MS-CR-TS) uses 15 input channels = S1(2) ++ S2(13) and predicts
13 means + 13 diagonal variances. LISS-IV Green/Red/NIR correspond to S2 B3/B4/B8, i.e.
S2 indices 2/3/7. The surgery below keeps those pretrained filters and drops the rest.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import torch
from torch import nn

N_SAR = 2
N_OPT = 3
N_IN = N_SAR + N_OPT
S2_BANDS = 13
S2_IDX_FOR_LISS4 = (2, 3, 7)  # B3 (green), B4 (red), B8 (NIR)


def stack_inputs(
    cloudy: np.ndarray,
    sar: np.ndarray,
    references: Sequence[np.ndarray],
    input_t: int,
    ref_gap_days: Sequence[int] | int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``x [T,5,H,W]`` and ``dates [T]``: oldest reference first, current cloudy pass last.

    With fewer references than ``input_t - 1``, the oldest available reference is repeated
    (UnCRtainTS needs a fixed T; repetition adds no information but keeps the layout).
    """
    if input_t < 1:
        raise ValueError("input_t must be >= 1")
    refs = list(references)
    gaps = [ref_gap_days] * len(refs) if isinstance(ref_gap_days, int) else list(ref_gap_days)
    need = input_t - 1
    if need > 0 and not refs:
        refs, gaps = [cloudy], [0]
    while len(refs) < need:
        refs.insert(0, refs[0])
        gaps.insert(0, gaps[0])
    if need == 0:
        refs, gaps = [], []
    else:
        refs, gaps = refs[-need:], gaps[-need:]
    frames = [*refs, cloudy]
    x = np.stack([np.concatenate([sar, f], axis=0) for f in frames]).astype(np.float32)
    oldest = max(gaps) if gaps else 0
    dates = np.asarray([oldest - g for g in gaps] + [oldest], np.float32)
    return x, dates


def _first_conv(module: nn.Module) -> nn.Conv2d:
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            return m
    raise ValueError("no Conv2d found")


def _last_conv(module: nn.Module) -> nn.Conv2d:
    convs = [m for m in module.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise ValueError("no Conv2d found")
    return convs[-1]


def _clone_conv(conv: nn.Conv2d, cin: int, cout: int) -> nn.Conv2d:
    """New conv with ``conv``'s geometry but different channel counts."""
    return nn.Conv2d(
        cin,
        cout,
        cast(Any, conv.kernel_size),
        cast(Any, conv.stride),
        cast(Any, conv.padding),
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )


def _sub(model: nn.Module, name: str) -> nn.Module:
    return cast(nn.Module, getattr(model, name))


def _replace(parent: nn.Module, old: nn.Module, new: nn.Module) -> None:
    for name, child in parent.named_children():
        if child is old:
            setattr(parent, name, new)
            return
        _replace(child, old, new)


@torch.no_grad()
def adapt_input_conv(model: nn.Module) -> None:
    """Replace UnCRtainTS' first conv with a 5-channel one, re-using pretrained filters.

    15-ch (S1+S2) source: keep SAR columns 0-1 and S2 columns for B3/B4/B8.
    13-ch (S2 only) source: SAR columns are initialised to the mean of all pretrained columns.
    """
    conv = _first_conv(_sub(model, "in_conv"))
    cin = conv.in_channels
    if cin == N_IN:
        return
    new = _clone_conv(conv, N_IN, conv.out_channels)
    w = conv.weight
    if cin == N_SAR + S2_BANDS:
        sar_w = w[:, :N_SAR]
        opt_w = w[:, [N_SAR + i for i in S2_IDX_FOR_LISS4]]
    elif cin == S2_BANDS:
        sar_w = w.mean(dim=1, keepdim=True).repeat(1, N_SAR, 1, 1)
        opt_w = w[:, list(S2_IDX_FOR_LISS4)]
    else:
        raise ValueError(f"unsupported pretrained input width {cin}")
    new.weight.copy_(torch.cat([sar_w, opt_w], dim=1))
    if conv.bias is not None and new.bias is not None:
        new.bias.copy_(conv.bias)
    _replace(_sub(model, "in_conv"), conv, new)


@torch.no_grad()
def adapt_output_head(model: nn.Module, covmode: str = "diag") -> None:
    """Reduce the 13-band (+13 variance) head to Green/Red/NIR (+3 or +1 variance)."""
    conv = _last_conv(_sub(model, "out_conv"))
    cout = conv.out_channels
    n_var = N_OPT if covmode == "diag" else 1
    want = N_OPT + n_var
    if cout != want:
        if cout == 2 * S2_BANDS and covmode == "diag":
            rows = list(S2_IDX_FOR_LISS4) + [S2_BANDS + i for i in S2_IDX_FOR_LISS4]
        elif cout == S2_BANDS + 1 and covmode == "iso":
            rows = [*S2_IDX_FOR_LISS4, S2_BANDS]
        else:
            raise ValueError(f"cannot adapt head with {cout} outputs to covmode={covmode}")
        new = _clone_conv(conv, conv.in_channels, want)
        new.weight.copy_(conv.weight[rows])
        if conv.bias is not None and new.bias is not None:
            new.bias.copy_(conv.bias[rows])
        _replace(_sub(model, "out_conv"), conv, new)
    set_band_layout(model, covmode)


def set_band_layout(model: nn.Module, covmode: str = "diag") -> None:
    """UnCRtainTS slices mean/var with the module constant S2_BANDS=13; point it at 3 bands."""
    vars_idx = N_OPT + (N_OPT if covmode == "diag" else 1)
    setattr(model, "mean_idx", N_OPT)  # noqa: B010 - attributes of the vendored module
    setattr(model, "vars_idx", vars_idx)  # noqa: B010
    setattr(model, "out_dims", vars_idx)  # noqa: B010
