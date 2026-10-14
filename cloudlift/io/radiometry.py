"""Bring optical and SAR inputs to the [0, 1] range the model works in."""

from __future__ import annotations

from typing import Any

import numpy as np

LISS4_DN_MAX = 1023.0  # LISS-IV is 10-bit radiometry
SAR_DB_MIN, SAR_DB_RANGE = -25.0, 25.0


def optical_to_unit(arr: np.ndarray, meta: dict[str, Any] | None = None) -> np.ndarray:
    """Integer DN -> divide by ``meta["scale"]`` (default 1023); floats assumed already reflectance."""
    meta = meta or {}
    a = arr.astype(np.float32)
    if np.issubdtype(arr.dtype, np.integer) or "scale" in meta:
        a = a / float(meta.get("scale", LISS4_DN_MAX))
    return np.clip(a, 0.0, 1.0)


def sar_to_unit(arr: np.ndarray, meta: dict[str, Any] | None = None) -> np.ndarray:
    """SAR backscatter -> ``(dB + 25) / 25`` clipped to [0, 1].

    ``meta["sar_units"]`` may be ``"db"``, ``"linear"`` or ``"unit"`` (already normalised, e.g.
    the synthetic fixture). Without it: negative values => dB, otherwise linear sigma0.
    """
    meta = meta or {}
    units = meta.get("sar_units")
    if units is None:
        units = "unit" if meta.get("synthetic") else ("db" if np.nanmin(arr) < 0 else "linear")
    a = arr.astype(np.float32)
    if units == "unit":
        return np.clip(a, 0.0, 1.0)
    if units == "linear":
        a = (10.0 * np.log10(np.maximum(a, 1e-6))).astype(np.float32)
    return np.clip((a - SAR_DB_MIN) / SAR_DB_RANGE, 0.0, 1.0)
