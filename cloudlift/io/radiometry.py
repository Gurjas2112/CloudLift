"""Bring optical and SAR inputs to the [0, 1] range the model works in.

Optical inputs can be any GeoTIFF that contains Green, Red and NIR bands:

* ``meta["bands"]``: 1-based indexes of (Green, Red, NIR) in the file, default ``[1, 2, 3]``
  (LISS-IV order). A 4-band Blue/Green/Red/NIR product would use ``[2, 3, 4]``.
* ``meta["scale"]``: divisor that maps stored values to reflectance, or ``"auto"`` (default).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

LISS4_DN_MAX = 1023.0  # LISS-IV is 10-bit radiometry
SAR_DB_MIN, SAR_DB_RANGE = -25.0, 25.0
DEFAULT_BANDS = (1, 2, 3)


def band_indexes(meta: dict[str, Any] | None, count: int | None = None) -> tuple[int, int, int]:
    """1-based (Green, Red, NIR) band indexes from ``meta["bands"]`` (list or ``"2,3,4"``)."""
    raw = (meta or {}).get("bands", DEFAULT_BANDS)
    if isinstance(raw, str):
        raw = [p for p in raw.replace(" ", "").split(",") if p]
    if not isinstance(raw, Sequence) or len(raw) != 3:
        raise ValueError(f"bands must list 3 indexes (Green, Red, NIR), got {raw!r}")
    try:
        idx = tuple(int(b) for b in raw)
    except (TypeError, ValueError) as e:
        raise ValueError(f"bands must be integers, got {raw!r}") from e
    if min(idx) < 1 or (count is not None and max(idx) > count):
        raise ValueError(f"band indexes {idx} out of range for a {count}-band file")
    return idx[0], idx[1], idx[2]


def select_bands(arr: np.ndarray, meta: dict[str, Any] | None) -> np.ndarray:
    """``[C,H,W]`` -> ``[3,H,W]`` in Green, Red, NIR order."""
    g, r, n = band_indexes(meta, arr.shape[0])
    return arr[[g - 1, r - 1, n - 1]]


def resolve_scale(arr: np.ndarray, meta: dict[str, Any] | None = None) -> float:
    """Divisor mapping stored values to [0, 1] reflectance.

    An explicit numeric ``meta["scale"]`` wins. ``"auto"`` (or no value) picks from the data range:
    integers <= 1023 -> 1023 (LISS-IV 10-bit), <= 10000 -> 10000 (common L2A scaling), else 65535;
    floats <= 1.5 are already reflectance, <= 100 are percent, else 10000.
    """
    s = (meta or {}).get("scale", "auto")
    if s not in (None, "", "auto"):
        return float(s)
    finite = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.floating) else arr
    hi = float(np.percentile(finite, 99.9)) if finite.size else 1.0
    if np.issubdtype(arr.dtype, np.integer):
        return LISS4_DN_MAX if hi <= LISS4_DN_MAX else (10000.0 if hi <= 10000 else 65535.0)
    if hi <= 1.5:
        return 1.0
    return 100.0 if hi <= 100 else 10000.0


def optical_to_unit(arr: np.ndarray, meta: dict[str, Any] | None = None) -> np.ndarray:
    """Divide by the resolved scale (see :func:`resolve_scale`) and clip to [0, 1]."""
    a = arr.astype(np.float32) / resolve_scale(arr, meta)
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
