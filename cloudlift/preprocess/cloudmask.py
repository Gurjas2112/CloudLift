"""Cloud and cloud-shadow detection for 3-band LISS-IV (Green, Red, NIR; no blue band).

``method="omnicloudmask"`` uses OmniCloudMask (Red/Green/NIR model) when installed.
``method="baseline"`` is the in-repo deterministic fallback: spectral thresholds, made
multi-temporal when a cloud-free reference is available (brightening = cloud, NIR
darkening near clouds = shadow), followed by morphological clean-up and a safety buffer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

log = logging.getLogger(__name__)
G, R, NIR = 0, 1, 2


@dataclass
class MaskResult:
    cloud: np.ndarray  # [H, W] bool
    shadow: np.ndarray  # [H, W] bool
    method: str

    @property
    def combined(self) -> np.ndarray:
        return self.cloud | self.shadow

    @property
    def cloud_fraction(self) -> float:
        return float(self.cloud.mean())

    @property
    def obstructed_fraction(self) -> float:
        return float(self.combined.mean())

    def as_uint8(self) -> np.ndarray:
        m = np.zeros(self.cloud.shape, np.uint8)
        m[self.shadow] = 2
        m[self.cloud] = 1
        return m


def omnicloudmask_available() -> bool:
    try:
        import omnicloudmask  # noqa: F401
    except Exception:
        return False
    return True


def _morph(mask: np.ndarray, open_px: int, dilate_px: int) -> np.ndarray:
    m = mask.astype(np.uint8)
    if open_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_px + 1, 2 * open_px + 1))
        m = np.asarray(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), np.uint8)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        m = np.asarray(cv2.dilate(m, k), np.uint8)
    return m.astype(bool)


def baseline_mask(
    optical: np.ndarray,
    reference: np.ndarray | None = None,
    *,
    cloud_abs: float = 0.2,
    cloud_delta: float = 0.03,
    shadow_ratio: float = 0.8,
    open_px: int = 1,
    dilate_px: int = 3,
) -> MaskResult:
    """Deterministic threshold + morphology cloud/shadow mask over reflectance in [0, 1]."""
    vis_min = np.minimum(optical[G], optical[R])
    cloud = vis_min > cloud_abs
    if reference is not None:
        ref_vis = np.minimum(reference[G], reference[R])
        cloud |= (vis_min - ref_vis) > cloud_delta
    cloud = _morph(cloud, open_px, dilate_px)

    near = _morph(cloud, 0, max(8, min(optical.shape[1:]) // 25))
    if reference is not None:
        darker = (optical[NIR] < shadow_ratio * reference[NIR]) & (reference[NIR] > 0.08)
        darker &= (optical[G] + optical[R]) < (reference[G] + reference[R])
    else:
        darker = (optical[NIR] < 0.12) & (optical[G] < 0.06) & (optical[R] < 0.06)
    shadow = _morph(darker & near & ~cloud, open_px, dilate_px) & ~cloud
    return MaskResult(
        cloud=cloud, shadow=shadow, method="baseline-multitemporal" if reference is not None else "baseline"
    )


def omnicloudmask_mask(optical: np.ndarray, dilate_px: int = 3) -> MaskResult:
    from omnicloudmask import predict_from_array

    rgn = np.stack([optical[R], optical[G], optical[NIR]]).astype(np.float32)
    pred = np.asarray(predict_from_array(rgn)).squeeze()
    cloud = _morph((pred == 1) | (pred == 2), 0, dilate_px)  # thick | thin
    shadow = _morph(pred == 3, 0, dilate_px) & ~cloud
    return MaskResult(cloud=cloud, shadow=shadow, method="omnicloudmask")


def detect(
    optical: np.ndarray,
    reference: np.ndarray | None = None,
    method: Literal["auto", "omnicloudmask", "baseline"] = "auto",
) -> MaskResult:
    if method in ("auto", "omnicloudmask") and omnicloudmask_available():
        try:
            return omnicloudmask_mask(optical)
        except Exception as e:  # model download / runtime failure -> deterministic fallback
            if method == "omnicloudmask":
                raise
            log.warning("OmniCloudMask failed (%s); using baseline mask", e)
    elif method == "omnicloudmask":
        raise RuntimeError("omnicloudmask not installed: pip install -r requirements-cloudmask.txt")
    return baseline_mask(optical, reference)
