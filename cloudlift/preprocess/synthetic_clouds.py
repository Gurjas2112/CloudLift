"""OpenSimplex cloud + shadow compositing for training pairs and the synthetic fixture.

Clouds are an fBm field thresholded at the quantile that gives exactly the requested cloud
fraction; opacity ramps from thin (edge haze) to thick (core). Shadows are a shifted,
blurred, darkened copy of the cloud opacity, applied only to clear pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from opensimplex import OpenSimplex

CLEAR, CLOUD, SHADOW = 0, 1, 2


def fbm_noise(
    shape: tuple[int, int],
    seed: int,
    base_cells: float = 4.0,
    octaves: int = 5,
    persistence: float = 0.5,
    max_grid: int = 160,
) -> np.ndarray:
    """Fractal OpenSimplex noise in [0, 1], shape ``(H, W)``.

    Each octave is sampled on a coarse grid (at most ``max_grid`` per side) and upsampled
    with bicubic interpolation, which keeps generation fast for 1024-px scenes.
    """
    h, w = shape
    gen = OpenSimplex(seed=int(seed) & 0x7FFFFFFF)
    total = np.zeros(shape, np.float64)
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        cells = base_cells * (2**o)
        gh = int(min(max_grid, max(8, round(cells * 8 * h / max(h, w)))))
        gw = int(min(max_grid, max(8, round(cells * 8 * w / max(h, w)))))
        ys = np.linspace(0, cells * h / max(h, w), gh) + 101.7 * o
        xs = np.linspace(0, cells * w / max(h, w), gw) + 57.3 * o
        layer = gen.noise2array(xs, ys)  # (gh, gw)
        total += amp * cv2.resize(layer.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        norm += amp
        amp *= persistence
    total /= norm
    lo, hi = total.min(), total.max()
    return ((total - lo) / (hi - lo + 1e-12)).astype(np.float32)


@dataclass
class CloudResult:
    cloudy: np.ndarray  # [C, H, W] float32
    mask: np.ndarray  # [H, W] uint8: 0 clear, 1 cloud, 2 shadow
    clean: np.ndarray  # [C, H, W] float32 (input, unchanged)
    alpha: np.ndarray  # [H, W] float32 cloud opacity in [0, 1]

    @property
    def cloud_fraction(self) -> float:
        return float((self.mask == CLOUD).mean())


def make_clouds(
    clean: np.ndarray,
    fraction: float,
    seed: int,
    *,
    thickness: float = 1.0,
    softness: float = 0.08,
    shadow: bool = True,
    shadow_offset: tuple[int, int] | None = None,
    shadow_strength: float = 0.55,
    cloud_color: tuple[float, ...] | None = None,
) -> CloudResult:
    """Composite clouds (and shadows) over ``clean`` ``[C, H, W]`` reflectance in [0, 1]."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    c, h, w = clean.shape
    rng = np.random.default_rng(seed)
    field = fbm_noise((h, w), seed=seed, base_cells=float(rng.uniform(2.5, 5.0)))

    if fraction <= 0.0:
        alpha = np.zeros((h, w), np.float32)
        cloud = np.zeros((h, w), bool)
    else:
        t = float(np.quantile(field, 1.0 - fraction))
        cloud = field > t
        ramp = np.clip((field - t) / softness, 0.0, 1.0)
        alpha = np.where(cloud, (0.15 + 0.85 * ramp) * thickness, 0.0).astype(np.float32)

    if cloud_color is None:
        cloud_color = (0.82, 0.84, 0.78)[:c] + (0.8,) * max(0, c - 3)
    color = np.asarray(cloud_color, np.float32)[:, None, None]
    texture = 0.04 * (fbm_noise((h, w), seed=seed + 7919, base_cells=24.0, octaves=3) - 0.5)
    cloud_rgb = np.clip(color + texture[None], 0.0, 1.0)

    shadow_alpha = np.zeros((h, w), np.float32)
    if shadow and fraction > 0.0:
        if shadow_offset is None:
            mag = max(4, int(0.03 * min(h, w)))
            shadow_offset = (int(rng.integers(mag // 2, mag + 1)), int(rng.integers(-mag, -mag // 2 + 1)))
        dy, dx = shadow_offset
        m = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
        shifted = np.asarray(
            cv2.warpAffine(alpha, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0.0), np.float32
        )
        k = max(3, (min(h, w) // 64) | 1)
        shadow_alpha = np.asarray(cv2.GaussianBlur(shifted, (k, k), 0), np.float32) * (~cloud)

    shaded = clean * (1.0 - shadow_strength * shadow_alpha)[None]
    cloudy = (shaded * (1.0 - alpha)[None] + cloud_rgb * alpha[None]).astype(np.float32)

    mask = np.zeros((h, w), np.uint8)
    mask[(shadow_alpha > 0.15) & ~cloud] = SHADOW
    mask[cloud] = CLOUD
    return CloudResult(cloudy=cloudy, mask=mask, clean=clean.astype(np.float32), alpha=alpha)
