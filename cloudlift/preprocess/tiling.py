"""Tile/untile with overlap and feathered blending (seam-free reconstruction)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TileGrid:
    height: int  # original image size
    width: int
    padded_h: int
    padded_w: int
    tile: int
    overlap: int
    positions: tuple[tuple[int, int], ...]  # (y, x) of each tile's top-left in the padded image

    def __len__(self) -> int:
        return len(self.positions)


def _starts(n: int, tile: int, stride: int) -> list[int]:
    if n <= tile:
        return [0]
    s = list(range(0, n - tile + 1, stride))
    if s[-1] != n - tile:
        s.append(n - tile)
    return s


def make_grid(height: int, width: int, tile: int = 256, overlap: int = 32) -> TileGrid:
    if not 0 <= overlap < tile:
        raise ValueError("overlap must be in [0, tile)")
    ph, pw = max(height, tile), max(width, tile)
    stride = tile - overlap
    pos = tuple((y, x) for y in _starts(ph, tile, stride) for x in _starts(pw, tile, stride))
    return TileGrid(height, width, ph, pw, tile, overlap, pos)


def pad_to_grid(arr: np.ndarray, grid: TileGrid) -> np.ndarray:
    """Reflect-pad ``[C,H,W]`` / ``[H,W]`` up to the grid's padded size (only when smaller than a tile)."""
    ph, pw = grid.padded_h - grid.height, grid.padded_w - grid.width
    if ph == 0 and pw == 0:
        return arr
    pad = [(0, ph), (0, pw)] if arr.ndim == 2 else [(0, 0), (0, ph), (0, pw)]
    if min(arr.shape[-2:]) > 1:
        return np.pad(arr, pad, mode="reflect")
    return np.pad(arr, pad, mode="edge")


def extract_tile(arr: np.ndarray, grid: TileGrid, i: int) -> np.ndarray:
    y, x = grid.positions[i]
    t = grid.tile
    return arr[..., y : y + t, x : x + t]


def tile(arr: np.ndarray, tile: int = 256, overlap: int = 32) -> tuple[np.ndarray, TileGrid]:
    """Split ``[C,H,W]`` into ``[N,C,tile,tile]``."""
    grid = make_grid(arr.shape[-2], arr.shape[-1], tile, overlap)
    padded = pad_to_grid(arr, grid)
    return np.stack([extract_tile(padded, grid, i) for i in range(len(grid))]), grid


def feather_weights(tile: int, overlap: int) -> np.ndarray:
    """2-D separable linear ramp over the overlap band; strictly positive everywhere."""
    w1 = np.ones(tile, np.float64)
    if overlap > 0:
        ramp = (np.arange(overlap, dtype=np.float64) + 0.5) / overlap
        w1[:overlap] = ramp
        w1[-overlap:] = ramp[::-1]
    return np.outer(w1, w1)


class Untiler:
    """Streaming accumulator: add tiles one at a time, then ``result()``."""

    def __init__(self, grid: TileGrid, channels: int, blend: bool = True) -> None:
        self.grid = grid
        self.acc = np.zeros((channels, grid.padded_h, grid.padded_w), np.float64)
        self.wsum = np.zeros((grid.padded_h, grid.padded_w), np.float64)
        self.w = feather_weights(grid.tile, grid.overlap) if blend else np.ones((grid.tile, grid.tile))

    def add(self, i: int, t: np.ndarray) -> None:
        y, x = self.grid.positions[i]
        s = self.grid.tile
        self.acc[:, y : y + s, x : x + s] += t.astype(np.float64) * self.w
        self.wsum[y : y + s, x : x + s] += self.w

    def result(self, dtype: type = np.float32) -> np.ndarray:
        out = self.acc / np.maximum(self.wsum, 1e-12)
        return out[:, : self.grid.height, : self.grid.width].astype(dtype)


def untile(tiles: np.ndarray, grid: TileGrid, blend: bool = True) -> np.ndarray:
    """Inverse of :func:`tile`. Identical overlapping tiles reproduce the input exactly."""
    u = Untiler(grid, tiles.shape[1], blend)
    for i, t in enumerate(tiles):
        u.add(i, t)
    return u.result(tiles.dtype.type)
