import numpy as np
import pytest

from cloudlift.preprocess.tiling import make_grid, tile, untile


@pytest.mark.parametrize("shape", [(3, 512, 512), (3, 700, 333), (2, 100, 90), (1, 256, 256)])
def test_roundtrip_exact(shape):
    arr = np.random.default_rng(0).random(shape, dtype=np.float32)
    tiles, grid = tile(arr, 256, 32)
    assert tiles.shape[1:] == (shape[0], 256, 256)
    assert np.abs(untile(tiles, grid, blend=False) - arr).max() < 1e-6
    assert np.abs(untile(tiles, grid, blend=True) - arr).max() < 1e-6


def test_grid_covers_edges():
    g = make_grid(700, 333, 256, 32)
    ys = sorted({y for y, _ in g.positions})
    xs = sorted({x for _, x in g.positions})
    assert ys[0] == 0 and ys[-1] == 700 - 256
    assert xs[0] == 0 and xs[-1] == 333 - 256


def test_feathering_hides_seams():
    arr = np.zeros((1, 480, 480), np.float32)
    tiles, grid = tile(arr, 256, 32)
    tiles = tiles + np.arange(len(tiles), dtype=np.float32)[:, None, None, None]  # per-tile offsets
    hard = untile(tiles, grid, blend=False)[0]
    soft = untile(tiles, grid, blend=True)[0]
    jump = lambda a: max(np.abs(np.diff(a, axis=0)).max(), np.abs(np.diff(a, axis=1)).max())  # noqa: E731
    assert jump(soft) < 0.25 * jump(hard)
