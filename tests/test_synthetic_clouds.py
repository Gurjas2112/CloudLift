import numpy as np
import pytest

from cloudlift.preprocess.synthetic_clouds import CLOUD, SHADOW, make_clouds


@pytest.fixture(scope="module")
def clean():
    rng = np.random.default_rng(0)
    return np.clip(rng.normal(0.15, 0.05, (3, 256, 256)), 0, 1).astype(np.float32)


def test_reproducible(clean):
    a = make_clouds(clean, 0.4, seed=5)
    b = make_clouds(clean, 0.4, seed=5)
    c = make_clouds(clean, 0.4, seed=6)
    assert np.array_equal(a.cloudy, b.cloudy) and np.array_equal(a.mask, b.mask)
    assert not np.array_equal(a.mask, c.mask)


@pytest.mark.parametrize("frac", [0.05, 0.2, 0.35, 0.5, 0.8])
def test_fraction_within_5pct(clean, frac):
    r = make_clouds(clean, frac, seed=11)
    assert abs(r.cloud_fraction - frac) <= 0.05


def test_shadow_and_range(clean):
    r = make_clouds(clean, 0.3, seed=2)
    assert (r.mask == SHADOW).any() and (r.mask == CLOUD).any()
    assert r.cloudy.min() >= 0 and r.cloudy.max() <= 1
    clear = r.mask == 0
    assert np.allclose(r.cloudy[:, clear], clean[:, clear], atol=0.06)
    assert np.array_equal(r.clean, clean)


def test_zero_fraction(clean):
    r = make_clouds(clean, 0.0, seed=1)
    assert r.cloud_fraction == 0 and np.array_equal(r.cloudy, clean)
