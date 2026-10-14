import numpy as np
import pytest

from cloudlift.io.radiometry import band_indexes, optical_to_unit, resolve_scale, select_bands


@pytest.mark.parametrize(
    ("arr", "expected"),
    [
        (np.array([0, 900], np.uint16), 1023.0),  # LISS-IV 10-bit DN
        (np.array([0, 4200], np.uint16), 10000.0),  # L2A-style reflectance x 10000
        (np.array([0, 40000], np.uint16), 65535.0),
        (np.array([0.0, 0.6], np.float32), 1.0),  # already reflectance
        (np.array([0.0, 55.0], np.float32), 100.0),  # percent
    ],
)
def test_auto_scale(arr, expected):
    assert resolve_scale(arr, {}) == expected
    assert resolve_scale(arr, {"scale": "auto"}) == expected


def test_explicit_scale_wins():
    arr = np.array([0, 900], np.uint16)
    assert resolve_scale(arr, {"scale": 10000}) == 10000.0
    assert optical_to_unit(arr, {"scale": "1000"}).max() == pytest.approx(0.9)


def test_band_selection():
    arr = np.stack([np.full((2, 2), v, np.float32) for v in (10, 20, 30, 40)])  # B, G, R, NIR
    out = select_bands(arr, {"bands": "2,3,4"})
    assert out.shape == (3, 2, 2) and out[:, 0, 0].tolist() == [20, 30, 40]
    assert band_indexes(None) == (1, 2, 3)
    with pytest.raises(ValueError, match="out of range"):
        band_indexes({"bands": [1, 2, 5]}, 4)
    with pytest.raises(ValueError, match="3 indexes"):
        band_indexes({"bands": [1, 2]})
