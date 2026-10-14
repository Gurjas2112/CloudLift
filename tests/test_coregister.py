import numpy as np
import pytest
from scipy.ndimage import shift as nd_shift

from cloudlift.io import geotiff
from cloudlift.preprocess import coregister


@pytest.mark.parametrize("true_shift", [(1.6, -0.8), (-2.3, 0.4), (0.3, 0.7)])
def test_known_shift_recovered(scene, true_shift):
    img, _ = geotiff.read(scene.clean)
    nir = img[2]
    moved = nd_shift(nir, true_shift, order=3, mode="nearest")
    est, n, _ = coregister.estimate_shift(nir, moved, tile=256)
    assert n > 0
    # estimate is the correction that registers ``moved`` back onto ``nir``
    assert abs(est[0] + true_shift[0]) < 0.25 and abs(est[1] + true_shift[1]) < 0.25


def test_refine_reference_on_fixture(scene):
    opt, _ = geotiff.read(scene.optical)
    ref, _ = geotiff.read(scene.reference)
    gt, _ = geotiff.read(scene.mask_gt)
    _, est = coregister.refine_reference(opt, ref, gt[0] == 0)
    # fixture reference was shifted by (1.6, -0.8) -> the correction is (-1.6, +0.8)
    assert abs(est.shift[0] + 1.6) < 0.25 and abs(est.shift[1] - 0.8) < 0.25
    assert est.residual_px < 0.25


def test_reproject_identity_and_partial(scene, tmp_path):
    meta = geotiff.read_meta(scene.optical)
    sar, valid = coregister.reproject_to_grid(scene.sar, meta)
    orig, _ = geotiff.read(scene.sar)
    assert valid.all()
    assert np.allclose(sar, orig, atol=1e-6)
    # SAR covering only the left half -> right half invalid
    half = orig[:, :, : orig.shape[2] // 2]
    p = geotiff.write(tmp_path / "half.tif", half, crs=meta.crs, transform=meta.transform)
    _, v = coregister.reproject_to_grid(p, meta)
    assert v[:, : half.shape[2] - 2].all() and not v[:, half.shape[2] + 2 :].any()
