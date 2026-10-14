import numpy as np

from cloudlift.io import geotiff
from cloudlift.preprocess import cloudmask


def test_baseline_matches_ground_truth(scene):
    opt, _ = geotiff.read(scene.optical)
    ref, _ = geotiff.read(scene.reference)
    gt, _ = geotiff.read(scene.mask_gt)
    m = cloudmask.detect(opt, ref, method="baseline")
    truth = gt[0] == 1
    iou = (m.cloud & truth).sum() / (m.cloud | truth).sum()
    assert iou > 0.8
    assert not (m.cloud & m.shadow).any()


def test_deterministic_and_single_date(scene):
    opt, _ = geotiff.read(scene.optical)
    a = cloudmask.baseline_mask(opt)
    b = cloudmask.baseline_mask(opt)
    assert np.array_equal(a.cloud, b.cloud) and a.method == "baseline"
    u = a.as_uint8()
    assert set(np.unique(u)) <= {0, 1, 2}


def test_clear_scene_has_no_cloud(scene):
    clean, _ = geotiff.read(scene.clean)
    assert cloudmask.baseline_mask(clean, clean).cloud_fraction < 0.01


def test_auto_falls_back(scene, monkeypatch):
    monkeypatch.setattr(cloudmask, "omnicloudmask_available", lambda: False)
    opt, _ = geotiff.read(scene.optical)
    assert cloudmask.detect(opt, None, "auto").method == "baseline"
