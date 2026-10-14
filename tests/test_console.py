import numpy as np

from cloudlift import pipeline
from cloudlift.ui import console


def test_metric_tiles_never_invent_numbers():
    empty = console.metrics_html(None)
    assert empty.count("—") == 4
    assert "—" in console.obstruction_card(None)


def test_panels_from_real_run(scene, cfg):
    res = pipeline.run(scene, None, cfg).to_json()
    html = console.metrics_html(res)
    assert f"{res['metrics']['ssim_masked']:.3f}" in html
    assert f"{res['inferred_fraction'] * 100:.1f}%" in html
    assert "untrained model" in html
    assert f"{res['cloud_fraction'] * 100:.1f}" in console.obstruction_card(res)
    before, after = console.slider_images(res, show_conf=False)
    _, overlay = console.slider_images(res, show_conf=True)
    assert before.shape == after.shape == overlay.shape and before.dtype == np.uint8
    assert not np.array_equal(after, overlay)  # confidence toggle changes the view
    stages = console.stages_html({3, 2}, 4)
    assert stages.count("cl-dot done") == 2 and stages.count("cl-dot run") == 1


def test_app_builds(scene, cfg):
    demo = console.build(cfg)
    assert demo is not None
    assert "BHOONIDHI · NRSC/ISRO" in console.status_bar()
    assert "Indian Space Policy 2023" in console.scene_panel(scene, cfg)


def test_upload_own_scene(scene_root, tmp_path, monkeypatch):
    import pytest

    from cloudlift.io.sources import UploadError

    monkeypatch.setattr(console, "UPLOADS", tmp_path / "uploads")
    d = scene_root / "SYN_TEST_0001"
    sc = console.add_uploaded_scene(str(d / "liss4.tif"), None, str(d / "reference.tif"), "2026-07-15", "")
    assert sc.scene_id.startswith("UPLOAD_") and sc.sar is None and sc.reference is not None
    assert sc.meta["acquired"] == "2026-07-15" and "reference_acquired" not in sc.meta
    again = console.add_uploaded_scene(str(d / "liss4.tif"), None, str(d / "reference.tif"), "2026-07-15", "")
    assert again.scene_id == sc.scene_id  # content-addressed: re-upload does not duplicate
    with pytest.raises(UploadError, match="required"):
        console.add_uploaded_scene(None, str(d / "sar.tif"), None, "", "")
    with pytest.raises(UploadError, match="expected at least 3 band"):
        console.add_uploaded_scene(str(d / "sar.tif"), None, None, "", "")
    bad = tmp_path / "bad.tif"
    bad.write_bytes(b"not a tiff")
    with pytest.raises(UploadError, match="not a readable GeoTIFF"):
        console.add_uploaded_scene(str(bad), None, None, "", "")


def test_fresh_install_generates_demo_scenes(tmp_path, monkeypatch, cfg):
    monkeypatch.setattr(console, "DEMO_SCENES", tmp_path / "demo")
    monkeypatch.setattr(console, "UPLOADS", tmp_path / "up")
    monkeypatch.setattr(console, "HEROES", [("HERO_T", 1, 0.3)])
    cfg.paths.data_root = tmp_path / "empty"
    assert console.scene_choices(cfg) == []
    console.ensure_demo_scenes(size=128)
    assert console.scene_choices(cfg) == ["HERO_T"]


def test_upload_band_order_and_scaling(tmp_path, monkeypatch, four_band_scene, cfg):
    import pytest

    from cloudlift.io.sources import UploadError

    monkeypatch.setattr(console, "UPLOADS", tmp_path / "uploads")
    opt, ref = four_band_scene
    sc = console.add_uploaded_scene(str(opt), None, str(ref), "", "", bands="2,3,4", scale="auto (detect)")
    assert sc.meta["bands"] == [2, 3, 4] and sc.meta["scale"] == "auto"
    assert "bands 2, 3, 4" in console.scene_panel(sc, cfg)
    assert "missing" in console.scene_panel(sc, cfg)  # no SAR
    assert console.preview_cloudy(sc).shape[-1] == 3
    with pytest.raises(UploadError, match="out of range"):
        console.add_uploaded_scene(str(opt), None, None, "", "", bands="2,3,9")
    with pytest.raises(UploadError, match="three numbers"):
        console.add_uploaded_scene(str(opt), None, None, "", "", bands="g,r,n")
