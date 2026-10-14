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
