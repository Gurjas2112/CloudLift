import csv
import json
import shutil

import numpy as np
import pytest
import rasterio
import yaml

from cloudlift import pipeline
from cloudlift.cli import main as cli_main
from cloudlift.io import geotiff
from cloudlift.io.aoi import parse_aoi
from cloudlift.io.sources import LocalFolderSource


def _strict(c):
    raise AssertionError(f"non-strict JSON constant {c}")


def test_end_to_end_synthetic(scene, cfg):
    stages = []
    r = pipeline.run(scene, None, cfg, progress=lambda i, n, s: stages.append((i, s)))
    assert [i for i, s in stages if s == "done"] == [1, 2, 3, 4, 5, 6, 7]
    with rasterio.open(r.geotiff) as ds:
        assert ds.count == 4
        assert ds.descriptions == ("Green", "Red", "NIR", "uncertainty")
        assert ds.crs.to_epsg() == 32646
        assert ds.nodata == pipeline.NODATA
        assert ds.overviews(1)
        tags = ds.tags()
        for k in (
            "MODEL_VERSION",
            "WEIGHTS_HASH",
            "CONFIG_HASH",
            "SEED",
            "SOURCE_SCENE_IDS",
            "PROCESSING_TIMESTAMP",
            "INFERRED_PIXEL_FRACTION",
        ):
            assert k in tags
        assert tags["SYNTHETIC"] == "true" and tags["MODEL_TRAINED"] == "false"
        src_meta = geotiff.read_meta(scene.optical)
        assert ds.transform == src_meta.transform
        unc = ds.read(4)
    assert (unc >= 0).all()
    assert r.extra["mode"] == "full-reference"
    m = r.metrics
    assert 0 < m["masked_fraction"] < 1 and m["cloud_detection_iou"] > 0.8
    assert {"psnr_masked", "ssim_masked", "sam_deg_masked", "ndvi_r2", "ndvi_rmse", "ece"} <= set(m)
    assert r.timing["ms_per_tile"] > 0
    json.loads((r.out_dir / "result.json").read_text(encoding="utf-8"), parse_constant=_strict)
    json.loads(r.metrics_json.read_text(encoding="utf-8"), parse_constant=_strict)
    assert r.panels.exists() and "Measured inference" in r.metrics_md.read_text(encoding="utf-8")
    # clear pixels are passed through untouched
    recon = np.load(r.out_dir / "recon.npy")
    opt, _ = geotiff.read(scene.optical)
    gt, _ = geotiff.read(scene.mask_gt)
    far = gt[0] == 0
    import cv2

    far = cv2.erode(far.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    far &= ~cv2.dilate(
        (np.load(r.out_dir / "uncertainty.npy") > 0).astype(np.uint8), np.ones((3, 3), np.uint8)
    ).astype(bool)
    assert np.array_equal(recon[:, far], opt[:, far])


def test_aoi_crop(scene, cfg):
    fp, crs = scene.footprint()
    x0, y0, x1, y1 = fp.bounds
    aoi = parse_aoi(f"bbox:{x0 + 300},{y0 + 400},{x0 + 1500},{y0 + 1700}@{crs}")
    r = pipeline.run(scene, aoi, cfg)
    with rasterio.open(r.geotiff) as ds:
        assert abs(ds.width - 1200 / 5.8) <= 2 and abs(ds.height - 1300 / 5.8) <= 2
        assert abs(ds.bounds.left - (x0 + 300)) <= 5.8


def test_real_scene_mode_and_missing_sar(scene_root, tmp_path, cfg):
    real = tmp_path / "raw" / "REAL_LIKE"
    real.mkdir(parents=True)
    src = scene_root / "SYN_TEST_0001"
    for f in ("liss4.tif", "reference.tif"):
        shutil.copy(src / f, real / f)
    sar, meta = geotiff.read(src / "sar.tif")
    half = sar.shape[2] // 2
    geotiff.write(real / "sar.tif", sar[:, :, :half], crs=meta.crs, transform=meta.transform)
    (real / "meta.json").write_text(
        json.dumps({"acquired": "2026-07-15", "reference_acquired": "2026-02-10", "sar_units": "unit"}),
        encoding="utf-8",
    )
    sc = LocalFolderSource(tmp_path / "raw").get_scene("REAL_LIKE")
    r = pipeline.run(sc, None, cfg)
    assert r.extra["mode"] == "real-scene"
    assert "ndvi_r2_vs_reference" in r.metrics and "psnr" not in r.metrics
    u = np.load(r.out_dir / "uncertainty.npy")
    mask, _ = geotiff.read(r.mask)
    right = (mask[0] > 0)[:, half + 2 :]
    assert right.any() and np.all(u[:, half + 2 :][right] == pipeline.UNCERTAINTY_MAX)
    assert "full-reference metrics are unavailable" in r.metrics_md.read_text(encoding="utf-8").lower()


def test_batch_resumable(scene_root, tmp_path, capsys):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "device": "cpu",
                "model": {"size": "tiny"},
                "paths": {"data_root": str(scene_root), "results_root": str(tmp_path / "results")},
            }
        ),
        encoding="utf-8",
    )
    aoi_csv = tmp_path / "aois.csv"
    with open(aoi_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["aoi"])
        w.writerow(["tile:SYN_TEST_0001"])
    args = ["batch", "--config", str(cfg_path), "--aoi-list", str(aoi_csv), "--source", "local"]
    assert cli_main(args) == 0
    first = capsys.readouterr().out
    assert '"processed": 1' in first
    assert cli_main(args) == 0
    second = capsys.readouterr().out
    assert '"processed": 0' in second and '"skipped": 1' in second


def test_any_sensor_band_order_and_scale(scene_root, four_band_scene, cfg, tmp_path):
    """A 4-band uint16 B,G,R,NIR product (x10000) gives the same reconstruction as the 3-band float file."""
    from cloudlift.io.sources import save_scene

    d = scene_root / "SYN_TEST_0001"
    opt, ref = four_band_scene
    meta = {
        "bands": [2, 3, 4],
        "scale": "auto",
        "sar_units": "unit",
        "acquired": "2026-07-15",
        "reference_acquired": "2026-02-10",
    }
    blobs = {"liss4": opt.read_bytes(), "reference": ref.read_bytes(), "sar": (d / "sar.tif").read_bytes()}
    four = save_scene(tmp_path / "up4", blobs, meta)
    blobs3 = {
        "liss4": (d / "liss4.tif").read_bytes(),
        "reference": (d / "reference.tif").read_bytes(),
        "sar": (d / "sar.tif").read_bytes(),
    }
    three = save_scene(
        tmp_path / "up3", blobs3, {k: v for k, v in meta.items() if k not in ("bands", "scale")}
    )
    r4 = pipeline.run(four, None, cfg, out_dir=tmp_path / "r4")
    r3 = pipeline.run(three, None, cfg, out_dir=tmp_path / "r3")
    assert r4.extra["radiometry"]["optical"] == {"bands": [2, 3, 4], "scale": 10000.0}
    a, b = np.load(r4.out_dir / "recon.npy"), np.load(r3.out_dir / "recon.npy")
    diff = np.abs(a - b)  # only uint16 quantisation differs (it can flip a few mask-edge pixels)
    assert diff.mean() < 1e-3 and np.percentile(diff, 99.9) < 1e-2
    with rasterio.open(r4.geotiff) as ds:
        assert ds.descriptions == ("Green", "Red", "NIR", "uncertainty")
        assert '"bands": [2, 3, 4]' in ds.tags()["INPUT_RADIOMETRY"]


def test_cli_infer_direct_files(four_band_scene, tmp_path, monkeypatch):
    opt, _ = four_band_scene
    monkeypatch.chdir(tmp_path)
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump({"device": "cpu", "model": {"size": "tiny"}}), encoding="utf-8")
    assert (
        cli_main(
            [
                "infer",
                "--config",
                str(cfg_path),
                "--optical",
                str(opt),
                "--bands",
                "2,3,4",
                "--out",
                str(tmp_path / "out"),
            ]
        )
        == 0
    )
    res = json.loads((tmp_path / "out" / "result.json").read_text(encoding="utf-8"))
    assert res["extra"]["mode"] == "real-scene" and res["extra"]["coregistration"]["sar"] == "missing"
    with rasterio.open(res["geotiff"]) as ds:
        assert ds.count == 4
        u = ds.read(4)
    mask = np.load(tmp_path / "out" / "recon.npy")  # exists
    assert mask.shape[0] == 3
    assert u.max() == pytest.approx(pipeline.UNCERTAINTY_MAX)  # no SAR -> sigma = max under cloud
