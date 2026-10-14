import numpy as np
import rasterio
from rasterio.transform import from_origin

from cloudlift.io import geotiff


def test_roundtrip_exact(tmp_path):
    rng = np.random.default_rng(0)
    arr = rng.random((4, 64, 80), dtype=np.float32)
    tr = from_origin(360000.0, 2895000.0, 5.8, 5.8)
    p = geotiff.write(
        tmp_path / "a.tif",
        arr,
        crs="EPSG:32646",
        transform=tr,
        nodata=-9999.0,
        descriptions=("Green", "Red", "NIR", "uncertainty"),
        tags={"SEED": 0, "X": "y"},
    )
    back, meta = geotiff.read(p)
    assert np.array_equal(back, arr)
    assert meta.crs.to_epsg() == 32646
    assert meta.transform == tr
    assert meta.nodata == -9999.0
    assert meta.descriptions == ("Green", "Red", "NIR", "uncertainty")
    assert meta.tags["SEED"] == "0" and meta.tags["X"] == "y"


def test_analysis_ready_profile(tmp_path):
    arr = np.ones((3, 512, 512), np.float32)
    p = geotiff.write(
        tmp_path / "b.tif", arr, crs="EPSG:32646", transform=from_origin(0, 0, 5.8, 5.8), analysis_ready=True
    )
    with rasterio.open(p) as ds:
        assert ds.compression.name.lower() == "deflate"
        assert ds.profile["tiled"] and ds.block_shapes[0] == (256, 256)
        assert ds.overviews(1)
    back, _ = geotiff.read(p)
    assert np.array_equal(back, arr)


def test_window_read(tmp_path):
    arr = np.arange(2 * 32 * 32, dtype=np.int16).reshape(2, 32, 32)
    p = geotiff.write(tmp_path / "c.tif", arr, crs="EPSG:4326", transform=from_origin(90, 27, 0.01, 0.01))
    from rasterio.windows import Window

    sub, meta = geotiff.read(p, window=Window(4, 8, 10, 6))
    assert np.array_equal(sub, arr[:, 8:14, 4:14])
    assert meta.transform.c == 90 + 4 * 0.01
