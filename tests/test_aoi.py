import math

import pytest
from pyproj import Transformer

from cloudlift.io.aoi import AOIError, parse_aoi
from cloudlift.io.sources import LocalFolderSource


def test_bbox_forms():
    a = parse_aoi("bbox:91.6,26.0,91.9,26.3")
    b = parse_aoi("91.6, 26.0, 91.9, 26.3")
    assert a.kind == b.kind == "bbox"
    assert a.geometry.bounds == pytest.approx((91.6, 26.0, 91.9, 26.3))
    c = parse_aoi("bbox:360000,2890000,365000,2895000@EPSG:32646")
    assert c.crs == "EPSG:32646"
    assert c.to_crs("EPSG:4326").bounds[0] == pytest.approx(91.6, abs=0.1)


def test_point_radius_area():
    a = parse_aoi("point:26.14,91.74,5km")
    area_m2 = a.to_crs("EPSG:32646").area
    assert area_m2 == pytest.approx(math.pi * 5000**2, rel=0.01)
    b = parse_aoi("26.14,91.74,500m")
    assert b.to_crs("EPSG:32646").area == pytest.approx(math.pi * 500**2, rel=0.01)


def test_tile_id_resolution(scene_root):
    src = LocalFolderSource(scene_root)
    a = parse_aoi("tile:SYN_TEST_0001", src.resolve_tile)
    assert a.kind == "tile" and a.geometry is not None and a.crs == "EPSG:32646"
    unresolved = parse_aoi("R2A_L4_104_053")
    assert unresolved.kind == "tile" and unresolved.geometry is None


@pytest.mark.parametrize("bad", ["", "bbox:1,2,3", "bbox:5,5,1,1", "point:95,91,5km", "tile:bad id!"])
def test_invalid(bad):
    with pytest.raises(AOIError):
        parse_aoi(bad)


def test_source_aoi_filter(scene_root):
    src = LocalFolderSource(scene_root)
    fp, crs = src.get_scene("SYN_TEST_0001").footprint()
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(fp.centroid.x, fp.centroid.y)
    assert 25 < lat < 27 and 91 < lon < 92.5  # North-Eastern Region
    hit = parse_aoi(f"point:{lat},{lon},1km")
    miss = parse_aoi("point:12.97,77.59,1km")  # Bengaluru
    assert [s.scene_id for s in src.list_scenes(hit)] == ["SYN_TEST_0001"]
    assert src.list_scenes(miss) == []
