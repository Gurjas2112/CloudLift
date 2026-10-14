import hashlib

from cloudlift.io.sources import LocalFolderSource, SyntheticSource
from cloudlift.io.synthetic import generate_scene

FILES = ("liss4.tif", "sar.tif", "reference.tif", "dem.tif", "clean.tif", "mask_gt.tif", "meta.json")


def _hashes(d):
    return {f: hashlib.sha256((d / f).read_bytes()).hexdigest() for f in FILES}


def test_fixture_deterministic(tmp_path):
    a = generate_scene(tmp_path / "a", "S", size=256, seed=7)
    b = generate_scene(tmp_path / "b", "S", size=256, seed=7)
    c = generate_scene(tmp_path / "c", "S", size=256, seed=8)
    assert _hashes(a) == _hashes(b)
    assert _hashes(a)["liss4.tif"] != _hashes(c)["liss4.tif"]


def test_fixture_labelled_synthetic(scene):
    assert scene.synthetic
    from cloudlift.io import geotiff

    _, meta = geotiff.read(scene.optical)
    assert meta.tags["SYNTHETIC"] == "true"
    assert meta.descriptions == ("Green", "Red", "NIR")


def test_local_folder_convention(scene_root):
    s = LocalFolderSource(scene_root).list_scenes()
    assert len(s) == 1
    sc = s[0]
    assert sc.sar and sc.reference and sc.dem and sc.clean and sc.mask_gt
    assert sc.reference_gap_days() == 155
    assert LocalFolderSource(scene_root / "missing").list_scenes() == []


def test_synthetic_source_generates(tmp_path):
    src = SyntheticSource(tmp_path, n_scenes=2, size=128, seed=1)
    ids = [s.scene_id for s in src.list_scenes()]
    assert ids == ["SYN_NER_0001", "SYN_NER_0002"]
