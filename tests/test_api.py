import io
import time

import pytest
import rasterio
from fastapi.testclient import TestClient

from cloudlift.config import load_config
from cloudlift.service.api import create_app


@pytest.fixture()
def client(scene_root, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # uploads land in tmp_path/data/uploads
    cfg = load_config(
        overrides={
            "device": "cpu",
            "model": {"size": "tiny"},
            "paths": {"data_root": str(scene_root), "results_root": str(tmp_path / "results")},
        }
    )
    with TestClient(create_app(cfg)) as c:
        yield c


def _wait(client, jid, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/jobs/{jid}").json()
        if j["status"] in ("done", "failed"):
            return j
        time.sleep(0.2)
    raise TimeoutError(jid)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert "untrained" in r.json()["model"]


def test_lifecycle_export_and_cache(client):
    r = client.post("/reconstruct", json={"scene_id": "SYN_TEST_0001", "source": "local"})
    assert r.status_code == 202
    jid = r.json()["job_id"]
    j = _wait(client, jid)
    assert j["status"] == "done" and j["stage"] == 7 and not j["cached"]
    assert [s["state"] for s in j["stages"]] == ["done"] * 7
    e = client.get(f"/export/{jid}")
    assert e.status_code == 200 and e.headers["content-type"] == "image/tiff"
    with rasterio.open(io.BytesIO(e.content)) as ds:
        assert ds.count == 4 and ds.descriptions[3] == "uncertainty" and ds.crs.to_epsg() == 32646
    # same inputs again -> new job record, served from cache
    r2 = client.post("/reconstruct", json={"scene_id": "SYN_TEST_0001", "source": "local"})
    j2 = r2.json()
    assert j2["job_id"] != jid and j2["cached"] and j2["status"] == "done"
    assert client.get(f"/jobs/{j2['job_id']}").json()["result"]["input_hash"] == j["input_hash"]


def test_idempotent_retry(client):
    h = {"Idempotency-Key": "abc-123"}
    a = client.post("/reconstruct", json={"scene_id": "SYN_TEST_0001", "source": "local"}, headers=h)
    b = client.post("/reconstruct", json={"scene_id": "SYN_TEST_0001", "source": "local"}, headers=h)
    assert a.json()["job_id"] == b.json()["job_id"] and b.status_code == 200
    c = client.post(
        "/reconstruct",
        json={"scene_id": "SYN_TEST_0001", "source": "local", "aoi": "point:26.15,91.62,0.5km"},
        headers=h,
    )
    assert c.status_code == 409
    _wait(client, a.json()["job_id"])


def test_different_input_never_cached(client, scene_root):
    d = scene_root / "SYN_TEST_0001"
    files = {
        "liss4": ("liss4.tif", (d / "liss4.tif").read_bytes(), "image/tiff"),
        "reference": ("reference.tif", (d / "reference.tif").read_bytes(), "image/tiff"),
        "sar": ("sar.tif", (d / "sar.tif").read_bytes(), "image/tiff"),
    }
    meta = '{"sar_units": "unit", "acquired": "2026-07-15", "reference_acquired": "2026-02-10"}'
    a = client.post("/reconstruct", files=files, data={"meta": meta})
    assert a.status_code == 202
    ja = _wait(client, a.json()["job_id"])
    assert ja["status"] == "done"
    # different input (no reference) -> different hash -> must not be served from cache
    files2 = {k: v for k, v in files.items() if k != "reference"}
    b = client.post("/reconstruct", files=files2, data={"meta": meta})
    jb = b.json()
    assert jb["input_hash"] != ja["input_hash"] and not jb["cached"]
    assert _wait(client, jb["job_id"])["status"] == "done"


def test_errors(client):
    assert client.get("/jobs/nope").status_code == 404
    assert client.post("/reconstruct", json={"scene_id": "MISSING", "source": "local"}).status_code == 404
    assert client.post("/reconstruct", files={"sar": ("s.tif", b"x", "image/tiff")}).status_code == 422
    bad = client.post("/reconstruct", files={"liss4": ("l.tif", b"not a tiff", "image/tiff")})
    assert bad.status_code == 422 and "GeoTIFF" in bad.json()["detail"]


def test_upload_any_sensor_with_band_fields(client, four_band_scene):
    opt, ref = four_band_scene
    files = {
        "liss4": ("o.tif", opt.read_bytes(), "image/tiff"),
        "reference": ("r.tif", ref.read_bytes(), "image/tiff"),
    }
    r = client.post("/reconstruct", files=files, data={"bands": "2,3,4", "scale": "10000"})
    assert r.status_code == 202
    j = _wait(client, r.json()["job_id"])
    assert j["status"] == "done"
    assert j["result"]["extra"]["radiometry"]["optical"] == {"bands": [2, 3, 4], "scale": 10000.0}
    bad = client.post("/reconstruct", files=files, data={"bands": "2,3,7"})
    assert bad.status_code == 422 and "out of range" in bad.json()["detail"]
