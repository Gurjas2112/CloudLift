"""FastAPI service: ``/health``, ``/reconstruct``, ``/jobs/{id}``, ``/export/{id}``.

* One worker thread processes jobs sequentially (single GPU).
* ``Idempotency-Key`` header: a retry with the same key returns the same job; reusing a key
  for different inputs is a 409.
* Result cache keyed by ``(input hash, model version + weights hash, config hash)``. A cache
  hit still creates a new job record (``cached: true``); a cached result is never served for
  a different input hash because the input hash is part of the key.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from cloudlift import MODEL_VERSION, __version__
from cloudlift.config import Config, load_config

log = logging.getLogger("cloudlift.api")
CacheKey = tuple[str, str, str]


class ReconstructRequest(BaseModel):
    scene_id: str | None = None
    aoi: str | None = None
    source: Literal["auto", "local", "synthetic"] = "auto"


@dataclass
class Job:
    id: str
    status: str  # queued | running | done | failed
    scene_id: str
    aoi: str | None
    input_hash: str
    cache_key: CacheKey
    cached: bool = False
    stage: int = 0
    stage_name: str = ""
    stages: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)

    def public(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("cache_key")
        return d


class JobManager:
    def __init__(self, cfg: Config, start_worker: bool = True) -> None:
        from cloudlift import pipeline

        self.cfg = cfg
        self.pipeline = pipeline
        self.jobs: dict[str, Job] = {}
        self.idem: dict[str, str] = {}
        self.cache: dict[CacheKey, Path] = {}
        self.lock = threading.Lock()
        self.q: queue.Queue[str | None] = queue.Queue()
        self.jobs_dir = Path(cfg.paths.results_root) / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._worker: threading.Thread | None = None
        self._scan_results()
        if start_worker:
            self.start()

    # -- cache ----------------------------------------------------------------------------
    def model_tag(self) -> str:
        w = self.cfg.paths.weights
        if w is None:
            return f"{MODEL_VERSION}+untrained-{self.cfg.model.size}"
        from cloudlift.model.uncrtaints_wrapper import file_hash

        return f"{MODEL_VERSION}+{file_hash(w)}"

    def key(self, ihash: str) -> CacheKey:
        return (ihash, self.model_tag(), self.cfg.config_hash())

    def _scan_results(self) -> None:
        root = Path(self.cfg.paths.results_root)
        for p in root.glob("*/result.json"):
            try:
                r = json.loads(p.read_text(encoding="utf-8"))
                wh = r["model"]["weights_hash"]
                tag = f"{MODEL_VERSION}+{wh if r['model']['trained'] else 'untrained-' + r['model']['size']}"
                if Path(r["geotiff"]).exists():
                    self.cache[(r["input_hash"], tag, r["config_hash"])] = p.parent
            except Exception:
                continue

    # -- jobs -----------------------------------------------------------------------------
    def start(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._loop, name="cloudlift-gpu-worker", daemon=True)
            self._worker.start()

    def stop(self) -> None:
        self.q.put(None)

    def _persist(self, job: Job) -> None:
        job.updated = time.time()
        (self.jobs_dir / f"{job.id}.json").write_text(json.dumps(job.public(), default=str), encoding="utf-8")

    def submit(self, scene: Any, aoi_text: str | None, idem_key: str | None) -> tuple[Job, bool]:
        """Returns ``(job, created)``."""
        from cloudlift.io.aoi import parse_aoi

        aoi = parse_aoi(aoi_text) if aoi_text else None
        ihash = self.pipeline.input_hash(scene, aoi)
        key = self.key(ihash)
        with self.lock:
            if idem_key and idem_key in self.idem:
                job = self.jobs[self.idem[idem_key]]
                if job.input_hash != ihash:
                    raise HTTPException(409, "Idempotency-Key reused with different inputs")
                return job, False
            job = Job(
                id=uuid.uuid4().hex[:12],
                status="queued",
                scene_id=scene.scene_id,
                aoi=aoi_text,
                input_hash=ihash,
                cache_key=key,
                stages=[
                    {"n": i + 1, "name": n, "state": "pending"} for i, n in enumerate(self.pipeline.STAGES)
                ],
            )
            self.jobs[job.id] = job
            if idem_key:
                self.idem[idem_key] = job.id
            hit = self.cache.get(key)
            if hit is not None and (hit / "result.json").exists():
                job.cached, job.status, job.stage = True, "done", 7
                for s in job.stages:
                    s["state"] = "done"
                job.result = json.loads((hit / "result.json").read_text(encoding="utf-8"))
                self._persist(job)
                return job, True
            job._scene, job._aoi = scene, aoi  # type: ignore[attr-defined]
            self._persist(job)
        self.q.put(job.id)
        return job, True

    def _loop(self) -> None:
        while True:
            jid = self.q.get()
            if jid is None:
                return
            job = self.jobs[jid]
            job.status = "running"
            self._persist(job)

            def progress(i: int, name: str, state: str, job: Job = job) -> None:
                job.stage, job.stage_name = i, name
                job.stages[i - 1]["state"] = state
                job.updated = time.time()

            try:
                res = self.pipeline.run(job._scene, job._aoi, self.cfg, progress=progress)  # type: ignore[attr-defined]
                job.result = res.to_json()
                job.status = "done"
                with self.lock:
                    self.cache[job.cache_key] = res.out_dir
            except Exception as e:  # report, never crash the worker
                log.exception("job %s failed", jid)
                job.status, job.error = "failed", f"{type(e).__name__}: {e}"
            self._persist(job)

    def get(self, jid: str) -> Job:
        job = self.jobs.get(jid)
        if job is None:
            raise HTTPException(404, f"job {jid} not found")
        return job


def _save_upload(form: Any, root: Path) -> Any:
    """Store uploaded files in the LocalFolderSource convention under ``data/uploads/<hash>``."""
    from cloudlift.io.sources import LocalFolderSource

    files = {}
    for name in ("liss4", "sar", "reference", "dem"):
        f = form.get(name)
        if f is not None and hasattr(f, "read"):
            files[name] = f
    if "liss4" not in files:
        raise HTTPException(422, "upload requires at least a 'liss4' GeoTIFF")
    blobs = {k: v.file.read() for k, v in files.items()}
    h = hashlib.sha256()
    for k in sorted(blobs):
        h.update(k.encode())
        h.update(hashlib.sha256(blobs[k]).digest())
    sid = f"UPLOAD_{h.hexdigest()[:12]}"
    d = root / sid
    d.mkdir(parents=True, exist_ok=True)
    for k, b in blobs.items():
        (d / f"{k}.tif").write_bytes(b)
    meta = form.get("meta")
    if isinstance(meta, str) and meta:
        (d / "meta.json").write_text(meta, encoding="utf-8")
    return LocalFolderSource(root).get_scene(sid)


def create_app(cfg: Config | None = None, start_worker: bool = True) -> FastAPI:
    cfg = cfg or load_config(os.environ.get("CLOUDLIFT_CONFIG"))
    state: dict[str, JobManager] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> Any:
        state["jm"] = JobManager(cfg, start_worker=start_worker)
        yield
        state["jm"].stop()

    app = FastAPI(
        title="CloudLift",
        version=__version__,
        lifespan=lifespan,
        description="Generative AI cloud removal for LISS-IV (UnCRtainTS + EOS-04 SAR + temporal fusion)",
    )

    def jm() -> JobManager:
        return state["jm"]

    @app.get("/health")
    def health() -> dict[str, Any]:
        m = jm()
        return {
            "status": "ok",
            "version": __version__,
            "model": m.model_tag(),
            "device": cfg.resolve_device(),
            "config_hash": cfg.config_hash(),
            "queue": m.q.qsize(),
            "jobs": len(m.jobs),
        }

    @app.post("/reconstruct", status_code=202)
    async def reconstruct(request: Request) -> JSONResponse:
        from cloudlift.cli import make_source

        idem = request.headers.get("Idempotency-Key")
        ctype = request.headers.get("content-type", "")
        if ctype.startswith("multipart/form-data"):
            form = await request.form()
            scene = _save_upload(form, Path("data") / "uploads")
            aoi = form.get("aoi")
            aoi_text = aoi if isinstance(aoi, str) and aoi else None
        else:
            body = ReconstructRequest.model_validate(await request.json())
            src = make_source(body.source, cfg)
            try:
                scene = src.get_scene(body.scene_id) if body.scene_id else src.list_scenes()[0]
            except (KeyError, IndexError) as e:
                raise HTTPException(404, f"scene not found: {body.scene_id}") from e
            aoi_text = body.aoi
        try:
            job, created = jm().submit(scene, aoi_text, idem)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        payload = {"job_id": job.id, "status": job.status, "cached": job.cached, "input_hash": job.input_hash}
        return JSONResponse(payload, status_code=202 if created else 200)

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        return jm().get(job_id).public()

    @app.get("/export/{job_id}")
    def export(job_id: str) -> FileResponse:
        job = jm().get(job_id)
        if job.status != "done" or not job.result:
            raise HTTPException(409, f"job {job_id} is {job.status}")
        path = Path(job.result["geotiff"])
        if not path.exists():
            raise HTTPException(410, "result file no longer available")
        return FileResponse(path, media_type="image/tiff", filename=path.name)

    return app


def __getattr__(name: str) -> Any:  # ``uvicorn cloudlift.service.api:app`` builds the app lazily
    if name == "app":
        return create_app()
    raise AttributeError(name)
