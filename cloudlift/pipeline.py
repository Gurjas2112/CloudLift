"""The 7-stage CloudLift run for one scene.

1 Ingest -> 2 Co-register & tile -> 3 Detect cloud + shadow -> 4 Reconstruct (UnCRtainTS, SAR-fused)
-> 5 Per-pixel uncertainty -> 6 Validate -> 7 Export analysis-ready GeoTIFF
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from rasterio.windows import Window, from_bounds

from cloudlift import BANDS, MODEL_VERSION
from cloudlift.config import Config
from cloudlift.determinism import set_determinism
from cloudlift.eval import metrics as M
from cloudlift.eval.report import json_safe, write_panels, write_report
from cloudlift.io import geotiff
from cloudlift.io.aoi import AOI
from cloudlift.io.radiometry import optical_to_unit, sar_to_unit
from cloudlift.io.sources import Scene
from cloudlift.model.adapter import stack_inputs
from cloudlift.model.uncrtaints_wrapper import CloudLiftNet, file_hash, load
from cloudlift.preprocess import cloudmask, coregister
from cloudlift.preprocess.tiling import Untiler, extract_tile, make_grid, pad_to_grid

log = logging.getLogger("cloudlift.pipeline")

STAGES = (
    "Ingest",
    "Co-register & tile",
    "Detect cloud + shadow",
    "Reconstruct (UnCRtainTS, SAR-fused)",
    "Per-pixel uncertainty",
    "Validate",
    "Export GeoTIFF",
)
NODATA = -9999.0
UNCERTAINTY_MAX = 1.0  # sigma in reflectance units; assigned where SAR is missing under cloud

ProgressFn = Callable[[int, str, str], None]  # (stage 1..7, stage name, "running" | "done")


@dataclass
class Result:
    run_id: str
    out_dir: Path
    geotiff: Path
    mask: Path
    panels: Path
    metrics_json: Path
    metrics_md: Path
    metrics: dict[str, Any]
    timing: dict[str, float]
    cloud_fraction: float
    obstructed_fraction: float
    inferred_fraction: float
    config_hash: str
    input_hash: str
    model: dict[str, Any]
    synthetic: bool
    scene_id: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = v.as_posix()
        return json_safe(json.loads(json.dumps(d, default=str)))

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Result:
        d = dict(d)
        for k in ("out_dir", "geotiff", "mask", "panels", "metrics_json", "metrics_md"):
            d[k] = Path(d[k])
        return cls(**d)


_NET_CACHE: dict[tuple[Any, ...], CloudLiftNet] = {}


def load_net(cfg: Config) -> CloudLiftNet:
    """Load (and memoise) the network for a config."""
    w = cfg.paths.weights
    if w is not None and not Path(w).exists():
        raise FileNotFoundError(f"weights not found: {w}")
    key = (str(w), cfg.model.size, cfg.model.covmode, cfg.seed)
    if key not in _NET_CACHE:
        _NET_CACHE[key] = load(w, size=cfg.model.size, covmode=cfg.model.covmode, seed=cfg.seed).eval()
    return _NET_CACHE[key]


def input_hash(scene: Scene, aoi: AOI | None) -> str:
    h = hashlib.sha256()
    for p in scene.input_paths():
        h.update(p.name.encode())
        h.update(file_hash(p).encode())
    h.update((aoi.text if aoi else "").encode())
    return h.hexdigest()[:16]


def run_id_for(scene: Scene, cfg: Config, ihash: str) -> str:
    return f"{scene.scene_id}_{cfg.config_hash()[:8]}_{ihash[:8]}"


def scene_window(scene: Scene, aoi: AOI | None) -> Window | None:
    """Pixel window of ``scene`` covered by ``aoi`` (``None`` = whole scene)."""
    if aoi is None or aoi.geometry is None:
        return None
    meta = geotiff.read_meta(scene.optical)
    geom = aoi.to_crs(meta.crs.to_string() if meta.crs else "EPSG:4326")
    x0, y0, x1, y1 = geom.bounds
    sx0, sy0, sx1, sy1 = meta.bounds
    x0, y0, x1, y1 = max(x0, sx0), max(y0, sy0), min(x1, sx1), min(y1, sy1)
    if x0 >= x1 or y0 >= y1:
        raise ValueError(f"AOI {aoi.text!r} does not overlap scene {scene.scene_id}")
    w = from_bounds(x0, y0, x1, y1, transform=meta.transform).round_offsets().round_lengths()
    return w.intersection(Window(0, 0, meta.width, meta.height))


def ingest(scene: Scene, win: Window | None) -> tuple[geotiff.RasterMeta, np.ndarray, np.ndarray]:
    """Stage 1: read the optical pass (optionally a window) as [0,1] reflectance + nodata mask."""
    raw, meta = geotiff.read(scene.optical, window=win)
    nodata_px = np.zeros(raw.shape[1:], bool)
    if meta.nodata is not None:
        nodata_px = np.all(raw == meta.nodata, axis=0)
    optical = optical_to_unit(raw, scene.meta)
    optical[:, nodata_px] = 0.0
    return meta, optical, nodata_px


def coregister_inputs(
    scene: Scene, meta: geotiff.RasterMeta, optical: np.ndarray, nodata_px: np.ndarray, tile: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Stage 2: SAR + temporal reference onto the LISS-IV grid; sub-pixel refine of the reference."""
    h, w = optical.shape[1:]
    if scene.sar is not None:
        sar_raw, sar_valid = coregister.reproject_to_grid(scene.sar, meta)
        sar = sar_to_unit(sar_raw, scene.meta) * sar_valid[None]
    else:
        sar, sar_valid = np.zeros((2, h, w), np.float32), np.zeros((h, w), bool)
    info: dict[str, Any] = {"sar": "reprojected (geocoded; no cross-modal refinement)"}
    if scene.reference is not None:
        ref_raw, ref_valid = coregister.reproject_to_grid(scene.reference, meta)
        reference = optical_to_unit(ref_raw, scene.meta) * ref_valid[None]
        prelim = cloudmask.baseline_mask(optical).combined
        reference, est = coregister.refine_reference(
            optical, reference, ~prelim & ref_valid & ~nodata_px, tile=tile
        )
        info["reference"] = est.as_dict()
    else:
        reference = optical.copy()
        info["reference"] = "missing: current pass used as its own reference (no temporal prior)"
    return sar, sar_valid, reference, info


def feather(mask: np.ndarray, px: int = 3) -> np.ndarray:
    """Soft compositing weight: 1 inside mask, ramps to 0 over ``px`` pixels outside."""
    m = mask.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    grown = cv2.dilate(m, k).astype(np.float32)
    soft = cv2.GaussianBlur(grown, (2 * px + 1, 2 * px + 1), 0)
    return np.maximum(soft, m.astype(np.float32))


def _baselines(
    cloudy: np.ndarray, reference: np.ndarray, clean: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    """Non-learned comparators on the same pixels (for the comparative assessment)."""
    fill = np.where(mask[None], reference, cloudy)
    return {
        "cloudy_input": {
            "psnr_masked": M.psnr(cloudy, clean, mask),
            "sam_deg_masked": M.sam_deg(cloudy, clean, mask),
            **M.ndvi_agreement(cloudy, clean, mask),
        },
        "temporal_fill": {
            "psnr_masked": M.psnr(fill, clean, mask),
            "sam_deg_masked": M.sam_deg(fill, clean, mask),
            "ssim_masked": M.ssim(fill, clean, mask),
            **M.ndvi_agreement(fill, clean, mask),
        },
    }


def run(
    scene: Scene,
    aoi: AOI | None,
    cfg: Config,
    *,
    out_dir: str | Path | None = None,
    net: CloudLiftNet | None = None,
    progress: ProgressFn | None = None,
) -> Result:
    """Execute all 7 stages for ``scene`` (optionally cropped to ``aoi``)."""
    set_determinism(cfg.seed)
    device = cfg.resolve_device()
    net = net if net is not None else load_net(cfg)
    net = net.to(device).eval()
    ihash = input_hash(scene, aoi)
    chash = cfg.config_hash()
    rid = run_id_for(scene, cfg, ihash)
    out = Path(out_dir) if out_dir is not None else cfg.paths.results_root / rid
    out.mkdir(parents=True, exist_ok=True)
    timing: dict[str, float] = {}

    def stage(i: int, state: str) -> None:
        if progress is not None:
            progress(i, STAGES[i - 1], state)

    # 1 -- Ingest ------------------------------------------------------------------
    t0 = time.perf_counter()
    stage(1, "running")
    win = scene_window(scene, aoi)
    meta, optical, nodata_px = ingest(scene, win)
    timing["1_ingest_s"] = time.perf_counter() - t0
    stage(1, "done")

    # 2 -- Co-register & tile -------------------------------------------------------
    t0 = time.perf_counter()
    stage(2, "running")
    h, w = optical.shape[1:]
    sar, sar_valid, reference, coreg_info = coregister_inputs(scene, meta, optical, nodata_px, cfg.data.tile)
    grid = make_grid(h, w, cfg.data.tile, cfg.data.overlap)
    timing["2_coregister_s"] = time.perf_counter() - t0
    stage(2, "done")

    # 3 -- Detect cloud + shadow ------------------------------------------------------
    t0 = time.perf_counter()
    stage(3, "running")
    masks = cloudmask.detect(
        optical, reference if scene.reference is not None else None, cfg.cloudmask.method
    )
    obstructed = masks.combined & ~nodata_px
    timing["3_cloudmask_s"] = time.perf_counter() - t0
    stage(3, "done")

    # 4 -- Reconstruct ------------------------------------------------------------------
    t0 = time.perf_counter()
    stage(4, "running")
    gap = scene.reference_gap_days()
    padded = {
        k: pad_to_grid(v, grid)
        for k, v in {
            "opt": optical,
            "sar": sar,
            "ref": reference,
            "mask": obstructed.astype(np.float32),
        }.items()
    }
    rec_acc, sig_acc = Untiler(grid, 3), Untiler(grid, 3)
    use_amp = device == "cuda" and cfg.infer.amp
    todo = [i for i in range(len(grid)) if extract_tile(padded["mask"], grid, i).any()]
    todo_set = set(todo)
    for i in range(len(grid)):  # tiles without cloud/shadow are passed through unchanged
        if i not in todo_set:
            rec_acc.add(i, extract_tile(padded["opt"], grid, i))
            sig_acc.add(i, np.zeros((3, grid.tile, grid.tile), np.float32))
    t_inf = time.perf_counter()
    bs = max(1, cfg.infer.batch_size)
    with torch.inference_mode():
        for b0 in range(0, len(todo), bs):
            idx = todo[b0 : b0 + bs]
            xs, ds = [], []
            for i in idx:
                x, d = stack_inputs(
                    extract_tile(padded["opt"], grid, i),
                    extract_tile(padded["sar"], grid, i),
                    [extract_tile(padded["ref"], grid, i)],
                    cfg.data.input_t,
                    gap,
                )
                xs.append(x)
                ds.append(d)
            xt = torch.from_numpy(np.stack(xs)).to(device)
            dt = torch.from_numpy(np.stack(ds)).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                mu, sg = net(xt, dt)
            mu_np, sg_np = mu.float().cpu().numpy(), sg.float().cpu().numpy()
            if sg_np.shape[1] == 1:
                sg_np = np.repeat(sg_np, 3, axis=1)
            for j, i in enumerate(idx):
                rec_acc.add(i, mu_np[j])
                sig_acc.add(i, sg_np[j])
    if device == "cuda":
        torch.cuda.synchronize()
    infer_s = time.perf_counter() - t_inf
    recon_raw = rec_acc.result()
    sigma = sig_acc.result()
    alpha = feather(obstructed)
    recon = (alpha[None] * recon_raw + (1 - alpha[None]) * optical).astype(np.float32)
    recon[:, nodata_px] = 0.0
    timing["4_reconstruct_s"] = time.perf_counter() - t0
    timing["ms_per_tile"] = 1000.0 * infer_s / max(1, len(todo))
    timing["tiles_inferred"] = float(len(todo))
    timing["tiles_total"] = float(len(grid))
    stage(4, "done")

    # 5 -- Per-pixel uncertainty --------------------------------------------------------
    t0 = time.perf_counter()
    stage(5, "running")
    u = (sigma.mean(axis=0) * alpha).astype(np.float32)
    sar_missing = obstructed & ~sar_valid
    u[sar_missing] = UNCERTAINTY_MAX
    sigma_eff = sigma * alpha[None]
    sigma_eff[:, sar_missing] = UNCERTAINTY_MAX
    u_in = u[obstructed]
    unc_stats = {
        "mean_sigma_masked": float(u_in.mean()) if u_in.size else float("nan"),
        "p90_sigma_masked": float(np.percentile(u_in, 90)) if u_in.size else float("nan"),
        "sar_missing_fraction": float(sar_missing.mean()),
    }
    timing["5_uncertainty_s"] = time.perf_counter() - t0
    stage(5, "done")

    # 6 -- Validate ----------------------------------------------------------------------
    t0 = time.perf_counter()
    stage(6, "running")
    scene_metrics: dict[str, Any]
    eval_mask = obstructed
    clean: np.ndarray | None = None
    baselines: dict[str, Any] = {}
    if scene.clean is not None:
        clean_raw, _ = geotiff.read(scene.clean, window=win)
        clean = optical_to_unit(clean_raw, scene.meta)
        det_iou = float("nan")
        if scene.mask_gt is not None:
            gt, _ = geotiff.read(scene.mask_gt, window=win)
            eval_mask = gt[0] > 0
            inter, union = (masks.cloud & (gt[0] == 1)).sum(), (masks.cloud | (gt[0] == 1)).sum()
            det_iou = float(inter / union) if union else 1.0
        mode = "full-reference"
        scene_metrics = M.full_reference(recon, clean, eval_mask, sigma_eff)
        scene_metrics["cloud_detection_iou"] = det_iou
        if eval_mask.any():
            baselines = _baselines(optical, reference, clean, eval_mask)
    else:
        mode = "real-scene"
        scene_metrics = M.reference_consistency(recon, reference, eval_mask) if scene.reference else {}
    scene_metrics.update(unc_stats)
    shown_ref = clean if clean is not None else (reference if scene.reference is not None else None)
    panels = write_panels(
        out / "panels.png",
        optical,
        recon,
        shown_ref,
        u,
        obstructed,
        title=f"{scene.scene_id}{'  [SYNTHETIC]' if scene.synthetic else ''}  ·  {mode}",
        ref_label="Cloud-free truth" if clean is not None else "Temporal reference",
    )
    card = net.card()
    report = {
        "run_id": rid,
        "mode": mode,
        "split": "inference",
        "synthetic": scene.synthetic,
        "model": card,
        "config_hash": chash,
        "seed": cfg.seed,
        "device": device,
        "timing": {"ms_per_tile": timing["ms_per_tile"], "tiles": len(todo), "tile": cfg.data.tile},
        "scenes": {scene.scene_id: {k: v for k, v in scene_metrics.items() if not isinstance(v, dict)}},
        "detail": {
            "ssim_bands_masked": scene_metrics.get("ssim_bands_masked"),
            "baselines": baselines,
            "coregistration": coreg_info,
            "cloudmask_method": masks.method,
        },
    }
    jpath, mdpath = write_report(out, report, [panels])
    timing["6_validate_s"] = time.perf_counter() - t0
    stage(6, "done")

    # 7 -- Export -------------------------------------------------------------------------
    t0 = time.perf_counter()
    stage(7, "running")
    inferred_fraction = float(obstructed.mean())
    stack = np.concatenate([recon, u[None]], axis=0).astype(np.float32)
    stack[:, nodata_px] = NODATA
    tags = {
        "MODEL_VERSION": MODEL_VERSION,
        "MODEL_TRAINED": str(card["trained"]).lower(),
        "WEIGHTS_HASH": card["weights_hash"],
        "WEIGHTS_SOURCE": card["weights_source"],
        "CONFIG_HASH": chash,
        "INPUT_HASH": ihash,
        "SEED": cfg.seed,
        "SOURCE_SCENE_IDS": scene.scene_id,
        "AOI": aoi.text if aoi else "full-scene",
        "PROCESSING_TIMESTAMP": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "INFERRED_PIXEL_FRACTION": f"{inferred_fraction:.6f}",
        "SYNTHETIC": str(scene.synthetic).lower(),
        "UNCERTAINTY_UNITS": "sigma, reflectance units; 0 = observed clear pixel; 1.0 = SAR missing under cloud",
        "PRODUCT": "CloudLift analysis-ready cloud-free LISS-IV",
    }
    gpath = geotiff.write(
        out / f"{rid}_cloudfree.tif",
        stack,
        crs=meta.crs,
        transform=meta.transform,
        nodata=NODATA,
        descriptions=(*BANDS, "uncertainty"),
        tags=tags,
        analysis_ready=True,
    )
    mask_path = geotiff.write(
        out / f"{rid}_mask.tif",
        masks.as_uint8(),
        crs=meta.crs,
        transform=meta.transform,
        nodata=255,
        descriptions=("cloud_mask: 0 clear, 1 cloud, 2 shadow",),
        tags={"CONFIG_HASH": chash, "METHOD": masks.method},
        analysis_ready=True,
    )
    np.save(out / "recon.npy", recon)
    np.save(out / "uncertainty.npy", u)
    np.save(out / "cloudy.npy", optical)
    timing["7_export_s"] = time.perf_counter() - t0
    stage(7, "done")

    res = Result(
        run_id=rid,
        out_dir=out,
        geotiff=gpath,
        mask=mask_path,
        panels=panels,
        metrics_json=jpath,
        metrics_md=mdpath,
        metrics=scene_metrics,
        timing=timing,
        cloud_fraction=masks.cloud_fraction,
        obstructed_fraction=masks.obstructed_fraction,
        inferred_fraction=inferred_fraction,
        config_hash=chash,
        input_hash=ihash,
        model=card,
        synthetic=scene.synthetic,
        scene_id=scene.scene_id,
        extra={
            "baselines": baselines,
            "coregistration": coreg_info,
            "mode": mode,
            "recon_sha256": hashlib.sha256(np.ascontiguousarray(recon).tobytes()).hexdigest(),
            "device": device,
            "cloudmask_method": masks.method,
        },
    )
    (out / "result.json").write_text(json.dumps(res.to_json(), indent=2), encoding="utf-8")
    log.info("run %s done: %s", rid, {k: round(v, 3) for k, v in timing.items()})
    return res


def existing_result(out_dir: Path, cfg: Config, ihash: str) -> Result | None:
    """Return the stored result if it was produced with the same config and input hash."""
    p = out_dir / "result.json"
    if not p.exists():
        return None
    try:
        r = Result.from_json(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        return None
    if r.config_hash == cfg.config_hash() and r.input_hash == ihash and r.geotiff.exists():
        return r
    return None
