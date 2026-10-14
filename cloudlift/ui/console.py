"""CloudLift operator console (Gradio). Run: ``python -m cloudlift.ui.console`` -> http://localhost:7860

Every number on screen comes from the current run's result; unavailable values show "—".
Hero scenes pre-computed by ``scripts/bake_demo.py`` are replayed from ``cache/demo/`` so the
demo works offline and instantly; anything else runs the live 7-stage pipeline.
"""

from __future__ import annotations

import html
import math
import os
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np

from cloudlift.config import REPO_ROOT, Config, load_config
from cloudlift.eval.report import fcc, stretch_limits
from cloudlift.io.aoi import AOIError, parse_aoi
from cloudlift.io.sources import LocalFolderSource, Scene

CHROME, PANEL, TEAL, AMBER, RED, TEXT = "#0C121D", "#151E2D", "#12B09C", "#EF9F27", "#E24B4A", "#E7EEF6"
DEMO_CACHE = Path("cache/demo")
DEMO_SCENES = Path("data/demo_scenes")
UPLOADS = Path("data/uploads")

# SYNTHETIC demo scenes generated on first launch when no other scene exists: (id, seed, cloud fraction)
HEROES = [
    ("HERO_BRAHMAPUTRA_01", 101, 0.25),
    ("HERO_KAMRUP_02", 102, 0.40),
    ("HERO_HILLS_03", 103, 0.55),
    ("HERO_MONSOON_04", 104, 0.70),
]

# console stage list (spec) -> pipeline stage numbers that complete it
CONSOLE_STAGES = [
    ("Cloud detection", 3),
    ("SAR fusion", 2),
    ("Reconstruction", 4),
    ("Confidence", 5),
    ("NDVI validation", 6),
    ("Export GeoTIFF", 7),
]

# The console is designed dark-only; force Gradio's dark theme regardless of the OS setting.
FORCE_DARK = "() => { document.body.classList.add('dark'); document.documentElement.classList.add('dark'); }"

CSS = f"""
:root, .dark {{ --cl-chrome:{CHROME}; --cl-panel:{PANEL}; --cl-teal:{TEAL}; --cl-amber:{AMBER};
  --cl-red:{RED}; --cl-text:{TEXT}; }}
.gradio-container, body {{ background:{CHROME} !important; color:{TEXT} !important; }}
.cl-panel {{ background:{PANEL} !important; border:1px solid #22304a !important; border-radius:10px !important; }}
.cl-bar {{ display:flex; justify-content:space-between; align-items:center; background:{PANEL};
  border:1px solid #22304a; border-radius:10px; padding:10px 16px; font-family:ui-monospace,Consolas,monospace; }}
.cl-brand {{ font-weight:700; font-size:18px; letter-spacing:.5px; color:{TEXT}; }}
.cl-brand b {{ color:{TEAL}; }}
.cl-dim {{ color:#8a9bb3; font-size:12px; letter-spacing:1px; }}
.cl-live {{ color:{TEAL}; font-size:12px; }}
.cl-kv, .cl-stage, .cl-tile .v, .cl-big, .cl-card, .cl-queue {{ color:{TEXT} !important; }}
.cl-kv {{ font-size:13px; line-height:1.5; margin-bottom:8px; }}
.cl-kv .k {{ color:#8a9bb3; font-size:11px; text-transform:uppercase; letter-spacing:1px; }}
.cl-card {{ background:{CHROME}; border:1px solid #22304a; border-radius:8px; padding:10px; }}
.cl-big {{ font-size:28px; font-weight:700; }}
.cl-bar-track {{ height:6px; background:#22304a; border-radius:3px; overflow:hidden; margin-top:6px; }}
.cl-bar-fill {{ height:6px; background:{RED}; }}
.cl-stage {{ display:flex; align-items:center; gap:8px; padding:6px 0; font-size:14px; }}
.cl-dot {{ width:10px; height:10px; border-radius:50%; background:#2b3a55; flex:none; }}
.cl-dot.run {{ background:{AMBER}; box-shadow:0 0 6px {AMBER}; }}
.cl-dot.done {{ background:{TEAL}; }}
.cl-tiles {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }}
.cl-tile {{ background:{CHROME}; border:1px solid #22304a; border-radius:8px; padding:8px 10px; }}
.cl-tile .v {{ font-size:20px; font-weight:700; color:{TEXT}; }}
.cl-tile .k {{ font-size:11px; color:#8a9bb3; text-transform:uppercase; letter-spacing:1px; }}
.cl-queue {{ display:flex; gap:8px; flex-wrap:wrap; }}
.cl-chip {{ border:1px solid #22304a; border-radius:14px; padding:4px 10px; font-size:12px; background:{CHROME}; }}
.cl-chip.done {{ border-color:{TEAL}; color:{TEAL}; }}
.cl-chip.queued {{ border-color:{AMBER}; color:{AMBER}; }}
.cl-badge {{ display:inline-block; border-radius:4px; padding:1px 6px; font-size:11px; margin-left:6px; }}
.cl-badge.syn {{ background:{AMBER}; color:{CHROME}; }}
.cl-badge.warn {{ background:{RED}; color:white; }}
.cl-legend span {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin:0 4px 0 10px; }}
button.primary {{ background:{TEAL} !important; border-color:{TEAL} !important; color:{CHROME} !important; }}
"""


# --------------------------------------------------------------------------------------------
# data helpers
# --------------------------------------------------------------------------------------------
def console_config() -> Config:
    path = os.environ.get("CLOUDLIFT_CONFIG")
    if path is None and (REPO_ROOT / "configs" / "demo.yaml").exists():
        path = str(REPO_ROOT / "configs" / "demo.yaml")
    cfg = load_config(path)
    if cfg.paths.weights is not None and not Path(cfg.paths.weights).exists():
        cfg.paths.weights = None  # fall back to the (clearly labelled) untrained model
    return cfg


def ensure_demo_scenes(size: int = 768) -> None:
    """Generate the synthetic hero scenes if missing (a few seconds each), so a fresh install works."""
    from cloudlift.io.synthetic import generate_scene

    for sid, seed, frac in HEROES:
        if not (DEMO_SCENES / sid / "meta.json").exists():
            generate_scene(DEMO_SCENES, sid, size=size, seed=seed, cloud_fraction=frac)


def sources(cfg: Config) -> list[LocalFolderSource]:
    out = [LocalFolderSource(UPLOADS), LocalFolderSource(cfg.paths.data_root), LocalFolderSource(DEMO_SCENES)]
    return [s for s in out if s.list_scenes()]


SCALE_CHOICES: dict[str, str | int] = {
    "auto (detect)": "auto",
    "10-bit DN (÷1023)": 1023,
    "÷10000": 10000,
    "already 0–1": 1,
}
MAX_FULL_SCENE_PX = 4096 * 4096  # larger scenes need an AOI crop (keeps RAM bounded)


def add_uploaded_scene(
    liss4: str | None,
    sar: str | None,
    reference: str | None,
    acquired: str,
    reference_acquired: str,
    bands: str = "1,2,3",
    scale: str = "auto (detect)",
) -> Scene:
    """Validate and store user GeoTIFFs (file paths from the UI) as a new scene."""
    from cloudlift.io.sources import UploadError, save_scene

    files = {"liss4": liss4, "sar": sar, "reference": reference}
    blobs = {k: Path(v).read_bytes() for k, v in files.items() if v}
    band_list = [b for b in (bands or "1,2,3").replace(" ", "").split(",") if b]
    if not all(b.isdigit() for b in band_list):
        raise UploadError(f"band order must be three numbers like 2,3,4 (got {bands!r})")
    meta = {
        "acquired": acquired.strip(),
        "reference_acquired": reference_acquired.strip(),
        "source": "user upload",
        "bands": [int(b) for b in band_list],
        "scale": SCALE_CHOICES.get(scale, "auto"),
    }
    return save_scene(UPLOADS, blobs, meta)


def scene_choices(cfg: Config) -> list[str]:
    return [s.scene_id for src in sources(cfg) for s in src.list_scenes()]


def find_scene(cfg: Config, scene_id: str) -> Scene | None:
    for src in sources(cfg):
        try:
            return src.get_scene(scene_id)
        except KeyError:
            continue
    return None


def baked(cfg: Config, scene: Scene) -> Any:
    from cloudlift import pipeline

    return pipeline.existing_result(DEMO_CACHE / scene.scene_id, cfg, pipeline.input_hash(scene, None))


# --------------------------------------------------------------------------------------------
# HTML fragments (only computed values; "—" otherwise)
# --------------------------------------------------------------------------------------------
def _num(v: Any, fmt: str) -> str:
    if v is None or isinstance(v, str) or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    return format(v, fmt)


def status_bar() -> str:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    return (
        f"<div class='cl-bar'><div><span class='cl-brand'>Cloud<b>Lift</b></span>"
        f"&nbsp;&nbsp;<span class='cl-dim'>NORTH-EAST REGION · GROUND STATION</span></div>"
        f"<div class='cl-dim'>{now}</div>"
        f"<div class='cl-live'>● BHOONIDHI · NRSC/ISRO</div></div>"
    )


def scene_panel(scene: Scene | None, cfg: Config) -> str:
    if scene is None:
        return "<div class='cl-kv'>No scene selected.</div>"
    syn = "<span class='cl-badge syn'>SYNTHETIC FIXTURE</span>" if scene.synthetic else ""
    upload = scene.meta.get("source") == "user upload"
    w, h = scene_size(scene)
    bands = scene.meta.get("bands", [1, 2, 3])
    sensor = (
        f"User optical GeoTIFF · G/R/NIR = bands {', '.join(str(b) for b in bands)}"
        if upload
        else "LISS-IV · 5.8 m · G/R/NIR"
    )
    missing = "<i>missing (higher uncertainty)</i>"
    ref_txt = (
        html.escape(str(scene.meta.get("reference_acquired") or "provided")) if scene.reference else missing
    )
    sar_txt = ("C-band, 2-pol" if upload else "EOS-04 / RISAT-1A · C-band") if scene.sar else missing
    src_txt = (
        "User-supplied files, kept on this server"
        if upload
        else "Bhoonidhi (NRSC/ISRO) · Indian Space Policy 2023"
    )
    rows = [
        (
            "Scene",
            f"{html.escape(scene.scene_id)}{syn}<br>{sensor}<br><span class='cl-dim'>{w} × {h} px</span>",
        ),
        ("Acquired", html.escape(scene.acquired)),
        ("Temporal reference", ref_txt),
        ("Auxiliary SAR", sar_txt),
        ("Source · compliance", src_txt),
    ]
    return "".join(f"<div class='cl-kv'><div class='k'>{k}</div>{v}</div>" for k, v in rows)


def model_badge(cfg: Config) -> str:
    if cfg.paths.weights is None:
        return "<div class='cl-kv'><div class='k'>Model</div>UnCRtainTS <span class='cl-badge warn'>UNTRAINED</span></div>"
    return (
        f"<div class='cl-kv'><div class='k'>Model</div>UnCRtainTS ({cfg.model.size})"
        f"<br><span class='cl-dim'>{html.escape(Path(cfg.paths.weights).name)}</span></div>"
    )


def obstruction_card(res: dict[str, Any] | None) -> str:
    cloud = res.get("cloud_fraction") if res else None
    obst = res.get("obstructed_fraction") if res else None
    pct = _num(cloud * 100 if cloud is not None else None, ".1f")
    width = 0 if cloud is None else min(100, cloud * 100)
    shadow = _num((obst - cloud) * 100 if (obst is not None and cloud is not None) else None, ".1f")
    return (
        f"<div class='cl-card'><div class='cl-dim'>OBSTRUCTION</div><div class='cl-big'>{pct}"
        f"{'%' if pct != '—' else ''}</div><div class='cl-dim'>cloud cover · shadow {shadow}"
        f"{'%' if shadow != '—' else ''}</div><div class='cl-bar-track'><div class='cl-bar-fill' "
        f"style='width:{width:.1f}%'></div></div></div>"
    )


def stages_html(done: set[int], running: int | None) -> str:
    out = []
    for name, n in CONSOLE_STAGES:
        cls = "done" if n in done else ("run" if running == n else "")
        out.append(f"<div class='cl-stage'><div class='cl-dot {cls}'></div>{name}</div>")
    return "".join(out)


def metrics_html(res: dict[str, Any] | None) -> str:
    m = (res or {}).get("metrics", {})
    ndvi = m.get("ndvi_r2", m.get("ndvi_r2_vs_reference"))
    sam = m.get("sam_deg_masked", m.get("sam_deg_vs_reference_masked"))
    ssim = m.get("ssim_masked")
    rec = (res or {}).get("inferred_fraction")
    tiles = [
        ("NDVI R²", _num(ndvi, ".3f")),
        ("SAM", _num(sam, ".2f") + ("°" if _num(sam, ".2f") != "—" else "")),
        ("SSIM", _num(ssim, ".3f")),
        ("Recovered", _num(rec * 100 if rec is not None else None, ".1f") + ("%" if rec is not None else "")),
    ]
    body = "".join(
        f"<div class='cl-tile'><div class='k'>{k}</div><div class='v'>{v}</div></div>" for k, v in tiles
    )
    note = ""
    if res:
        mode = res.get("extra", {}).get("mode", "")
        note = (
            "<div class='cl-dim' style='margin-top:6px'>vs cloud-free truth over cloud/shadow pixels</div>"
            if mode == "full-reference"
            else "<div class='cl-dim' style='margin-top:6px'>vs temporal reference (no ground truth)</div>"
        )
        if not res.get("model", {}).get("trained", False):
            note += "<div class='cl-dim' style='color:#E24B4A'>untrained model: pipeline test only</div>"
        ms = res.get("timing", {}).get("ms_per_tile")
        note += f"<div class='cl-dim'>{_num(ms, '.0f')} ms/tile · {html.escape(res.get('extra', {}).get('device', ''))}</div>"
    return f"<div class='cl-tiles'>{body}</div>{note}"


def queue_html(cfg: Config, active: str | None = None) -> str:
    chips = []
    for sid in scene_choices(cfg):
        sc = find_scene(cfg, sid)
        st = "done" if sc is not None and baked(cfg, sc) is not None else "queued"
        if sid == active:
            st = "done"
        chips.append(f"<span class='cl-chip {st}'>{html.escape(sid)} · {st}</span>")
    return "<div class='cl-dim'>BATCH TILE QUEUE</div><div class='cl-queue'>" + "".join(chips) + "</div>"


# --------------------------------------------------------------------------------------------
# images
# --------------------------------------------------------------------------------------------
def _load_arrays(res: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = Path(res["out_dir"])
    return np.load(d / "cloudy.npy"), np.load(d / "recon.npy"), np.load(d / "uncertainty.npy")


def _limits(recon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return stretch_limits(recon, 1.0, 99.0)


def scene_size(scene: Scene) -> tuple[int, int]:
    from cloudlift.io import geotiff

    m = geotiff.read_meta(scene.optical)
    return m.width, m.height


def preview_cloudy(scene: Scene, max_px: int = 1024) -> np.ndarray:
    """Decimated false-colour preview, so even full-size scenes load quickly."""
    import rasterio

    from cloudlift.io.radiometry import optical_to_unit
    from cloudlift.pipeline import radiometry

    opt_meta, _ = radiometry(scene)
    with rasterio.open(scene.optical) as ds:
        f = max(1, -(-max(ds.width, ds.height) // max_px))
        raw = ds.read(opt_meta["bands"], out_shape=(3, max(1, ds.height // f), max(1, ds.width // f)))
    opt = optical_to_unit(raw, opt_meta)
    return fcc(opt, stretch_limits(opt, 1.0, 60.0))


def slider_images(res: dict[str, Any], show_conf: bool) -> tuple[np.ndarray, np.ndarray]:
    cloudy, recon, u = _load_arrays(res)
    lim = _limits(recon)
    before, after = fcc(cloudy, lim), fcc(recon, lim)
    if show_conf:
        inside = u > 0
        if inside.any():
            vmax = float(np.percentile(u[inside], 95))
            t = np.clip(u / max(vmax, 1e-9), 0, 1)[..., None]
            teal, amber = np.array([0x12, 0xB0, 0x9C], np.float32), np.array([0xEF, 0x9F, 0x27], np.float32)
            color = (1 - t) * teal + t * amber
            a = 0.55 * inside[..., None]
            after = (after * (1 - a) + color * a).astype(np.uint8)
    return before, after


# --------------------------------------------------------------------------------------------
# app
# --------------------------------------------------------------------------------------------
def build(cfg: Config | None = None) -> gr.Blocks:
    cfg = cfg or console_config()
    if not scene_choices(cfg):
        ensure_demo_scenes()
    choices = scene_choices(cfg)

    with gr.Blocks(title="CloudLift — operator console") as demo:
        state = gr.State(None)
        clock = gr.HTML(status_bar())
        gr.Timer(1.0).tick(status_bar, outputs=clock)

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=260, elem_classes="cl-panel"):
                gr.HTML("<div class='cl-dim'>ACTIVE TILE</div>")
                scene_dd = gr.Dropdown(choices=choices, value=choices[0] if choices else None, label="Scene")
                aoi_tb = gr.Textbox(
                    label="AOI (any location)",
                    placeholder="bbox:91.6,26.0,91.9,26.3 · "
                    "point:26.14,91.74,5km · tile:<ID>  (blank = full scene)",
                )
                info = gr.HTML(scene_panel(find_scene(cfg, choices[0]) if choices else None, cfg))
                gr.HTML(model_badge(cfg))
                obstruction = gr.HTML(obstruction_card(None))
                with gr.Accordion("Use your own scene", open=False):
                    gr.Markdown(
                        "GeoTIFFs with a CRS. The **optical** file (LISS-IV, or any sensor with Green, Red and "
                        "NIR bands) is required. SAR (VV, VH) and an earlier cloud-free pass of the same area "
                        "are optional but strongly improve results. Files stay on this server."
                    )
                    up_liss4 = gr.File(
                        label="Optical, cloudy (3+ bands incl. Green, Red, NIR)",
                        file_types=[".tif", ".tiff"],
                        type="filepath",
                    )
                    up_sar = gr.File(
                        label="SAR, optional (2 bands)", file_types=[".tif", ".tiff"], type="filepath"
                    )
                    up_ref = gr.File(
                        label="Earlier cloud-free pass, optional (3 bands)",
                        file_types=[".tif", ".tiff"],
                        type="filepath",
                    )
                    with gr.Row():
                        up_bands = gr.Textbox(
                            label="Band order (G,R,NIR)", value="1,2,3", info="e.g. 2,3,4 for B,G,R,NIR files"
                        )
                        up_scale = gr.Dropdown(
                            choices=list(SCALE_CHOICES), value="auto (detect)", label="Scaling"
                        )
                    with gr.Row():
                        up_acq = gr.Textbox(label="Acquired (YYYY-MM-DD)", placeholder="optional")
                        up_ref_acq = gr.Textbox(label="Reference acquired", placeholder="optional")
                    up_btn = gr.Button("Add scene")
                    up_msg = gr.Markdown("")

            with gr.Column(scale=3, min_width=420):
                slider = gr.ImageSlider(
                    label="Cloudy  ↔  Reconstructed  (false colour: NIR/Red/Green)",
                    type="numpy",
                    max_height=640,
                    slider_position=50,
                )
                with gr.Row():
                    go = gr.Button("Initiate recovery", variant="primary")
                    reset = gr.Button("Reset")
                    export = gr.DownloadButton("↓ Export GeoTIFF", interactive=False)
                note = gr.Markdown("")

            with gr.Column(scale=1, min_width=260, elem_classes="cl-panel"):
                gr.HTML("<div class='cl-dim'>PIPELINE</div>")
                stages = gr.HTML(stages_html(set(), None))
                conf = gr.Checkbox(label="Show confidence", value=False)
                gr.HTML(
                    f"<div class='cl-legend cl-dim'><span style='background:{TEAL}'></span>SAR-constrained · "
                    f"low σ<span style='background:{AMBER}'></span>interpolated · high σ</div>"
                )
                gr.HTML("<div class='cl-dim' style='margin-top:10px'>METRICS (this run)</div>")
                metrics = gr.HTML(metrics_html(None))

        queue = gr.HTML(queue_html(cfg), elem_classes="cl-panel")

        def on_scene(sid: str | None) -> tuple[str, Any, str, Any, str, str, Any]:
            sc = find_scene(cfg, sid) if sid else None
            img = preview_cloudy(sc) if sc is not None else None
            return (
                scene_panel(sc, cfg),
                (img, img) if img is not None else None,
                obstruction_card(None),
                gr.update(value=None, interactive=False),
                stages_html(set(), None),
                metrics_html(None),
                None,
            )

        def recover(sid: str | None, aoi_text: str, show: bool) -> Iterator[tuple[Any, ...]]:
            from cloudlift import pipeline

            sc = find_scene(cfg, sid) if sid else None
            if sc is None:
                yield (
                    gr.update(),
                    stages_html(set(), None),
                    metrics_html(None),
                    obstruction_card(None),
                    gr.update(interactive=False),
                    None,
                    "Select a scene first.",
                    queue_html(cfg),
                )
                return
            aoi = None
            if aoi_text and aoi_text.strip():
                try:
                    aoi = parse_aoi(aoi_text, LocalFolderSource(sc.optical.parent.parent).resolve_tile)
                except AOIError as e:
                    yield (
                        gr.update(),
                        stages_html(set(), None),
                        metrics_html(None),
                        obstruction_card(None),
                        gr.update(interactive=False),
                        None,
                        f"AOI error: {e}",
                        queue_html(cfg),
                    )
                    return
            w, h = scene_size(sc)
            if aoi is None and w * h > MAX_FULL_SCENE_PX:
                yield (
                    gr.update(),
                    stages_html(set(), None),
                    metrics_html(None),
                    obstruction_card(None),
                    gr.update(interactive=False),
                    None,
                    f"This scene is {w} × {h} px. Enter an AOI to crop it (e.g. `point:lat,lon,5km`), "
                    "or process the full scene with the CLI or API.",
                    queue_html(cfg),
                )
                return
            hit = baked(cfg, sc) if aoi is None else None
            done: set[int] = set()
            if hit is not None:
                res = hit.to_json()
                for _, n in CONSOLE_STAGES:  # replay stage list for a cached bake
                    yield (
                        gr.update(),
                        stages_html(done, n),
                        metrics_html(None),
                        obstruction_card(None),
                        gr.update(interactive=False),
                        None,
                        "Replaying pre-computed result…",
                        queue_html(cfg),
                    )
                    time.sleep(0.15)
                    done.add(n)
                source_note = (
                    f"Cached result from `scripts/bake_demo.py` (same code, config `{res['config_hash']}`)"
                    f" · computed {res['timing'].get('ms_per_tile', 0):.0f} ms/tile on "
                    f"{res['extra'].get('device', '?')}."
                )
            else:
                status: dict[str, Any] = {"running": None, "done": set(), "res": None, "err": None}

                def progress(i: int, _name: str, st: str) -> None:
                    if st == "running":
                        status["running"] = i
                    else:
                        status["done"].add(i)

                def work() -> None:
                    try:
                        out = Path(cfg.paths.results_root) / "console"
                        r = pipeline.run(
                            sc,
                            aoi,
                            cfg,
                            progress=progress,
                            out_dir=out / pipeline.run_id_for(sc, cfg, pipeline.input_hash(sc, aoi)),
                        )
                        status["res"] = r.to_json()
                    except Exception as e:  # surfaced in the UI
                        status["err"] = f"{type(e).__name__}: {e}"

                th = threading.Thread(target=work, daemon=True)
                th.start()
                while th.is_alive():
                    yield (
                        gr.update(),
                        stages_html(set(status["done"]), status["running"]),
                        metrics_html(None),
                        obstruction_card(None),
                        gr.update(interactive=False),
                        None,
                        "Running live pipeline…",
                        queue_html(cfg),
                    )
                    time.sleep(0.3)
                if status["err"]:
                    yield (
                        gr.update(),
                        stages_html(set(status["done"]), None),
                        metrics_html(None),
                        obstruction_card(None),
                        gr.update(interactive=False),
                        None,
                        f"Run failed: {status['err']}",
                        queue_html(cfg),
                    )
                    return
                res = status["res"]
                done = {n for _, n in CONSOLE_STAGES}
                source_note = f"Live run `{res['run_id']}` on {res['extra'].get('device', '?')}."
            if res.get("synthetic"):
                source_note += " **Synthetic fixture scene**: not real satellite data."
            yield (
                slider_images(res, show),
                stages_html(done, None),
                metrics_html(res),
                obstruction_card(res),
                gr.update(value=res["geotiff"], interactive=True),
                res,
                source_note,
                queue_html(cfg, sid),
            )

        def on_upload(
            liss4: str | None,
            sar: str | None,
            ref: str | None,
            acq: str,
            ref_acq: str,
            bands: str,
            scale: str,
        ) -> tuple[Any, str]:
            from cloudlift.io.sources import UploadError

            try:
                sc = add_uploaded_scene(liss4, sar, ref, acq or "", ref_acq or "", bands, scale)
            except UploadError as e:
                return gr.update(), f"Upload rejected: {e}"
            missing = [n for n, p in (("SAR", sar), ("earlier pass", ref)) if not p]
            extra = f" Missing {', '.join(missing)}: uncertainty will be higher." if missing else ""
            return (
                gr.update(choices=scene_choices(cfg), value=sc.scene_id),
                f"Added `{sc.scene_id}`. Press **Initiate recovery**.{extra}",
            )

        def on_conf(show: bool, res: dict[str, Any] | None) -> Any:
            return slider_images(res, show) if res else gr.update()

        def on_reset(sid: str | None) -> tuple[Any, ...]:
            return (*on_scene(sid), "", False)

        outs = [slider, stages, metrics, obstruction, export, state, note, queue]
        scene_dd.change(on_scene, scene_dd, [info, slider, obstruction, export, stages, metrics, state])
        go.click(recover, [scene_dd, aoi_tb, conf], outs)
        conf.change(on_conf, [conf, state], slider)
        up_btn.click(
            on_upload,
            [up_liss4, up_sar, up_ref, up_acq, up_ref_acq, up_bands, up_scale],
            [scene_dd, up_msg],
        )
        reset.click(
            on_reset, scene_dd, [info, slider, obstruction, export, stages, metrics, state, note, conf]
        )
        demo.load(on_scene, scene_dd, [info, slider, obstruction, export, stages, metrics, state])
    return demo


def launch(host: str = "127.0.0.1", port: int = 7860) -> None:
    cfg = console_config()
    allowed = [str(Path(p).resolve()) for p in (cfg.paths.results_root, DEMO_CACHE, UPLOADS)]
    ensure_demo_scenes()
    build(cfg).launch(
        server_name=host,
        server_port=port,
        css=CSS,
        theme=gr.themes.Base(),
        js=FORCE_DARK,
        allowed_paths=allowed,
        footer_links=[],
    )


if __name__ == "__main__":
    launch(
        host=os.environ.get("CLOUDLIFT_HOST", "127.0.0.1"), port=int(os.environ.get("CLOUDLIFT_PORT", "7860"))
    )
