"""Capture screenshots 01-22 and a full-length screen recording of the CloudLift console, API and CLI.

Prerequisites (running locally):
    pip install playwright            # uses the installed Chrome (channel="chrome"), no browser download
    CLOUDLIFT_CONFIG=configs/demo.yaml python -m cloudlift.ui.console      # :7860
    cloudlift serve --port 8000                                             # :8000

    python scripts/capture_walkthrough.py --out app_screenshots_screen_recording

The sample "user" files are SYNTHETIC: a procedural scene re-encoded as a 4-band uint16
Blue/Green/Red/NIR GeoTIFF (x10000) to demonstrate the any-sensor path.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
from playwright.sync_api import Page, sync_playwright

from cloudlift.io import geotiff
from cloudlift.io.synthetic import generate_scene

VIEW = {"width": 1440, "height": 900}
CONSOLE = "http://127.0.0.1:7860"
API = "http://127.0.0.1:8000"


def make_sample(root: Path) -> dict[str, Path]:
    """A synthetic scene written as a typical 4-band uint16 product + 2-band SAR + earlier pass."""
    d = generate_scene(root, "SAMPLE_USER_SCENE", size=768, seed=321, cloud_fraction=0.45)
    out: dict[str, Path] = {"sar": d / "sar.tif"}
    for name, src in (("optical", "liss4.tif"), ("reference", "reference.tif")):
        arr, meta = geotiff.read(d / src)
        four = np.concatenate([np.clip(arr[:1] * 0.8, 0, 1), arr]) * 10000
        out[name] = geotiff.write(
            root / f"user_{name}_BGRN_uint16.tif",
            np.round(four).astype(np.uint16),
            crs=meta.crs,
            transform=meta.transform,
            descriptions=("Blue", "Green", "Red", "NIR"),
        )
    return out


def shot(page: Page, out: Path, name: str, pause: float = 1.2) -> None:
    time.sleep(pause)
    page.screenshot(path=str(out / name))
    print("saved", name)


def wait_text(page: Page, text: str, timeout: float = 180) -> None:
    page.get_by_text(text).first.wait_for(timeout=timeout * 1000)


_SLIDER = {"pos": 0.5}


def _slider_box(page: Page) -> dict[str, float] | None:
    return page.evaluate(
        "() => { const imgs=[...document.querySelectorAll('img')].map(i=>i.getBoundingClientRect())"
        ".filter(r=>r.width>200).sort((a,b)=>b.width*b.height-a.width*a.height); const r=imgs[0];"
        " return r ? {x:r.x,y:r.y,w:r.width,h:r.height} : null }"
    )


def slide_to(page: Page, *targets: float) -> None:
    """Drag the before/after handle through ``targets`` (fractions of the image width)."""
    box = _slider_box(page)
    if not box:
        return
    y = box["y"] + box["h"] / 2
    page.mouse.move(box["x"] + box["w"] * _SLIDER["pos"], y)
    page.mouse.down()
    for frac in targets:
        page.mouse.move(box["x"] + box["w"] * frac, y, steps=15)
        time.sleep(0.3)
        _SLIDER["pos"] = frac
    page.mouse.up()


def demo_flow(page: Page, out: Path) -> None:
    """Screenshots 01-12: demo scene, slider, confidence, export, AOI live run."""
    page.goto(CONSOLE + "/")
    wait_text(page, "Initiate recovery")
    shot(page, out, "01_console_home.png", 3.0)
    page.get_by_label("Scene").click()
    shot(page, out, "02_select_scene.png", 1.0)
    page.get_by_role("option", name="HERO_KAMRUP_02").click()
    _SLIDER["pos"] = 0.5
    shot(page, out, "03_scene_loaded_cloudy.png", 3.0)
    page.get_by_role("button", name="Initiate recovery").click()
    shot(page, out, "04_pipeline_running.png", 0.35)
    wait_text(page, "Cached result", 300)
    shot(page, out, "05_reconstructed_slider.png", 2.0)
    slide_to(page, 0.7, 0.92)
    shot(page, out, "06_slider_cloudy_side.png", 1.0)
    slide_to(page, 0.5, 0.25, 0.08)
    shot(page, out, "07_slider_reconstructed_side.png", 1.0)
    slide_to(page, 0.3, 0.5)
    page.get_by_label("Show confidence").check()
    shot(page, out, "08_confidence_overlay.png", 2.5)
    page.get_by_label("Show confidence").uncheck()
    page.get_by_text("Export GeoTIFF").first.hover()
    shot(page, out, "09_export_geotiff.png", 1.5)
    aoi = page.get_by_label("AOI (any location)")
    aoi.fill("point:26.1477,91.6217,0.8km")
    shot(page, out, "10_aoi_input.png", 1.0)
    page.get_by_role("button", name="Initiate recovery").click()
    shot(page, out, "11_live_run_progress.png", 0.6)
    wait_text(page, "Live run", 300)
    _SLIDER["pos"] = 0.5
    shot(page, out, "12_live_aoi_result.png", 2.0)
    slide_to(page, 0.15, 0.85, 0.5)
    aoi.fill("")
    time.sleep(1)


def console_flow(page: Page, out: Path, sample: dict[str, Path]) -> None:
    page.goto(CONSOLE + "/")
    wait_text(page, "Initiate recovery")
    page.get_by_text("Use your own scene").click()
    page.get_by_text("Band order (G,R,NIR)").scroll_into_view_if_needed()
    shot(page, out, "13_upload_panel.png")
    # each File component drops its <input> once a file is attached, so always fill the first remaining one
    for key in ("optical", "sar", "reference"):
        before = page.locator("input[type=file]").count()
        page.locator("input[type=file]").first.set_input_files(str(sample[key]))
        page.wait_for_function(
            f"document.querySelectorAll('input[type=file]').length < {before}", timeout=60000
        )
        time.sleep(0.8)
    band_box = page.get_by_label("Band order (G,R,NIR)")
    band_box.fill("2,3,4")
    page.get_by_label("Acquired (YYYY-MM-DD)").fill("2026-07-15")
    page.get_by_label("Reference acquired").fill("2026-02-10")
    shot(page, out, "14_upload_files_attached.png", 2.0)
    page.get_by_role("button", name="Add scene").click()
    wait_text(page, "Press Initiate recovery", 60)
    page.mouse.wheel(0, -2000)
    shot(page, out, "15_upload_scene_added.png", 2.5)
    page.get_by_role("button", name="Initiate recovery").click()
    time.sleep(2.0)
    shot(page, out, "16_upload_recovery_running.png", 0.2)
    wait_text(page, "Live run", 300)
    shot(page, out, "17_upload_reconstructed.png", 2.0)
    page.get_by_label("Show confidence").check()
    shot(page, out, "18_upload_confidence.png", 2.0)
    page.get_by_label("Show confidence").uncheck()
    time.sleep(1.5)


def api_flow(page: Page, out: Path, sample: dict[str, Path]) -> None:
    page.goto(API + "/docs")
    page.get_by_text("/reconstruct").first.wait_for(timeout=60000)
    shot(page, out, "19_api_docs.png", 2.0)
    boundary = "----cloudlift"
    parts = []
    for role, key in (("liss4", "optical"), ("sar", "sar")):
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{role}"; filename="{role}.tif"\r\n'
            "Content-Type: image/tiff\r\n\r\n".encode()
            + sample[key].read_bytes()
            + b"\r\n"
        )
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="bands"\r\n\r\n2,3,4\r\n'.encode())
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        API + "/reconstruct",
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    job = json.loads(urllib.request.urlopen(req).read())["job_id"]
    for _ in range(300):
        j = json.loads(urllib.request.urlopen(f"{API}/jobs/{job}").read())
        if j["status"] in ("done", "failed"):
            break
        time.sleep(1)
    page.goto(f"{API}/jobs/{job}")
    shot(page, out, "20_api_job_done.png", 2.0)


def eval_panel(page: Page, out: Path) -> None:
    panels = sorted(Path("results/console").glob("UPLOAD_*/panels.png"), key=lambda p: p.stat().st_mtime)
    if panels:
        page.goto(panels[-1].resolve().as_uri())
        shot(page, out, "21_evaluation_panels.png", 1.5)


def cli_shot(page: Page, out: Path, sample: dict[str, Path]) -> None:
    """Render the real output of `cloudlift infer --optical ...` as a terminal-style page."""
    cmd = [
        sys.executable,
        "-m",
        "cloudlift.cli",
        "infer",
        "--config",
        "configs/demo.yaml",
        "--optical",
        str(sample["optical"]),
        "--sar",
        str(sample["sar"]),
        "--reference",
        str(sample["reference"]),
        "--bands",
        "2,3,4",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    text = proc.stdout[proc.stdout.find("{") :] if "{" in proc.stdout else proc.stdout + proc.stderr
    shown = (
        "$ cloudlift infer --config configs/demo.yaml --optical user_optical_BGRN_uint16.tif \\\n"
        "    --sar sar.tif --reference user_reference_BGRN_uint16.tif --bands 2,3,4\n\n" + text
    )
    html = (
        "<html><body style='margin:0;background:#0C121D;color:#E7EEF6;font:14px/1.45 Consolas,monospace'>"
        "<pre style='padding:24px;white-space:pre-wrap'>"
        + shown.replace("&", "&amp;").replace("<", "&lt;")
        + "</pre></body></html>"
    )
    page.goto("about:blank")
    page.set_content(html, wait_until="domcontentloaded")
    shot(page, out, "22_cli_infer_own_files.png", 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="app_screenshots_screen_recording")
    ap.add_argument("--video-name", default="cloudlift_full_walkthrough.mp4")
    ap.add_argument("--console", default=CONSOLE)
    ap.add_argument("--api", default=API)
    a = ap.parse_args()
    globals().update(CONSOLE=a.console.rstrip("/"), API=a.api.rstrip("/"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="cloudlift_capture_"))
    sample = make_sample(tmp / "sample")
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome")
        ctx = browser.new_context(viewport=VIEW, record_video_dir=str(tmp / "video"), record_video_size=VIEW)
        page = ctx.new_page()
        demo_flow(page, out)
        console_flow(page, out, sample)
        api_flow(page, out, sample)
        eval_panel(page, out)
        cli_shot(page, out, sample)
        video = page.video
        ctx.close()
        browser.close()
        webm = Path(video.path()) if video else None
    if webm and webm.exists() and shutil.which("ffmpeg"):
        mp4 = Path(a.out) / a.video_name
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(webm),
                "-c:v",
                "libx264",
                "-crf",
                "28",
                "-preset",
                "slow",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(mp4),
            ],
            check=True,
        )
        print("saved", mp4)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
