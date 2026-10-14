"""Build (and optionally upload) the free *static* Hugging Face Space from baked results.

Hugging Face hosts Docker/Gradio Spaces on free CPU only for PRO accounts; static Spaces are
free. This page therefore shows results PRE-COMPUTED by ``scripts/bake_demo.py`` (the real
pipeline + shipped weights) -- no inference runs in the browser. Every number on the page is
read from each run's ``result.json``; nothing is typed in by hand.

    python scripts/bake_demo.py
    python scripts/build_static_demo.py [--upload gurjasgandhi76/cloudlift]
"""

from __future__ import annotations

import argparse
import html
import json
import math
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from cloudlift.ui.console import DEMO_CACHE, DEMO_SCENES, slider_images

OUT = Path("build/hf_static")
REPO_URL = "https://github.com/Gurjas2112/CloudLift"


def _f(v: Any, fmt: str, suffix: str = "") -> str:
    if v is None or isinstance(v, str) or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    return format(v, fmt) + suffix


def scene_record(res: dict[str, Any], meta: dict[str, Any], assets: Path) -> dict[str, Any]:
    sid = res["scene_id"]
    d = assets / sid
    d.mkdir(parents=True, exist_ok=True)
    before, after = slider_images(res, show_conf=False)
    _, conf = slider_images(res, show_conf=True)
    for name, arr in (("before", before), ("after", after), ("confidence", conf)):
        Image.fromarray(arr).save(d / f"{name}.webp", quality=90)
    shutil.copyfile(res["panels"], d / "panels.png")
    shutil.copyfile(res["geotiff"], d / f"{sid}_cloudfree.tif")
    m, base = res["metrics"], res["extra"].get("baselines", {})
    tf = base.get("temporal_fill", {})
    return {
        "id": sid,
        "acquired": meta.get("acquired", "—"),
        "reference": meta.get("reference_acquired", "—"),
        "cloud": _f(res["cloud_fraction"] * 100, ".1f", "%"),
        "shadow": _f((res["obstructed_fraction"] - res["cloud_fraction"]) * 100, ".1f", "%"),
        "cloud_pct": round(res["cloud_fraction"] * 100, 1),
        "tiles": {
            "NDVI R²": _f(m.get("ndvi_r2"), ".3f"),
            "SAM": _f(m.get("sam_deg_masked"), ".2f", "°"),
            "SSIM": _f(m.get("ssim_masked"), ".3f"),
            "Recovered": _f(res["inferred_fraction"] * 100, ".1f", "%"),
        },
        "compare": [
            [
                "CloudLift UnCRtainTS",
                _f(m.get("psnr_masked"), ".2f"),
                _f(m.get("sam_deg_masked"), ".2f"),
                _f(m.get("ndvi_r2"), ".3f"),
            ],
            [
                "Temporal fill (copy reference)",
                _f(tf.get("psnr_masked"), ".2f"),
                _f(tf.get("sam_deg_masked"), ".2f"),
                _f(tf.get("ndvi_r2"), ".3f"),
            ],
        ],
        "ece": _f(m.get("ece"), ".3f"),
        "iou": _f(m.get("cloud_detection_iou"), ".3f"),
        "ms": _f(res["timing"].get("ms_per_tile"), ".0f"),
        "device": res["extra"].get("device", "?"),
        "weights": res["model"]["weights_hash"],
        "config": res["config_hash"],
        "run": res["run_id"],
    }


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CloudLift console</title>
<style>
:root{--chrome:#0C121D;--panel:#151E2D;--line:#22304a;--teal:#12B09C;--amber:#EF9F27;--red:#E24B4A;--text:#E7EEF6;--dim:#8a9bb3}
*{box-sizing:border-box}
body{margin:0;background:var(--chrome);color:var(--text);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding:16px}
.bar{display:flex;flex-wrap:wrap;gap:8px 16px;justify-content:space-between;align-items:center;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 16px;font-family:ui-monospace,Consolas,monospace}
.brand{font-weight:700;font-size:18px}.brand b{color:var(--teal)}
.dim{color:var(--dim);font-size:12px;letter-spacing:1px}.live{color:var(--teal);font-size:12px}
.note{margin:12px 0;padding:10px 14px;border-left:3px solid var(--amber);background:var(--panel);border-radius:6px;font-size:13px}
.note a{color:var(--teal)}
.grid{display:grid;grid-template-columns:260px minmax(0,1fr) 260px;gap:12px;margin-top:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
.k{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:1px}
.kv{margin-bottom:10px}
select{width:100%;background:var(--chrome);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:8px;font:inherit}
.big{font-size:28px;font-weight:700}
.track{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin-top:6px}.fill{height:6px;background:var(--red)}
.badge{display:inline-block;background:var(--amber);color:var(--chrome);border-radius:4px;padding:1px 6px;font-size:11px;font-weight:600}
.slider{position:relative;width:100%;aspect-ratio:1/1;overflow:hidden;border-radius:8px;border:1px solid var(--line);background:#000;user-select:none;touch-action:none}
.slider img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;pointer-events:none}
.slider .top{clip-path:inset(0 0 0 var(--pos,50%))}
.handle{position:absolute;top:0;bottom:0;left:var(--pos,50%);width:2px;background:var(--text);box-shadow:0 0 6px #000}
.handle::after{content:"\\2194";position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);background:var(--teal);color:var(--chrome);border-radius:50%;width:30px;height:30px;display:grid;place-items:center;font-weight:700}
.lbl{position:absolute;top:8px;padding:2px 8px;border-radius:4px;background:#0C121Dcc;font-size:12px}
.lbl.l{left:8px}.lbl.r{right:8px}
.btns{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
.btn{background:var(--chrome);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:8px 12px;font:inherit;cursor:pointer;text-decoration:none}
.btn.primary{background:var(--teal);border-color:var(--teal);color:var(--chrome);font-weight:600}
.stage{display:flex;gap:8px;align-items:center;padding:5px 0}.dot{width:10px;height:10px;border-radius:50%;background:var(--teal)}
.tiles{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.tile{background:var(--chrome);border:1px solid var(--line);border-radius:8px;padding:8px 10px}.tile .v{font-size:20px;font-weight:700}
label.tog{display:flex;gap:8px;align-items:center;margin:10px 0;cursor:pointer}
.legend span{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 6px}
table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}
.queue{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}
.chip{border:1px solid var(--teal);color:var(--teal);border-radius:14px;padding:4px 10px;font-size:12px;background:var(--chrome);cursor:pointer}
.chip.on{background:var(--teal);color:var(--chrome)}
.panels{width:100%;border-radius:8px;border:1px solid var(--line);margin-top:8px;background:#fff}
.mono{font-family:ui-monospace,Consolas,monospace;font-size:12px;color:var(--dim);word-break:break-all}
@media (max-width:980px){.grid{grid-template-columns:1fr}}
</style></head>
<body><div class="wrap">
<div class="bar"><div><span class="brand">Cloud<b>Lift</b></span>&nbsp;&nbsp;<span class="dim">NORTH-EAST REGION · GROUND STATION</span></div>
<div class="dim" id="clock">—</div><div class="live">● BHOONIDHI · NRSC/ISRO</div></div>

<div class="note"><b>Pre-computed demo.</b> These four <b>synthetic</b> scenes (procedural NER-like terrain, not real LISS-IV)
were processed by the real CloudLift pipeline with weights fine-tuned on synthetic data only. Every number below is read from
that run's output. Nothing runs in your browser. For live runs on your own AOI, use the Docker or local console:
<a href="__REPO__">source and setup</a>.</div>

<div class="grid">
 <div class="card">
  <div class="k">Active tile</div>
  <select id="scene" aria-label="Scene"></select>
  <div class="kv" style="margin-top:10px"><div class="k">Scene</div><span id="sid"></span> <span class="badge">SYNTHETIC</span><br>LISS-IV · 5.8 m · G/R/NIR</div>
  <div class="kv"><div class="k">Acquired</div><span id="acq"></span></div>
  <div class="kv"><div class="k">Temporal reference</div><span id="ref"></span></div>
  <div class="kv"><div class="k">Auxiliary SAR</div>EOS-04 / RISAT-1A · C-band</div>
  <div class="kv"><div class="k">Source · compliance</div>Bhoonidhi (NRSC/ISRO) · Indian Space Policy 2023</div>
  <div class="tile"><div class="k">Obstruction</div><div class="big" id="cloud"></div><div class="dim">cloud cover · shadow <span id="shadow"></span></div>
   <div class="track"><div class="fill" id="fill"></div></div></div>
 </div>

 <div class="card">
  <div class="slider" id="slider" role="slider" aria-label="Before and after" aria-valuemin="0" aria-valuemax="100" aria-valuenow="50" tabindex="0">
   <img id="imgBefore" alt="Cloudy input, false colour">
   <img id="imgAfter" class="top" alt="Reconstructed, false colour">
   <div class="handle"></div><span class="lbl l">Cloudy</span><span class="lbl r">Reconstructed</span>
  </div>
  <div class="dim" style="margin-top:6px">LISS-IV standard false colour (R = NIR, G = Red, B = Green). Drag to compare.</div>
  <div class="btns"><a class="btn primary" id="dl" download>↓ Export GeoTIFF</a><a class="btn" id="dlp" target="_blank">Evaluation panels</a></div>
  <img class="panels" id="panels" alt="cloudy | reconstructed | truth | uncertainty | NDVI difference">
 </div>

 <div class="card">
  <div class="k">Pipeline</div>
  <div class="stage"><span class="dot"></span>Cloud detection</div><div class="stage"><span class="dot"></span>SAR fusion</div>
  <div class="stage"><span class="dot"></span>Reconstruction</div><div class="stage"><span class="dot"></span>Confidence</div>
  <div class="stage"><span class="dot"></span>NDVI validation</div><div class="stage"><span class="dot"></span>Export GeoTIFF</div>
  <label class="tog"><input type="checkbox" id="conf"> Show confidence</label>
  <div class="dim legend"><span style="background:var(--teal)"></span>low σ <span style="background:var(--amber)"></span>high σ</div>
  <div class="k" style="margin-top:12px">Metrics (this run, cloud + shadow pixels vs truth)</div>
  <div class="tiles" id="tiles"></div>
  <div class="dim" style="margin-top:8px">ECE <span id="ece"></span> · mask IoU <span id="iou"></span><br><span id="ms"></span> ms/tile on <span id="dev"></span></div>
 </div>
</div>

<div class="card" style="margin-top:12px">
 <div class="k">Same pixels, comparison with a non-learned baseline</div>
 <table><thead><tr><th>method</th><th>PSNR dB</th><th>SAM °</th><th>NDVI R²</th></tr></thead><tbody id="cmp"></tbody></table>
 <div class="dim" style="margin-top:6px">On this synthetic fixture the reference pass is nearly identical to the truth, so copying it is hard to beat.
 The network adds calibrated σ and SAR evidence for changed or missing references.</div>
 <div class="k" style="margin-top:12px">Batch tile queue</div><div class="queue" id="queue"></div>
 <div class="mono" style="margin-top:10px" id="prov"></div>
</div>
</div>
<script>
const S=__DATA__;
const $=id=>document.getElementById(id);
const sel=$("scene");let cur=0;
S.forEach((s,i)=>{const o=document.createElement("option");o.value=i;o.textContent=s.id;sel.appendChild(o)});
function show(i){cur=i;const s=S[i];sel.value=i;
 $("sid").textContent=s.id;$("acq").textContent=s.acquired;$("ref").textContent=s.reference;
 $("cloud").textContent=s.cloud;$("shadow").textContent=s.shadow;$("fill").style.width=s.cloud_pct+"%";
 $("imgBefore").src=`scenes/${s.id}/before.webp`;
 $("imgAfter").src=`scenes/${s.id}/${$("conf").checked?"confidence":"after"}.webp`;
 $("panels").src=`scenes/${s.id}/panels.png`;$("dlp").href=`scenes/${s.id}/panels.png`;
 $("dl").href=`scenes/${s.id}/${s.id}_cloudfree.tif`;
 $("tiles").innerHTML=Object.entries(s.tiles).map(([k,v])=>`<div class="tile"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
 $("ece").textContent=s.ece;$("iou").textContent=s.iou;$("ms").textContent=s.ms;$("dev").textContent=s.device;
 $("cmp").innerHTML=s.compare.map(r=>`<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td><td>${r[3]}</td></tr>`).join("");
 $("queue").innerHTML=S.map((q,j)=>`<span class="chip ${j===i?"on":""}" data-i="${j}">${q.id} · done</span>`).join("");
 $("prov").textContent=`run ${s.run} · weights ${s.weights} · config ${s.config}`;}
sel.onchange=e=>show(+e.target.value);
$("queue").onclick=e=>{const j=e.target.dataset.i;if(j!==undefined)show(+j)};
$("conf").onchange=()=>show(cur);
const sl=$("slider");
function setPos(p){p=Math.max(0,Math.min(100,p));sl.style.setProperty("--pos",p+"%");sl.setAttribute("aria-valuenow",Math.round(p))}
function fromEvt(e){const r=sl.getBoundingClientRect();setPos((e.clientX-r.left)/r.width*100)}
let drag=false;sl.addEventListener("pointerdown",e=>{drag=true;sl.setPointerCapture(e.pointerId);fromEvt(e)});
sl.addEventListener("pointermove",e=>{if(drag)fromEvt(e)});sl.addEventListener("pointerup",()=>drag=false);
sl.addEventListener("keydown",e=>{const v=+sl.getAttribute("aria-valuenow");if(e.key==="ArrowLeft")setPos(v-5);if(e.key==="ArrowRight")setPos(v+5)});
function tick(){$("clock").textContent=new Date().toISOString().replace("T"," ").slice(0,19)+" UTC"}
tick();setInterval(tick,1000);setPos(50);show(0);
</script></body></html>
"""

SPACE_README = """---
title: CloudLift
emoji: ☁️
colorFrom: indigo
colorTo: green
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Gen-AI cloud removal for LISS-IV (UnCRtainTS + SAR)
---

# CloudLift: pre-computed demo

Static showcase of CloudLift, a Generative AI cloud-removal pipeline for LISS-IV that fuses EOS-04 SAR
and a temporal reference with an adapted UnCRtainTS network.

* Scenes are **synthetic** (procedural); no real Bhoonidhi data is published.
* Results were **pre-computed** by the real pipeline with weights fine-tuned on synthetic data only.
  Every number shown is read from that run's output.
* Live inference: run the Docker/local console from the source repository: __REPO__
"""


def build() -> Path:
    if OUT.exists():
        shutil.rmtree(OUT)
    assets = OUT / "scenes"
    index = json.loads((DEMO_CACHE / "index.json").read_text(encoding="utf-8"))
    records = []
    for e in index:
        res = json.loads((DEMO_CACHE / e["scene_id"] / "result.json").read_text(encoding="utf-8"))
        meta = json.loads((DEMO_SCENES / e["scene_id"] / "meta.json").read_text(encoding="utf-8"))
        records.append(scene_record(res, meta, assets))
    data = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    page = PAGE.replace("__DATA__", data).replace("__REPO__", html.escape(REPO_URL))
    (OUT / "index.html").write_text(page, encoding="utf-8")
    (OUT / "README.md").write_text(SPACE_README.replace("__REPO__", REPO_URL), encoding="utf-8")
    return OUT


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--upload", metavar="USER/SPACE", help="create/update a static HF Space")
    a = ap.parse_args()
    out = build()
    print(f"built {out.resolve()}")
    if a.upload:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(a.upload, repo_type="space", space_sdk="static", exist_ok=True)
        api.upload_folder(
            folder_path=out,
            repo_id=a.upload,
            repo_type="space",
            commit_message="Deploy CloudLift pre-computed demo",
        )
        print(f"https://huggingface.co/spaces/{a.upload}")


if __name__ == "__main__":
    main()
