"""``cloudlift`` command line: env-check | ingest | prep | train | infer | eval | export | batch | serve | console."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

from cloudlift.config import Config, load_config

log = logging.getLogger("cloudlift")

SYNTHETIC_ROOT = Path("data/synthetic")


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


def env_check() -> dict[str, Any]:
    info: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        info["cuda_runtime"] = torch.version.cuda
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = p.name
            info["vram_gb"] = round(p.total_memory / 2**30, 2)
    except ImportError:
        info["torch"] = "NOT INSTALLED"
    try:
        import rasterio

        info["rasterio"] = rasterio.__version__
        info["gdal"] = rasterio.__gdal_version__
    except ImportError:
        info["rasterio"] = "NOT INSTALLED"
    from cloudlift.model.uncrtaints_wrapper import UNCRTAINTS_ROOT
    from cloudlift.preprocess.cloudmask import omnicloudmask_available

    info["uncrtaints_submodule"] = (UNCRTAINTS_ROOT / "src" / "backbones" / "uncrtaints.py").exists()
    info["omnicloudmask"] = omnicloudmask_available()
    return info


def _cfg(args: argparse.Namespace) -> Config:
    over: dict[str, Any] = {}
    if getattr(args, "device", None):
        over["device"] = args.device
    if getattr(args, "weights", None):
        over.setdefault("paths", {})["weights"] = args.weights
    if getattr(args, "seed", None) is not None:
        over["seed"] = args.seed
    if getattr(args, "epochs", None):
        over.setdefault("train", {})["max_epochs"] = args.epochs
    return load_config(args.config, over)


def make_source(kind: str, cfg: Config, n_synthetic: int = 1, seed: int = 0, root: Path | None = None) -> Any:
    from cloudlift.io.sources import LocalFolderSource, SyntheticSource

    local = LocalFolderSource(cfg.paths.data_root)
    if kind == "local" or (kind == "auto" and local.list_scenes()):
        return local
    if kind == "auto":
        log.warning("no scenes in %s; using the SYNTHETIC fixture source", cfg.paths.data_root)
    return SyntheticSource(root or SYNTHETIC_ROOT / f"seed{seed}", n_scenes=n_synthetic, size=1024, seed=seed)


def cmd_infer(args: argparse.Namespace) -> int:
    from cloudlift import pipeline
    from cloudlift.io.aoi import parse_aoi

    cfg = _cfg(args)
    src = make_source(args.source, cfg)
    scene = src.get_scene(args.scene) if args.scene else src.list_scenes()[0]
    aoi = parse_aoi(args.aoi, src.resolve_tile) if args.aoi else None
    res = pipeline.run(scene, aoi, cfg, out_dir=args.out)
    _print(
        {
            "run_id": res.run_id,
            "geotiff": res.geotiff,
            "panels": res.panels,
            "report": res.metrics_md,
            "trained_model": res.model["trained"],
            "synthetic": res.synthetic,
            "timing": res.timing,
            "metrics": {k: v for k, v in res.metrics.items() if not isinstance(v, dict)},
        }
    )
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from cloudlift import pipeline
    from cloudlift.eval.report import summarize, write_report

    cfg = _cfg(args)
    run_dir = Path(cfg.paths.results_root) / args.run
    if not getattr(args, "weights", None) and (run_dir / "cloudlift_weights.pt").exists():
        cfg.paths.weights = run_dir / "cloudlift_weights.pt"
    if args.source == "local":
        src = make_source("local", cfg)
        scenes = src.list_scenes()
        split_file = Path(cfg.paths.data_root) / "splits.json"
        if split_file.exists():
            wanted = set(json.loads(split_file.read_text(encoding="utf-8")).get(args.split, []))
            scenes = [s for s in scenes if s.scene_id in wanted]
    else:  # held-out synthetic scenes: seeds disjoint from training (0..999)
        seed = 1000 if args.split == "val" else 2000
        src = make_source(
            "synthetic",
            cfg,
            n_synthetic=args.n_scenes,
            seed=seed,
            root=SYNTHETIC_ROOT / f"{args.split}_seed{seed}",
        )
        scenes = src.list_scenes()
    out = run_dir / f"eval_{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    per_scene: dict[str, dict[str, Any]] = {}
    baselines: dict[str, Any] = {}
    panels: list[Path] = []
    ms: list[float] = []
    model: dict[str, Any] = {}
    synthetic = True
    mode = "full-reference"
    for s in scenes:
        r = pipeline.run(s, None, cfg, out_dir=out / s.scene_id)
        per_scene[s.scene_id] = {k: v for k, v in r.metrics.items() if not isinstance(v, dict)}
        baselines[s.scene_id] = r.extra.get("baselines", {})
        p = out / f"panels_{s.scene_id}.png"
        shutil.copyfile(r.panels, p)
        panels.append(p)
        ms.append(r.timing["ms_per_tile"])
        model, synthetic, mode = r.model, synthetic and r.synthetic, r.extra["mode"]
    flat_base = {f"{sid}:{name}": vals for sid, b in baselines.items() for name, vals in b.items()}
    report = {
        "run_id": args.run,
        "split": args.split,
        "mode": mode if scenes else "no-scenes",
        "synthetic": synthetic,
        "model": model,
        "config_hash": cfg.config_hash(),
        "seed": cfg.seed,
        "device": cfg.resolve_device(),
        "timing": {
            "ms_per_tile": sum(ms) / len(ms) if ms else float("nan"),
            "tiles": "per scene",
            "tile": cfg.data.tile,
        },
        "scenes": per_scene,
        "summary": summarize(per_scene),
        "baselines": flat_base,
    }
    j, m = write_report(out, report, panels)
    _print({"eval_json": j, "eval_md": m, "summary": report["summary"]})
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    import rasterio

    cfg = _cfg(args)
    run_dir = Path(cfg.paths.results_root) / args.run
    res = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    src = Path(res["geotiff"])
    dst = Path(args.out) if args.out else Path("exports") / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    with rasterio.open(dst) as ds:
        info = {
            "path": dst,
            "crs": str(ds.crs),
            "bands": ds.descriptions,
            "size": [ds.width, ds.height],
            "overviews": ds.overviews(1),
            "tags": ds.tags(),
        }
    _print(info)
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    """CSV columns: ``aoi`` (required; any AOI form) and optional ``scene_id``. Resumable."""
    from cloudlift import pipeline
    from cloudlift.io.aoi import parse_aoi

    cfg = _cfg(args)
    src = make_source(args.source, cfg)
    done = skipped = 0
    with open(args.aoi_list, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        aoi_txt = (row.get("aoi") or "").strip()
        aoi = parse_aoi(aoi_txt, src.resolve_tile) if aoi_txt else None
        scenes = [src.get_scene(row["scene_id"])] if row.get("scene_id") else src.list_scenes(aoi)
        for s in scenes:
            ih = pipeline.input_hash(s, aoi)
            out = Path(cfg.paths.results_root) / pipeline.run_id_for(s, cfg, ih)
            if pipeline.existing_result(out, cfg, ih) is not None:
                skipped += 1
                print(f"skip  {s.scene_id} ({aoi_txt or 'full'}): up to date")
                continue
            r = pipeline.run(s, aoi, cfg, out_dir=out)
            done += 1
            print(f"done  {s.scene_id} ({aoi_txt or 'full'}) -> {r.geotiff}")
    _print({"processed": done, "skipped": skipped})
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="cloudlift", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--config", default=None, help="YAML merged over configs/default.yaml")
        sp.add_argument("--device", choices=["auto", "cpu", "cuda"])
        sp.add_argument("--weights", help="model weights (.pth.tar upstream, .pt CloudLift, .ckpt Lightning)")
        sp.add_argument("--seed", type=int)

    sub.add_parser("env-check", help="print Python/torch/CUDA/GPU/GDAL versions")

    sp = sub.add_parser(
        "ingest", help="list scenes available for an AOI (generates the synthetic fixture if needed)"
    )
    common(sp)
    sp.add_argument("--source", choices=["auto", "local", "synthetic"], default="auto")
    sp.add_argument("--aoi")

    sp = sub.add_parser("prep", help="co-register and cache training tiles (preprocess once)")
    common(sp)
    sp.add_argument("--source", choices=["auto", "local", "synthetic"], default="auto")
    sp.add_argument("--n-synthetic", type=int, default=8)

    sp = sub.add_parser("train", help="fine-tune UnCRtainTS")
    common(sp)
    sp.add_argument("--source", choices=["auto", "local", "synthetic"], default="auto")
    sp.add_argument("--n-synthetic", type=int, default=8)
    sp.add_argument("--run")
    sp.add_argument("--epochs", type=int)

    sp = sub.add_parser("infer", help="run the 7-stage pipeline on one scene")
    common(sp)
    sp.add_argument("--source", choices=["auto", "local", "synthetic"], default="auto")
    sp.add_argument("--scene")
    sp.add_argument("--aoi", help="bbox:..., point:lat,lon,5km or tile:<ID>")
    sp.add_argument("--out")

    sp = sub.add_parser("eval", help="evaluate a training run on val/test scenes")
    common(sp)
    sp.add_argument("--run", required=True)
    sp.add_argument("--split", choices=["val", "test"], default="val")
    sp.add_argument("--source", choices=["synthetic", "local"], default="synthetic")
    sp.add_argument("--n-scenes", type=int, default=2)

    sp = sub.add_parser("export", help="copy a run's analysis-ready GeoTIFF and print its metadata")
    common(sp)
    sp.add_argument("--run", required=True)
    sp.add_argument("--out")

    sp = sub.add_parser("batch", help="resumable batch over an AOI list CSV")
    common(sp)
    sp.add_argument("--aoi-list", required=True)
    sp.add_argument("--source", choices=["auto", "local", "synthetic"], default="auto")

    sp = sub.add_parser("serve", help="start the FastAPI service")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)

    sp = sub.add_parser("console", help="start the Gradio operator console")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=7860)

    args = p.parse_args(argv)
    if args.cmd == "env-check":
        _print(env_check())
        return 0
    if args.cmd == "ingest":
        from cloudlift.io.aoi import parse_aoi

        cfg = _cfg(args)
        src = make_source(args.source, cfg)
        aoi = parse_aoi(args.aoi, src.resolve_tile) if args.aoi else None
        _print(
            {
                "source": repr(src),
                "aoi": aoi.text if aoi else None,
                "scenes": [
                    {
                        "scene_id": s.scene_id,
                        "acquired": s.acquired,
                        "synthetic": s.synthetic,
                        "sar": bool(s.sar),
                        "reference": bool(s.reference),
                    }
                    for s in src.list_scenes(aoi)
                ],
            }
        )
        return 0
    if args.cmd == "prep":
        from cloudlift.model.datamodule import build_tile_cache

        cfg = _cfg(args)
        root = build_tile_cache(make_source(args.source, cfg, n_synthetic=args.n_synthetic), cfg)
        _print(
            {"cache": root, "index": json.loads((root / "index.json").read_text(encoding="utf-8"))["scenes"]}
        )
        return 0
    if args.cmd == "train":
        from cloudlift.model.lit_module import fit

        cfg = _cfg(args)
        summary = fit(cfg, make_source(args.source, cfg, n_synthetic=args.n_synthetic), args.run)
        _print({k: v for k, v in summary.items() if k != "history"} | {"history": summary["history"][-3:]})
        return 0
    if args.cmd == "infer":
        return cmd_infer(args)
    if args.cmd == "eval":
        return cmd_eval(args)
    if args.cmd == "export":
        return cmd_export(args)
    if args.cmd == "batch":
        return cmd_batch(args)
    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("cloudlift.service.api:app", host=args.host, port=args.port)
        return 0
    if args.cmd == "console":
        from cloudlift.ui.console import launch

        launch(host=args.host, port=args.port)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
