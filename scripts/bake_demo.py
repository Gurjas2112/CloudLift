"""Pre-compute hero scenes into cache/demo/ so the console demo is instant and works offline.

Scenes are SYNTHETIC (procedural NER-like terrain); every result is produced by the real
pipeline with the configured weights, nothing is hand-edited.

    python scripts/bake_demo.py [--config configs/demo.yaml] [--size 768]
"""

from __future__ import annotations

import argparse
import json

from cloudlift import pipeline
from cloudlift.io.sources import LocalFolderSource
from cloudlift.io.synthetic import generate_scene
from cloudlift.ui.console import DEMO_CACHE, DEMO_SCENES, HEROES, console_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--size", type=int, default=768)
    a = ap.parse_args()
    import os

    if a.config:
        os.environ["CLOUDLIFT_CONFIG"] = a.config
    cfg = console_config()
    for sid, seed, frac in HEROES:
        if not (DEMO_SCENES / sid / "meta.json").exists():
            generate_scene(DEMO_SCENES, sid, size=a.size, seed=seed, cloud_fraction=frac)
    src = LocalFolderSource(DEMO_SCENES)
    index = []
    for sid, _, _ in HEROES:
        sc = src.get_scene(sid)
        out = DEMO_CACHE / sid
        r = pipeline.existing_result(out, cfg, pipeline.input_hash(sc, None)) or pipeline.run(
            sc, None, cfg, out_dir=out
        )
        index.append(
            {
                "scene_id": sid,
                "run_id": r.run_id,
                "config_hash": r.config_hash,
                "trained": r.model["trained"],
                "weights": r.model["weights_source"],
                "ms_per_tile": r.timing["ms_per_tile"],
                "device": r.extra["device"],
            }
        )
        print(f"baked {sid}: {r.geotiff}")
    DEMO_CACHE.mkdir(parents=True, exist_ok=True)
    (DEMO_CACHE / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
