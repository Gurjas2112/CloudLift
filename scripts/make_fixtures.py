"""Generate the deterministic SYNTHETIC fixture scene(s) (procedural; not real LISS-IV).

python scripts/make_fixtures.py --out data/synthetic/fixture --n 1 --size 1024 --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cloudlift.io.synthetic import generate_scene


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/synthetic/fixture")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cloud-fraction", type=float, default=0.35)
    a = ap.parse_args()
    report = {}
    for i in range(a.n):
        d = generate_scene(
            a.out, f"SYN_NER_{i + 1:04d}", size=a.size, seed=a.seed + i, cloud_fraction=a.cloud_fraction
        )
        report[d.name] = {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16] for f in sorted(d.iterdir())
        }
    print(json.dumps({"out": str(Path(a.out).resolve()), "scenes": report}, indent=2))


if __name__ == "__main__":
    main()
