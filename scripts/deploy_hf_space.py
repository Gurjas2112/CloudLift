"""Stage and upload the CPU demo to a Hugging Face Space (Docker SDK).

hf auth login            # once; token stays in the HF CLI store, never in this repo
python scripts/deploy_hf_space.py --repo <user>/cloudlift
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
INCLUDE = [
    "pyproject.toml",
    "requirements.txt",
    "LICENSE",
    "MODEL_CARD.md",
    "cloudlift",
    "configs",
    "scripts",
    "assets",
]
UNCRTAINTS = "third_party/UnCRtainTS/model/src/backbones"


def _origin_url() -> str:
    try:
        url = subprocess.run(
            ["git", "remote", "get-url", "origin"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "(source repository)"
    return url.removesuffix(".git").replace("git@github.com:", "https://github.com/")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--private", action="store_true")
    ap.add_argument(
        "--source-url", default=None, help="link shown on the Space (default: git remote 'origin')"
    )
    a = ap.parse_args()
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for item in INCLUDE:
            src = ROOT / item
            if src.is_dir():
                shutil.copytree(src, stage / item, ignore=ignore)
            elif src.exists():
                shutil.copy2(src, stage / item)
        shutil.copytree(ROOT / UNCRTAINTS, stage / UNCRTAINTS, ignore=ignore)
        shutil.copy2(ROOT / "deploy/hf_space/Dockerfile", stage / "Dockerfile")
        readme = (ROOT / "deploy/hf_space/README.md").read_text(encoding="utf-8")
        (stage / "README.md").write_text(
            readme.replace("__SOURCE_URL__", a.source_url or _origin_url()), encoding="utf-8"
        )
        api = HfApi()
        api.create_repo(a.repo, repo_type="space", space_sdk="docker", private=a.private, exist_ok=True)
        api.upload_folder(
            folder_path=stage,
            repo_id=a.repo,
            repo_type="space",
            commit_message="Deploy CloudLift live console (Docker, CPU)",
            delete_patterns=["*"],  # mirror the staged folder (e.g. replaces a previous static demo)
        )
    print(f"https://huggingface.co/spaces/{a.repo}")


if __name__ == "__main__":
    main()
