#!/usr/bin/env bash
# CloudLift setup for Linux / WSL2. Run from the repository root:
#   bash scripts/setup_linux.sh [--cpu]
set -euo pipefail

git submodule update --init --recursive

TORCH_INDEX="https://download.pytorch.org/whl/cu124"
[[ "${1:-}" == "--cpu" ]] && TORCH_INDEX="https://download.pytorch.org/whl/cpu"

if command -v conda >/dev/null 2>&1; then
  conda env create -f environment.yml || conda env update -f environment.yml
  PY="$(conda run -n cloudlift python -c 'import sys; print(sys.executable)')"
else
  [[ -d .venv ]] || python3.12 -m venv .venv
  PY=".venv/bin/python"
fi

"$PY" -m pip install --upgrade pip
# cu124 wheel bundles its own CUDA runtime; no CUDA toolkit needed (only the NVIDIA driver).
"$PY" -m pip install torch torchvision --index-url "$TORCH_INDEX"
"$PY" -m pip install -r requirements-dev.txt
"$PY" -m pip install --no-deps -e .
"$PY" -m cloudlift.cli env-check
echo "Done. Activate with: source .venv/bin/activate  (or: conda activate cloudlift)"
