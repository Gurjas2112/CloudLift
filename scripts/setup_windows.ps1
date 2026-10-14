# CloudLift setup for Windows 11 (PowerShell). Run from the repository root:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 [-Cpu]
# Uses conda (conda-forge GDAL/Rasterio) when available, otherwise a venv: Rasterio wheels
# on PyPI bundle GDAL, so conda is optional on Windows.
param([switch]$Cpu)
$ErrorActionPreference = "Stop"

git submodule update --init --recursive

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { $Cpu = $true }  # no NVIDIA GPU -> CPU wheels
$torchIndex = if ($Cpu) { "https://download.pytorch.org/whl/cpu" } else { "https://download.pytorch.org/whl/cu124" }

if (Get-Command conda -ErrorAction SilentlyContinue) {
    conda env create -f environment.yml
    if (-not $?) { conda env update -f environment.yml }
    $py = (conda run -n cloudlift python -c "import sys; print(sys.executable)").Trim()
} else {
    if (-not (Test-Path .venv)) {
        if (Get-Command py -ErrorAction SilentlyContinue) { py -3.12 -m venv .venv } else { python -m venv .venv }
    }
    $py = Join-Path (Resolve-Path .venv) "Scripts\python.exe"
}

& $py -m pip install --upgrade pip
# cu124 wheel bundles its own CUDA runtime and runs on newer drivers (e.g. 13.x). Do NOT install a CUDA toolkit.
& $py -m pip install torch torchvision --index-url $torchIndex
& $py -m pip install -r requirements-dev.txt
& $py -m pip install --no-deps -e .
& $py -m cloudlift.cli env-check
Write-Host "Done. Activate with .venv\Scripts\Activate.ps1 (or 'conda activate cloudlift')."
