# CloudLift — Build Prompt for a Coding Agent

> **How to use this file.** Place it at the repository root as `AGENT_PROMPT.md` (or `.github/copilot-instructions.md` for GitHub Copilot). Then tell your agent: *"Read AGENT_PROMPT.md and build the system phase by phase. Stop at the end of each phase, run its checks, and report."* It is IDE-agnostic: it works with GitHub Copilot agent mode (VS Code / JetBrains / Visual Studio), Cursor, Windsurf, Claude Code, or any agent that can edit files and run a terminal.

---

## 0. Your role and rules

You are a senior ML + geospatial engineer building **CloudLift**: a Generative AI framework that removes cloud and cloud-shadow from **LISS-IV** satellite imagery and reconstructs the hidden surface into **analysis-ready, cloud-free GeoTIFFs**, using SAR and temporal fusion so that thick-cloud regions are *recovered, not invented*.

Follow these rules for the entire build:

1. **Work in phases (Section 6).** Finish one phase, run its acceptance checks, report results, then continue. Never skip ahead.
2. **Single recommended path.** Use exactly the tools in Section 3. Do not swap libraries or add frameworks unless a check fails and you explain why.
3. **Determinism is a requirement, not a nice-to-have.** Same input → byte-identical output. No MC-dropout, no ensembles, no stochastic sampling at inference.
4. **Never fabricate results.** Do not hard-code metric values, fake outputs, or claim a model is trained when it is not. If real data or weights are missing, build a clearly-labelled stub/fixture path and say so.
5. **Respect the hardware (Section 2).** The dev GPU has 6 GB VRAM. Every training/inference default must fit it.
6. **Keep secrets out of git.** Data credentials and keys go in `.env` (git-ignored); commit `.env.example` only.
7. **Small, tested commits.** Every module ships with unit tests. Run `pytest` before reporting a phase complete.
8. **Ask only when blocked.** If a decision is genuinely ambiguous and expensive to reverse, ask. Otherwise pick the default in this file and note it.

---

## 1. What the system must do (PS-2 requirements → features)

| PS requirement | Feature you build |
|---|---|
| Automated cloud removal in LISS-IV | `cloudlift.model` inference on tiles, batch over scenes |
| Reconstruct while preserving spatial + spectral detail | UnCRtainTS fine-tuned with L1 + SSIM + **SAM** loss |
| Leverage spatial, spectral, temporal info / SAR fusion | Inputs = cloudy LISS-IV + EOS-04 SAR + temporal reference |
| Visually consistent, analysis-ready output | Georeferenced GeoTIFF + per-pixel **uncertainty band** |
| Quantitative + qualitative evaluation | PSNR, SSIM, SAM, NDVI agreement, uncertainty calibration; side-by-side panels |
| Cloud masks + co-registration | `cloudlift.preprocess` (mask, reproject, phase-correlation align, tile) |
| Transfer learning / fine-tuning | Load SEN12MS-CR-TS-pretrained UnCRtainTS, 3-band adapter, fine-tune |
| Scalable operational workflow | CLI batch runner + FastAPI service + Docker |
| Documentation + deployment | README, model card, Dockerfile, runbook |

**Pipeline (must match exactly, 7 stages):**
`1 Ingest → 2 Co-register & tile → 3 Detect cloud + shadow → 4 Reconstruct (UnCRtainTS, SAR-fused) → 5 Per-pixel uncertainty → 6 Validate (PSNR/SSIM/SAM/NDVI) → 7 Export analysis-ready GeoTIFF`

---

## 2. Target environment (develop against this)

| Item | Value | Implication |
|---|---|---|
| OS | Windows 11 (also must run on Linux/WSL2) | Use `pathlib`, no hard-coded separators; provide `.ps1` and `.sh` scripts |
| GPU | NVIDIA RTX 3060 Laptop, **6 GB** | 256×256 tiles, batch 1, AMP, gradient checkpointing, grad accumulation |
| CPU / RAM | i7-11800H, 16 GB | `num_workers ≤ 4`; stream tiles, never load full scenes into RAM |
| CUDA driver | 13.2 | Install PyTorch **cu124** wheel — it bundles its own CUDA runtime and runs on the 13.2 driver. **Do not** install a CUDA toolkit for PyTorch. |
| Python | 3.12 | Pin `requires-python = ">=3.12,<3.13"` |

Also provide a **CPU fallback** so tests and the demo run on machines without a GPU (slow but working).

Heavy jobs (full fine-tune, benchmark) are expected to run on a rented India GPU (A100/L40S). Make device, batch size and precision configurable so the same code runs there unchanged.

---

## 3. Tech stack (use exactly these)

- **Language:** Python 3.12
- **DL:** PyTorch + torchvision (cu124 index), PyTorch Lightning, torchmetrics
- **Core model:** UnCRtainTS — `https://github.com/PatrickTUM/UnCRtainTS` (vendored as a git submodule under `third_party/UnCRtainTS`; wrap, don't fork-edit)
- **Geospatial:** GDAL, Rasterio (install via conda-forge on Windows, or apt/pip on WSL2), QGIS for manual QA only
- **Supporting:** NumPy, OpenCV (`opencv-python-headless`), scikit-image, Albumentations, OpenSimplex
- **Cloud masking:** OmniCloudMask (fallback: threshold+morphology baseline in-repo)
- **Service:** FastAPI + Uvicorn
- **UI:** Gradio (operator console)
- **Config:** Pydantic Settings + YAML
- **Quality:** pytest, ruff, mypy (strict on `cloudlift/`), pre-commit
- **Packaging:** Docker (CUDA runtime base image for GPU, slim for CPU)
- **Optional:** Google Earth Engine client for auxiliary pulls (behind a feature flag, off by default)

---

## 4. Data sources and compliance

All operational inputs come from **Bhoonidhi (NRSC/ISRO)** under the **Indian Space Policy 2023**. Data must stay in-country.

| Role | Dataset | Notes |
|---|---|---|
| Primary optical | LISS-IV (ResourceSat-2/2A), 5.8 m, **3 bands: Green, Red, NIR — no blue** | Band order in code: `G, R, NIR` |
| Through-cloud SAR | EOS-04 / RISAT-1A, C-band, 2 polarisations | Resample to the 5.8 m LISS-IV grid |
| Temporal reference | Prior cloud-free LISS-IV pass, same AOI | `T−N` days |
| Terrain | Cartosat DEM (CartoDEM) | Optional input; used for shadow geometry |
| Pretraining only | SEN12MS-CR-TS | Weights initialisation only; never an inference input |

Implement a `DataSource` interface with a `LocalFolderSource` (reads files the user downloaded from Bhoonidhi) as the default. **Do not scrape Bhoonidhi or bypass its login.** If no real data is present, generate a **synthetic fixture scene** (procedural terrain + river + clouds, 3-band + SAR) so every phase is testable; label it clearly as synthetic everywhere it appears.

AOI must accept **any location**: a bounding box, a point + radius, or a Bhoonidhi tile ID.

---

## 5. Repository layout (create this)

```
cloudlift/
├─ AGENT_PROMPT.md                 # this file
├─ README.md
├─ MODEL_CARD.md
├─ pyproject.toml
├─ environment.yml                 # conda-forge: gdal, rasterio, python=3.12
├─ requirements.txt                # pip deps (torch installed separately, see README)
├─ .env.example
├─ .pre-commit-config.yaml
├─ configs/
│  ├─ default.yaml                 # paths, tile=256, input_t=2, seed=0, device=auto
│  ├─ train_laptop.yaml            # 6 GB profile
│  └─ train_cloud.yaml             # A100 profile (bigger batch, tile 512 optional)
├─ cloudlift/
│  ├─ __init__.py
│  ├─ config.py                    # Pydantic settings, YAML loader
│  ├─ determinism.py               # seed + cuDNN + deterministic algorithms
│  ├─ io/
│  │  ├─ sources.py                # DataSource, LocalFolderSource, SyntheticSource
│  │  ├─ aoi.py                    # bbox / point+radius / tile-ID parsing
│  │  └─ geotiff.py                # read/write with CRS, transform, nodata, band tags
│  ├─ preprocess/
│  │  ├─ coregister.py             # reproject to LISS-IV grid + phase-correlation refine
│  │  ├─ tiling.py                 # tile/untile with overlap + feathered blending
│  │  ├─ cloudmask.py              # OmniCloudMask wrapper + baseline fallback
│  │  └─ synthetic_clouds.py       # OpenSimplex cloud + shadow compositing
│  ├─ model/
│  │  ├─ adapter.py                # 3-band LISS-IV + 2-pol SAR input adapter
│  │  ├─ uncrtaints_wrapper.py     # load pretrained, forward, return (recon, sigma)
│  │  ├─ losses.py                 # L1, SSIM, SAM, NLL
│  │  ├─ datamodule.py             # Lightning DataModule over tiles
│  │  └─ lit_module.py             # Lightning training module
│  ├─ eval/
│  │  ├─ metrics.py                # PSNR, SSIM, SAM, NDVI R²/RMSE, calibration error
│  │  └─ report.py                 # JSON + Markdown report + comparison panels
│  ├─ pipeline.py                  # the 7-stage run() for one scene
│  ├─ cli.py                       # `cloudlift ingest|prep|train|infer|eval|export|batch`
│  ├─ service/
│  │  └─ api.py                    # FastAPI: /health, /reconstruct, /jobs/{id}, /export/{id}
│  └─ ui/
│     └─ console.py                # Gradio operator console
├─ scripts/
│  ├─ setup_windows.ps1
│  ├─ setup_linux.sh
│  ├─ make_fixtures.py             # synthetic scene for tests/demo
│  └─ bake_demo.py                 # pre-compute hero scenes into cache/
├─ tests/
│  ├─ conftest.py                  # fixtures: synthetic scene, tiny model, tmp dirs
│  ├─ test_geotiff.py
│  ├─ test_coregister.py
│  ├─ test_tiling.py
│  ├─ test_cloudmask.py
│  ├─ test_synthetic_clouds.py
│  ├─ test_adapter.py
│  ├─ test_losses.py
│  ├─ test_metrics.py
│  ├─ test_determinism.py          # golden-hash reproducibility
│  ├─ test_pipeline.py
│  └─ test_api.py
├─ docker/
│  ├─ Dockerfile.gpu
│  └─ Dockerfile.cpu
├─ third_party/UnCRtainTS/         # git submodule
└─ data/  cache/  results/          # git-ignored
```

---

## 6. Build phases (do them in order)

Each phase lists **deliverables** and **acceptance checks**. A phase is done only when all checks pass.

### Phase 1 — Scaffold and environment
- Create the layout above, `pyproject.toml`, `environment.yml`, `requirements.txt`, pre-commit (ruff, mypy), `.gitignore`, `.env.example`.
- `scripts/setup_windows.ps1`: create conda env from `environment.yml`, `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`, then `pip install -r requirements.txt`, then `git submodule update --init`.
- `scripts/setup_linux.sh`: same for Linux/WSL2.
- `cloudlift/determinism.py` with `set_determinism(seed)`: sets `PYTHONHASHSEED`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, seeds `random/numpy/torch`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, `torch.use_deterministic_algorithms(True)`.
- `cloudlift env-check` CLI command prints Python, torch, CUDA availability, GPU name, VRAM, GDAL/Rasterio versions.

**Checks:** `cloudlift env-check` runs; on the dev laptop it reports `cuda: True` and `NVIDIA GeForce RTX 3060 Laptop GPU`; `pytest` passes (placeholder test); `ruff` and `mypy` clean.

### Phase 2 — Data I/O and AOI
- `io/geotiff.py`: read/write multi-band GeoTIFF preserving CRS, transform, nodata; write band descriptions (`Green`, `Red`, `NIR`, `uncertainty`).
- `io/aoi.py`: parse bbox / point+radius / tile ID into a `shapely` geometry + CRS.
- `io/sources.py`: `LocalFolderSource` (expects a documented folder convention for LISS-IV, SAR, reference, DEM) and `SyntheticSource`.
- `scripts/make_fixtures.py`: produce a small synthetic scene (e.g. 1024×1024, 3-band + 2-pol SAR + clean reference + cloud mask) as GeoTIFFs in a fixed CRS.

**Checks:** round-trip write→read preserves values and georeferencing exactly; AOI parser unit-tested for all three forms; fixture scene generated deterministically (same seed → same file hash).

### Phase 3 — Preprocessing
- `coregister.py`: reproject SAR/reference/DEM onto the LISS-IV grid (Rasterio `reproject`, bilinear for continuous, nearest for masks), then per-tile sub-pixel refine with `skimage.registration.phase_cross_correlation`; report residual shift in pixels.
- `tiling.py`: tile with overlap (default 256 px, 32 px overlap) and untile with feathered blending so seams are invisible.
- `cloudmask.py`: OmniCloudMask wrapper returning cloud and shadow masks; deterministic threshold+morphology fallback if the package is unavailable.
- `synthetic_clouds.py`: OpenSimplex-based cloud alpha (thin→thick), shadow = shifted, darkened, blurred copy; returns `(cloudy, mask, clean)`.

**Checks:** a known synthetic shift is recovered within 0.25 px; tile→untile of an image reproduces it (max abs error < 1e-6 without blending, smooth with); synthetic clouds are reproducible from a seed; cloud fraction of generated clouds is within ±5% of the requested fraction.

### Phase 4 — Model wrapper and adapter
- `adapter.py`: maps inputs `[G, R, NIR] + [SAR_pol1, SAR_pol2]` × `input_t` time steps into the tensor layout UnCRtainTS expects; replaces first conv for 5 input channels and output head for 3 channels; initialises new weights sensibly (e.g. copy mean of pretrained RGB/NIR filters).
- `uncrtaints_wrapper.py`: `load(weights_path | None)`, `forward(batch) -> (recon[B,3,H,W], sigma[B,3,H,W] or [B,1,H,W])`. If no weights are available, load a **tiny random-init config** for tests and log a loud warning; never pretend it is trained.
- `losses.py`: L1, SSIM, SAM (spectral angle, numerically stable), heteroscedastic NLL; a weighted combined loss with weights in config, optionally up-weighting masked (cloud) pixels.

**Checks:** shape tests for adapter and wrapper; a forward pass on a 256×256 batch of 1 runs within 6 GB VRAM under AMP (assert peak memory via `torch.cuda.max_memory_allocated` when CUDA is available, skip otherwise); SAM of identical inputs = 0, SAM is invariant to positive scaling; loss is differentiable (backward runs).

### Phase 5 — Training (fine-tune)
- `datamodule.py`: dataset over cached tiles producing synthetic-cloud pairs on the fly (seeded per index for reproducibility); Albumentations geometric aug only (flips/rot90) — **no colour jitter** (it breaks spectral integrity).
- `lit_module.py`: Lightning module with the combined loss, AdamW, LR `1e-3` with exponential decay `0.8`/epoch, logging of PSNR/SSIM/SAM on validation.
- `configs/train_laptop.yaml`: `tile=256, batch_size=1, accumulate_grad_batches=8, precision=16-mixed, input_t=2, num_workers=4, gradient_checkpointing=true, max_epochs=20`.
- `configs/train_cloud.yaml`: larger batch, `accumulate_grad_batches=1`, optional `tile=512`.
- `cloudlift train --config ...` command; checkpoints to `results/<run>/`.
- A **preprocess-once cache**: co-registered tiles written once to `cache/<preproc_version>/` as `.npy`/`.tif`; training only reads the cache. Version the folder name so changing preprocessing never mixes old and new tiles.

**Checks:** a 2-epoch smoke run on the synthetic fixture completes on CPU and on the 3060 without OOM; training loss decreases; checkpoint reload yields identical validation metrics.

### Phase 6 — Evaluation
- `metrics.py`: PSNR, SSIM (per band and mean), SAM (degrees), RMSE, NDVI = (NIR−R)/(NIR+R+ε) with R² and RMSE **computed over cloud-masked pixels** (where reconstruction matters), and expected-calibration-error for the uncertainty.
- `report.py`: writes `results/<run>/eval.json` and `eval.md`, plus PNG panels: cloudy | reconstructed | reference | uncertainty | NDVI difference.
- `cloudlift eval --run <id> --split val|test` command.
- Real-scene mode: when no clean reference exists (thick cloud), produce qualitative panels and NDVI-consistency checks only, and state that full-reference metrics are unavailable.

**Checks:** metric functions verified against hand-computed small arrays; identical images → PSNR=inf handled, SSIM=1, SAM=0; report files are created and valid JSON/Markdown.

### Phase 7 — End-to-end pipeline and export
- `pipeline.py`: `run(scene, aoi, config) -> Result` executing all 7 stages; returns paths to the reconstruction GeoTIFF, uncertainty band, mask, metrics, and panels; logs per-stage timing.
- Export: **analysis-ready GeoTIFF** with bands `Green, Red, NIR, uncertainty`, original CRS and transform, LZW/DEFLATE compression, internal tiling, overviews, nodata set, and metadata tags (model version, weights hash, config hash, seed, source scene IDs, processing timestamp, "inferred pixel fraction").
- `cloudlift infer`, `cloudlift export`, and `cloudlift batch --aoi-list file.csv` (resumable: skip scenes whose output already exists with a matching config hash).

**Checks:** pipeline runs end-to-end on the synthetic scene; output opens in Rasterio with correct CRS/bands/tags; `test_determinism.py` asserts two runs produce **byte-identical** reconstruction arrays (golden SHA-256); batch run is resumable.

### Phase 8 — Service API
- `service/api.py` (FastAPI): `GET /health`, `POST /reconstruct` (AOI + scene refs or upload) → job id, `GET /jobs/{id}` (stage progress 1–7), `GET /export/{id}` (GeoTIFF download). Single GPU worker processes jobs sequentially; requests are idempotent by `Idempotency-Key` header.
- Cache: key results by `(input hash, model version, config hash)`; a cache hit must still create a new job record, and a cached result is never served for a different input hash.

**Checks:** `test_api.py` with `httpx`/TestClient covers health, job lifecycle, idempotent retry returns the same job, export returns a valid GeoTIFF, wrong input hash never returns a cached result.

### Phase 9 — Operator console (Gradio)
Build a single-screen console matching this layout (it mirrors the approved wireframe and HTML prototype):
- **Status bar:** brand `CloudLift`, `NORTH-EAST REGION · GROUND STATION`, live UTC clock, `BHOONIDHI · NRSC/ISRO` indicator.
- **Left rail:** active tile — Scene (`LISS-IV · 5.8 m · G/R/NIR`), **AOI input (any location)**, Acquired, Auxiliary SAR (`EOS-04 / RISAT-1A · C-band`), Source · compliance (`Bhoonidhi (NRSC/ISRO) · Indian Space Policy 2023`); Obstruction card with cloud %.
- **Centre:** before/after **slider** (cloudy ↔ reconstructed, use `gradio_imageslider`), buttons `Initiate recovery`, `Reset`, `↓ Export GeoTIFF` (enabled only after a run).
- **Right rail:** 6-stage progress list (`Cloud detection, SAR fusion, Reconstruction, Confidence, NDVI validation, Export GeoTIFF`), **Show confidence** toggle overlaying uncertainty (teal = SAR-constrained / low uncertainty, amber = interpolated / high uncertainty), metric tiles (NDVI R², SAM, SSIM, recovered %).
- **Bottom:** batch tile queue (done / queued).
- Colours: chrome `#0C121D`, panel `#151E2D`, teal `#12B09C`, amber `#EF9F27`, red `#E24B4A`, text `#E7EEF6`.
- Metric tiles display **only computed values** from the current run; show `—` when unavailable. Never hard-code numbers.
- Demo safety: `scripts/bake_demo.py` pre-computes 3–4 hero scenes into `cache/demo/`; the console offers them as examples so the live demo works offline and instantly.

**Checks:** `python -m cloudlift.ui.console` launches on `http://localhost:7860`; a full run on the synthetic scene updates all panels; export downloads a GeoTIFF; toggle switches the overlay; works with GPU and CPU.

### Phase 10 — Packaging, docs, CI
- `docker/Dockerfile.gpu` (PyTorch CUDA runtime base, GDAL from apt/conda) and `Dockerfile.cpu`; `docker compose` with `api` and `console` services; GPU via `--gpus all`.
- `README.md`: overview, the honest-core statement, setup for Windows (conda-forge + cu124 wheel) and Linux/WSL2, data folder convention, every CLI command with examples, laptop vs cloud training profiles, troubleshooting (CUDA 13.2 driver + cu124 wheel, GDAL on Windows, OOM tips).
- `MODEL_CARD.md`: intended use, inputs/outputs, training data (synthetic pairs + pretraining source), evaluation protocol, known limitations (domain shift S2→LISS-IV, thick-cloud ground truth unavailable, SAR speckle), determinism statement, compliance/data-residency statement.
- CI (`.github/workflows/ci.yml`): ruff, mypy, pytest on CPU with the synthetic fixture.

**Checks:** both images build; `docker compose up` serves API and console; CI workflow passes locally via `act` or equivalent dry-run; README commands are copy-paste correct.

---

## 7. Non-functional requirements

- **Reproducibility:** every output carries model version, weights hash, config hash and seed; rerunning with the same tuple reproduces it byte-for-byte.
- **Memory:** inference peak VRAM < 5 GB at default settings on the 3060; stream tiles; never hold a full scene of all bands in RAM if it exceeds 2 GB.
- **Performance target (laptop, 256² tile, AMP):** report measured ms/tile in `eval.md`; do not invent a number.
- **Robustness:** handle nodata, scene edges, partial overlap between LISS-IV and SAR (mark as uncertainty = max where SAR is missing).
- **Logging:** structured logs with stage, tile index, timing; no PII; no data uploaded anywhere by default.
- **Portability:** no OS-specific paths; scripts for both PowerShell and bash.

---

## 8. Definition of done

- [ ] All 10 phases complete with their checks passing.
- [ ] `pytest` green on CPU; GPU-specific tests green on the RTX 3060.
- [ ] End-to-end run on the synthetic scene produces a valid, georeferenced 4-band GeoTIFF and an evaluation report.
- [ ] Determinism golden test passes.
- [ ] Console demonstrates: AOI entry, recovery, slider reveal, confidence toggle, metrics from the actual run, GeoTIFF export.
- [ ] README and MODEL_CARD complete; Docker images build.
- [ ] No hard-coded or fabricated metrics anywhere in code, UI, or docs.

---

## 9. What to report after each phase

Reply with: (1) files created/changed, (2) commands run and their output summary, (3) check results (pass/fail), (4) any default you chose because something was ambiguous, (5) anything blocked that needs the human (e.g. real Bhoonidhi data, pretrained weights download, cloud GPU access).

---

## 10. Things the human will supply (do not invent them)

- Real LISS-IV / EOS-04 / reference / CartoDEM scenes downloaded from Bhoonidhi into `data/raw/` following the README folder convention.
- Pretrained UnCRtainTS weights (from the repo's release/project page) into `checkpoints/`.
- Access to a cloud GPU for the full fine-tune, if desired.

Until these arrive, build and test everything against the synthetic fixture and say so clearly in every report.
