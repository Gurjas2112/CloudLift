# CloudLift

**Generative AI cloud removal and surface reconstruction for LISS-IV imagery**:

CloudLift removes cloud and cloud shadow from **LISS-IV** (ResourceSat-2/2A, 5.8 m, Green/Red/NIR)
scenes and writes **analysis-ready, cloud-free GeoTIFFs**. It fuses the cloudy pass with
**EOS-04 C-band SAR**, which sees through cloud, and a **prior cloud-free LISS-IV pass**, using an
adapted [UnCRtainTS](https://github.com/PatrickTUM/UnCRtainTS) network. Every reconstructed pixel
also gets a **per-pixel uncertainty** (σ) band.

> **Honest core.** Thick-cloud regions are *recovered* from SAR and time, not *invented*. Clear pixels
> are passed through untouched, and only cloud/shadow pixels are replaced. Each replaced pixel carries
> the model's own uncertainty: 1.0 where no SAR was available. Inference is a single deterministic
> forward pass (no sampling, no MC-dropout, no ensembles), so the same input gives byte-identical output.
> Metrics are computed by the pipeline, never typed in.

**Live demo:** https://huggingface.co/spaces/gurjasgandhi76/cloudlift is the full operator console as a Docker Space on
free CPU hardware. Four synthetic hero scenes replay instantly from the build-time bake, and any AOI typed in runs the live
7-stage pipeline on CPU (about 1–2.5 s per 256² tile).

---

## Status

| Item | State |
|---|---|
| Real Bhoonidhi LISS-IV / EOS-04 data | **not included.** Drop scenes in `data/raw/` (convention below) |
| Pretrained SEN12MS-CR-TS UnCRtainTS weights | **not included.** The loader + 3-band adapter are tested; put the `.pth.tar` in `checkpoints/` |
| Shipped weights `assets/weights/cloudlift_synthetic_full_v1.pt` | full UnCRtainTS fine-tuned **only on synthetic pairs**; demo use only (see `MODEL_CARD.md`) |

Until real data arrives, everything is exercised on a **synthetic fixture**: procedural NER-like terrain,
river, fields and OpenSimplex clouds, with matching SAR. It is labelled `SYNTHETIC` in every file,
report and on screen.

## Pipeline

```
1 Ingest → 2 Co-register & tile → 3 Detect cloud + shadow → 4 Reconstruct (UnCRtainTS, SAR-fused)
        → 5 Per-pixel uncertainty → 6 Validate (PSNR/SSIM/SAM/NDVI) → 7 Export analysis-ready GeoTIFF
```

| Stage | Implementation |
|---|---|
| 1 Ingest | `LocalFolderSource` (Bhoonidhi downloads) or `SyntheticSource`; AOI crop: bbox, point + radius, or tile ID |
| 2 Co-register | Rasterio `reproject` of SAR/reference/DEM onto the LISS-IV grid; sub-pixel phase-correlation refinement of the temporal reference (median of per-tile shifts, residual reported) |
| 3 Cloud + shadow | OmniCloudMask (optional extra) or the in-repo deterministic multi-temporal threshold + morphology baseline |
| 4 Reconstruct | UnCRtainTS, input `[B, T=2, 5, H, W]` = (VV, VH, G, R, NIR) × (reference, cloudy pass); 256 px tiles, 32 px overlap, feathered blending; clear tiles skipped |
| 5 Uncertainty | predicted σ (diagonal Gaussian head), feathered to 0 on observed pixels, `1.0` where SAR is missing |
| 6 Validate | full-reference metrics over cloud/shadow pixels when truth exists; otherwise NDVI consistency with the reference only |
| 7 Export | 4-band float32 GeoTIFF `Green, Red, NIR, uncertainty`, original CRS/transform, DEFLATE, 256 px internal tiles, overviews, nodata `-9999`, provenance tags |

**3-band adapter.** Upstream UnCRtainTS takes S1(2) + S2(13) channels and predicts 13 means + 13 variances.
The adapter keeps the pretrained SAR filters and the S2 B3/B4/B8 filters (= LISS-IV Green/Red/NIR) for
both the first conv and the output head. It then re-points the module's band slicing (hard-coded to 13
upstream) to 3. The vendored repository is wrapped and never edited.

## Setup

Python 3.12. The PyTorch **cu124** wheel bundles its own CUDA runtime and runs on newer drivers
(tested on a 13.x-series driver), so **do not install a CUDA toolkit**.

**Windows 11 (PowerShell)**

```powershell
git clone --recurse-submodules https://github.com/Gurjas2112/CloudLift.git
cd CloudLift
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1        # add -Cpu for CPU-only
.venv\Scripts\Activate.ps1
cloudlift env-check
```

The script uses conda (`environment.yml`, conda-forge GDAL) if `conda` is on PATH. Otherwise it uses a
venv: Rasterio's PyPI wheels bundle GDAL, so conda is optional on Windows.

**Linux / WSL2**

```bash
git clone --recurse-submodules https://github.com/Gurjas2112/CloudLift.git && cd CloudLift
bash scripts/setup_linux.sh            # --cpu for CPU-only
source .venv/bin/activate && cloudlift env-check
```

Manual equivalent: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`,
then `pip install -r requirements-dev.txt` and `pip install --no-deps -e .`.
Optional learned cloud masking: `pip install -r requirements-cloudmask.txt`.

## Data folder convention (Bhoonidhi downloads)

CloudLift never scrapes Bhoonidhi or bypasses its login. Download scenes yourself (Indian Space
Policy 2023: data stays in-country) and place each scene in its own folder:

```
data/raw/<scene_id>/
  liss4.tif       required  LISS-IV, bands Green, Red, NIR (10-bit DN or reflectance)
  sar.tif         required  EOS-04 / RISAT-1A, 2 polarisations, any grid (reprojected)
  reference.tif   required  earlier cloud-free LISS-IV pass of the same area
  dem.tif         optional  CartoDEM
  meta.json       optional  {"acquired": "2026-07-15", "reference_acquired": "2026-02-10",
                             "scale": 1023, "sar_units": "db" | "linear"}
data/raw/splits.json  optional  {"val": [...scene ids], "test": [...]}
```

`<scene_id>` doubles as the AOI tile ID (`tile:<scene_id>`). Integer optical data is divided by
`scale` (default 1023). SAR is converted to `(dB + 25) / 25`.

## CLI

| Command | Example |
|---|---|
| environment | `cloudlift env-check` |
| list scenes for an AOI | `cloudlift ingest --aoi "point:26.14,91.74,5km"` |
| build the tile cache (once) | `cloudlift prep --source synthetic --n-synthetic 16` |
| train / fine-tune | `cloudlift train --config configs/train_laptop.yaml --source synthetic --n-synthetic 16 --run my_run` |
| fine-tune from upstream weights | `cloudlift train --config configs/train_laptop.yaml --weights checkpoints/uncrtaints.pth.tar --source local` |
| inference (one scene) | `cloudlift infer --config configs/demo.yaml --scene SYN_NER_0001 --aoi "bbox:91.60,26.14,91.63,26.17"` |
| evaluate a training run | `cloudlift eval --run my_run --split test --n-scenes 3` |
| export a run's GeoTIFF | `cloudlift export --run <run_id> --out exports/scene.tif` |
| resumable batch | `cloudlift batch --aoi-list aois.csv` (CSV columns `aoi[,scene_id]`; finished scenes with the same config hash are skipped) |
| API | `cloudlift serve --host 0.0.0.0 --port 8000` |
| console | `cloudlift console` or `python -m cloudlift.ui.console` → http://localhost:7860 |

`--source auto` (the default) uses `data/raw/` when it has scenes. Otherwise it generates the
synthetic fixture in `data/synthetic/` and says so in the log. AOI forms: `bbox:minlon,minlat,maxlon,maxlat`
(optionally `@EPSG:xxxx`), `point:lat,lon,5km` or `500m`, and `tile:<ID>`.

### Training profiles

| | `configs/train_laptop.yaml` (RTX 3060, 6 GB) | `configs/train_cloud.yaml` (A100 / L40S) |
|---|---|---|
| tile / batch / accumulation | 256 / 1 / 8 | 256 (512 optional) / 8 / 1 |
| precision | 16-mixed | bf16-mixed |
| gradient checkpointing | on | off |
| workers | 4 | 8 |

The same code runs on both; only the YAML changes. Training reads a **preprocess-once cache**
(`cache/<preproc_version>/`); bump `data.preproc_version` when preprocessing changes.

### Service API

`GET /health` · `POST /reconstruct` (JSON `{"scene_id", "aoi", "source"}` or multipart upload of
`liss4`, `sar`, `reference` [+ `meta`]) → `job_id` · `GET /jobs/{id}` (stage 1–7 progress) ·
`GET /export/{id}` (GeoTIFF). One worker processes jobs sequentially. Sending the same
`Idempotency-Key` returns the same job. Results are cached by (input hash, model version + weights
hash, config hash), so a different input never gets a cached result.

## Results (measured)

All numbers below come from `cloudlift eval --config configs/demo.yaml --run demo_full_synthetic --split test --n-scenes 3`
(`results/demo_full_synthetic/eval_test/eval.json`). They were measured on **3 held-out SYNTHETIC test scenes** (seeds 2000–2002, 1024² px,
31–51 % cloud + shadow). The shipped weights were trained on synthetic scenes with seeds 0–15. **These are not real LISS-IV results.**

Training: `configs/train_laptop.yaml`, full UnCRtainTS, 204 train / 52 val tiles, RTX 3060 Laptop. The 20-epoch run was
interrupted during epoch 11. The exported weights are the best validation-loss checkpoint (epoch 9; val masked PSNR 40.7 dB).

| over cloud + shadow pixels (mean of 3 scenes) | PSNR (dB) | SSIM | SAM (°) | NDVI R² | NDVI RMSE |
|---|---|---|---|---|---|
| cloudy input (no reconstruction) | 5.60 | — | 25.93 | −2.965 | 0.561 |
| temporal fill (copy co-registered reference pass) | **41.60** | **0.986** | **1.25** | **0.983** | **0.037** |
| **CloudLift UnCRtainTS** (SAR + reference + cloudy pass) | 41.01 | 0.977 | 1.47 | 0.961 | 0.056 |

Whole-scene (including untouched clear pixels): PSNR 44.51 dB, SSIM 0.989, SAM 0.67°. Uncertainty calibration: coverage
ECE 0.073, UCE 0.0020 (mean σ over masked pixels 0.0056). Cloud-detection IoU of the baseline mask: 0.938.
Measured inference: **78 ms per 256² tile** on the RTX 3060 Laptop (fp16 autocast, full model) and **1.12 s per tile** on CPU.

**Reading this honestly.** On the synthetic fixture, the reference pass differs from the current pass only by a mild crop-vigour
shift and noise, so copying the reference is already near-optimal. The trained network does not beat it there. What the network
adds is a calibrated per-pixel σ and SAR-driven evidence when the reference is stale, missing or changed (floods, harvest). The
synthetic fixture does not stress that case. Real-LISS-IV fine-tuning and evaluation, with genuinely changed surfaces, is the
next step and needs Bhoonidhi data.

## Docker

```bash
docker compose up --build                                   # CPU: API :8000 + console :7860
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build   # NVIDIA GPU
```

`docker/Dockerfile.cpu` (python:3.12-slim + CPU torch) bakes the demo scenes at build time.
`docker/Dockerfile.gpu` uses `nvidia/cuda:12.6.3-runtime-ubuntu24.04` (Python 3.12) + the cu124 wheel.
The live Space is deployed with `python scripts/deploy_hf_space.py --repo <user>/cloudlift`, which uses `deploy/hf_space/`.
Docker/Gradio Spaces on free CPU need a Hugging Face PRO account. Without PRO, `python scripts/build_static_demo.py --upload
<user>/cloudlift` publishes a free static page of the pre-computed hero scenes instead.

## Determinism

`cloudlift.determinism.set_determinism(seed)` seeds Python/NumPy/torch and sets `CUBLAS_WORKSPACE_CONFIG`,
cuDNN deterministic mode and `torch.use_deterministic_algorithms(True)`. `tests/test_determinism.py`
runs the full pipeline twice and checks the reconstruction SHA-256s are identical. It also compares
against `tests/golden_hashes.json` for the recorded environment. Hashes are device- and
library-specific, so CI verifies run-to-run identity and skips the golden comparison on other
environments. The only relaxation: CUDA *training* uses `warn_only=True`, because UnCRtainTS'
`AdaptiveMaxPool2d` has no deterministic CUDA backward kernel. Inference stays strict.

Each GeoTIFF records `MODEL_VERSION`, `WEIGHTS_HASH`, `CONFIG_HASH`, `INPUT_HASH`, `SEED`,
`SOURCE_SCENE_IDS`, `PROCESSING_TIMESTAMP`, `INFERRED_PIXEL_FRACTION`, `MODEL_TRAINED` and `SYNTHETIC`.

## Choices that differ from the build prompt

* **venv instead of conda by default** on Windows (conda optional): PyPI Rasterio wheels bundle GDAL.
* **Gradio ≥ 5 native `gr.ImageSlider`** instead of the `gradio_imageslider` custom component, which targets Gradio 4.
* **Training on CUDA uses `warn_only` determinism** (see above).
* **SAR is reprojected but not phase-correlation refined**: SAR↔optical correlation is cross-modal and
  unreliable. Only the optical temporal reference is sub-pixel aligned.
* **Comparative assessment**: one generative architecture (UnCRtainTS) is trained. Every evaluation
  also scores two non-learned baselines on the same pixels (the cloudy input and temporal
  reference substitution). `MODEL_CARD.md` explains why UnCRtainTS was chosen over GAN/diffusion models.

## Troubleshooting

* **`cuda: False` on a 13.x driver**: install the cu124 wheel from the PyTorch index, not the default PyPI wheel. Run `cloudlift env-check`.
* **GDAL/Rasterio import errors on Windows**: use the PyPI Rasterio wheel in a clean venv, or conda-forge `gdal rasterio`. Never mix OSGeo4W GDAL DLLs on PATH.
* **CUDA OOM on 6 GB**: keep `tile: 256`, `batch_size: 1`, `gradient_checkpointing: true`, `precision: 16-mixed`; raise `accumulate_grad_batches` instead of the batch size. Inference peaks at about 0.5 GB for the full model at 256 px with AMP.
* **Windows `UnicodeEncodeError`** in custom scripts: set `PYTHONUTF8=1`.
* **`UnCRtainTS submodule missing`**: `git submodule update --init --recursive`.

## Repository layout

`cloudlift/` (package: `io`, `preprocess`, `model`, `eval`, `service`, `ui`, `pipeline.py`, `cli.py`),
`configs/`, `scripts/`, `tests/`, `docker/`, `deploy/hf_space/`, `assets/weights/`,
`third_party/UnCRtainTS` (git submodule), `AGENT_PROMPT.md` (build specification), `MODEL_CARD.md`.

## Licence

MIT (see `LICENSE`). UnCRtainTS © Patrick Ebel, MIT-licensed upstream.
