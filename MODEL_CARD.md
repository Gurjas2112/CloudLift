# Model card: CloudLift UnCRtainTS (LISS-IV + EOS-04 SAR + temporal reference)

## Summary

| | |
|---|---|
| Architecture | UnCRtainTS (Ebel et al., CVPR-W 2023): MBConv encoder, lightweight temporal attention (L-TAE), MBConv decoder, diagonal-Gaussian output head (mean + variance per band) |
| Code | `cloudlift/model/` wraps the unmodified upstream code in `third_party/UnCRtainTS` (git submodule) |
| Size | "full" = paper configuration, 566,150 parameters after the 3-band adaptation; "tiny" = test configuration, 21,926 parameters |
| Inputs | `x [B, T, 5, H, W]`: per time step SAR VV, SAR VH, LISS-IV Green, Red, NIR (all scaled to [0, 1]); `dates [B, T]` days since the oldest observation. Default `T = 2`: (temporal reference, current cloudy pass) |
| Outputs | reconstruction `[B, 3, H, W]` (Green, Red, NIR reflectance) and σ `[B, 3, H, W]` (`covmode=diag`) or `[B, 1, H, W]` (`iso`) |
| Inference | single deterministic forward pass; no sampling, dropout or ensembling |
| Shipped weights | `assets/weights/cloudlift_synthetic_full_v1.pt`: full configuration, **trained on synthetic pairs only** |

## Intended use

* Reconstructing cloud- and shadow-covered pixels of LISS-IV scenes (ResourceSat-2/2A) where a
  co-located EOS-04 SAR acquisition and an earlier cloud-free LISS-IV pass exist, to produce
  analysis-ready mosaics for LULC mapping, disaster monitoring and environmental assessment.
* The uncertainty band is part of the product. Downstream users should mask or down-weight pixels with
  high σ, and treat `uncertainty = 1.0` (SAR missing) as *no information*.

**Not intended for:** replacing observations in clear areas (they are passed through untouched),
change detection *inside* reconstructed areas without checking σ, or any use of the shipped synthetic
weights on real imagery without first fine-tuning and validating on real LISS-IV.

## Training data

| Stage | Data | Status in this repository |
|---|---|---|
| Pre-training (weights initialisation only) | SEN12MS-CR-TS (Sentinel-1 + Sentinel-2) via upstream UnCRtainTS checkpoints | supported (`--weights *.pth.tar`, adapter keeps S1 + S2 B3/B4/B8 filters); **weights not supplied**, not used for the shipped model |
| Fine-tuning | cloud-free LISS-IV tiles with on-the-fly synthetic clouds (OpenSimplex opacity, thin→thick; shifted/blurred shadows), seeded per tile | **real LISS-IV not available yet**. The shipped weights were trained on the procedural **synthetic fixture** (16 scenes of 1024² px, 256 cached tiles) |

The synthetic fixture models NER-like terrain (fBm DEM, a meandering river, forest on hills, crop
fields with per-field vigour, a small urban patch), LISS-IV-like class reflectances, a seasonal shift
between reference and current pass, a known sub-pixel misregistration of the reference, and 4-look
speckled C-band SAR derived from land cover and slope. It is a test bed, not a substitute for real data.

Training: AdamW, lr 1e-3 with ×0.8 decay per epoch; loss = L1 (cloud/shadow pixels up-weighted ×3)
+ 0.2·(1−SSIM) + 0.1·SAM + 0.05·Gaussian NLL. Augmentation is geometric only (flips, 90° rotations),
with no colour jitter, to preserve spectral integrity. Profile `configs/train_laptop.yaml`: 256 px tiles,
batch 1, accumulation 8, fp16 mixed precision, gradient checkpointing, RTX 3060 Laptop 6 GB.

## Evaluation protocol

* **Full-reference mode** (synthetic, or any scene with a cloud-free truth): PSNR, SSIM (per band and
  mean), SAM (degrees), RMSE, NDVI R² and RMSE, all computed **over cloud + shadow pixels**. Whole-scene
  values are reported too. Uncertainty calibration: coverage ECE (|observed − nominal| coverage of
  Gaussian intervals, averaged over 10…90 %) and binned UCE.
* **Baselines on the same pixels**: the cloudy input, and *temporal fill* (substituting the
  co-registered reference pass). These anchor the comparative assessment.
* **Real-scene mode** (no truth under thick cloud): only NDVI and SAM consistency with the temporal
  reference, explicitly labelled as not ground truth, plus qualitative panels.
* Held-out synthetic scenes use seeds disjoint from training (val 1000+, test 2000+).
* Measured inference speed (ms per 256² tile) is written to every `eval.md`.

## Results

Held-out **synthetic** test scenes (3 scenes, seeds 2000–2002), metrics over cloud + shadow pixels; source
`results/demo_full_synthetic/eval_test/eval.json`. Shipped weights = epoch 9 (best val loss) of a `train_laptop` run interrupted in epoch 11.

| method | PSNR (dB) | SSIM | SAM (°) | NDVI R² |
|---|---|---|---|---|
| cloudy input | 5.60 | — | 25.93 | −2.965 |
| temporal fill (reference substitution) | 41.60 | 0.986 | 1.25 | 0.983 |
| CloudLift UnCRtainTS | 41.01 | 0.977 | 1.47 | 0.961 |

Calibration: coverage ECE 0.073, UCE 0.0020. Speed: 78 ms/tile (RTX 3060 Laptop, fp16) and 1.12 s/tile (CPU), 256² tiles.
On this fixture the network does **not** outperform temporal fill, because the fixture's reference is nearly identical to the
truth. No real-LISS-IV result exists yet.

## Why UnCRtainTS (architecture choice)

PS-2 suggests diffusion models, GANs, transformers or multi-modal fusion. CloudLift uses one
generative architecture, chosen for operational fit:

| Requirement | UnCRtainTS | GAN (e.g. pix2pix/DSen2-CR-style) | Diffusion |
|---|---|---|---|
| Deterministic, reproducible output | yes, single forward pass | yes | no, sampling is stochastic (deterministic samplers trade quality) |
| Calibrated per-pixel uncertainty | native (heteroscedastic head) | no | only via repeated sampling (costly) |
| SAR + multi-temporal fusion | native (S1 + S2 time series) | single date typically | possible, heavy |
| 6 GB VRAM training / CPU demo | 0.57 M params | moderate | heavy |

No GAN or diffusion model was trained here. The comparison in this repository is therefore against
the non-learned baselines above, not against other trained networks.

## Known limitations

* **Domain shift**: upstream pre-training is Sentinel-2 (10 m, 13 bands); LISS-IV is 5.8 m with 3 bands
  and different spectral response functions. The adapter re-uses B3/B4/B8 filters, but fine-tuning on
  real LISS-IV is required before operational use.
* **Synthetic-only training of the shipped weights**: scores on the synthetic fixture say nothing
  about real-scene accuracy. Real clouds (cirrus, haze, convective towers) and real SAR (EOS-04
  geometry, speckle statistics, terrain effects in the NER hills) differ from the simulation.
* **Ground truth under thick cloud does not exist**: real-scene evaluation can only check
  consistency with the reference pass or use synthetic clouds over clear scenes.
* **SAR speckle and geometry**: SAR is reprojected but not sub-pixel refined against the optical
  pass. Foreshortening/layover in steep terrain degrades the SAR constraint, and σ should rise there.
* **Temporal change**: if the land changed between the reference and current pass (flood, harvest),
  the reference pulls the reconstruction towards the old state. SAR is the only current-date evidence.
* **Cloud masking**: the default deterministic baseline relies on a cloud-free reference. Single-date
  thresholds miss thin haze. OmniCloudMask (optional) is recommended for real data.

## Determinism

Same input, weights, config and seed give a byte-identical reconstruction on the same device and
library versions. This is verified by `tests/test_determinism.py` against golden SHA-256 hashes. CUDA
training relaxes `torch.use_deterministic_algorithms` to warn-only because `AdaptiveMaxPool2d` has no
deterministic CUDA backward. Inference is strict.

## Compliance and data residency

Operational inputs come from **Bhoonidhi (NRSC/ISRO)** under the **Indian Space Policy 2023**. CloudLift
never downloads from or scrapes Bhoonidhi, never uploads data anywhere by default, and keeps all
processing local or on an India-region GPU. Real scenes are git-ignored and are not part of the public
demo, which shows synthetic scenes only.
