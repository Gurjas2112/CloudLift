# CloudLift: screenshots and screen recording

`cloudlift_full_walkthrough.mp4` (about 70 s, 1440×900) records the whole flow below in one session.
Captured locally against the console (`configs/demo.yaml`, trained weights, RTX 3060) and the API.

All imagery shown is **synthetic** (procedural scenes). The "own data" files in steps 13–22 are a synthetic
scene re-encoded as a 4-band uint16 Blue/Green/Red/NIR GeoTIFF (×10000), to show the any-sensor path.
Every metric on screen is computed by that run.

| # | Screenshot | Step |
|---|---|---|
| 01 | `01_console_home.png` | Console on launch: status bar, active tile, pipeline, metrics (all `—` before a run) |
| 02 | `02_select_scene.png` | Choosing a scene from the dropdown |
| 03 | `03_scene_loaded_cloudy.png` | Scene loaded: cloudy false-colour preview, scene details |
| 04 | `04_pipeline_running.png` | *Initiate recovery*: stage list in progress |
| 05 | `05_reconstructed_slider.png` | Result: cloudy ↔ reconstructed slider, metrics, obstruction card |
| 06 | `06_slider_cloudy_side.png` | Slider moved towards the cloudy input |
| 07 | `07_slider_reconstructed_side.png` | Slider moved towards the reconstruction |
| 08 | `08_confidence_overlay.png` | *Show confidence*: teal = low σ, amber = high σ |
| 09 | `09_export_geotiff.png` | *Export GeoTIFF* enabled after the run |
| 10 | `10_aoi_input.png` | Typing an AOI (`point:lat,lon,radius`) to crop any location |
| 11 | `11_live_run_progress.png` | Live pipeline run on the AOI |
| 12 | `12_live_aoi_result.png` | AOI result with its own metrics |
| 13 | `13_upload_panel.png` | *Use your own scene* panel |
| 14 | `14_upload_files_attached.png` | Optical, SAR and earlier-pass GeoTIFFs attached; band order `2,3,4`, auto scaling |
| 15 | `15_upload_scene_added.png` | Uploaded scene selected: band mapping, size, inputs present |
| 16 | `16_upload_recovery_running.png` | Recovery running on the uploaded scene |
| 17 | `17_upload_reconstructed.png` | Reconstruction of the uploaded scene (metrics vs the temporal reference) |
| 18 | `18_upload_confidence.png` | Confidence overlay on the uploaded scene |
| 19 | `19_api_docs.png` | FastAPI docs (`/docs`) |
| 20 | `20_api_job_done.png` | `POST /reconstruct` with files + `bands=2,3,4`, then `GET /jobs/{id}`: all 7 stages done |
| 21 | `21_evaluation_panels.png` | Evaluation panels: cloudy, reconstructed, reference, uncertainty, NDVI difference |
| 22 | `22_cli_infer_own_files.png` | `cloudlift infer --optical … --bands 2,3,4`: real command output |

## Regenerate

```bash
pip install -r requirements-dev.txt          # includes playwright; uses the installed Chrome
CLOUDLIFT_CONFIG=configs/demo.yaml python -m cloudlift.ui.console &
CLOUDLIFT_CONFIG=configs/demo.yaml cloudlift serve --port 8000 &
python scripts/capture_walkthrough.py         # writes this folder; needs ffmpeg for the mp4
```
