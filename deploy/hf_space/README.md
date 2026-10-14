---
title: CloudLift
emoji: ☁️
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Gen-AI cloud removal for LISS-IV (UnCRtainTS + SAR)
---

# CloudLift — operator console (public demo)

Generative AI cloud and cloud-shadow removal for **LISS-IV** (Green, Red, NIR; 5.8 m), fusing
**EOS-04 C-band SAR** and a **temporal reference pass** with an adapted **UnCRtainTS** network
that outputs a reconstruction *and* a per-pixel uncertainty band.

**What this demo is — read before judging the numbers**

* The four hero scenes are **SYNTHETIC** procedural scenes (NER-like terrain, river, fields,
  OpenSimplex clouds). They are not real LISS-IV / EOS-04 data: Bhoonidhi data is not
  redistributed and must stay in-country.
* The model weights were fine-tuned **only on synthetic pairs**. They have never seen a real
  LISS-IV scene. See `MODEL_CARD.md` in the source repository.
* Every metric on screen is computed by the pipeline for that run (against the synthetic
  cloud-free truth). Nothing is hard-coded. This Space runs on CPU.

Source code, setup, training and evaluation: https://github.com/Gurjas2112/CloudLift
