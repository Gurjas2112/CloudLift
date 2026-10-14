"""Evaluation report: ``eval.json``, ``eval.md`` and comparison panels (PNG)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cloudlift.eval.metrics import ndvi  # noqa: E402


def stretch_limits(arr: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> tuple[np.ndarray, np.ndarray]:
    flat = arr.reshape(arr.shape[0], -1)
    return np.percentile(flat, lo, axis=1), np.percentile(flat, hi, axis=1)


def fcc(arr: np.ndarray, limits: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """LISS-IV standard false-colour composite (R=NIR, G=Red, B=Green) as uint8 ``[H,W,3]``."""
    lo, hi = limits if limits is not None else stretch_limits(arr)
    a = (arr - lo[:, None, None]) / np.maximum(hi - lo, 1e-6)[:, None, None]
    a = np.clip(a, 0, 1)
    return (np.stack([a[2], a[1], a[0]], axis=-1) * 255).astype(np.uint8)


def uncertainty_rgb(u: np.ndarray, vmax: float | None = None) -> np.ndarray:
    """Teal (low, SAR/temporally constrained) -> amber (high, interpolated) ramp, uint8 ``[H,W,3]``."""
    vmax = float(np.percentile(u, 99)) if vmax is None else vmax
    t = np.clip(u / max(vmax, 1e-9), 0, 1)[..., None]
    teal = np.array([0x12, 0xB0, 0x9C], np.float32)
    amber = np.array([0xEF, 0x9F, 0x27], np.float32)
    return ((1 - t) * teal + t * amber).astype(np.uint8)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        if math.isinf(v):
            return "inf"
        if math.isnan(v):
            return "—"
        return f"{v:.4f}"
    if isinstance(v, dict):
        return ", ".join(f"{k}={_fmt(x)}" for k, x in v.items())
    return str(v)


def json_safe(o: Any) -> Any:
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None if math.isnan(o) else ("inf" if o > 0 else "-inf")
    if isinstance(o, dict):
        return {k: json_safe(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [json_safe(v) for v in o]
    if isinstance(o, np.generic):
        return json_safe(o.item())
    return o


def write_panels(
    path: str | Path,
    cloudy: np.ndarray,
    recon: np.ndarray,
    reference: np.ndarray | None,
    uncertainty: np.ndarray,
    mask: np.ndarray,
    title: str = "",
    ref_label: str = "Reference",
) -> Path:
    """cloudy | reconstructed | reference | uncertainty | NDVI difference."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    limits = stretch_limits(reference if reference is not None else recon)
    fig, ax = plt.subplots(1, 5, figsize=(22, 4.8), constrained_layout=True)
    ax[0].imshow(fcc(cloudy, limits))
    ax[0].contour(mask, levels=[0.5], colors="#E24B4A", linewidths=0.6)
    ax[0].set_title("Cloudy input (FCC) + mask")
    ax[1].imshow(fcc(recon, limits))
    ax[1].set_title("Reconstructed")
    if reference is not None:
        ax[2].imshow(fcc(reference, limits))
        ax[2].set_title(ref_label)
        nd = ndvi(recon) - ndvi(reference)
        im = ax[4].imshow(nd, cmap="RdBu", vmin=-0.3, vmax=0.3)
        fig.colorbar(im, ax=ax[4], fraction=0.046)
        ax[4].set_title(f"NDVI(recon) - NDVI({ref_label.lower()})")
    else:
        ax[2].text(0.5, 0.5, "no reference", ha="center", va="center")
        ax[4].text(0.5, 0.5, "n/a", ha="center", va="center")
    im = ax[3].imshow(uncertainty, cmap="viridis")
    fig.colorbar(im, ax=ax[3], fraction=0.046)
    ax[3].set_title("Uncertainty (σ)")
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    if title:
        fig.suptitle(title)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def write_report(
    out_dir: str | Path, report: dict[str, Any], panels: list[Path] | None = None
) -> tuple[Path, Path]:
    """Write ``eval.json`` and a human-readable ``eval.md``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jpath = out / "eval.json"
    jpath.write_text(json.dumps(json_safe(report), indent=2, sort_keys=True), encoding="utf-8")

    lines = [f"# CloudLift evaluation — {report.get('run_id', '')}", ""]
    if report.get("synthetic"):
        lines += [
            "> **SYNTHETIC DATA.** Scores below are computed on the procedural fixture, not on real LISS-IV.",
            "",
        ]
    model = report.get("model", {})
    if model and not model.get("trained", True):
        lines += ["> **UNTRAINED MODEL.** Random-init weights; numbers only test the pipeline.", ""]
    lines += [
        f"- Mode: **{report.get('mode', '')}**",
        f"- Split: {report.get('split', '')}",
        f"- Model: {_fmt(model)}",
        f"- Config hash: `{report.get('config_hash', '')}`",
        f"- Seed: {report.get('seed', '')}",
        f"- Device: {report.get('device', '')}",
    ]
    timing = report.get("timing", {})
    if timing:
        lines.append(
            f"- Measured inference: {_fmt(timing.get('ms_per_tile'))} ms/tile "
            f"({timing.get('tiles', '')} tiles of {timing.get('tile', '')} px)"
        )
    lines += ["", "| scene | metric | value |", "|---|---|---|"]
    for sid, m in report.get("scenes", {}).items():
        for k, v in m.items():
            lines.append(f"| {sid} | {k} | {_fmt(v)} |")
    if report.get("summary"):
        lines += ["", "## Mean over scenes", "", "| metric | value |", "|---|---|"]
        lines += [f"| {k} | {_fmt(v)} |" for k, v in report["summary"].items()]
    if report.get("mode") == "real-scene":
        lines += [
            "",
            "Full-reference metrics are unavailable: no cloud-free truth exists under thick cloud. "
            "Only NDVI consistency against the temporal reference is reported (not a ground-truth score).",
        ]
    if panels:
        lines += ["", "## Panels", ""] + [f"![{p.stem}]({p.name})" for p in panels]
    mpath = out / "eval.md"
    mpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jpath, mpath


def summarize(scenes: dict[str, dict[str, Any]]) -> dict[str, float]:
    keys = {k for m in scenes.values() for k, v in m.items() if isinstance(v, int | float)}
    out = {}
    for k in sorted(keys):
        vals = [float(m[k]) for m in scenes.values() if isinstance(m.get(k), int | float)]
        finite = [v for v in vals if math.isfinite(v)]
        out[k] = float(np.mean(finite)) if finite else (vals[0] if vals else float("nan"))
    return out
