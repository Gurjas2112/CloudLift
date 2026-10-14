"""Typed configuration: YAML files merged on top of ``configs/default.yaml``.

Environment variables prefixed ``CLOUDLIFT_`` (see ``.env.example``) override the
device and weights path so the same YAML runs on the laptop, a cloud GPU or CPU.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


class Paths(BaseModel):
    data_root: Path = Path("data/raw")
    cache_root: Path = Path("cache")
    results_root: Path = Path("results")
    weights: Path | None = None


class DataCfg(BaseModel):
    tile: int = 256
    overlap: int = 32
    input_t: int = Field(default=2, ge=1)
    preproc_version: str = "v1"


class ModelCfg(BaseModel):
    size: Literal["tiny", "full"] = "tiny"
    covmode: Literal["diag", "iso"] = "diag"
    gradient_checkpointing: bool = False


class LossCfg(BaseModel):
    l1: float = 1.0
    ssim: float = 0.2
    sam: float = 0.1
    nll: float = 0.05
    cloud_weight: float = 3.0


class CloudMaskCfg(BaseModel):
    method: Literal["auto", "omnicloudmask", "baseline"] = "auto"


class InferCfg(BaseModel):
    batch_size: int = 1
    amp: bool = True


class TrainCfg(BaseModel):
    batch_size: int = 1
    accumulate_grad_batches: int = 1
    precision: str = "32-true"
    num_workers: int = Field(default=0, ge=0)
    max_epochs: int = 2
    lr: float = 1e-3
    lr_decay: float = 0.8
    val_fraction: float = 0.2
    cloud_fraction_min: float = 0.2
    cloud_fraction_max: float = 0.7


class Config(BaseModel):
    seed: int = 0
    device: Literal["auto", "cpu", "cuda"] = "auto"
    paths: Paths = Paths()
    data: DataCfg = DataCfg()
    model: ModelCfg = ModelCfg()
    loss: LossCfg = LossCfg()
    cloudmask: CloudMaskCfg = CloudMaskCfg()
    infer: InferCfg = InferCfg()
    train: TrainCfg = TrainCfg()

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def config_hash(self) -> str:
        """SHA-256 (16 hex) of the result-affecting settings.

        Paths other than the weights file and dataloader/runtime knobs are excluded so that
        moving the output folder does not invalidate cached results.
        """
        d = self.model_dump(mode="json")
        d["paths"] = {"weights": d["paths"]["weights"]}
        d["train"] = {k: d["train"][k] for k in ("lr", "lr_decay", "max_epochs")}
        blob = json.dumps(d, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


class EnvOverrides(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLOUDLIFT_", env_file=".env", extra="ignore")

    device: Literal["auto", "cpu", "cuda"] | None = None
    weights: str | None = None


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load default.yaml, merge ``path`` (if any), env overrides, then explicit ``overrides``."""
    raw: dict[str, Any] = {}
    if DEFAULT_CONFIG.exists():
        raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8")) or {}
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        raw = _deep_merge(raw, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    env = EnvOverrides()
    if env.device:
        raw["device"] = env.device
    if env.weights:
        raw.setdefault("paths", {})["weights"] = env.weights
    if overrides:
        raw = _deep_merge(raw, overrides)
    return Config.model_validate(raw)
