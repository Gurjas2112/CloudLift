"""Lightning fine-tuning module (AdamW, lr 1e-3, exponential decay 0.8 / epoch)."""

from __future__ import annotations

import json
import logging
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import lightning as L
import torch

from cloudlift.config import Config
from cloudlift.determinism import set_determinism
from cloudlift.io.sources import DataSource
from cloudlift.model.losses import CombinedLoss, spectral_angle, ssim_map
from cloudlift.model.uncrtaints_wrapper import CloudLiftNet, load, save_cloudlift_weights

log = logging.getLogger(__name__)


class CloudLiftLit(L.LightningModule):
    def __init__(self, cfg_dict: dict[str, Any], size: str = "tiny", covmode: str = "diag") -> None:
        super().__init__()
        self.save_hyperparameters()
        self.cfg = Config.model_validate(cfg_dict)
        c = self.cfg
        self.net: CloudLiftNet = load(
            c.paths.weights,
            size=size,
            covmode=covmode,
            gradient_checkpointing=c.model.gradient_checkpointing,
            seed=c.seed,
        )
        self.loss_fn = CombinedLoss(c.loss)

    def forward(self, x: torch.Tensor, dates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net(x, dates)

    def training_step(self, batch: dict[str, torch.Tensor], _: int) -> torch.Tensor:
        recon, sigma = self(batch["x"], batch["dates"])
        loss, parts = self.loss_fn(recon, sigma, batch["target"], batch["mask"])
        self.log_dict(
            {f"train_{k}": v for k, v in parts.items()},
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=batch["x"].shape[0],
        )
        self.log(
            "train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch["x"].shape[0]
        )
        return loss

    @torch.no_grad()
    def validation_step(self, batch: dict[str, torch.Tensor], _: int) -> None:
        recon, sigma = self(batch["x"], batch["dates"])
        target, mask = batch["target"].float(), batch["mask"].float()
        recon = recon.float()
        loss, _ = self.loss_fn(recon, sigma, target, mask)
        m = mask.expand_as(target)
        mse_m = ((recon - target) ** 2 * m).sum() / m.sum().clamp_min(1)
        psnr_m = 10 * torch.log10(1.0 / mse_m.clamp_min(1e-12))
        mse = ((recon - target) ** 2).mean()
        sam = torch.rad2deg(spectral_angle(recon, target))
        sam_m = (sam * mask[:, 0]).sum() / mask.sum().clamp_min(1)
        bs = target.shape[0]
        self.log_dict(
            {
                "val_loss": loss,
                "val_psnr": 10 * torch.log10(1.0 / mse.clamp_min(1e-12)),
                "val_psnr_masked": psnr_m,
                "val_ssim": ssim_map(recon, target).mean(),
                "val_sam_deg": sam.mean(),
                "val_sam_deg_masked": sam_m,
            },
            prog_bar=True,
            on_epoch=True,
            batch_size=bs,
        )

    def configure_optimizers(self) -> Any:
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.cfg.train.lr)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=self.cfg.train.lr_decay)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


def fit(cfg: Config, source: DataSource, run_name: str | None = None) -> dict[str, Any]:
    """Build the tile cache if needed, train, and write ``results/<run>/`` artefacts."""
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger

    from cloudlift.model.datamodule import TileDataModule, build_tile_cache

    device = cfg.resolve_device()
    set_determinism(cfg.seed, warn_only=device == "cuda")
    root = build_tile_cache(source, cfg)
    dm = TileDataModule(cfg, root)
    run = run_name or f"train_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    out = Path(cfg.paths.results_root) / run
    out.mkdir(parents=True, exist_ok=True)
    precision = cfg.train.precision if device == "cuda" else "32-true"
    module = CloudLiftLit(cfg.model_dump(mode="json"), size=cfg.model.size, covmode=cfg.model.covmode)
    ckpt = ModelCheckpoint(
        dirpath=out / "checkpoints",
        filename="epoch{epoch:02d}",
        monitor="val_loss",
        mode="min",
        save_last=True,
        auto_insert_metric_name=False,
    )
    trainer = L.Trainer(
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        max_epochs=cfg.train.max_epochs,
        precision=cast(Any, precision),
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        deterministic="warn" if device == "cuda" else True,
        logger=CSVLogger(out, name="logs"),
        callbacks=[ckpt],
        enable_progress_bar=True,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
    )
    trainer.fit(module, datamodule=dm)
    weights = save_cloudlift_weights(
        module.net,
        out / "cloudlift_weights.pt",
        trained_on=f"{type(source).__name__}:{cfg.data.preproc_version}:{len(dm.train_names)}tiles",
    )
    history = _read_history(out)
    summary = {
        "run": run,
        "device": device,
        "precision": precision,
        "train_tiles": len(dm.train_names),
        "val_tiles": len(dm.val_names),
        "epochs": cfg.train.max_epochs,
        "best_checkpoint": ckpt.best_model_path,
        "weights": weights.as_posix(),
        "config_hash": cfg.config_hash(),
        "size": cfg.model.size,
        "history": history,
        "source": repr(source),
    }
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def _read_history(out: Path) -> list[dict[str, float]]:
    """Per-epoch metrics from the CSV logger (epoch-level rows only)."""
    import csv

    rows: dict[int, dict[str, float]] = {}
    for f in sorted(out.glob("logs/version_*/metrics.csv")):
        with open(f, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if not r.get("epoch"):
                    continue
                e = int(float(r["epoch"]))
                for k, v in r.items():
                    if v not in ("", None) and (k.endswith("_epoch") or k.startswith("val_")):
                        try:
                            fv = float(v)
                        except ValueError:
                            continue
                        if math.isfinite(fv):
                            rows.setdefault(e, {"epoch": e})[k] = fv
    return [rows[e] for e in sorted(rows)]
