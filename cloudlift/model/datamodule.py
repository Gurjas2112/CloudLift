"""Training data: a preprocess-once tile cache + on-the-fly synthetic cloud pairs.

Cache layout (versioned so changed preprocessing never mixes with old tiles)::

    cache/<preproc_version>/tiles/<scene_id>_<yyyy>_<xxxx>.npz   target[3], reference[3], sar[2], gap
    cache/<preproc_version>/index.json

Targets are cloud-free: the synthetic fixture's ``clean.tif``, or for real scenes only tiles
whose optical pass is > 99 % clear. Clouds are composited per item with a seed derived from
the item index, so every epoch and every rerun sees identical pairs. Augmentation is
geometric only (flips / rot90): colour jitter would break spectral integrity.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import albumentations as A
import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from cloudlift.config import Config
from cloudlift.io.sources import DataSource
from cloudlift.model.adapter import stack_inputs
from cloudlift.preprocess import cloudmask
from cloudlift.preprocess.synthetic_clouds import make_clouds

log = logging.getLogger(__name__)


def cache_dir(cfg: Config) -> Path:
    return Path(cfg.paths.cache_root) / cfg.data.preproc_version


def build_tile_cache(source: DataSource, cfg: Config, *, max_obstructed: float = 0.01) -> Path:
    """Run stages 1-2 once per scene and write non-overlapping training tiles (idempotent)."""
    from cloudlift.pipeline import coregister_inputs, ingest, read_truth

    root = cache_dir(cfg)
    index_path = root / "index.json"
    tiles_dir = root / "tiles"
    index: dict[str, Any] = {"tile": cfg.data.tile, "scenes": {}, "tiles": []}
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if index.get("tile") != cfg.data.tile:
            raise RuntimeError(f"{root} was built with tile={index.get('tile')}; bump data.preproc_version")
    tiles_dir.mkdir(parents=True, exist_ok=True)
    t = cfg.data.tile
    for scene in source.list_scenes():
        if scene.scene_id in index["scenes"]:
            continue
        meta, optical, nodata = ingest(scene, None)
        sar, _, reference, _ = coregister_inputs(scene, meta, optical, nodata, t)
        truth = read_truth(scene, None)
        if truth is not None:
            target, usable = truth, ~nodata
        else:
            m = cloudmask.detect(optical, reference, cfg.cloudmask.method)
            target, usable = optical, ~(m.combined | nodata)
        h, w = optical.shape[1:]
        n = 0
        for y in range(0, h - t + 1, t):
            for x in range(0, w - t + 1, t):
                if usable[y : y + t, x : x + t].mean() < 1.0 - max_obstructed:
                    continue
                name = f"{scene.scene_id}_{y:05d}_{x:05d}.npz"
                np.savez(
                    tiles_dir / name,
                    target=target[:, y : y + t, x : x + t].astype(np.float32),
                    reference=reference[:, y : y + t, x : x + t].astype(np.float32),
                    sar=sar[:, y : y + t, x : x + t].astype(np.float32),
                    gap=np.int32(scene.reference_gap_days()),
                )
                index["tiles"].append(name)
                n += 1
        index["scenes"][scene.scene_id] = {"tiles": n, "synthetic": scene.synthetic}
        log.info("cached %d tiles from %s", n, scene.scene_id)
    index["tiles"] = sorted(index["tiles"])
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return root


class CloudPairDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, root: Path, names: list[str], cfg: Config, *, augment: bool, seed_offset: int) -> None:
        self.root, self.names, self.cfg = root, names, cfg
        self.augment, self.seed_offset = augment, seed_offset

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        seed = cfg.seed * 1_000_003 + self.seed_offset + i
        rng = np.random.default_rng(seed)
        with np.load(self.root / "tiles" / self.names[i]) as z:
            target, ref, sar, gap = z["target"], z["reference"], z["sar"], int(z["gap"])
        frac = float(rng.uniform(cfg.train.cloud_fraction_min, cfg.train.cloud_fraction_max))
        clouds = make_clouds(target, frac, seed=seed, thickness=float(rng.uniform(0.7, 1.0)))
        mask = (clouds.mask > 0).astype(np.float32)
        if self.augment:
            hwc = np.concatenate([target, clouds.cloudy, ref, sar], axis=0).transpose(1, 2, 0)
            aug = A.Compose(
                [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)], seed=seed
            )
            o = aug(image=hwc, mask=mask)
            chw = np.ascontiguousarray(o["image"].transpose(2, 0, 1))
            target, cloudy, ref, sar = chw[0:3], chw[3:6], chw[6:9], chw[9:11]
            mask = np.ascontiguousarray(o["mask"])
        else:
            cloudy = clouds.cloudy
        x, dates = stack_inputs(cloudy, sar, [ref], cfg.data.input_t, gap)
        return {
            "x": torch.from_numpy(x),
            "dates": torch.from_numpy(dates),
            "target": torch.from_numpy(np.ascontiguousarray(target)),
            "mask": torch.from_numpy(mask)[None],
        }


def split_names(names: list[str], val_fraction: float) -> tuple[list[str], list[str]]:
    """Deterministic split: every k-th tile goes to validation."""
    if len(names) < 2 or val_fraction <= 0:
        return names, names[:1]
    k = max(2, round(1.0 / val_fraction))
    val = [n for j, n in enumerate(names) if j % k == 0]
    train = [n for j, n in enumerate(names) if j % k != 0]
    return train, val


class TileDataModule(L.LightningDataModule):
    def __init__(self, cfg: Config, root: Path) -> None:
        super().__init__()
        self.cfg, self.root = cfg, root
        names = json.loads((root / "index.json").read_text(encoding="utf-8"))["tiles"]
        if not names:
            raise RuntimeError(f"tile cache {root} is empty")
        self.train_names, self.val_names = split_names(names, cfg.train.val_fraction)

    def _loader(self, ds: CloudPairDataset, shuffle: bool) -> DataLoader[dict[str, torch.Tensor]]:
        g = torch.Generator()
        g.manual_seed(self.cfg.seed)
        nw = self.cfg.train.num_workers
        return DataLoader(
            ds,
            batch_size=self.cfg.train.batch_size,
            shuffle=shuffle,
            num_workers=nw,
            generator=g,
            persistent_workers=nw > 0,
            pin_memory=torch.cuda.is_available(),
        )

    def train_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        return self._loader(
            CloudPairDataset(self.root, self.train_names, self.cfg, augment=True, seed_offset=0), True
        )

    def val_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        ds = CloudPairDataset(self.root, self.val_names, self.cfg, augment=False, seed_offset=500_000)
        return self._loader(ds, False)
