"""Global determinism switch: same input -> byte-identical output on the same device."""

from __future__ import annotations

import os
import random

import numpy as np


def set_determinism(seed: int = 0, *, warn_only: bool = False) -> None:
    """Seed every RNG and force deterministic kernels.

    ``warn_only=True`` is used only for CUDA *training*: UnCRtainTS' ``AdaptiveMaxPool2d``
    has no deterministic CUDA backward kernel. Inference always runs strict.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds the legacy global RNG used by third-party code

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
