"""CloudLift: Generative AI cloud removal and reconstruction for LISS-IV imagery."""

import os

# Must be set before cuBLAS initialises, i.e. before any CUDA work, for deterministic GEMMs.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

__version__ = "0.1.0"
MODEL_VERSION = f"cloudlift-uncrtaints-{__version__}"
BANDS = ("Green", "Red", "NIR")
