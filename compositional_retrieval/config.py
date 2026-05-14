"""Centralized paths, device, and CoOp config — all env-overridable.

Defaults assume the package is run from the repository root with a local
`./data` (for raw CelebA + benchmark JSON) and a local `./cache` (for embeddings
and CoOp context checkpoints).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = Path(os.environ.get("CIR_DATA_ROOT", REPO_ROOT / "data"))
CACHE_DIR = Path(os.environ.get("CIR_CACHE_DIR", REPO_ROOT / "cache"))
ANNOTATIONS_PATH = Path(
    os.environ.get("CIR_ANNOTATIONS_PATH", DATA_ROOT / "celeba_evaluation.json")
)

CLIP_MODEL_NAME = os.environ.get("CIR_CLIP_MODEL", "openai/clip-vit-base-patch32")

# CoOp: "colab" = 16-shot subset, "vm" = full train split. Match the notebook.
COOP_CONFIG = os.environ.get("CIR_COOP_CONFIG", "colab")


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def embeddings_path(split: str) -> Path:
    """Where the cached CLIP image embeddings for a CelebA split are stored."""
    return CACHE_DIR / f"embeddings_{split}.pt"


def coop_ctx_path(config: str) -> Path:
    """Where the trained CoOp context vectors are stored."""
    return CACHE_DIR / f"coop_ctx_{config}.pt"


def results_path(method: str) -> Path:
    """Per-method evaluation results, used by compare_methods.py."""
    return CACHE_DIR / f"results_{method}.pt"
