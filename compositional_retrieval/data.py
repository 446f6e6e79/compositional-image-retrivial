"""CelebA loading + attribute helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import CelebA


def load_celeba(split: str = "test", root: str | Path | None = None) -> CelebA:
    """Load a CelebA split. `root` should be the parent of the `celeba/` folder."""
    if root is None:
        from .config import DATA_ROOT
        root = DATA_ROOT
    return CelebA(root=str(root), split=split, download=False)


def build_attribute_indices(celeba: CelebA) -> tuple[dict[int, str], dict[str, int]]:
    """Return (idx2attribute, attribute2idx) for the 40 CelebA attributes."""
    idx2attribute = {idx: name for idx, name in enumerate(celeba.attr_names)}
    attribute2idx = {name: idx for idx, name in enumerate(celeba.attr_names)}
    return idx2attribute, attribute2idx


def collect_labels(celeba: CelebA) -> np.ndarray:
    """Iterate the dataset once and return an (N, 40) numpy array of binary labels."""
    return np.array([labels for _, labels in celeba])


def compute_attribute_stats(celeba: CelebA) -> dict[str, np.ndarray]:
    """Per-attribute positive count and frequency over the split."""
    all_labels = collect_labels(celeba)
    return {
        "attr_names": np.array(list(celeba.attr_names)),
        "counts": all_labels.sum(axis=0),
        "frequencies": all_labels.mean(axis=0),
        "all_labels": all_labels,
    }


def retrieve_by_attributes(
    celeba: CelebA,
    attribute2idx: dict[str, int],
    parameters: dict[str, str],
) -> list[int]:
    """Indices of images satisfying every condition in `parameters`.

    Query format: dict mapping attribute name to "+" (must be present) or "-"
    (must be absent), e.g. ``{"Bald": "+", "Eyeglasses": "-"}``.
    """
    valid_indices = set(range(len(celeba)))

    for attr_name, value in parameters.items():
        attr_idx = attribute2idx[attr_name]
        if value == "+":
            for idx in valid_indices.copy():
                if celeba[idx][1][attr_idx] == 0:
                    valid_indices.remove(idx)
        elif value == "-":
            for idx in valid_indices.copy():
                if celeba[idx][1][attr_idx] == 1:
                    valid_indices.remove(idx)
        else:
            raise ValueError(f"Invalid value for attribute condition: {value}. Use '+' or '-'.")

    return list(valid_indices)


def active_attributes(celeba: CelebA, idx2attribute: dict[int, str], image_idx: int) -> list[str]:
    """Names of attributes that are positive (label == 1) for image `image_idx`."""
    _, labels = celeba[image_idx]
    return [idx2attribute[i] for i, value in enumerate(labels) if value == 1]
