"""Load CLIP and cache image embeddings for a CelebA split."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import CLIPModel, CLIPProcessor

from .encoding import encode_images


def load_clip(model_name: str | None = None, device: str | None = None):
    """Load CLIP model + processor and put the model in eval() mode on `device`."""
    from .config import CLIP_MODEL_NAME, default_device

    if model_name is None:
        model_name = CLIP_MODEL_NAME
    if device is None:
        device = default_device()
    print(f"Loading CLIP model {model_name} on {device}...")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    print("Model loaded.")
    return model, processor, device


def load_or_extract_embeddings(
    celeba,
    model,
    processor,
    device,
    cache_path: str | Path,
    batch_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (embeddings, labels) for the dataset, using `cache_path` as the disk cache.

    embeddings: (N, D) on `device`, L2-normalized per row.
    labels:     (N, 40) on CPU.
    """
    cache_path = Path(cache_path)

    if cache_path.exists():
        print(f"Found cached embeddings at {cache_path}")
        data = torch.load(cache_path)
        embeddings = data["embeddings"].to(device)
        labels = data["labels"]
        print(f"Loaded embeddings with shape: {tuple(embeddings.shape)}")
        print(f"Loaded labels with shape:     {tuple(labels.shape)}")
        return embeddings, labels

    print(f"Cache miss at {cache_path}. Encoding dataset...")
    # Iterate twice over the map-style dataset: cheap, and avoids holding all PIL
    # images in memory at once.
    imgs = (image for image, _label in celeba)
    labels = torch.stack([label for _image, label in celeba], dim=0)   # (N, 40)
    embeddings = encode_images(imgs, model, processor, device, batch_size=batch_size)
    print("Encoding completed.")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embeddings": embeddings, "labels": labels}, cache_path)
    print(f"Saved embeddings to {cache_path}")

    return embeddings.to(device), labels
