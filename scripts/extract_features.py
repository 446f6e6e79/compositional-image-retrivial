"""Cache CLIP image embeddings for a CelebA split.

Usage:
    python scripts/extract_features.py --split test
    python scripts/extract_features.py --split train
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from compositional_retrieval.config import DATA_ROOT, embeddings_path
from compositional_retrieval.data import load_celeba
from compositional_retrieval.features import load_clip, load_or_extract_embeddings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "valid", "test"], default="test")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    celeba = load_celeba(split=args.split, root=DATA_ROOT)
    print(f"CelebA {args.split} split size: {len(celeba)}")

    model, processor, device = load_clip()
    cache_path = embeddings_path(args.split)
    embeddings, labels = load_or_extract_embeddings(
        celeba, model, processor, device, cache_path, batch_size=args.batch_size
    )
    print(f"Done. embeddings: {tuple(embeddings.shape)}  labels: {tuple(labels.shape)}")
    print(f"Cache file: {cache_path}")


if __name__ == "__main__":
    main()
