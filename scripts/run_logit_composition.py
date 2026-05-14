"""Evaluate Method 2 — per-attribute logit composition (training-free)."""

from __future__ import annotations

import torch

import _bootstrap  # noqa: F401

from compositional_retrieval.annotations import load_annotations
from compositional_retrieval.config import DATA_ROOT, embeddings_path, results_path
from compositional_retrieval.data import build_attribute_indices, load_celeba
from compositional_retrieval.features import load_clip, load_or_extract_embeddings
from compositional_retrieval.methods.logit_composition import (
    evaluate_logit_composition,
    precompute_attribute_text_embeddings,
)
from compositional_retrieval.metrics import compute_query_average_results


def main():
    celeba = load_celeba(split="test", root=DATA_ROOT)
    _idx2attribute, attribute2idx = build_attribute_indices(celeba)
    attr_names = list(celeba.attr_names)

    model, processor, device = load_clip()
    embeddings, _labels = load_or_extract_embeddings(
        celeba, model, processor, device, embeddings_path("test")
    )

    attr_text_embs = precompute_attribute_text_embeddings(attr_names, model, processor, device).to(embeddings.device)
    annotations = load_annotations()

    evaluation_results = evaluate_logit_composition(
        annotations=annotations,
        embeddings=embeddings,
        attr_text_embs=attr_text_embs,
        attribute2idx=attribute2idx,
    )
    average_results_per_query = [
        compute_query_average_results(q) for q in evaluation_results
    ]

    out = results_path("logit_composition")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"per_query": average_results_per_query, "raw": evaluation_results}, out)
    print(f"Saved logit-composition results to {out}")


if __name__ == "__main__":
    main()
