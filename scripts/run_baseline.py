"""Evaluate the baseline (text + image fusion) method on the benchmark."""

from __future__ import annotations

import torch

import _bootstrap  # noqa: F401

from compositional_retrieval.annotations import load_annotations
from compositional_retrieval.config import DATA_ROOT, embeddings_path, results_path
from compositional_retrieval.data import load_celeba
from compositional_retrieval.evaluation import evaluate
from compositional_retrieval.features import load_clip, load_or_extract_embeddings
from compositional_retrieval.methods.baseline import baseline_embed_query, baseline_fusion
from compositional_retrieval.metrics import compute_query_average_results


def main():
    celeba = load_celeba(split="test", root=DATA_ROOT)
    model, processor, device = load_clip()
    embeddings, _labels = load_or_extract_embeddings(
        celeba, model, processor, device, embeddings_path("test")
    )
    annotations = load_annotations()

    evaluation_results = evaluate(
        fusion_mechanism=baseline_fusion,
        query_embedding_function=baseline_embed_query,
        annotations=annotations,
        model=model,
        processor=processor,
        device=device,
        embeddings=embeddings,
    )
    average_results_per_query = [
        compute_query_average_results(q) for q in evaluation_results
    ]

    out = results_path("baseline")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"per_query": average_results_per_query, "raw": evaluation_results}, out)
    print(f"Saved baseline results to {out}")


if __name__ == "__main__":
    main()
