"""Evaluate Method 3 — prompt ensembling v2 (training-free)."""

from __future__ import annotations

import torch

import _bootstrap  # noqa: F401

from compositional_retrieval.annotations import load_annotations
from compositional_retrieval.config import DATA_ROOT, embeddings_path, results_path
from compositional_retrieval.data import build_attribute_indices, load_celeba
from compositional_retrieval.features import load_clip, load_or_extract_embeddings
from compositional_retrieval.methods.prompt_ensemble import (
    evaluate_pos_neg_logit_composition,
    precompute_attribute_pos_neg_embeddings,
)
from compositional_retrieval.metrics import compute_query_average_results
from compositional_retrieval.prompts import (
    humanized_mappings_neg,
    humanized_mappings_pos,
    prompt_templates_v2,
)


def main():
    celeba = load_celeba(split="test", root=DATA_ROOT)
    _idx2attribute, attribute2idx = build_attribute_indices(celeba)
    attr_names = list(celeba.attr_names)

    model, processor, device = load_clip()
    embeddings, _labels = load_or_extract_embeddings(
        celeba, model, processor, device, embeddings_path("test")
    )

    print("Precomputing pos/neg attribute embeddings with the expanded template bank...")
    E_pos, E_neg = precompute_attribute_pos_neg_embeddings(
        attr_names=attr_names,
        humanized_mappings_pos=humanized_mappings_pos,
        humanized_mappings_neg=humanized_mappings_neg,
        templates=prompt_templates_v2,
        model=model,
        processor=processor,
        device=device,
    )
    E_pos = E_pos.to(embeddings.device)
    E_neg = E_neg.to(embeddings.device)
    print(f"E_pos: {tuple(E_pos.shape)}  E_neg: {tuple(E_neg.shape)}")

    annotations = load_annotations()
    evaluation_results = evaluate_pos_neg_logit_composition(
        annotations=annotations,
        embeddings=embeddings,
        E_pos=E_pos,
        E_neg=E_neg,
        attr_names=attr_names,
        attribute2idx=attribute2idx,
    )
    average_results_per_query = [
        compute_query_average_results(q) for q in evaluation_results
    ]

    out = results_path("prompt_ensemble")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"per_query": average_results_per_query, "raw": evaluation_results}, out)
    print(f"Saved prompt-ensemble results to {out}")


if __name__ == "__main__":
    main()
