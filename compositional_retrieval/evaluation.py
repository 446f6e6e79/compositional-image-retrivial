"""Generic retrieval evaluation loop.

`evaluate(...)` is the shared scaffold used by the baseline. Each method-specific
evaluator (logit composition, prompt ensemble, CoOp) lives next to its method
because they don't fit the (fuse text + source image) interface this loop
assumes.
"""

from __future__ import annotations

from typing import Callable

import torch

from .annotations import (
    get_ground_truth_indices,
    get_source_image_idxs,
    get_text_query,
)
from .metrics import evaluate_retrieval


def evaluate(
    fusion_mechanism: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    query_embedding_function: Callable,
    annotations: list[dict],
    model,
    processor,
    device,
    embeddings: torch.Tensor,
) -> list[dict]:
    """Evaluate a (text + source image) fusion retrieval method on the benchmark.

    Args:
        fusion_mechanism: maps (text_emb, image_emb) -> a single query embedding.
        query_embedding_function: maps (text_query, model, processor, device) ->
            a tensor or list of tensors that `fusion_mechanism` accepts.
        annotations: benchmark annotations.
        model, processor, device: CLIP model state.
        embeddings: (N, D) pre-extracted, L2-normalized image embeddings for the
            full retrieval pool.

    Returns:
        Per-query nested dict: {source_image_idx: {k: {Recall@k, Precision@k}}}.
    """
    evaluation_results = []

    for i, annotation in enumerate(annotations):
        print(f"Evaluating query Q{i+1}: {annotation.get('query', '')}")

        text_query = get_text_query(annotation)
        embedded_text_query = query_embedding_function(text_query, model, processor, device)
        query_image_idxs = get_source_image_idxs(annotation)

        query_evaluation_results: dict = {}
        for query_image_idx in query_image_idxs:
            embedded_query_image = embeddings[query_image_idx]
            fused_embedding = fusion_mechanism(embedded_text_query, embedded_query_image)
            fused_embedding = torch.nn.functional.normalize(fused_embedding, dim=0)

            similarities = torch.matmul(embeddings, fused_embedding)

            # Top-10 most similar, excluding the source image itself.
            _, topk_indices = torch.topk(similarities, k=11)
            topk_indices = topk_indices[topk_indices != query_image_idx][:10]
            retrieved_indices = topk_indices.tolist()

            image_query_eval_metrics: dict = {}
            for k in [1, 5, 10]:
                image_query_eval_metrics[k] = evaluate_retrieval(
                    retrieved_indices=retrieved_indices,
                    ground_truth_indices=get_ground_truth_indices(annotation, query_image_idx),
                    k=k,
                )
            query_evaluation_results[query_image_idx] = image_query_eval_metrics

        evaluation_results.append(query_evaluation_results)
    return evaluation_results
