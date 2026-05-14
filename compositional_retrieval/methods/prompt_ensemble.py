"""Method 3 — prompt ensembling v2 (training-free).

We build separate positive and negative text embeddings for each attribute by
mean-pooling (phrase x template) prompts and then re-normalizing. Per-attribute
scoring at evaluation time:

    + attr  contributes  cos(img, E_pos[attr])
    - attr  contributes  cos(img, E_neg[attr])
"""

from __future__ import annotations

import torch

from ..annotations import (
    get_ground_truth_indices,
    get_source_image_idxs,
    get_text_query,
)
from ..encoding import encode_texts
from ..metrics import evaluate_retrieval


@torch.no_grad()
def _encode_phrases_through_templates(
    phrases: list[str],
    templates: list[str],
    model,
    processor,
    device,
) -> torch.Tensor:
    """Encode every (phrase x template) pair, L2-normalize each, mean-pool, re-normalize."""
    prompts = [template.format(phrase=phrase) for phrase in phrases for template in templates]
    embs = encode_texts(prompts, model, processor, device)   # (P, D), per-row normalized
    mean_emb = embs.mean(dim=0)
    return mean_emb / mean_emb.norm()


@torch.no_grad()
def precompute_attribute_pos_neg_embeddings(
    attr_names: list[str],
    humanized_mappings_pos: dict[str, list[str]],
    humanized_mappings_neg: dict[str, list[str]],
    templates: list[str],
    model,
    processor,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (E_pos, E_neg), each (n_attrs, D) and L2-normalized."""
    pos_embs, neg_embs = [], []
    for name in attr_names:
        pos_embs.append(_encode_phrases_through_templates(humanized_mappings_pos[name], templates, model, processor, device))
        neg_embs.append(_encode_phrases_through_templates(humanized_mappings_neg[name], templates, model, processor, device))
    return torch.stack(pos_embs, dim=0), torch.stack(neg_embs, dim=0)


def evaluate_pos_neg_logit_composition(
    annotations: list[dict],
    embeddings: torch.Tensor,
    E_pos: torch.Tensor,
    E_neg: torch.Tensor,
    attr_names: list[str],
    attribute2idx: dict[str, int],
) -> list[dict]:
    """Per-attribute logit composition with separate +/- text embeddings."""
    img_pos = embeddings @ E_pos.T   # (N, n_attrs)
    img_neg = embeddings @ E_neg.T   # (N, n_attrs)

    evaluation_results = []
    for i, annotation in enumerate(annotations):
        print(f"Evaluating query Q{i+1}: {annotation.get('query', '')}")
        pos_mask = torch.zeros(len(attr_names), device=embeddings.device)
        neg_mask = torch.zeros(len(attr_names), device=embeddings.device)
        for component in get_text_query(annotation).split(","):
            component = component.strip()
            if not component:
                continue
            sign_char, attr_name = component[0], component[1:].strip()
            j = attribute2idx[attr_name]
            if sign_char == "+":
                pos_mask[j] = 1.0
            elif sign_char == "-":
                neg_mask[j] = 1.0

        scores = (img_pos @ pos_mask) + (img_neg @ neg_mask)

        query_image_idxs = get_source_image_idxs(annotation)
        query_evaluation_results: dict = {}
        for query_image_idx in query_image_idxs:
            _, topk_indices = torch.topk(scores, k=11)
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
