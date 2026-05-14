"""Method 2 — per-attribute logit composition (training-free).

For each candidate image, the score is sum_i sign_i * cos(img, attr_text_emb[i]),
where sign_i comes from the signed query (+attr / -attr). The source image
embedding is intentionally not used: this is a pure text-driven retrieval.
"""

from __future__ import annotations

import torch

from ..annotations import (
    get_ground_truth_indices,
    get_source_image_idxs,
    get_text_query,
)
from ..encoding import encode_attribute_prompt
from ..metrics import evaluate_retrieval


@torch.no_grad()
def precompute_attribute_text_embeddings(
    attr_names: list[str],
    model,
    processor,
    device,
) -> torch.Tensor:
    """Return a (n_attrs, D) L2-normalized matrix of CLIP text embeddings, one row per attribute."""
    embs = torch.stack(
        [encode_attribute_prompt(name, model, processor, device) for name in attr_names], dim=0
    )
    embs = embs / embs.norm(dim=-1, keepdim=True)
    return embs


def parse_signed_query(
    text_query: str,
    attribute2idx: dict[str, int],
    n_attrs: int,
    device,
) -> torch.Tensor:
    """Parse '+Bald, +Smiling, -Eyeglasses' into a signed weight vector of length n_attrs.

    Each entry is +1, -1, or 0.
    """
    w = torch.zeros(n_attrs, device=device)
    for component in text_query.split(","):
        component = component.strip()
        if not component:
            continue
        sign_char, attr_name = component[0], component[1:].strip()
        if attr_name not in attribute2idx:
            raise KeyError(f"Unknown attribute '{attr_name}' in query '{text_query}'")
        idx = attribute2idx[attr_name]
        if idx >= n_attrs:
            continue
        sign = 1.0 if sign_char == "+" else -1.0 if sign_char == "-" else 0.0
        w[idx] = sign
    return w


def evaluate_logit_composition(
    annotations: list[dict],
    embeddings: torch.Tensor,
    attr_text_embs: torch.Tensor,
    attribute2idx: dict[str, int],
) -> list[dict]:
    """Score every candidate image by sum_i sign_i * cos(img, attr_text_embs[i])."""
    n_attrs = attr_text_embs.shape[0]
    # (N_imgs, n_attrs) attribute logit matrix; cos(img, attr) since both are L2-normalized.
    img_attr_logits = embeddings @ attr_text_embs.T

    evaluation_results = []
    for i, annotation in enumerate(annotations):
        print(f"Evaluating query Q{i+1}: {annotation.get('query', '')}")
        w = parse_signed_query(get_text_query(annotation), attribute2idx, n_attrs, embeddings.device)
        scores = img_attr_logits @ w   # (N,)

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
