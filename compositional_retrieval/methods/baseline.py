"""Baseline method: text + image embedding sum, normalized."""

from __future__ import annotations

import torch

from ..encoding import encode_text


def baseline_fusion(
    normalized_text_embedding,
    normalized_image_embedding: torch.Tensor,
    ALPHA: float = 1.0,
    BETA: float = 1.0,
) -> torch.Tensor:
    """Weighted sum of (list of) text embeddings and a single image embedding."""
    if isinstance(normalized_text_embedding, list):
        normalized_text_embedding = torch.stack(
            [embedding.view(-1) for embedding in normalized_text_embedding]
        ).sum(dim=0)
    else:
        normalized_text_embedding = normalized_text_embedding.view(-1)

    return ALPHA * normalized_text_embedding + BETA * normalized_image_embedding.view(-1)


def baseline_embed_query(text_query: str, model, processor, device) -> list[torch.Tensor]:
    """Encode the full text query as a single normalized 1D tensor, wrapped in a list."""
    return [encode_text(text_query, model, processor, device)]
