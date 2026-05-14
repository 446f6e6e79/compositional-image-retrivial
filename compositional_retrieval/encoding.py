"""CLIP encoding utilities — text and image embeddings, L2-normalized per row.

All functions take `model`, `processor`, and `device` explicitly so they can be
reused across modules without relying on any module-level state.
"""

from __future__ import annotations

import torch


def _as_feature_tensor(out) -> torch.Tensor:
    """Normalize CLIPModel.get_text_features / get_image_features outputs into a Tensor.

    Different transformers versions return either a plain Tensor or a
    BaseModelOutputWithPooling-style object exposing .text_embeds / .image_embeds /
    .pooler_output.
    """
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("text_embeds", "image_embeds", "pooler_output"):
        v = getattr(out, attr, None)
        if v is not None:
            return v
    if isinstance(out, tuple) and len(out) > 0:
        return out[0]
    raise TypeError(f"Unexpected feature output type: {type(out)}")


@torch.no_grad()
def encode_texts(prompts: list[str], model, processor, device) -> torch.Tensor:
    """Encode a batch of prompts in one call. Returns (P, D), L2-normalized per row, on `device`."""
    inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    embs = _as_feature_tensor(model.get_text_features(**inputs))   # (P, D)
    return embs / embs.norm(dim=-1, keepdim=True)


@torch.no_grad()
def encode_text(prompt: str, model, processor, device) -> torch.Tensor:
    """Tokenize, encode, and L2-normalize a single text prompt. Returns shape (D,) on `device`."""
    return encode_texts([prompt], model, processor, device).view(-1)


@torch.no_grad()
def encode_images(images, model, processor, device, batch_size: int = 128, to_cpu: bool = True) -> torch.Tensor:
    """Encode an iterable of PIL images. Returns (N, D), L2-normalized per row.

    Streams in `batch_size` chunks so callers don't have to hand-roll batching.
    By default each batch is moved to CPU before being concatenated, which keeps
    GPU memory bounded when encoding large datasets. Pass `to_cpu=False` to keep
    the result on `device` (useful for single-image / small-batch calls).
    """
    feats: list[torch.Tensor] = []
    buf: list = []

    def _flush():
        inputs = processor(images=buf, return_tensors="pt").to(device)
        e = _as_feature_tensor(model.get_image_features(**inputs))
        e = e / e.norm(dim=-1, keepdim=True)
        feats.append(e.cpu() if to_cpu else e)
        buf.clear()

    total = len(images) if hasattr(images, "__len__") else None
    for img in images:
        buf.append(img)
        if len(buf) >= batch_size:
            if total is not None:
                print(f"Encoding batch of {len(buf)} images...")
            _flush()
            if total is not None:
                print(f"Encoded {len(feats) * batch_size}  / {total} images...")
    if buf:
        _flush()
    return torch.cat(feats, dim=0) if feats else torch.empty(0)


@torch.no_grad()
def encode_image(image, model, processor, device) -> torch.Tensor:
    """Encode and L2-normalize a single PIL image. Returns shape (D,) on `device`."""
    return encode_images([image], model, processor, device, to_cpu=False).view(-1)


def encode_attribute_prompt(attr_name: str, model, processor, device) -> torch.Tensor:
    """Encode 'A picture of a person with {attr}' for one CelebA attribute name."""
    readable = attr_name.replace("_", " ").lower()
    return encode_text(f"A picture of a person with {readable}", model, processor, device)
