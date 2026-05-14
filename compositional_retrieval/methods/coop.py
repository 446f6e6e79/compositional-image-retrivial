"""CoOp prompt learner (training-based).

For each CelebA attribute we build a positive and a negative prompt of the form
    [SOS] [ctx_1] ... [ctx_M] a person with/without {attr}. [EOS] [PAD] ...
The token-embedding lookup happens once at construction time; the forward pass
replaces positions [1:1+M] with `self.ctx` before running the (frozen) CLIP text
transformer.

CoOp does **not** route through `encode_texts()` — it operates one level below
the encoder API, which is why it tokenizes via `processor.tokenizer` directly
and reimplements the encoder forward in `_encode_text_with_ctx`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


class CoOpPromptLearner(nn.Module):
    """Shared-context CoOp on top of HuggingFace CLIP."""

    def __init__(
        self,
        clip_model: CLIPModel,
        processor: CLIPProcessor,
        attr_names: list[str],
        M: int = 16,
    ):
        super().__init__()
        self.clip = clip_model
        self.M = M
        self.attr_names = attr_names
        self.num_attrs = len(attr_names)

        token_emb_layer = clip_model.text_model.embeddings.token_embedding
        embedding_dim = token_emb_layer.embedding_dim

        # M learnable context vectors, standard CoOp Gaussian init.
        ctx = torch.empty(M, embedding_dim)
        nn.init.normal_(ctx, std=0.02)
        self.ctx = nn.Parameter(ctx)

        tokenizer = processor.tokenizer
        max_len = clip_model.config.text_config.max_position_embeddings   # 77 for ViT-B/32
        placeholder = " ".join(["X"] * M)

        readable = [name.replace("_", " ").lower() for name in attr_names]
        pos_prompts = [f"{placeholder} a person with {name}." for name in readable]
        neg_prompts = [f"{placeholder} a person without {name}." for name in readable]

        pos = tokenizer(pos_prompts, padding="max_length", max_length=max_len, truncation=True, return_tensors="pt")
        neg = tokenizer(neg_prompts, padding="max_length", max_length=max_len, truncation=True, return_tensors="pt")
        self.register_buffer("pos_ids", pos["input_ids"])
        self.register_buffer("neg_ids", neg["input_ids"])

        for p in self.clip.parameters():
            p.requires_grad_(False)

    def _encode_text_with_ctx(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized text features (N, D) for a batch of prompt token-id rows."""
        text_model = self.clip.text_model
        device_ = self.ctx.device
        input_ids = input_ids.to(device_)
        bsz, seq_len = input_ids.shape

        token_emb = text_model.embeddings.token_embedding(input_ids)
        token_emb = torch.cat(
            [
                token_emb[:, :1, :],
                self.ctx.unsqueeze(0).expand(bsz, -1, -1),
                token_emb[:, 1 + self.M:, :],
            ],
            dim=1,
        )

        position_ids = torch.arange(seq_len, device=device_).unsqueeze(0)
        position_emb = text_model.embeddings.position_embedding(position_ids)
        hidden_states = token_emb + position_emb

        causal_mask = torch.full((seq_len, seq_len), float("-inf"), device=device_)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(bsz, 1, seq_len, seq_len)

        encoder_outputs = text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=None,
            causal_attention_mask=causal_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        last_hidden = text_model.final_layer_norm(encoder_outputs.last_hidden_state)

        # Pool at the EOS token (the highest-id token in the row).
        eos_pos = input_ids.argmax(dim=-1)
        pooled = last_hidden[torch.arange(bsz, device=device_), eos_pos]
        text_features = self.clip.text_projection(pooled)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def forward(self):
        e_pos = self._encode_text_with_ctx(self.pos_ids)
        e_neg = self._encode_text_with_ctx(self.neg_ids)
        return e_pos, e_neg


@torch.no_grad()
def macro_auc(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Macro-averaged AUC over the 40 attributes, ignoring attributes with no positives or no negatives in the batch.

    Runs entirely on `logits.device`; only the final mean scalar is materialized on the host.
    """
    logits = logits.detach()
    targets = targets.detach()
    aucs: list[torch.Tensor] = []
    n = targets.shape[0]
    for j in range(targets.shape[1]):
        y = targets[:, j]
        n_pos = int(y.sum().item())
        if n_pos == 0 or n_pos == n:
            continue
        n_neg = n - n_pos
        # Mann-Whitney AUC via ranks (no sklearn dependency).
        order = torch.argsort(logits[:, j])
        ranks = torch.empty_like(order, dtype=torch.float32)
        ranks[order] = torch.arange(n, device=order.device, dtype=torch.float32)
        pos_ranks = ranks[y == 1].sum()
        aucs.append((pos_ranks - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))
    if not aucs:
        return float("nan")
    return float(torch.stack(aucs).mean())


def _select_few_shot_subset(
    train_labels: torch.Tensor,
    k_shot: int = 16,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """16-shot per attribute (each polarity) from the training labels, deduplicated.

    Returns a 1-D tensor of unique sorted indices on `device`.
    """
    labels = train_labels.to(device) if isinstance(train_labels, torch.Tensor) else torch.as_tensor(train_labels, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    chosen = torch.zeros(labels.shape[0], dtype=torch.bool, device=device)
    for j in range(labels.shape[1]):
        for value in (1, 0):
            idxs = (labels[:, j] == value).nonzero(as_tuple=True)[0]
            if idxs.numel() == 0:
                continue
            k = min(k_shot, int(idxs.numel()))
            pick = idxs[torch.randperm(idxs.numel(), generator=generator, device=device)[:k]]
            chosen[pick] = True
    return chosen.nonzero(as_tuple=True)[0]


def train_coop(
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    attr_names: list[str],
    device: str,
    config: str = "colab",
    M: int = 16,
    val_fraction: float = 0.05,
    save_path: str | Path | None = None,
) -> tuple[CoOpPromptLearner, float, Path | None]:
    """Train the CoOp context vectors. Returns (learner with best ctx restored, best_val_macro_auc, save_path)."""
    torch.manual_seed(42)
    np.random.seed(42)

    train_features = train_features.to(device)
    train_labels_dev = (
        train_labels.to(device)
        if isinstance(train_labels, torch.Tensor)
        else torch.as_tensor(np.asarray(train_labels), device=device)
    )

    if config == "colab":
        selected = _select_few_shot_subset(train_labels_dev, k_shot=16, seed=42, device=device)
        train_feats_sel = train_features[selected]
        train_labels_sel = train_labels_dev[selected]
        epochs, batch, lr = 30, 64, 2e-3
        use_amp = False
    elif config == "vm":
        train_feats_sel = train_features
        train_labels_sel = train_labels_dev
        epochs, batch, lr = 10, 256, 2e-3
        use_amp = (str(device) == "cuda")
    else:
        raise ValueError(f"Unknown CoOp config: {config}")

    N = train_feats_sel.shape[0]
    perm = torch.randperm(N, device=device)
    val_size = max(1, int(round(val_fraction * N)))
    val_idx = perm[:val_size]
    trn_idx = perm[val_size:]
    X_tr = train_feats_sel[trn_idx]
    Y_tr = train_labels_sel[trn_idx].float()
    X_va = train_feats_sel[val_idx]
    Y_va = train_labels_sel[val_idx].float()

    print(f"CoOp config: {config}  |  train: {tuple(X_tr.shape)}  val: {tuple(X_va.shape)}")
    print(f"epochs={epochs} batch={batch} lr={lr} M={M} amp={use_amp}")

    learner = CoOpPromptLearner(clip_model, processor, attr_names, M=M).to(device)
    optimizer = torch.optim.SGD([learner.ctx], lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    logit_scale = clip_model.logit_scale.exp().detach()

    best_val = -1.0
    best_ctx = learner.ctx.detach().clone()
    n_train = X_tr.shape[0]

    print("Starting CoOp training...")
    for epoch in range(epochs):
        learner.train()
        perm_ = torch.randperm(n_train, device=device)
        total_loss = 0.0
        for start in range(0, n_train, batch):
            idx = perm_[start:start + batch]
            x = X_tr[idx]
            y = Y_tr[idx]
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                e_pos, e_neg = learner()
                logits = logit_scale * (x @ e_pos.T - x @ e_neg.T)
                loss = F.binary_cross_entropy_with_logits(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * x.shape[0]

        scheduler.step()

        learner.eval()
        with torch.no_grad():
            e_pos, e_neg = learner()
            val_logits = logit_scale * (X_va @ e_pos.T - X_va @ e_neg.T)
            val_auc = macro_auc(val_logits, Y_va)

        avg_loss = total_loss / max(1, n_train)
        print(f"Epoch {epoch+1:3d}/{epochs}  loss={avg_loss:.4f}  val_macroAUC={val_auc:.4f}")

        if val_auc > best_val:
            best_val = val_auc
            best_ctx = learner.ctx.detach().clone()

    learner.ctx.data.copy_(best_ctx)
    print(f"Best validation macro-AUC: {best_val:.4f}")

    save_path_out: Path | None = None
    if save_path is not None:
        save_path_out = Path(save_path)
        save_path_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"ctx": best_ctx.cpu(), "M": M, "config": config, "val_macro_auc": best_val},
            save_path_out,
        )
        print(f"Saved context to {save_path_out}")

    return learner, best_val, save_path_out
