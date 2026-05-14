"""Train CoOp and evaluate it on the benchmark via the prompt-ensemble evaluator."""

from __future__ import annotations

import argparse

import torch

import _bootstrap  # noqa: F401

from compositional_retrieval.annotations import load_annotations
from compositional_retrieval.config import (
    DATA_ROOT,
    coop_ctx_path,
    embeddings_path,
    results_path,
)
from compositional_retrieval.data import build_attribute_indices, load_celeba
from compositional_retrieval.features import load_clip, load_or_extract_embeddings
from compositional_retrieval.methods.coop import train_coop
from compositional_retrieval.methods.prompt_ensemble import evaluate_pos_neg_logit_composition
from compositional_retrieval.metrics import compute_query_average_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["colab", "vm"], default="colab")
    ap.add_argument("--M", type=int, default=16)
    args = ap.parse_args()

    model, processor, device = load_clip()

    # Test split — retrieval pool
    celeba_test = load_celeba(split="test", root=DATA_ROOT)
    _idx2attribute, attribute2idx = build_attribute_indices(celeba_test)
    attr_names = list(celeba_test.attr_names)
    embeddings, _labels = load_or_extract_embeddings(
        celeba_test, model, processor, device, embeddings_path("test")
    )

    # Train split — for CoOp
    celeba_train = load_celeba(split="train", root=DATA_ROOT)
    train_features, train_labels = load_or_extract_embeddings(
        celeba_train, model, processor, device, embeddings_path("train")
    )

    coop, best_val, save_path = train_coop(
        clip_model=model,
        processor=processor,
        train_features=train_features,
        train_labels=train_labels,
        attr_names=attr_names,
        device=device,
        config=args.config,
        M=args.M,
        save_path=coop_ctx_path(args.config),
    )
    print(f"Saved trained context to: {save_path}  (val_macroAUC={best_val:.4f})")

    # Retrieval evaluation with the learned context
    coop.eval()
    with torch.no_grad():
        E_pos_coop, E_neg_coop = coop()
    E_pos_coop = E_pos_coop.to(embeddings.device)
    E_neg_coop = E_neg_coop.to(embeddings.device)

    annotations = load_annotations()
    evaluation_results = evaluate_pos_neg_logit_composition(
        annotations=annotations,
        embeddings=embeddings,
        E_pos=E_pos_coop,
        E_neg=E_neg_coop,
        attr_names=attr_names,
        attribute2idx=attribute2idx,
    )
    average_results_per_query = [
        compute_query_average_results(q) for q in evaluation_results
    ]

    out = results_path(f"coop_{args.config}")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "per_query": average_results_per_query,
            "raw": evaluation_results,
            "config": args.config,
            "val_macro_auc": best_val,
        },
        out,
    )
    print(f"Saved CoOp results to {out}")


if __name__ == "__main__":
    main()
