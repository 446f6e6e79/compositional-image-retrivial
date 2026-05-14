"""Load all method results from cache/ and plot the final comparison figure."""

from __future__ import annotations

import argparse

import torch

import _bootstrap  # noqa: F401

from compositional_retrieval.config import COOP_CONFIG, results_path
from compositional_retrieval.visualization import plot_methods_comparison


def _load(method: str):
    path = results_path(method)
    if not path.exists():
        print(f"WARNING: {path} not found; skipping '{method}'")
        return None
    return torch.load(path)["per_query"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coop-config", default=COOP_CONFIG, help="CoOp config to load (default from env)")
    args = ap.parse_args()

    methods = {
        "Baseline": _load("baseline"),
        "Method 2 — Logit Compose": _load("logit_composition"),
        "Method 3 — Prompt Ens. v2": _load("prompt_ensemble"),
        f"CoOp ({args.coop_config}, M=16)": _load(f"coop_{args.coop_config}"),
    }
    methods = {k: v for k, v in methods.items() if v is not None}
    if not methods:
        raise SystemExit("No method results found in cache/. Run the run_*.py scripts first.")

    plot_methods_comparison(
        methods,
        title="Final Method Comparison — per-query Recall@K and Precision@K",
    )


if __name__ == "__main__":
    main()
