"""All matplotlib helpers — image grids, per-query metric bars, method comparisons, heatmap."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from .encoding import encode_attribute_prompt


def plot_images(
    celeba_dataset,
    indices: list[int],
    n_cols: int,
    n_rows: int,
    figsize: tuple[int, int] = (20, 10),
):
    """Grid of images from a dataset."""
    if len(indices) > n_cols * n_rows:
        raise ValueError("Number of indices exceeds the grid capacity")

    _, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    for counter, img_idx in enumerate(indices):
        img, _ = celeba_dataset[img_idx]
        if n_rows == 1:
            ax = axes[counter % n_cols]
        else:
            ax = axes[counter // n_cols, counter % n_cols]
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_image_with_attributes(
    celeba_dataset,
    idx2attribute: dict[int, str],
    idx: int,
    figsize: tuple[int, int] = (10, 5),
):
    """Single image with its active attribute names listed next to it."""
    img, labels = celeba_dataset[idx]
    active_attrs = [idx2attribute[i] for i, value in enumerate(labels) if value == 1]

    fig, (ax_img, ax_text) = plt.subplots(1, 2, figsize=figsize)
    ax_img.imshow(img)
    ax_img.axis("off")
    ax_text.axis("off")
    ax_text.text(0.5, 0.5, "\n".join(active_attrs), fontsize=10, ha="center", va="center")
    plt.tight_layout()
    plt.show()


def plot_metrics_across_k(
    average_results_per_query: list[dict],
    title: str = "Retrieval Metrics across K",
):
    """Per-query grouped bars for Recall@K and Precision@K (K = 1, 5, 10) with 95% CIs."""
    k_values = [1, 5, 10]
    n_queries = len(average_results_per_query)
    x = np.arange(n_queries)
    width = 0.25
    offsets = [-width, 0.0, width]
    colors = [plt.cm.tab10(i) for i in range(len(k_values))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title)

    for k, offset, color in zip(k_values, offsets, colors):
        recall_means = [q[f"Recall@{k}"] for q in average_results_per_query]
        recall_cis = [q[f"Recall@{k}_CI"] for q in average_results_per_query]
        precision_means = [q[f"Precision@{k}"] for q in average_results_per_query]
        precision_cis = [q[f"Precision@{k}_CI"] for q in average_results_per_query]

        ax1.bar(x + offset, recall_means, width, yerr=recall_cis, capsize=4, ecolor="black", color=color, label=f"K={k}")
        ax2.bar(x + offset, precision_means, width, yerr=precision_cis, capsize=4, ecolor="black", color=color, label=f"K={k}")

    for ax, metric in [(ax1, "Recall"), (ax2, "Precision")]:
        ax.set_xlabel("Query")
        ax.set_ylabel(f"{metric}@K")
        ax.set_title(f"{metric}@K per query")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Q{i+1}" for i in range(n_queries)])
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(title="K")

    plt.tight_layout()
    plt.show()


def plot_methods_comparison(
    method_results: dict[str, list[dict]],
    title: str = "Method Comparison across queries",
):
    """2 x 3 grid (rows: Recall, Precision; cols: K) of per-query method comparisons."""
    k_values = [1, 5, 10]
    method_names = list(method_results.keys())
    n_methods = len(method_names)
    if n_methods == 0:
        raise ValueError("method_results must contain at least one method.")

    n_queries = len(next(iter(method_results.values())))
    x = np.arange(n_queries)
    colors = [plt.cm.tab10(i % 10) for i in range(n_methods)]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    fig.suptitle(title)

    for row_idx, metric in enumerate(["Recall", "Precision"]):
        for col_idx, k in enumerate(k_values):
            ax = axes[row_idx, col_idx]
            for method, color in zip(method_names, colors):
                ys = [q[f"{metric}@{k}"] for q in method_results[method]]
                ax.plot(x, ys, marker="o", color=color, label=method)
            ax.set_title(f"{metric}@{k}")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(x)
            ax.set_xticklabels([f"Q{i+1}" for i in range(n_queries)], rotation=45, ha="right")

    for ax in axes[:, 0]:
        ax.set_ylabel("Score")
    for ax in axes[-1, :]:
        ax.set_xlabel("Query")

    axes[0, 0].legend(title="Method", loc="best")
    plt.tight_layout()
    plt.show()


def _select_pure_image_idx(
    attr_idx: int,
    all_labels: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """Pick one image where attr_{attr_idx} is positive and other positives are minimal."""
    pos_mask = all_labels[:, attr_idx] == 1
    candidates = np.where(pos_mask)[0]
    if len(candidates) == 0:
        return int(rng.integers(0, all_labels.shape[0]))
    other_counts = all_labels[candidates].sum(axis=1) - 1
    min_count = other_counts.min()
    purest = candidates[other_counts == min_count]
    return int(rng.choice(purest))


def plot_class_image_cosine_heatmap(
    embeddings: torch.Tensor,
    all_labels: np.ndarray,
    attr_names: list[str],
    model,
    processor,
    device,
    seed: int = 0,
):
    """Heatmap of CLIP cosine similarity between attribute text prompts and 'pure' images."""
    if all_labels.ndim != 2:
        raise RuntimeError(
            f"`all_labels` is not 2-D (got shape {all_labels.shape}). Pass a (N, n_attrs) numpy array."
        )
    n_attrs = all_labels.shape[1]
    attr_names = list(attr_names)[:n_attrs]
    assert len(attr_names) == n_attrs

    text_embs = torch.stack(
        [encode_attribute_prompt(name, model, processor, device) for name in attr_names], dim=0
    )

    rng = np.random.default_rng(seed=seed)
    selected_idxs = [_select_pure_image_idx(i, all_labels, rng) for i in range(n_attrs)]
    selected_img_embs = embeddings[selected_idxs].to(text_embs.device)

    cos_mat = (text_embs @ selected_img_embs.T).detach().cpu().numpy()

    diag = np.diag(cos_mat)
    off_diag_mean = (cos_mat.sum() - diag.sum()) / (cos_mat.size - cos_mat.shape[0])
    row_argmax = cos_mat.argmax(axis=1)
    diag_argmax_rate = float((row_argmax == np.arange(cos_mat.shape[0])).mean())
    print(f"Mean diagonal cosine:      {diag.mean():.4f}")
    print(f"Mean off-diagonal cosine:  {off_diag_mean:.4f}")
    print(f"Diagonal-argmax rate:      {diag_argmax_rate:.2%}")

    fig, ax = plt.subplots(figsize=(14, 14))
    im = ax.imshow(cos_mat, cmap="viridis", aspect="equal")
    ax.set_xticks(np.arange(n_attrs))
    ax.set_yticks(np.arange(n_attrs))
    ax.set_xticklabels(attr_names, rotation=90, fontsize=8)
    ax.set_yticklabels(attr_names, fontsize=8)
    ax.set_xlabel("Sampled image (chosen as 'pure' positive for this attribute)")
    ax.set_ylabel("Text prompt: 'A picture of a person with {attr}'")
    ax.set_title(f"CLIP cosine similarity: {n_attrs} attribute prompts × {n_attrs} sampled images")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine similarity")
    plt.tight_layout()
    plt.show()
    return cos_mat
