"""Retrieval metrics — per-query and per-image."""

from __future__ import annotations

import numpy as np


def evaluate_retrieval(
    retrieved_indices: list[int],
    ground_truth_indices: list[int],
    k: int,
) -> dict:
    """Recall@K (hit rate) and Precision@K for a single source-image retrieval.

    `retrieved_indices` must be ordered by similarity in descending order.
    """
    top_k_retrieved = retrieved_indices[:k]
    hits = set(top_k_retrieved).intersection(set(ground_truth_indices))
    num_hits = len(hits)

    recall_at_k = 1 if num_hits > 0 else 0
    precision_at_k = num_hits / k

    return {
        f"Recall@{k}": recall_at_k,
        f"Precision@{k}": precision_at_k,
    }


def compute_query_average_results(query_evaluation_results: dict) -> dict:
    """Average Recall@K / Precision@K across all source images of a single query.

    Also returns 95% normal-approximation confidence intervals.
    """
    average_results: dict = {}
    for k in [1, 5, 10]:
        recall_sum = 0
        precision_sum = 0
        num_images = len(query_evaluation_results)

        for _, eval_metrics_per_k in query_evaluation_results.items():
            recall_sum += eval_metrics_per_k[k][f"Recall@{k}"]
            precision_sum += eval_metrics_per_k[k][f"Precision@{k}"]

        average_results[f"Recall@{k}"] = recall_sum / num_images
        average_results[f"Precision@{k}"] = precision_sum / num_images

        recall_std_error = np.sqrt(
            (average_results[f"Recall@{k}"] * (1 - average_results[f"Recall@{k}"])) / num_images
        )
        precision_std_error = np.sqrt(
            (average_results[f"Precision@{k}"] * (1 - average_results[f"Precision@{k}"])) / num_images
        )
        average_results[f"Recall@{k}_CI"] = 1.96 * recall_std_error
        average_results[f"Precision@{k}_CI"] = 1.96 * precision_std_error

    return average_results
