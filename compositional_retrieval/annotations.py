"""Benchmark annotation utilities — load the evaluation JSON and pull fields out."""

from __future__ import annotations

import json
from pathlib import Path


def load_annotations(path: str | Path | None = None) -> list[dict]:
    """Load the benchmark annotations JSON file."""
    if path is None:
        from .config import ANNOTATIONS_PATH
        path = ANNOTATIONS_PATH
    with open(path, "r") as f:
        return json.load(f)


def get_text_query(annotation: dict) -> str:
    """Extract the signed text query (e.g. '+Bald, -Eyeglasses') from an annotation."""
    return annotation.get("query", "")


def get_source_image_idxs(annotation: dict) -> list[int]:
    """Source image IDs for an annotation. JSON keys are strings, so convert to int."""
    return [int(key) for key in annotation.get("ground_truth", {}).keys()]


def get_ground_truth_indices(annotation: dict, source_image_idx: int) -> list[int]:
    """Valid target IDs for a given (annotation, source image) pair."""
    return annotation.get("ground_truth", {}).get(str(source_image_idx), [])
