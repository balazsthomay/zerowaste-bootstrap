"""Confidence filtering for pseudo-labels."""

import json
import logging
from collections import Counter
from pathlib import Path

from zerowaste_bootstrap.config import ZEROWASTE_CLASSES

logger = logging.getLogger(__name__)


def filter_pseudo_labels(
    raw_json: Path,
    output_json: Path,
    confidence_threshold: float = 0.7,
    min_mask_area: int = 100,
) -> Path:
    """Filter pseudo-labels by confidence score and mask area.

    Args:
        raw_json: Path to raw COCO JSON with scores.
        output_json: Path to save filtered COCO JSON.
        confidence_threshold: Minimum confidence score to keep.
        min_mask_area: Minimum mask area in pixels.

    Returns:
        Path to the filtered JSON file.
    """
    with open(raw_json) as f:
        data = json.load(f)

    total = len(data["annotations"])
    kept = []
    removed_conf = 0
    removed_area = 0
    class_counts: Counter[int] = Counter()

    for ann in data["annotations"]:
        score = ann.get("score", 1.0)
        area = ann.get("area", 0)

        if score < confidence_threshold:
            removed_conf += 1
            continue
        if area < min_mask_area:
            removed_area += 1
            continue

        kept.append(ann)
        class_counts[ann["category_id"]] += 1

    # Remap annotation IDs to be contiguous
    for i, ann in enumerate(kept):
        ann["id"] = i

    # Keep only images that have at least one annotation
    kept_image_ids = {ann["image_id"] for ann in kept}
    kept_images = [img for img in data["images"] if img["id"] in kept_image_ids]

    filtered = {
        "images": kept_images,
        "annotations": kept,
        "categories": data["categories"],
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(filtered, f)

    # Log stats
    logger.info("Filtering pseudo-labels: threshold=%.2f, min_area=%d", confidence_threshold, min_mask_area)
    logger.info("  Total: %d → Kept: %d (%.1f%%)", total, len(kept), 100 * len(kept) / max(total, 1))
    logger.info("  Removed (low confidence): %d", removed_conf)
    logger.info("  Removed (small area): %d", removed_area)
    logger.info("  Images with annotations: %d / %d", len(kept_images), len(data["images"]))

    for cat_id, count in sorted(class_counts.items()):
        cat_name = ZEROWASTE_CLASSES.get(cat_id, f"class_{cat_id}")
        logger.info("  %s: %d instances", cat_name, count)

    return output_json


def analyze_pseudo_labels(json_path: Path) -> dict:
    """Analyze pseudo-label statistics.

    Returns:
        Dictionary with class distribution and score statistics.
    """
    with open(json_path) as f:
        data = json.load(f)

    annotations = data["annotations"]
    if not annotations:
        return {"total": 0, "class_distribution": {}, "score_stats": {}}

    class_counts: Counter[int] = Counter()
    scores: list[float] = []

    for ann in annotations:
        class_counts[ann["category_id"]] += 1
        if "score" in ann:
            scores.append(ann["score"])

    class_distribution = {}
    for cat_id, count in sorted(class_counts.items()):
        cat_name = ZEROWASTE_CLASSES.get(cat_id, f"class_{cat_id}")
        class_distribution[cat_name] = count

    score_stats = {}
    if scores:
        import numpy as np
        score_arr = np.array(scores)
        score_stats = {
            "mean": float(np.mean(score_arr)),
            "std": float(np.std(score_arr)),
            "min": float(np.min(score_arr)),
            "max": float(np.max(score_arr)),
            "median": float(np.median(score_arr)),
        }

    return {
        "total": len(annotations),
        "num_images": len(data["images"]),
        "class_distribution": class_distribution,
        "score_stats": score_stats,
    }
