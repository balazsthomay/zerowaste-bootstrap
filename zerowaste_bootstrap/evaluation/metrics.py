"""COCO-style evaluation metrics for instance segmentation."""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_util
from tqdm import tqdm

from zerowaste_bootstrap.config import ZEROWASTE_CLASSES

logger = logging.getLogger(__name__)


def _masks_to_rle(masks: np.ndarray) -> list[dict]:
    """Convert binary masks (N, H, W) to COCO RLE format."""
    rles = []
    for mask in masks:
        # Ensure Fortran order for pycocotools
        rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("utf-8")
        rles.append(rle)
    return rles


def evaluate_model(
    model: torch.nn.Module,
    processor,
    dataset,
    device: str = "cpu",
) -> dict:
    """Evaluate a model on a dataset using COCO metrics.

    Args:
        model: Trained Mask2Former model.
        processor: Mask2FormerImageProcessor.
        dataset: Dataset with COCO annotations.
        device: Device to run inference on.

    Returns:
        Dictionary with AP, AP50, AP75, per-class AP, AR metrics.
    """
    model.eval()
    model.to(device)

    coco_gt = dataset.coco
    results = []

    for idx in tqdm(range(len(dataset)), desc="Evaluating"):
        img_id = dataset.image_ids[idx]

        # Get raw image for inference
        image = dataset.load_image(idx)
        inputs = processor(images=[image], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process predictions
        target_size = (image.height, image.width)
        pred = processor.post_process_instance_segmentation(
            outputs,
            target_sizes=[target_size],
            threshold=0.5,
        )[0]

        segments = pred["segments_info"]
        if len(segments) == 0:
            continue

        # Convert each segment to COCO result format
        pred_masks = pred["segmentation"]
        for seg_info in segments:
            seg_id = seg_info["id"]
            category_id = seg_info["label_id"]
            score = seg_info["score"]

            # Extract binary mask for this segment
            binary_mask = (pred_masks == seg_id).cpu().numpy().astype(np.uint8)
            rle = mask_util.encode(np.asfortranarray(binary_mask))
            rle["counts"] = rle["counts"].decode("utf-8")

            results.append({
                "image_id": img_id,
                "category_id": category_id,
                "segmentation": rle,
                "score": float(score),
            })

    if not results:
        logger.warning("No predictions generated")
        return _empty_metrics()

    # Run COCO evaluation
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metrics = {
        "AP": float(coco_eval.stats[0]),
        "AP50": float(coco_eval.stats[1]),
        "AP75": float(coco_eval.stats[2]),
        "AR": float(coco_eval.stats[8]),
    }

    # Per-class AP
    for cat_id, cat_name in ZEROWASTE_CLASSES.items():
        coco_eval_cls = COCOeval(coco_gt, coco_dt, "segm")
        coco_eval_cls.params.catIds = [cat_id]
        coco_eval_cls.evaluate()
        coco_eval_cls.accumulate()
        coco_eval_cls.summarize()
        metrics[f"AP_{cat_name}"] = float(coco_eval_cls.stats[0])

    return metrics


def _empty_metrics() -> dict:
    """Return zero metrics when no predictions exist."""
    metrics = {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR": 0.0}
    for cat_name in ZEROWASTE_CLASSES.values():
        metrics[f"AP_{cat_name}"] = 0.0
    return metrics


def evaluate_from_results(
    gt_json: Path,
    results_json: Path,
) -> dict:
    """Evaluate from pre-computed COCO result JSON files."""
    coco_gt = COCO(str(gt_json))

    with open(results_json) as f:
        results = json.load(f)

    if not results:
        return _empty_metrics()

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metrics = {
        "AP": float(coco_eval.stats[0]),
        "AP50": float(coco_eval.stats[1]),
        "AP75": float(coco_eval.stats[2]),
        "AR": float(coco_eval.stats[8]),
    }

    for cat_id, cat_name in ZEROWASTE_CLASSES.items():
        coco_eval_cls = COCOeval(coco_gt, coco_dt, "segm")
        coco_eval_cls.params.catIds = [cat_id]
        coco_eval_cls.evaluate()
        coco_eval_cls.accumulate()
        coco_eval_cls.summarize()
        metrics[f"AP_{cat_name}"] = float(coco_eval_cls.stats[0])

    return metrics


def evaluate_model_cli(
    checkpoint: Path,
    data_dir: Path,
    split: str,
    output_dir: Path,
    device: str = "auto",
    visualize: bool = False,
    num_vis: int = 10,
) -> dict:
    """CLI entry point for evaluation."""
    from zerowaste_bootstrap.modeling.model import get_device, load_processor
    from transformers import Mask2FormerForUniversalSegmentation
    from zerowaste_bootstrap.data.dataset import ZeroWasteDataset

    device = get_device(device)
    processor = load_processor()
    model = Mask2FormerForUniversalSegmentation.from_pretrained(str(checkpoint))

    split_dir = data_dir / "zerowaste-f" / split
    dataset = ZeroWasteDataset(root_dir=split_dir, processor=processor)

    metrics = evaluate_model(model, processor, dataset, device=device)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"metrics_{split}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Metrics: %s", json.dumps(metrics, indent=2))

    if visualize:
        from zerowaste_bootstrap.evaluation.visualize import visualize_model_predictions

        visualize_model_predictions(
            model=model,
            processor=processor,
            dataset=dataset,
            device=device,
            output_dir=output_dir / "vis",
            num_images=num_vis,
        )

    return metrics
