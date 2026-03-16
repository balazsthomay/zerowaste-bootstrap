"""Batch inference for pseudo-label generation."""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_util
from tqdm import tqdm

from zerowaste_bootstrap.config import ZEROWASTE_CLASSES
from zerowaste_bootstrap.modeling.model import get_device

logger = logging.getLogger(__name__)


def _find_images(image_dir: Path) -> list[Path]:
    """Find all image files in a directory."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = []
    for ext in extensions:
        images.extend(image_dir.glob(f"*{ext}"))
        images.extend(image_dir.glob(f"*{ext.upper()}"))
    return sorted(set(images))


def generate_pseudo_labels(
    model_path: Path,
    image_dir: Path,
    output_json: Path,
    device: str = "auto",
    batch_size: int = 4,
) -> Path:
    """Generate pseudo-labels on unlabeled images using a trained model.

    Args:
        model_path: Path to trained model checkpoint.
        image_dir: Directory containing unlabeled images.
        output_json: Path to save raw COCO JSON with all predictions.
        device: Device for inference.
        batch_size: Batch size for inference.

    Returns:
        Path to the output JSON file.
    """
    from transformers import Mask2FormerForUniversalSegmentation
    from zerowaste_bootstrap.modeling.model import load_processor

    device = get_device(device)
    logger.info("Loading model from %s", model_path)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(str(model_path))
    model.to(device)
    model.eval()

    processor = load_processor()

    # Find all images
    image_paths = _find_images(image_dir)
    logger.info("Found %d images in %s", len(image_paths), image_dir)

    # Check for resume (previously processed images)
    processed_images: set[str] = set()
    existing_results: dict = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": k, "name": v} for k, v in ZEROWASTE_CLASSES.items()
        ],
    }

    if output_json.exists():
        with open(output_json) as f:
            existing_results = json.load(f)
        processed_images = {img["file_name"] for img in existing_results["images"]}
        logger.info("Resuming: %d images already processed", len(processed_images))

    remaining = [p for p in image_paths if p.name not in processed_images]
    logger.info("Processing %d remaining images", len(remaining))

    images_list = existing_results["images"]
    annotations = existing_results["annotations"]
    ann_id = max((a["id"] for a in annotations), default=-1) + 1
    img_id = max((i["id"] for i in images_list), default=-1) + 1

    output_json.parent.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(0, len(remaining), batch_size), desc="Pseudo-labeling"):
        batch_paths = remaining[i : i + batch_size]
        batch_images = []

        for path in batch_paths:
            img = Image.open(path).convert("RGB")
            batch_images.append(img)

        # Process batch
        inputs = processor(images=batch_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process each image
        target_sizes = [(img.height, img.width) for img in batch_images]
        preds = processor.post_process_instance_segmentation(
            outputs, target_sizes=target_sizes, threshold=0.0
        )

        for j, (path, img, pred) in enumerate(zip(batch_paths, batch_images, preds)):
            images_list.append({
                "id": img_id,
                "file_name": path.name,
                "width": img.width,
                "height": img.height,
            })

            seg_map = pred["segmentation"].cpu().numpy()

            for seg_info in pred["segments_info"]:
                seg_id_val = seg_info["id"]
                category_id = seg_info["label_id"]
                score = float(seg_info["score"])

                binary_mask = (seg_map == seg_id_val).astype(np.uint8)
                rle = mask_util.encode(np.asfortranarray(binary_mask))
                rle["counts"] = rle["counts"].decode("utf-8")
                area = float(mask_util.area(rle))
                bbox = mask_util.toBbox(rle).tolist()

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(category_id),
                    "segmentation": rle,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                    "score": score,
                })
                ann_id += 1

            img_id += 1

        # Periodic save for resume support
        if (i // batch_size) % 50 == 0:
            _save_results(output_json, images_list, annotations, existing_results["categories"])

    _save_results(output_json, images_list, annotations, existing_results["categories"])
    logger.info(
        "Generated %d annotations for %d images → %s",
        len(annotations), len(images_list), output_json,
    )
    return output_json


def _save_results(
    output_json: Path,
    images: list[dict],
    annotations: list[dict],
    categories: list[dict],
) -> None:
    """Save current results to JSON."""
    result = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    with open(output_json, "w") as f:
        json.dump(result, f)
