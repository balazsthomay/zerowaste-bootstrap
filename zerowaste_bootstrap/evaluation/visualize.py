"""Visualization utilities for predictions and annotations."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO

from zerowaste_bootstrap.config import ZEROWASTE_CLASSES

logger = logging.getLogger(__name__)

# Distinct colors for each class
CLASS_COLORS = {
    1: (1.0, 0.2, 0.2, 0.5),   # rigid_plastic - red
    2: (0.2, 0.6, 1.0, 0.5),   # cardboard - blue
    3: (0.8, 0.8, 0.2, 0.5),   # metal - yellow
    4: (0.2, 1.0, 0.4, 0.5),   # soft_plastic - green
}


def visualize_predictions(
    image: Image.Image,
    predictions: dict,
    gt: dict | None = None,
    output_path: Path | None = None,
) -> plt.Figure:
    """Overlay predicted masks and bboxes on an image.

    Args:
        image: PIL image.
        predictions: Dict with 'segmentation' (tensor), 'segments_info' (list of dicts).
        gt: Optional ground truth in same format.
        output_path: Optional path to save the figure.
    """
    fig, axes = plt.subplots(1, 2 if gt else 1, figsize=(12, 6))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # Plot predictions
    ax = axes[0]
    ax.imshow(image)
    ax.set_title("Predictions")

    seg_map = predictions["segmentation"]
    if isinstance(seg_map, torch.Tensor):
        seg_map = seg_map.cpu().numpy()

    for seg_info in predictions["segments_info"]:
        seg_id = seg_info["id"]
        cat_id = seg_info["label_id"]
        score = seg_info.get("score", 1.0)
        cat_name = ZEROWASTE_CLASSES.get(cat_id, f"class_{cat_id}")

        mask = seg_map == seg_id
        color = CLASS_COLORS.get(cat_id, (0.5, 0.5, 0.5, 0.5))

        overlay = np.zeros((*mask.shape, 4))
        overlay[mask] = color
        ax.imshow(overlay)

        # Find bbox from mask
        ys, xs = np.where(mask)
        if len(xs) > 0:
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            rect = plt.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=1, edgecolor=color[:3], facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(x_min, y_min - 2, f"{cat_name} {score:.2f}",
                    fontsize=7, color="white",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=color[:3], alpha=0.7))
    ax.axis("off")

    # Plot ground truth if available
    if gt and len(axes) > 1:
        ax = axes[1]
        ax.imshow(image)
        ax.set_title("Ground Truth")

        gt_seg = gt["segmentation"]
        if isinstance(gt_seg, torch.Tensor):
            gt_seg = gt_seg.cpu().numpy()

        for seg_info in gt["segments_info"]:
            seg_id = seg_info["id"]
            cat_id = seg_info["label_id"]
            cat_name = ZEROWASTE_CLASSES.get(cat_id, f"class_{cat_id}")

            mask = gt_seg == seg_id
            color = CLASS_COLORS.get(cat_id, (0.5, 0.5, 0.5, 0.5))
            overlay = np.zeros((*mask.shape, 4))
            overlay[mask] = color
            ax.imshow(overlay)

            ys, xs = np.where(mask)
            if len(xs) > 0:
                ax.text(xs.min(), ys.min() - 2, cat_name,
                        fontsize=7, color="white",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor=color[:3], alpha=0.7))
        ax.axis("off")

    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def create_comparison_grid(
    images: list[Image.Image],
    predictions_list: list[list[dict]],
    names: list[str],
    output_path: Path,
) -> None:
    """Create a side-by-side comparison grid of predictions from different models.

    Args:
        images: List of PIL images.
        predictions_list: List of prediction lists (one per model).
        names: Model/experiment names.
        output_path: Path to save the grid.
    """
    n_images = len(images)
    n_models = len(predictions_list)

    fig, axes = plt.subplots(n_images, n_models, figsize=(5 * n_models, 5 * n_images))
    if n_images == 1:
        axes = axes[np.newaxis, :]
    if n_models == 1:
        axes = axes[:, np.newaxis]

    for i, image in enumerate(images):
        for j, (preds, name) in enumerate(zip(predictions_list, names)):
            ax = axes[i, j]
            ax.imshow(image)
            if i == 0:
                ax.set_title(name, fontsize=12)

            if i < len(preds):
                pred = preds[i]
                seg_map = pred["segmentation"]
                if isinstance(seg_map, torch.Tensor):
                    seg_map = seg_map.cpu().numpy()
                for seg_info in pred["segments_info"]:
                    seg_id = seg_info["id"]
                    cat_id = seg_info["label_id"]
                    color = CLASS_COLORS.get(cat_id, (0.5, 0.5, 0.5, 0.5))
                    mask = seg_map == seg_id
                    overlay = np.zeros((*mask.shape, 4))
                    overlay[mask] = color
                    ax.imshow(overlay)
            ax.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Comparison grid saved to %s", output_path)


def visualize_model_predictions(
    model,
    processor,
    dataset,
    device: str,
    output_dir: Path,
    num_images: int = 10,
) -> None:
    """Generate prediction visualizations for a dataset subset."""
    model.eval()
    model.to(device)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = min(num_images, len(dataset))
    for idx in range(n):
        image = dataset.load_image(idx)
        inputs = processor(images=[image], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        target_size = (image.height, image.width)
        pred = processor.post_process_instance_segmentation(
            outputs, target_sizes=[target_size], threshold=0.5
        )[0]

        visualize_predictions(
            image=image,
            predictions=pred,
            output_path=output_dir / f"pred_{idx:04d}.png",
        )

    logger.info("Saved %d visualizations to %s", n, output_dir)


def visualize_annotations(
    annotations_path: Path,
    image_dir: Path,
    output_dir: Path,
    num_images: int = 20,
) -> None:
    """Visualize COCO annotations overlaid on images."""
    coco = COCO(str(annotations_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    img_ids = coco.getImgIds()[:num_images]

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        image = Image.open(image_dir / img_info["file_name"])
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.imshow(image)

        for ann in anns:
            cat_id = ann["category_id"]
            cat_name = ZEROWASTE_CLASSES.get(cat_id, f"class_{cat_id}")
            color = CLASS_COLORS.get(cat_id, (0.5, 0.5, 0.5, 0.5))

            # Draw polygon
            for seg in ann["segmentation"]:
                poly = np.array(seg).reshape(-1, 2)
                ax.fill(poly[:, 0], poly[:, 1], alpha=0.3, color=color[:3])
                ax.plot(
                    np.append(poly[:, 0], poly[0, 0]),
                    np.append(poly[:, 1], poly[0, 1]),
                    color=color[:3], linewidth=1,
                )

            # Label
            bbox = ann["bbox"]
            ax.text(bbox[0], bbox[1] - 2, cat_name,
                    fontsize=7, color="white",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=color[:3], alpha=0.7))

        ax.axis("off")
        ax.set_title(img_info["file_name"])
        fig.savefig(output_dir / f"ann_{img_id:04d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    logger.info("Saved %d annotation visualizations to %s", len(img_ids), output_dir)
