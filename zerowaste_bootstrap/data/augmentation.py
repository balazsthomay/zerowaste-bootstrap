"""Cut-paste augmentation for synthetic training data generation."""

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from zerowaste_bootstrap.config import AugmentConfig, ZEROWASTE_CLASSES

logger = logging.getLogger(__name__)


def build_object_bank(
    coco_json: Path,
    image_dir: Path,
    output_dir: Path,
) -> Path:
    """Extract individual object instances as RGBA PNGs from labeled data.

    Args:
        coco_json: Path to COCO JSON annotations.
        image_dir: Path to images directory.
        output_dir: Path to save extracted objects.

    Returns:
        Path to the object bank directory.
    """
    coco = COCO(str(coco_json))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create class subdirectories
    for cat_name in ZEROWASTE_CLASSES.values():
        (output_dir / cat_name).mkdir(exist_ok=True)

    metadata = []
    ann_ids = coco.getAnnIds()
    logger.info("Extracting %d objects from %s", len(ann_ids), coco_json)

    for ann_id in tqdm(ann_ids, desc="Building object bank"):
        ann = coco.loadAnns(ann_id)[0]
        img_info = coco.loadImgs(ann["image_id"])[0]

        cat_id = ann["category_id"]
        cat_name = ZEROWASTE_CLASSES.get(cat_id)
        if cat_name is None:
            continue

        # Load image
        img_path = image_dir / img_info["file_name"]
        if not img_path.exists():
            continue
        image = Image.open(img_path).convert("RGB")
        img_array = np.array(image)

        # Create binary mask from annotation
        mask = coco.annToMask(ann)

        # Crop to bounding box with small padding
        bbox = ann["bbox"]  # [x, y, w, h]
        x, y, w, h = [int(v) for v in bbox]
        pad = 2
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(img_array.shape[1], x + w + pad)
        y1 = min(img_array.shape[0], y + h + pad)

        cropped_img = img_array[y0:y1, x0:x1]
        cropped_mask = mask[y0:y1, x0:x1]

        # Create RGBA image
        rgba = np.zeros((*cropped_img.shape[:2], 4), dtype=np.uint8)
        rgba[:, :, :3] = cropped_img
        rgba[:, :, 3] = (cropped_mask * 255).astype(np.uint8)

        # Save
        obj_path = output_dir / cat_name / f"{ann_id}.png"
        Image.fromarray(rgba).save(obj_path)

        metadata.append({
            "id": ann_id,
            "category_id": cat_id,
            "category_name": cat_name,
            "file": str(obj_path.relative_to(output_dir)),
            "original_area": float(ann["area"]),
            "width": x1 - x0,
            "height": y1 - y0,
        })

    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Object bank: %d objects in %s", len(metadata), output_dir)
    return output_dir


def _apply_transforms(
    obj_rgba: np.ndarray,
    rng: np.random.RandomState,
    config: AugmentConfig,
) -> np.ndarray:
    """Apply random transforms to an RGBA object crop."""
    img = Image.fromarray(obj_rgba)

    # Random scale
    scale = rng.uniform(*config.scale_range)
    new_w = max(1, int(img.width * scale))
    new_h = max(1, int(img.height * scale))
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # Random rotation
    angle = rng.uniform(*config.rotation_range)
    img = img.rotate(angle, expand=True, resample=Image.BILINEAR)

    # Random horizontal flip
    if rng.random() < config.hflip_prob:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Random brightness
    brightness = rng.uniform(*config.brightness_range)
    rgb = img.convert("RGB")
    rgb = ImageEnhance.Brightness(rgb).enhance(brightness)
    result = np.array(img)
    result[:, :, :3] = np.array(rgb)

    return result


def _smooth_alpha(alpha: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian smoothing to alpha channel for blended edges."""
    smoothed = gaussian_filter(alpha.astype(np.float64), sigma=sigma)
    return np.clip(smoothed, 0, 255).astype(np.uint8)


def generate_synthetic_images(
    object_bank_dir: Path,
    background_dir: Path,
    output_dir: Path,
    num_images: int = 1000,
    seed: int = 42,
    config: AugmentConfig | None = None,
    visualize: bool = False,
) -> Path:
    """Generate synthetic images via cut-paste augmentation.

    Args:
        object_bank_dir: Path to object bank with class subdirectories.
        background_dir: Path to background images.
        output_dir: Path to save synthetic images and annotations.
        num_images: Number of synthetic images to generate.
        seed: Random seed.
        config: Augmentation configuration.
        visualize: Whether to save visualization of first few images.

    Returns:
        Path to the output directory.
    """
    if config is None:
        config = AugmentConfig(seed=seed)

    rng = np.random.RandomState(seed)

    # Load object bank
    with open(object_bank_dir / "metadata.json") as f:
        obj_metadata = json.load(f)

    if not obj_metadata:
        logger.warning("Object bank is empty")
        return output_dir

    # Group objects by class
    objects_by_class: dict[str, list[dict]] = {}
    for obj in obj_metadata:
        cat = obj["category_name"]
        objects_by_class.setdefault(cat, []).append(obj)

    all_objects = obj_metadata

    # Find backgrounds
    bg_paths = sorted(
        p for p in background_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")
    )
    if not bg_paths:
        logger.error("No background images found in %s", background_dir)
        return output_dir

    # Output directories
    img_output_dir = output_dir / "images"
    img_output_dir.mkdir(parents=True, exist_ok=True)

    images_list = []
    annotations = []
    ann_id = 0

    logger.info("Generating %d synthetic images", num_images)

    for img_id in tqdm(range(num_images), desc="Generating synthetic images"):
        # Random background
        bg_path = bg_paths[rng.randint(len(bg_paths))]
        background = Image.open(bg_path).convert("RGB")
        canvas = np.array(background)
        h, w = canvas.shape[:2]

        # Instance tracking for occlusion
        instance_mask = np.zeros((h, w), dtype=np.int32)  # pixel → instance_id
        img_annotations = []

        # Sample number of objects (Poisson)
        n_objects = max(1, rng.poisson(config.objects_per_image_mean))

        for obj_idx in range(n_objects):
            # Random object
            obj_info = all_objects[rng.randint(len(all_objects))]
            obj_path = object_bank_dir / obj_info["file"]
            if not obj_path.exists():
                continue

            obj_rgba = np.array(Image.open(obj_path).convert("RGBA"))
            obj_rgba = _apply_transforms(obj_rgba, rng, config)

            obj_h, obj_w = obj_rgba.shape[:2]
            if obj_h < 4 or obj_w < 4:
                continue

            # Random placement
            max_x = max(1, w - obj_w)
            max_y = max(1, h - obj_h)
            paste_x = rng.randint(0, max_x)
            paste_y = rng.randint(0, max_y)

            # Clip to canvas bounds
            src_x0 = 0
            src_y0 = 0
            src_x1 = min(obj_w, w - paste_x)
            src_y1 = min(obj_h, h - paste_y)

            obj_rgb = obj_rgba[src_y0:src_y1, src_x0:src_x1, :3]
            obj_alpha = obj_rgba[src_y0:src_y1, src_x0:src_x1, 3]

            # Smooth alpha edges
            obj_alpha = _smooth_alpha(obj_alpha, config.edge_sigma)

            # Alpha blend onto canvas
            alpha_f = obj_alpha.astype(np.float32) / 255.0
            for c in range(3):
                region = canvas[paste_y:paste_y + src_y1, paste_x:paste_x + src_x1, c]
                canvas[paste_y:paste_y + src_y1, paste_x:paste_x + src_x1, c] = (
                    alpha_f * obj_rgb[:, :, c] + (1 - alpha_f) * region
                ).astype(np.uint8)

            # Update instance mask (later objects occlude earlier ones)
            instance_id = obj_idx + 1
            obj_binary = obj_alpha > 127
            instance_mask[paste_y:paste_y + src_y1, paste_x:paste_x + src_x1][obj_binary] = instance_id

            img_annotations.append({
                "instance_id": instance_id,
                "category_id": obj_info["category_id"],
                "category_name": obj_info["category_name"],
            })

        # Convert final instance masks to annotations
        for ann_info in img_annotations:
            inst_id = ann_info["instance_id"]
            binary_mask = (instance_mask == inst_id).astype(np.uint8)

            # Skip if fully occluded
            if binary_mask.sum() < 10:
                continue

            rle = mask_util.encode(np.asfortranarray(binary_mask))
            rle["counts"] = rle["counts"].decode("utf-8")
            area = float(mask_util.area(rle))
            bbox = mask_util.toBbox(rle).tolist()

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": ann_info["category_id"],
                "segmentation": rle,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
            })
            ann_id += 1

        # Save synthetic image
        fname = f"synthetic_{img_id:06d}.png"
        Image.fromarray(canvas).save(img_output_dir / fname)

        images_list.append({
            "id": img_id,
            "file_name": fname,
            "width": w,
            "height": h,
        })

    # Save COCO annotations
    coco_data = {
        "images": images_list,
        "annotations": annotations,
        "categories": [
            {"id": k, "name": v} for k, v in ZEROWASTE_CLASSES.items()
        ],
    }
    with open(output_dir / "annotations.json", "w") as f:
        json.dump(coco_data, f)

    logger.info(
        "Generated %d images with %d annotations → %s",
        len(images_list), len(annotations), output_dir,
    )

    if visualize:
        from zerowaste_bootstrap.evaluation.visualize import visualize_annotations
        visualize_annotations(
            annotations_path=output_dir / "annotations.json",
            image_dir=img_output_dir,
            output_dir=output_dir / "vis",
            num_images=min(10, num_images),
        )

    return output_dir
