"""Shared test fixtures."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory structure."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def synthetic_coco_dir(tmp_path: Path) -> Path:
    """Create a synthetic COCO dataset with tiny images and annotations.

    Returns path to a directory containing data/ and labels.json.
    """
    dataset_dir = tmp_path / "synthetic_dataset"
    image_dir = dataset_dir / "data"
    image_dir.mkdir(parents=True)

    img_w, img_h = 64, 64
    num_images = 5
    categories = [
        {"id": 0, "name": "rigid_plastic"},
        {"id": 1, "name": "cardboard"},
        {"id": 2, "name": "metal"},
        {"id": 3, "name": "soft_plastic"},
    ]

    images = []
    annotations = []
    ann_id = 0

    rng = np.random.RandomState(42)

    for img_id in range(num_images):
        # Create a random RGB image
        arr = rng.randint(0, 255, (img_h, img_w, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        fname = f"image_{img_id:04d}.png"
        img.save(image_dir / fname)

        images.append({
            "id": img_id,
            "file_name": fname,
            "width": img_w,
            "height": img_h,
        })

        # Add 2-3 instances per image
        num_instances = rng.randint(2, 4)
        for _ in range(num_instances):
            cat_id = int(rng.randint(0, 4))
            # Random bbox
            x = int(rng.randint(0, img_w - 16))
            y = int(rng.randint(0, img_h - 16))
            w = int(rng.randint(8, min(16, img_w - x)))
            h = int(rng.randint(8, min(16, img_h - y)))

            # Simple polygon (rectangle)
            segmentation = [[x, y, x + w, y, x + w, y + h, x, y + h]]

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "segmentation": segmentation,
                "area": float(w * h),
                "bbox": [float(x), float(y), float(w), float(h)],
                "iscrowd": 0,
            })
            ann_id += 1

    coco_json = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    with open(dataset_dir / "labels.json", "w") as f:
        json.dump(coco_json, f)

    return dataset_dir


@pytest.fixture
def synthetic_coco_json(synthetic_coco_dir: Path) -> Path:
    """Path to the synthetic COCO JSON file."""
    return synthetic_coco_dir / "labels.json"


@pytest.fixture
def synthetic_image_dir(synthetic_coco_dir: Path) -> Path:
    """Path to the synthetic images directory."""
    return synthetic_coco_dir / "data"
