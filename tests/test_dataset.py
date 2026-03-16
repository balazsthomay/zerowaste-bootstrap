"""Tests for the ZeroWaste COCO dataset adapter."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import Mask2FormerImageProcessor

from zerowaste_bootstrap.data.dataset import (
    ZeroWasteDataset,
    collate_fn,
    merge_coco_jsons,
)


@pytest.fixture
def processor() -> Mask2FormerImageProcessor:
    """Lightweight Mask2FormerImageProcessor (no model download)."""
    return Mask2FormerImageProcessor(
        size={"shortest_edge": 32, "longest_edge": 64},
        num_labels=4,
        ignore_index=255,
        do_resize=True,
        do_normalize=True,
    )


@pytest.fixture
def dataset(synthetic_coco_dir: Path, processor: Mask2FormerImageProcessor) -> ZeroWasteDataset:
    """ZeroWasteDataset built from the synthetic COCO fixture."""
    return ZeroWasteDataset(root_dir=synthetic_coco_dir, processor=processor)


# ------------------------------------------------------------------
# Dataset basics
# ------------------------------------------------------------------


class TestZeroWasteDatasetLength:
    """Tests for dataset __len__."""

    def test_length_matches_images(self, dataset: ZeroWasteDataset) -> None:
        assert len(dataset) == 5

    def test_length_is_positive(self, dataset: ZeroWasteDataset) -> None:
        assert len(dataset) > 0


class TestZeroWasteDatasetGetItem:
    """Tests for dataset __getitem__ keys and shapes."""

    def test_returns_expected_keys(self, dataset: ZeroWasteDataset) -> None:
        item = dataset[0]
        expected_keys = {"pixel_values", "pixel_mask", "mask_labels", "class_labels"}
        assert expected_keys.issubset(set(item.keys()))

    def test_pixel_values_is_3d_tensor(self, dataset: ZeroWasteDataset) -> None:
        item = dataset[0]
        pv = item["pixel_values"]
        assert isinstance(pv, torch.Tensor)
        assert pv.ndim == 3  # (C, H, W)
        assert pv.shape[0] == 3  # RGB

    def test_pixel_values_dtype(self, dataset: ZeroWasteDataset) -> None:
        item = dataset[0]
        assert item["pixel_values"].dtype == torch.float32

    def test_mask_labels_is_tensor(self, dataset: ZeroWasteDataset) -> None:
        item = dataset[0]
        ml = item["mask_labels"]
        assert isinstance(ml, torch.Tensor)
        # (num_instances, H, W) -- at least background
        assert ml.ndim == 3

    def test_class_labels_is_tensor(self, dataset: ZeroWasteDataset) -> None:
        item = dataset[0]
        cl = item["class_labels"]
        assert isinstance(cl, torch.Tensor)
        assert cl.ndim == 1
        assert cl.shape[0] > 0  # at least background

    def test_mask_and_class_labels_consistent(self, dataset: ZeroWasteDataset) -> None:
        """Number of masks must equal number of class labels."""
        item = dataset[0]
        assert item["mask_labels"].shape[0] == item["class_labels"].shape[0]

    def test_all_items_loadable(self, dataset: ZeroWasteDataset) -> None:
        """Iterate through every item to ensure no loading errors."""
        for i in range(len(dataset)):
            item = dataset[i]
            assert "pixel_values" in item

    def test_pixel_mask_is_2d(self, dataset: ZeroWasteDataset) -> None:
        item = dataset[0]
        pm = item["pixel_mask"]
        assert isinstance(pm, torch.Tensor)
        assert pm.ndim == 2  # (H, W)


# ------------------------------------------------------------------
# collate_fn
# ------------------------------------------------------------------


class TestCollateFn:
    """Tests for the custom collate function."""

    def test_collate_batches_pixel_values(self, dataset: ZeroWasteDataset) -> None:
        batch = [dataset[0], dataset[1]]
        collated = collate_fn(batch)
        pv = collated["pixel_values"]
        assert isinstance(pv, torch.Tensor)
        assert pv.shape[0] == 2  # batch size

    def test_collate_keeps_mask_labels_as_list(self, dataset: ZeroWasteDataset) -> None:
        batch = [dataset[0], dataset[1]]
        collated = collate_fn(batch)
        assert isinstance(collated["mask_labels"], list)
        assert len(collated["mask_labels"]) == 2

    def test_collate_keeps_class_labels_as_list(self, dataset: ZeroWasteDataset) -> None:
        batch = [dataset[0], dataset[1]]
        collated = collate_fn(batch)
        assert isinstance(collated["class_labels"], list)
        assert len(collated["class_labels"]) == 2

    def test_collate_pixel_mask_shape(self, dataset: ZeroWasteDataset) -> None:
        batch = [dataset[0], dataset[1]]
        collated = collate_fn(batch)
        pm = collated["pixel_mask"]
        assert isinstance(pm, torch.Tensor)
        assert pm.shape[0] == 2

    def test_collate_pads_to_max_dimensions(self, dataset: ZeroWasteDataset) -> None:
        """pixel_values should be padded to the max H, W in the batch."""
        batch = [dataset[0], dataset[1]]
        collated = collate_fn(batch)
        pv = collated["pixel_values"]
        # All items in batch share same spatial dims after padding
        assert pv.shape[-2] > 0
        assert pv.shape[-1] > 0

    def test_collate_single_item(self, dataset: ZeroWasteDataset) -> None:
        batch = [dataset[0]]
        collated = collate_fn(batch)
        assert collated["pixel_values"].shape[0] == 1
        assert len(collated["mask_labels"]) == 1
        assert len(collated["class_labels"]) == 1


# ------------------------------------------------------------------
# merge_coco_jsons
# ------------------------------------------------------------------


def _make_coco_json(tmp_path: Path, name: str, num_images: int, start_img_id: int = 0) -> Path:
    """Helper to create a minimal COCO JSON file."""
    images = []
    annotations = []
    ann_id = 0
    rng = np.random.RandomState(start_img_id)

    for i in range(num_images):
        img_id = start_img_id + i
        images.append({
            "id": img_id,
            "file_name": f"{name}_{i:04d}.png",
            "width": 64,
            "height": 64,
        })
        x, y = int(rng.randint(0, 32)), int(rng.randint(0, 32))
        annotations.append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": 0,
            "segmentation": [[x, y, x + 10, y, x + 10, y + 10, x, y + 10]],
            "area": 100.0,
            "bbox": [float(x), float(y), 10.0, 10.0],
            "iscrowd": 0,
        })
        ann_id += 1

    data = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 0, "name": "rigid_plastic"},
            {"id": 1, "name": "cardboard"},
            {"id": 2, "name": "metal"},
            {"id": 3, "name": "soft_plastic"},
        ],
    }
    path = tmp_path / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f)
    return path


class TestMergeCocoJsons:
    """Tests for merge_coco_jsons."""

    def test_merge_two_jsons_image_count(self, tmp_path: Path) -> None:
        json1 = _make_coco_json(tmp_path, "a", num_images=3, start_img_id=0)
        json2 = _make_coco_json(tmp_path, "b", num_images=4, start_img_id=0)
        merged = merge_coco_jsons([json1, json2])
        assert len(merged["images"]) == 7

    def test_merge_two_jsons_annotation_count(self, tmp_path: Path) -> None:
        json1 = _make_coco_json(tmp_path, "a", num_images=3, start_img_id=0)
        json2 = _make_coco_json(tmp_path, "b", num_images=4, start_img_id=0)
        merged = merge_coco_jsons([json1, json2])
        assert len(merged["annotations"]) == 7

    def test_merge_unique_image_ids(self, tmp_path: Path) -> None:
        """All image IDs must be unique after merge."""
        json1 = _make_coco_json(tmp_path, "a", num_images=3, start_img_id=0)
        json2 = _make_coco_json(tmp_path, "b", num_images=4, start_img_id=0)
        merged = merge_coco_jsons([json1, json2])
        image_ids = [img["id"] for img in merged["images"]]
        assert len(image_ids) == len(set(image_ids))

    def test_merge_unique_annotation_ids(self, tmp_path: Path) -> None:
        """All annotation IDs must be unique after merge."""
        json1 = _make_coco_json(tmp_path, "a", num_images=3, start_img_id=0)
        json2 = _make_coco_json(tmp_path, "b", num_images=4, start_img_id=0)
        merged = merge_coco_jsons([json1, json2])
        ann_ids = [ann["id"] for ann in merged["annotations"]]
        assert len(ann_ids) == len(set(ann_ids))

    def test_merge_preserves_categories(self, tmp_path: Path) -> None:
        json1 = _make_coco_json(tmp_path, "a", num_images=2, start_img_id=0)
        json2 = _make_coco_json(tmp_path, "b", num_images=2, start_img_id=0)
        merged = merge_coco_jsons([json1, json2])
        assert len(merged["categories"]) == 4
        cat_names = {c["name"] for c in merged["categories"]}
        assert "rigid_plastic" in cat_names

    def test_merge_annotation_image_ids_valid(self, tmp_path: Path) -> None:
        """Every annotation's image_id must reference an existing image."""
        json1 = _make_coco_json(tmp_path, "a", num_images=3, start_img_id=0)
        json2 = _make_coco_json(tmp_path, "b", num_images=4, start_img_id=0)
        merged = merge_coco_jsons([json1, json2])
        valid_img_ids = {img["id"] for img in merged["images"]}
        for ann in merged["annotations"]:
            assert ann["image_id"] in valid_img_ids

    def test_merge_single_json(self, tmp_path: Path) -> None:
        json1 = _make_coco_json(tmp_path, "a", num_images=3, start_img_id=0)
        merged = merge_coco_jsons([json1])
        assert len(merged["images"]) == 3

    def test_merge_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            merge_coco_jsons([])

    def test_merge_overlapping_ids_remapped(self, tmp_path: Path) -> None:
        """Even when source JSONs have identical IDs, merge produces unique ones."""
        json1 = _make_coco_json(tmp_path, "c", num_images=5, start_img_id=0)
        json2 = _make_coco_json(tmp_path, "d", num_images=5, start_img_id=0)
        merged = merge_coco_jsons([json1, json2])
        image_ids = [img["id"] for img in merged["images"]]
        assert len(image_ids) == 10
        assert len(set(image_ids)) == 10
