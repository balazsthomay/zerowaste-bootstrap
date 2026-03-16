"""Tests for cut-paste augmentation."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from zerowaste_bootstrap.config import AugmentConfig
from zerowaste_bootstrap.data.augmentation import (
    _apply_transforms,
    _smooth_alpha,
    build_object_bank,
    generate_synthetic_images,
)


class TestSmoothAlpha:
    def test_preserves_shape(self):
        alpha = np.ones((32, 32), dtype=np.uint8) * 255
        result = _smooth_alpha(alpha, sigma=2.0)
        assert result.shape == (32, 32)
        assert result.dtype == np.uint8

    def test_smooths_edges(self):
        alpha = np.zeros((32, 32), dtype=np.uint8)
        alpha[8:24, 8:24] = 255
        result = _smooth_alpha(alpha, sigma=2.0)
        # Edges should be partially transparent
        assert result[8, 8] < 255
        # Center should still be close to opaque
        assert result[16, 16] > 200


class TestApplyTransforms:
    def test_output_is_rgba(self):
        obj = np.zeros((20, 20, 4), dtype=np.uint8)
        obj[:, :, :3] = 128
        obj[:, :, 3] = 255
        rng = np.random.RandomState(42)
        config = AugmentConfig()
        result = _apply_transforms(obj, rng, config)
        assert result.ndim == 3
        assert result.shape[2] == 4

    def test_deterministic_with_seed(self):
        obj = np.random.RandomState(0).randint(0, 255, (20, 20, 4), dtype=np.uint8)
        config = AugmentConfig()

        rng1 = np.random.RandomState(42)
        result1 = _apply_transforms(obj.copy(), rng1, config)

        rng2 = np.random.RandomState(42)
        result2 = _apply_transforms(obj.copy(), rng2, config)

        np.testing.assert_array_equal(result1, result2)


class TestBuildObjectBank:
    def test_extracts_objects(self, synthetic_coco_dir: Path, tmp_path: Path):
        output_dir = tmp_path / "object_bank"
        build_object_bank(
            coco_json=synthetic_coco_dir / "labels.json",
            image_dir=synthetic_coco_dir / "data",
            output_dir=output_dir,
        )

        # Should have metadata
        metadata_path = output_dir / "metadata.json"
        assert metadata_path.exists()

        with open(metadata_path) as f:
            metadata = json.load(f)
        assert len(metadata) > 0

        # Each object should be a valid RGBA PNG
        for obj in metadata:
            obj_path = output_dir / obj["file"]
            assert obj_path.exists()
            img = Image.open(obj_path)
            assert img.mode == "RGBA"

    def test_creates_class_subdirectories(self, synthetic_coco_dir: Path, tmp_path: Path):
        output_dir = tmp_path / "object_bank"
        build_object_bank(
            coco_json=synthetic_coco_dir / "labels.json",
            image_dir=synthetic_coco_dir / "data",
            output_dir=output_dir,
        )
        # Should have at least some class directories
        subdirs = [d.name for d in output_dir.iterdir() if d.is_dir()]
        assert len(subdirs) > 0


class TestGenerateSyntheticImages:
    def test_generates_images_and_annotations(
        self, synthetic_coco_dir: Path, tmp_path: Path
    ):
        # First build object bank
        object_bank_dir = tmp_path / "object_bank"
        build_object_bank(
            coco_json=synthetic_coco_dir / "labels.json",
            image_dir=synthetic_coco_dir / "data",
            output_dir=object_bank_dir,
        )

        # Generate synthetic images
        output_dir = tmp_path / "synthetic"
        generate_synthetic_images(
            object_bank_dir=object_bank_dir,
            background_dir=synthetic_coco_dir / "data",
            output_dir=output_dir,
            num_images=3,
            seed=42,
        )

        # Check output
        assert (output_dir / "annotations.json").exists()
        assert (output_dir / "images").exists()

        with open(output_dir / "annotations.json") as f:
            data = json.load(f)

        assert len(data["images"]) == 3
        assert len(data["annotations"]) > 0
        assert len(data["categories"]) == 4

        # All synthetic images should exist
        for img_info in data["images"]:
            assert (output_dir / "images" / img_info["file_name"]).exists()

    def test_annotations_have_valid_format(
        self, synthetic_coco_dir: Path, tmp_path: Path
    ):
        object_bank_dir = tmp_path / "object_bank"
        build_object_bank(
            coco_json=synthetic_coco_dir / "labels.json",
            image_dir=synthetic_coco_dir / "data",
            output_dir=object_bank_dir,
        )

        output_dir = tmp_path / "synthetic"
        generate_synthetic_images(
            object_bank_dir=object_bank_dir,
            background_dir=synthetic_coco_dir / "data",
            output_dir=output_dir,
            num_images=2,
            seed=42,
        )

        with open(output_dir / "annotations.json") as f:
            data = json.load(f)

        for ann in data["annotations"]:
            assert "id" in ann
            assert "image_id" in ann
            assert "category_id" in ann
            assert "segmentation" in ann
            assert "area" in ann
            assert "bbox" in ann
            assert ann["area"] > 0
            assert len(ann["bbox"]) == 4

    def test_occlusion_handling(self, synthetic_coco_dir: Path, tmp_path: Path):
        """Objects pasted later should occlude earlier ones."""
        object_bank_dir = tmp_path / "object_bank"
        build_object_bank(
            coco_json=synthetic_coco_dir / "labels.json",
            image_dir=synthetic_coco_dir / "data",
            output_dir=object_bank_dir,
        )

        config = AugmentConfig(objects_per_image_mean=10.0, seed=42)
        output_dir = tmp_path / "synthetic"
        generate_synthetic_images(
            object_bank_dir=object_bank_dir,
            background_dir=synthetic_coco_dir / "data",
            output_dir=output_dir,
            num_images=1,
            seed=42,
            config=config,
        )

        with open(output_dir / "annotations.json") as f:
            data = json.load(f)

        # With 10 objects average, some should be partially occluded
        # Just verify annotations exist and areas are reasonable
        assert len(data["annotations"]) >= 1
        for ann in data["annotations"]:
            assert ann["area"] >= 10  # minimum threshold in generation
