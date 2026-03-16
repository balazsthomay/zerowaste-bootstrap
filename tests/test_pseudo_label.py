"""Tests for pseudo-label generation and filtering."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from zerowaste_bootstrap.pseudo_label.filter import (
    analyze_pseudo_labels,
    filter_pseudo_labels,
)
from zerowaste_bootstrap.pseudo_label.generate import (
    _find_images,
    _save_results,
    generate_pseudo_labels,
)


@pytest.fixture
def raw_pseudo_labels(tmp_path: Path) -> Path:
    """Create a raw pseudo-label COCO JSON with scores."""
    categories = [
        {"id": 0, "name": "rigid_plastic"},
        {"id": 1, "name": "cardboard"},
        {"id": 2, "name": "metal"},
        {"id": 3, "name": "soft_plastic"},
    ]
    images = [
        {"id": 0, "file_name": "img_0.png", "width": 64, "height": 64},
        {"id": 1, "file_name": "img_1.png", "width": 64, "height": 64},
        {"id": 2, "file_name": "img_2.png", "width": 64, "height": 64},
    ]
    annotations = [
        # High confidence, large area → keep
        {"id": 0, "image_id": 0, "category_id": 0, "area": 500, "score": 0.95, "bbox": [0, 0, 20, 25], "iscrowd": 0,
         "segmentation": {"size": [64, 64], "counts": "0"}},
        # Low confidence → remove
        {"id": 1, "image_id": 0, "category_id": 1, "area": 300, "score": 0.3, "bbox": [10, 10, 15, 20], "iscrowd": 0,
         "segmentation": {"size": [64, 64], "counts": "0"}},
        # High confidence, tiny area → remove
        {"id": 2, "image_id": 1, "category_id": 2, "area": 50, "score": 0.9, "bbox": [5, 5, 5, 10], "iscrowd": 0,
         "segmentation": {"size": [64, 64], "counts": "0"}},
        # High confidence, large area → keep
        {"id": 3, "image_id": 1, "category_id": 3, "area": 200, "score": 0.8, "bbox": [20, 20, 10, 20], "iscrowd": 0,
         "segmentation": {"size": [64, 64], "counts": "0"}},
        # Borderline confidence → remove (below 0.7)
        {"id": 4, "image_id": 2, "category_id": 0, "area": 400, "score": 0.69, "bbox": [0, 0, 20, 20], "iscrowd": 0,
         "segmentation": {"size": [64, 64], "counts": "0"}},
    ]

    data = {"images": images, "annotations": annotations, "categories": categories}
    json_path = tmp_path / "raw.json"
    with open(json_path, "w") as f:
        json.dump(data, f)
    return json_path


class TestFindImages:
    def test_finds_images(self, tmp_path: Path):
        (tmp_path / "img1.jpg").touch()
        (tmp_path / "img2.png").touch()
        (tmp_path / "img3.PNG").touch()
        (tmp_path / "notes.txt").touch()

        found = _find_images(tmp_path)
        names = {p.name for p in found}
        assert "img1.jpg" in names
        assert "img2.png" in names
        assert "img3.PNG" in names
        assert "notes.txt" not in names

    def test_empty_directory(self, tmp_path: Path):
        found = _find_images(tmp_path)
        assert len(found) == 0


class TestFilterPseudoLabels:
    def test_filters_by_confidence(self, raw_pseudo_labels: Path, tmp_path: Path):
        output = tmp_path / "filtered.json"
        filter_pseudo_labels(raw_pseudo_labels, output, confidence_threshold=0.7, min_mask_area=0)

        with open(output) as f:
            data = json.load(f)

        # Annotations with score >= 0.7: ids 0 (0.95), 2 (0.9), 3 (0.8)
        assert len(data["annotations"]) == 3

    def test_filters_by_area(self, raw_pseudo_labels: Path, tmp_path: Path):
        output = tmp_path / "filtered.json"
        filter_pseudo_labels(raw_pseudo_labels, output, confidence_threshold=0.0, min_mask_area=100)

        with open(output) as f:
            data = json.load(f)

        # Annotations with area >= 100: ids 0 (500), 1 (300), 3 (200), 4 (400)
        assert len(data["annotations"]) == 4

    def test_combined_filter(self, raw_pseudo_labels: Path, tmp_path: Path):
        output = tmp_path / "filtered.json"
        filter_pseudo_labels(raw_pseudo_labels, output, confidence_threshold=0.7, min_mask_area=100)

        with open(output) as f:
            data = json.load(f)

        # score >= 0.7 AND area >= 100: ids 0 (0.95, 500) and 3 (0.8, 200)
        assert len(data["annotations"]) == 2

    def test_images_filtered_to_match(self, raw_pseudo_labels: Path, tmp_path: Path):
        output = tmp_path / "filtered.json"
        filter_pseudo_labels(raw_pseudo_labels, output, confidence_threshold=0.7, min_mask_area=100)

        with open(output) as f:
            data = json.load(f)

        # Only images 0 and 1 have kept annotations
        img_ids = {img["id"] for img in data["images"]}
        assert img_ids == {0, 1}

    def test_annotation_ids_remapped(self, raw_pseudo_labels: Path, tmp_path: Path):
        output = tmp_path / "filtered.json"
        filter_pseudo_labels(raw_pseudo_labels, output)

        with open(output) as f:
            data = json.load(f)

        ids = [ann["id"] for ann in data["annotations"]]
        assert ids == list(range(len(ids)))


class TestAnalyzePseudoLabels:
    def test_analyze(self, raw_pseudo_labels: Path):
        stats = analyze_pseudo_labels(raw_pseudo_labels)
        assert stats["total"] == 5
        assert stats["num_images"] == 3
        assert "rigid_plastic" in stats["class_distribution"]
        assert stats["score_stats"]["min"] == pytest.approx(0.3)
        assert stats["score_stats"]["max"] == pytest.approx(0.95)

    def test_analyze_empty(self, tmp_path: Path):
        empty_json = tmp_path / "empty.json"
        data = {"images": [], "annotations": [], "categories": []}
        with open(empty_json, "w") as f:
            json.dump(data, f)

        stats = analyze_pseudo_labels(empty_json)
        assert stats["total"] == 0


class TestSaveResults:
    def test_save_results(self, tmp_path: Path):
        """Test _save_results writes valid JSON."""
        output = tmp_path / "output.json"
        images = [{"id": 0, "file_name": "img.png", "width": 64, "height": 64}]
        annotations = [{"id": 0, "image_id": 0, "category_id": 0, "area": 100}]
        categories = [{"id": 0, "name": "rigid_plastic"}]

        _save_results(output, images, annotations, categories)

        assert output.exists()
        with open(output) as f:
            data = json.load(f)
        assert len(data["images"]) == 1
        assert len(data["annotations"]) == 1
        assert len(data["categories"]) == 1

    def test_save_results_creates_parent_dirs(self, tmp_path: Path):
        """Test that _save_results works when parent dir exists."""
        output = tmp_path / "sub" / "dir" / "output.json"
        output.parent.mkdir(parents=True)
        _save_results(output, [], [], [])
        assert output.exists()


class TestGeneratePseudoLabels:
    def _create_test_images(self, image_dir: Path, num_images: int = 3) -> list[Path]:
        """Create small test images and return their paths."""
        image_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in range(num_images):
            img = Image.fromarray(
                np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            )
            path = image_dir / f"test_{i:04d}.png"
            img.save(path)
            paths.append(path)
        return paths

    @patch("zerowaste_bootstrap.modeling.model.load_processor")
    @patch(
        "transformers.Mask2FormerForUniversalSegmentation"
    )
    @patch("zerowaste_bootstrap.pseudo_label.generate.get_device")
    def test_generate_pseudo_labels_basic(
        self, mock_get_device, mock_model_cls, mock_load_processor, tmp_path: Path
    ):
        """Test generate_pseudo_labels with fully mocked model."""
        mock_get_device.return_value = "cpu"

        # Create test images
        image_dir = tmp_path / "images"
        self._create_test_images(image_dir, num_images=2)

        # Mock model
        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        # Create a segmentation map with one segment (id=1)
        seg_map = torch.zeros((32, 32), dtype=torch.int64)
        seg_map[5:15, 5:15] = 1

        # Mock model forward pass output
        mock_outputs = MagicMock()
        mock_model.__call__ = MagicMock(return_value=mock_outputs)
        mock_model.return_value = mock_outputs

        # Mock processor
        mock_processor = MagicMock()
        mock_load_processor.return_value = mock_processor
        mock_processor.return_value = {
            "pixel_values": torch.randn(2, 3, 32, 32),
        }

        # Mock post-processing: return one segment per image
        pred_result = {
            "segmentation": seg_map,
            "segments_info": [
                {"id": 1, "label_id": 0, "score": 0.85},
            ],
        }
        mock_processor.post_process_instance_segmentation.return_value = [
            pred_result,
            pred_result,
        ]

        output_json = tmp_path / "output" / "raw.json"
        result = generate_pseudo_labels(
            model_path=tmp_path / "checkpoint",
            image_dir=image_dir,
            output_json=output_json,
            device="cpu",
            batch_size=4,
        )

        assert result == output_json
        assert output_json.exists()

        with open(output_json) as f:
            data = json.load(f)

        assert len(data["images"]) == 2
        assert len(data["annotations"]) == 2
        assert len(data["categories"]) == 4  # 4 zerowaste classes

        # Check annotation structure
        ann = data["annotations"][0]
        assert "image_id" in ann
        assert "category_id" in ann
        assert "segmentation" in ann
        assert "area" in ann
        assert "bbox" in ann
        assert "score" in ann
        assert ann["iscrowd"] == 0

    @patch("zerowaste_bootstrap.modeling.model.load_processor")
    @patch(
        "transformers.Mask2FormerForUniversalSegmentation"
    )
    @patch("zerowaste_bootstrap.pseudo_label.generate.get_device")
    def test_generate_pseudo_labels_no_segments(
        self, mock_get_device, mock_model_cls, mock_load_processor, tmp_path: Path
    ):
        """Test generate_pseudo_labels when model predicts no segments."""
        mock_get_device.return_value = "cpu"

        image_dir = tmp_path / "images"
        self._create_test_images(image_dir, num_images=1)

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        mock_outputs = MagicMock()
        mock_model.return_value = mock_outputs

        mock_processor = MagicMock()
        mock_load_processor.return_value = mock_processor
        mock_processor.return_value = {
            "pixel_values": torch.randn(1, 3, 32, 32),
        }

        # No segments predicted
        pred_result = {
            "segmentation": torch.zeros((32, 32), dtype=torch.int64),
            "segments_info": [],
        }
        mock_processor.post_process_instance_segmentation.return_value = [pred_result]

        output_json = tmp_path / "output" / "raw.json"
        generate_pseudo_labels(
            model_path=tmp_path / "checkpoint",
            image_dir=image_dir,
            output_json=output_json,
            device="cpu",
            batch_size=4,
        )

        with open(output_json) as f:
            data = json.load(f)

        assert len(data["images"]) == 1
        assert len(data["annotations"]) == 0

    @patch("zerowaste_bootstrap.modeling.model.load_processor")
    @patch(
        "transformers.Mask2FormerForUniversalSegmentation"
    )
    @patch("zerowaste_bootstrap.pseudo_label.generate.get_device")
    def test_generate_pseudo_labels_resume(
        self, mock_get_device, mock_model_cls, mock_load_processor, tmp_path: Path
    ):
        """Test that generate_pseudo_labels resumes from existing output."""
        mock_get_device.return_value = "cpu"

        image_dir = tmp_path / "images"
        self._create_test_images(image_dir, num_images=3)

        # Pre-populate output with one already-processed image
        output_json = tmp_path / "output" / "raw.json"
        output_json.parent.mkdir(parents=True)
        existing = {
            "images": [
                {"id": 0, "file_name": "test_0000.png", "width": 32, "height": 32}
            ],
            "annotations": [
                {
                    "id": 0,
                    "image_id": 0,
                    "category_id": 0,
                    "segmentation": {"size": [32, 32], "counts": "0"},
                    "area": 100,
                    "bbox": [5, 5, 10, 10],
                    "iscrowd": 0,
                    "score": 0.9,
                }
            ],
            "categories": [
                {"id": 0, "name": "rigid_plastic"},
                {"id": 1, "name": "cardboard"},
                {"id": 2, "name": "metal"},
                {"id": 3, "name": "soft_plastic"},
            ],
        }
        with open(output_json, "w") as f:
            json.dump(existing, f)

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        seg_map = torch.zeros((32, 32), dtype=torch.int64)
        seg_map[5:15, 5:15] = 1

        mock_outputs = MagicMock()
        mock_model.return_value = mock_outputs

        mock_processor = MagicMock()
        mock_load_processor.return_value = mock_processor
        mock_processor.return_value = {
            "pixel_values": torch.randn(2, 3, 32, 32),
        }

        pred_result = {
            "segmentation": seg_map,
            "segments_info": [{"id": 1, "label_id": 1, "score": 0.75}],
        }
        mock_processor.post_process_instance_segmentation.return_value = [
            pred_result,
            pred_result,
        ]

        generate_pseudo_labels(
            model_path=tmp_path / "checkpoint",
            image_dir=image_dir,
            output_json=output_json,
            device="cpu",
            batch_size=4,
        )

        with open(output_json) as f:
            data = json.load(f)

        # Should have 3 images total: 1 existing + 2 new
        assert len(data["images"]) == 3
        # 1 existing + 2 new annotations
        assert len(data["annotations"]) == 3

    @patch("zerowaste_bootstrap.modeling.model.load_processor")
    @patch(
        "transformers.Mask2FormerForUniversalSegmentation"
    )
    @patch("zerowaste_bootstrap.pseudo_label.generate.get_device")
    def test_generate_with_empty_image_dir(
        self, mock_get_device, mock_model_cls, mock_load_processor, tmp_path: Path
    ):
        """Test with no images in directory."""
        mock_get_device.return_value = "cpu"

        image_dir = tmp_path / "empty_images"
        image_dir.mkdir()

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        mock_processor = MagicMock()
        mock_load_processor.return_value = mock_processor

        output_json = tmp_path / "output" / "raw.json"
        generate_pseudo_labels(
            model_path=tmp_path / "checkpoint",
            image_dir=image_dir,
            output_json=output_json,
            device="cpu",
            batch_size=4,
        )

        with open(output_json) as f:
            data = json.load(f)
        assert len(data["images"]) == 0
        assert len(data["annotations"]) == 0
