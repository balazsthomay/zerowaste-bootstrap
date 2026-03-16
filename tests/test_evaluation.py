"""Tests for evaluation metrics, comparison, and visualization."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from zerowaste_bootstrap.evaluation.compare import (
    compare_experiments,
    compare_experiments_cli,
)
from zerowaste_bootstrap.evaluation.visualize import (
    create_comparison_grid,
    visualize_annotations,
    visualize_predictions,
)
from zerowaste_bootstrap.evaluation.metrics import (
    _empty_metrics,
    _masks_to_rle,
    evaluate_from_results,
)


class TestCompareExperiments:
    def test_empty_results(self):
        result = compare_experiments({})
        assert result == "No results to compare."

    def test_single_experiment(self):
        results = {
            "baseline": {"AP": 0.35, "AP50": 0.55, "AP75": 0.30, "AR": 0.40},
        }
        table = compare_experiments(results)
        assert "baseline" in table
        assert "0.3500" in table

    def test_two_experiments_with_delta(self):
        results = {
            "baseline": {"AP": 0.35, "AP50": 0.55, "AP75": 0.30, "AR": 0.40},
            "pseudo": {"AP": 0.40, "AP50": 0.60, "AP75": 0.35, "AR": 0.45},
        }
        table = compare_experiments(results)
        assert "baseline" in table
        assert "pseudo" in table
        assert "delta" in table
        assert "+0.0500" in table  # AP delta

    def test_multiple_experiments(self):
        results = {
            "baseline": {"AP": 0.35},
            "pseudo": {"AP": 0.40},
            "augment": {"AP": 0.38},
            "both": {"AP": 0.42},
        }
        table = compare_experiments(results)
        for name in results:
            assert name in table

    def test_negative_delta(self):
        results = {
            "baseline": {"AP": 0.40, "AP50": 0.60, "AP75": 0.35, "AR": 0.45},
            "worse": {"AP": 0.30, "AP50": 0.50, "AP75": 0.25, "AR": 0.35},
        }
        table = compare_experiments(results)
        assert "-0.1000" in table  # negative delta for AP


class TestCompareExperimentsCli:
    def test_with_metrics_files(self, tmp_path: Path):
        """Test compare_experiments_cli reads metrics and writes output."""
        baseline_dir = tmp_path / "baseline"
        baseline_dir.mkdir()
        with open(baseline_dir / "metrics_test.json", "w") as f:
            json.dump({"AP": 0.35, "AP50": 0.55, "AP75": 0.30, "AR": 0.40}, f)

        pseudo_dir = tmp_path / "pseudo"
        pseudo_dir.mkdir()
        with open(pseudo_dir / "metrics_test.json", "w") as f:
            json.dump({"AP": 0.40, "AP50": 0.60, "AP75": 0.35, "AR": 0.45}, f)

        compare_experiments_cli(
            experiment_names=["baseline", "pseudo"],
            output_dir=tmp_path,
        )

        assert (tmp_path / "comparison.md").exists()
        assert (tmp_path / "comparison.json").exists()

        with open(tmp_path / "comparison.json") as f:
            data = json.load(f)
        assert "baseline" in data
        assert "pseudo" in data
        assert data["baseline"]["AP"] == 0.35
        assert data["pseudo"]["AP"] == 0.40

    def test_no_experiments_found(self, tmp_path: Path):
        """Test graceful handling when no metrics files exist."""
        compare_experiments_cli(
            experiment_names=["nonexistent"],
            output_dir=tmp_path,
        )
        # Should not create output files
        assert not (tmp_path / "comparison.md").exists()

    def test_eval_subdirectory_fallback(self, tmp_path: Path):
        """Test that it falls back to output_dir/eval/metrics_test_{name}.json."""
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()
        with open(eval_dir / "metrics_test_baseline.json", "w") as f:
            json.dump({"AP": 0.35, "AP50": 0.55, "AP75": 0.30, "AR": 0.40}, f)

        compare_experiments_cli(
            experiment_names=["baseline"],
            output_dir=tmp_path,
        )

        assert (tmp_path / "comparison.md").exists()

    def test_partial_experiments(self, tmp_path: Path):
        """Test when only some experiments have metrics."""
        baseline_dir = tmp_path / "baseline"
        baseline_dir.mkdir()
        with open(baseline_dir / "metrics_test.json", "w") as f:
            json.dump({"AP": 0.35, "AP50": 0.55, "AP75": 0.30, "AR": 0.40}, f)

        compare_experiments_cli(
            experiment_names=["baseline", "nonexistent"],
            output_dir=tmp_path,
        )

        # Only baseline should appear in results
        with open(tmp_path / "comparison.json") as f:
            data = json.load(f)
        assert "baseline" in data
        assert "nonexistent" not in data


class TestVisualizePredictions:
    def test_produces_figure(self):
        image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        seg_map = np.zeros((64, 64), dtype=np.int32)
        seg_map[10:30, 10:30] = 1

        predictions = {
            "segmentation": seg_map,
            "segments_info": [
                {"id": 1, "label_id": 0, "score": 0.9},
            ],
        }
        fig = visualize_predictions(image, predictions)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_saves_to_file(self, tmp_path: Path):
        image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        seg_map = np.zeros((64, 64), dtype=np.int32)
        seg_map[10:30, 10:30] = 1

        predictions = {
            "segmentation": seg_map,
            "segments_info": [
                {"id": 1, "label_id": 0, "score": 0.9},
            ],
        }
        out_path = tmp_path / "test_vis.png"
        visualize_predictions(image, predictions, output_path=out_path)
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_with_ground_truth(self, tmp_path: Path):
        image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        seg_map = np.zeros((64, 64), dtype=np.int32)
        seg_map[10:30, 10:30] = 1

        predictions = {
            "segmentation": seg_map,
            "segments_info": [{"id": 1, "label_id": 0, "score": 0.9}],
        }
        gt = {
            "segmentation": seg_map.copy(),
            "segments_info": [{"id": 1, "label_id": 0}],
        }
        out_path = tmp_path / "test_vis_gt.png"
        visualize_predictions(image, predictions, gt=gt, output_path=out_path)
        assert out_path.exists()

    def test_empty_predictions(self, tmp_path: Path):
        image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        predictions = {
            "segmentation": np.zeros((64, 64), dtype=np.int32),
            "segments_info": [],
        }
        fig = visualize_predictions(image, predictions)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_with_torch_tensor_segmentation(self, tmp_path: Path):
        """Test that torch.Tensor segmentation maps work correctly."""
        import torch

        image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        seg_map = torch.zeros((64, 64), dtype=torch.int32)
        seg_map[10:30, 10:30] = 1

        predictions = {
            "segmentation": seg_map,
            "segments_info": [{"id": 1, "label_id": 0, "score": 0.85}],
        }
        fig = visualize_predictions(image, predictions)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestVisualizeAnnotations:
    def test_visualize_annotations(self, synthetic_coco_json: Path, synthetic_image_dir: Path, tmp_path: Path):
        """Test visualize_annotations with synthetic COCO data."""
        output_dir = tmp_path / "vis_output"
        visualize_annotations(
            annotations_path=synthetic_coco_json,
            image_dir=synthetic_image_dir,
            output_dir=output_dir,
            num_images=3,
        )
        assert output_dir.exists()
        # Should have created visualization files
        vis_files = list(output_dir.glob("ann_*.png"))
        assert len(vis_files) == 3

    def test_visualize_annotations_all_images(self, synthetic_coco_json: Path, synthetic_image_dir: Path, tmp_path: Path):
        """Test with num_images larger than dataset."""
        output_dir = tmp_path / "vis_output"
        visualize_annotations(
            annotations_path=synthetic_coco_json,
            image_dir=synthetic_image_dir,
            output_dir=output_dir,
            num_images=100,  # More than available
        )
        assert output_dir.exists()
        vis_files = list(output_dir.glob("ann_*.png"))
        assert len(vis_files) == 5  # Only 5 images in synthetic dataset


class TestCreateComparisonGrid:
    def test_creates_grid_file(self, tmp_path: Path):
        """Test that create_comparison_grid saves a file."""
        images = [
            Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            for _ in range(2)
        ]

        # Two models, each with predictions for 2 images
        seg_map1 = np.zeros((64, 64), dtype=np.int32)
        seg_map1[10:30, 10:30] = 1
        seg_map2 = np.zeros((64, 64), dtype=np.int32)
        seg_map2[20:40, 20:40] = 1

        preds_model1 = [
            {"segmentation": seg_map1, "segments_info": [{"id": 1, "label_id": 0, "score": 0.9}]},
            {"segmentation": seg_map2, "segments_info": [{"id": 1, "label_id": 1, "score": 0.8}]},
        ]
        preds_model2 = [
            {"segmentation": seg_map2, "segments_info": [{"id": 1, "label_id": 2, "score": 0.7}]},
            {"segmentation": seg_map1, "segments_info": [{"id": 1, "label_id": 3, "score": 0.6}]},
        ]

        output_path = tmp_path / "grid.png"
        create_comparison_grid(
            images=images,
            predictions_list=[preds_model1, preds_model2],
            names=["Model A", "Model B"],
            output_path=output_path,
        )
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_single_image_two_models(self, tmp_path: Path):
        """Test grid with a single image and two models."""
        images = [Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))]
        seg_map = np.zeros((64, 64), dtype=np.int32)
        seg_map[5:15, 5:15] = 1
        preds_a = [
            {"segmentation": seg_map, "segments_info": [{"id": 1, "label_id": 0, "score": 0.9}]},
        ]
        preds_b = [
            {"segmentation": seg_map.copy(), "segments_info": [{"id": 1, "label_id": 1, "score": 0.8}]},
        ]

        output_path = tmp_path / "grid_1img_2models.png"
        create_comparison_grid(
            images=images,
            predictions_list=[preds_a, preds_b],
            names=["Model A", "Model B"],
            output_path=output_path,
        )
        assert output_path.exists()

    def test_grid_with_torch_tensors(self, tmp_path: Path):
        """Test grid handles torch tensor segmentation maps."""
        import torch

        images = [
            Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)),
            Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)),
        ]
        seg_map = torch.zeros((64, 64), dtype=torch.int32)
        seg_map[5:15, 5:15] = 1
        preds = [
            {"segmentation": seg_map, "segments_info": [{"id": 1, "label_id": 0, "score": 0.9}]},
            {"segmentation": seg_map.clone(), "segments_info": [{"id": 1, "label_id": 1, "score": 0.8}]},
        ]

        output_path = tmp_path / "grid_torch.png"
        create_comparison_grid(
            images=images,
            predictions_list=[preds],
            names=["Torch Model"],
            output_path=output_path,
        )
        assert output_path.exists()


class TestEmptyMetrics:
    def test_empty_metrics(self):
        metrics = _empty_metrics()
        assert metrics["AP"] == 0.0
        assert metrics["AP50"] == 0.0
        assert "AP_rigid_plastic" in metrics
        assert "AP_soft_plastic" in metrics
        assert "AP_cardboard" in metrics
        assert "AP_metal" in metrics
        assert metrics["AR"] == 0.0

    def test_empty_metrics_all_zero(self):
        metrics = _empty_metrics()
        for val in metrics.values():
            assert val == 0.0


class TestMasksToRle:
    def test_single_mask(self):
        """Test RLE encoding of a single binary mask."""
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:30, 10:30] = 1
        masks = mask[np.newaxis, ...]  # Shape: (1, 64, 64)

        rles = _masks_to_rle(masks)
        assert len(rles) == 1
        assert "counts" in rles[0]
        assert "size" in rles[0]
        assert isinstance(rles[0]["counts"], str)
        assert rles[0]["size"] == [64, 64]

    def test_multiple_masks(self):
        """Test RLE encoding of multiple binary masks."""
        masks = np.zeros((3, 64, 64), dtype=np.uint8)
        masks[0, 5:15, 5:15] = 1
        masks[1, 20:40, 20:40] = 1
        masks[2, 50:60, 50:60] = 1

        rles = _masks_to_rle(masks)
        assert len(rles) == 3
        for rle in rles:
            assert "counts" in rle
            assert isinstance(rle["counts"], str)

    def test_empty_mask(self):
        """Test RLE encoding of an all-zeros mask."""
        masks = np.zeros((1, 64, 64), dtype=np.uint8)
        rles = _masks_to_rle(masks)
        assert len(rles) == 1
        assert "counts" in rles[0]

    def test_full_mask(self):
        """Test RLE encoding of an all-ones mask."""
        masks = np.ones((1, 32, 32), dtype=np.uint8)
        rles = _masks_to_rle(masks)
        assert len(rles) == 1
        assert rles[0]["size"] == [32, 32]

    def test_rle_decodable(self):
        """Test that encoded RLE can be decoded back to the original mask."""
        from pycocotools import mask as mask_util

        original = np.zeros((64, 64), dtype=np.uint8)
        original[10:30, 10:30] = 1
        masks = original[np.newaxis, ...]

        rles = _masks_to_rle(masks)
        # Re-encode counts to bytes for decoding
        rle_for_decode = {
            "counts": rles[0]["counts"].encode("utf-8"),
            "size": rles[0]["size"],
        }
        decoded = mask_util.decode(rle_for_decode)
        np.testing.assert_array_equal(decoded, original)


class TestEvaluateFromResults:
    def _make_coco_gt_json(self, tmp_path: Path) -> Path:
        """Create a minimal COCO ground-truth JSON."""
        from pycocotools import mask as mask_util

        # Create a ground truth with one image and one annotation
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:30, 10:30] = 1
        rle = mask_util.encode(np.asfortranarray(mask))
        area = float(mask_util.area(rle))
        bbox = mask_util.toBbox(rle).tolist()

        gt = {
            "images": [
                {"id": 1, "file_name": "img.png", "width": 64, "height": 64},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 0,
                    "segmentation": {
                        "counts": rle["counts"].decode("utf-8"),
                        "size": list(rle["size"]),
                    },
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                },
            ],
            "categories": [
                {"id": 0, "name": "rigid_plastic"},
                {"id": 1, "name": "cardboard"},
                {"id": 2, "name": "metal"},
                {"id": 3, "name": "soft_plastic"},
            ],
        }
        gt_path = tmp_path / "gt.json"
        with open(gt_path, "w") as f:
            json.dump(gt, f)
        return gt_path

    def test_with_empty_results(self, tmp_path: Path):
        """Test evaluate_from_results with empty results returns zero metrics."""
        gt_path = self._make_coco_gt_json(tmp_path)
        results_path = tmp_path / "results.json"
        with open(results_path, "w") as f:
            json.dump([], f)

        metrics = evaluate_from_results(gt_path, results_path)
        assert metrics["AP"] == 0.0
        assert metrics["AP50"] == 0.0
        assert "AP_rigid_plastic" in metrics

    def test_with_matching_prediction(self, tmp_path: Path):
        """Test evaluate_from_results with a prediction matching the GT."""
        from pycocotools import mask as mask_util

        gt_path = self._make_coco_gt_json(tmp_path)

        # Create a prediction that matches the GT exactly
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:30, 10:30] = 1
        rle = mask_util.encode(np.asfortranarray(mask))

        results = [
            {
                "image_id": 1,
                "category_id": 0,
                "segmentation": {
                    "counts": rle["counts"].decode("utf-8"),
                    "size": list(rle["size"]),
                },
                "score": 0.99,
            },
        ]
        results_path = tmp_path / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f)

        metrics = evaluate_from_results(gt_path, results_path)
        # Perfect prediction should yield high AP
        assert metrics["AP"] > 0.0
        assert metrics["AP50"] > 0.0
        assert isinstance(metrics["AR"], float)
        # Per-class AP should exist
        assert "AP_rigid_plastic" in metrics

    def test_with_wrong_category_prediction(self, tmp_path: Path):
        """Test evaluate_from_results with wrong category prediction."""
        from pycocotools import mask as mask_util

        gt_path = self._make_coco_gt_json(tmp_path)

        # Predict the right mask but wrong category
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:30, 10:30] = 1
        rle = mask_util.encode(np.asfortranarray(mask))

        results = [
            {
                "image_id": 1,
                "category_id": 2,  # metal, but GT is rigid_plastic
                "segmentation": {
                    "counts": rle["counts"].decode("utf-8"),
                    "size": list(rle["size"]),
                },
                "score": 0.99,
            },
        ]
        results_path = tmp_path / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f)

        metrics = evaluate_from_results(gt_path, results_path)
        # Wrong category should yield low AP for rigid_plastic
        assert metrics["AP_rigid_plastic"] == pytest.approx(-1.0) or metrics["AP_rigid_plastic"] == 0.0 or metrics["AP_rigid_plastic"] == -1.0
