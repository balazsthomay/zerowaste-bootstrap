"""Tests for CLI entry points via typer CliRunner."""

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from zerowaste_bootstrap.cli import app

runner = CliRunner()


class TestCLIHelp:
    """Test that all CLI commands respond to --help."""

    def test_main_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Bootstrap waste instance segmentation" in result.output

    def test_download_help(self):
        result = runner.invoke(app, ["download", "--help"])
        assert result.exit_code == 0
        assert "Download the ZeroWaste dataset" in result.output

    def test_train_help(self):
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        assert "Train a Mask2Former model" in result.output

    def test_evaluate_help(self):
        result = runner.invoke(app, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "Evaluate a trained model" in result.output

    def test_pseudo_label_help(self):
        result = runner.invoke(app, ["pseudo-label", "--help"])
        assert result.exit_code == 0
        assert "Generate pseudo-labels" in result.output

    def test_augment_help(self):
        result = runner.invoke(app, ["augment", "--help"])
        assert result.exit_code == 0
        assert "Generate synthetic training data" in result.output

    def test_compare_help(self):
        result = runner.invoke(app, ["compare", "--help"])
        assert result.exit_code == 0
        assert "Compare results across experiments" in result.output

    def test_visualize_help(self):
        result = runner.invoke(app, ["visualize", "--help"])
        assert result.exit_code == 0
        assert "Visualize annotations" in result.output


class TestCompareCommand:
    """Test the compare CLI command with real data on disk."""

    def test_compare_with_metrics_files(self, tmp_path: Path):
        """Test compare command reads metrics files and produces output."""
        # Create experiment directories with metrics
        baseline_dir = tmp_path / "baseline"
        baseline_dir.mkdir()
        baseline_metrics = {"AP": 0.35, "AP50": 0.55, "AP75": 0.30, "AR": 0.40}
        with open(baseline_dir / "metrics_test.json", "w") as f:
            json.dump(baseline_metrics, f)

        pseudo_dir = tmp_path / "pseudo"
        pseudo_dir.mkdir()
        pseudo_metrics = {"AP": 0.40, "AP50": 0.60, "AP75": 0.35, "AR": 0.45}
        with open(pseudo_dir / "metrics_test.json", "w") as f:
            json.dump(pseudo_metrics, f)

        result = runner.invoke(
            app,
            [
                "compare",
                "--experiments", "baseline,pseudo",
                "--output-dir", str(tmp_path),
            ],
        )
        assert result.exit_code == 0

        # Check that comparison files were generated
        assert (tmp_path / "comparison.md").exists()
        assert (tmp_path / "comparison.json").exists()

        # Check content
        with open(tmp_path / "comparison.json") as f:
            data = json.load(f)
        assert "baseline" in data
        assert "pseudo" in data

    def test_compare_no_experiments_found(self, tmp_path: Path):
        """Test compare command gracefully handles missing experiment data."""
        result = runner.invoke(
            app,
            [
                "compare",
                "--experiments", "nonexistent",
                "--output-dir", str(tmp_path),
            ],
        )
        # Should not crash, just log a warning
        assert result.exit_code == 0

    def test_compare_partial_experiments(self, tmp_path: Path):
        """Test compare when only some experiments have metrics."""
        baseline_dir = tmp_path / "baseline"
        baseline_dir.mkdir()
        with open(baseline_dir / "metrics_test.json", "w") as f:
            json.dump({"AP": 0.35, "AP50": 0.55, "AP75": 0.30, "AR": 0.40}, f)

        result = runner.invoke(
            app,
            [
                "compare",
                "--experiments", "baseline,nonexistent",
                "--output-dir", str(tmp_path),
            ],
        )
        assert result.exit_code == 0
        assert (tmp_path / "comparison.md").exists()


class TestDownloadCommand:
    """Test download command with mocked download function."""

    @patch("zerowaste_bootstrap.data.download.download_zerowaste")
    def test_download_invokes_function(self, mock_download, tmp_path: Path):
        result = runner.invoke(
            app,
            ["download", "--data-dir", str(tmp_path)],
        )
        assert result.exit_code == 0
        mock_download.assert_called_once_with(tmp_path)


class TestTrainCommand:
    """Test train command with mocked training function."""

    @patch("zerowaste_bootstrap.modeling.trainer.train")
    def test_train_invokes_function(self, mock_train, tmp_path: Path):
        mock_train.return_value = tmp_path / "best"
        result = runner.invoke(
            app,
            [
                "train",
                "--data-dir", str(tmp_path),
                "--output-dir", str(tmp_path / "output"),
                "--epochs", "1",
                "--batch-size", "2",
                "--device", "cpu",
            ],
        )
        assert result.exit_code == 0
        mock_train.assert_called_once()


class TestEvaluateCommand:
    """Test evaluate command with mocked evaluate function."""

    @patch("zerowaste_bootstrap.evaluation.metrics.evaluate_model_cli")
    def test_evaluate_invokes_function(self, mock_eval, tmp_path: Path):
        mock_eval.return_value = {"AP": 0.5}
        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()
        result = runner.invoke(
            app,
            [
                "evaluate",
                "--checkpoint", str(checkpoint),
                "--data-dir", str(tmp_path),
                "--device", "cpu",
            ],
        )
        assert result.exit_code == 0
        mock_eval.assert_called_once()


class TestPseudoLabelCommand:
    """Test pseudo-label command with mocked functions."""

    @patch("zerowaste_bootstrap.pseudo_label.filter.filter_pseudo_labels")
    @patch("zerowaste_bootstrap.pseudo_label.generate.generate_pseudo_labels")
    def test_pseudo_label_invokes_functions(
        self, mock_generate, mock_filter, tmp_path: Path
    ):
        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [
                "pseudo-label",
                "--checkpoint", str(checkpoint),
                "--data-dir", str(tmp_path),
                "--output-dir", str(output_dir),
                "--device", "cpu",
            ],
        )
        assert result.exit_code == 0
        mock_generate.assert_called_once()
        mock_filter.assert_called_once()


class TestAugmentCommand:
    """Test augment command with mocked functions."""

    @patch("zerowaste_bootstrap.data.augmentation.generate_synthetic_images")
    @patch("zerowaste_bootstrap.data.augmentation.build_object_bank")
    def test_augment_invokes_functions(
        self, mock_bank, mock_synth, tmp_path: Path
    ):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [
                "augment",
                "--object-sources", str(source_dir),
                "--output-dir", str(output_dir),
                "--num-synthetic", "10",
            ],
        )
        assert result.exit_code == 0
        mock_bank.assert_called_once()
        mock_synth.assert_called_once()


class TestVisualizeCommand:
    """Test visualize command with mocked function."""

    @patch("zerowaste_bootstrap.evaluation.visualize.visualize_annotations")
    def test_visualize_invokes_function(self, mock_vis, tmp_path: Path):
        ann_path = tmp_path / "labels.json"
        ann_path.touch()
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        result = runner.invoke(
            app,
            [
                "visualize",
                "--annotations", str(ann_path),
                "--data-dir", str(img_dir),
                "--output-dir", str(tmp_path / "vis"),
                "--num", "5",
            ],
        )
        assert result.exit_code == 0
        mock_vis.assert_called_once()
