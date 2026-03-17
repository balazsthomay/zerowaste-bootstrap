"""Tests for model loading and training utilities."""

import pytest
import torch

from zerowaste_bootstrap.config import NUM_CLASSES, ZEROWASTE_CLASSES
from zerowaste_bootstrap.modeling.model import get_device, load_model, load_processor
from zerowaste_bootstrap.modeling.trainer import Mask2FormerCollator, _build_training_args
from zerowaste_bootstrap.config import TrainConfig


class TestGetDevice:
    def test_explicit_cpu(self):
        assert get_device("cpu") == "cpu"

    def test_explicit_cuda(self):
        assert get_device("cuda") == "cuda"

    def test_auto_resolves(self):
        device = get_device("auto")
        assert device in ("cuda", "mps", "cpu")


@pytest.mark.slow
class TestLoadModel:
    def test_model_loads(self):
        model = load_model()
        assert model is not None

    def test_model_num_classes(self):
        model = load_model()
        # The model's config should reflect our class count
        assert model.config.num_labels == NUM_CLASSES

    def test_model_id2label(self):
        model = load_model()
        for class_id, name in ZEROWASTE_CLASSES.items():
            assert model.config.id2label[class_id] == name

    def test_forward_pass(self):
        model = load_model()
        model.eval()
        # Random input
        pixel_values = torch.randn(1, 3, 256, 256)
        pixel_mask = torch.ones(1, 256, 256, dtype=torch.long)
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        assert hasattr(outputs, "class_queries_logits")
        assert hasattr(outputs, "masks_queries_logits")


@pytest.mark.slow
class TestLoadProcessor:
    def test_processor_loads(self):
        processor = load_processor()
        assert processor is not None


class TestMask2FormerCollator:
    def test_collate(self):
        collator = Mask2FormerCollator()
        batch = [
            {
                "pixel_values": torch.randn(3, 64, 64),
                "pixel_mask": torch.ones(64, 64, dtype=torch.long),
                "mask_labels": torch.randint(0, 2, (3, 64, 64), dtype=torch.float),
                "class_labels": torch.tensor([0, 1, 2]),
            },
            {
                "pixel_values": torch.randn(3, 64, 64),
                "pixel_mask": torch.ones(64, 64, dtype=torch.long),
                "mask_labels": torch.randint(0, 2, (2, 64, 64), dtype=torch.float),
                "class_labels": torch.tensor([1, 3]),
            },
        ]
        result = collator(batch)
        assert result["pixel_values"].shape == (2, 3, 64, 64)
        assert result["pixel_mask"].shape == (2, 64, 64)
        assert len(result["mask_labels"]) == 2
        assert len(result["class_labels"]) == 2
        # Variable number of masks per image
        assert result["mask_labels"][0].shape[0] == 3
        assert result["mask_labels"][1].shape[0] == 2


class TestBuildTrainingArgs:
    def test_default_config(self):
        config = TrainConfig(output_dir="/tmp/test_train")
        args = _build_training_args(config, device="cpu")
        assert args.num_train_epochs == 50
        assert args.per_device_train_batch_size == 8

    def test_smoke_test_config(self):
        config = TrainConfig(output_dir="/tmp/test_train", smoke_test=True)
        args = _build_training_args(config, device="cpu")
        assert args.num_train_epochs == 2

    def test_cuda_enables_fp16(self):
        config = TrainConfig(output_dir="/tmp/test_train")
        args = _build_training_args(config, device="cuda")
        assert args.fp16 is True
