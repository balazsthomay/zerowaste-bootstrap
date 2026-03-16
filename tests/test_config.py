"""Tests for configuration."""

from pathlib import Path

from zerowaste_bootstrap.config import (
    AugmentConfig,
    DataConfig,
    LABEL2ID,
    NUM_CLASSES,
    PseudoLabelConfig,
    TrainConfig,
    ZEROWASTE_CLASSES,
)


class TestConstants:
    def test_num_classes(self):
        assert NUM_CLASSES == 4

    def test_class_names(self):
        expected = {"rigid_plastic", "cardboard", "metal", "soft_plastic"}
        assert set(ZEROWASTE_CLASSES.values()) == expected

    def test_label2id_inverse(self):
        for class_id, name in ZEROWASTE_CLASSES.items():
            assert LABEL2ID[name] == class_id


class TestDataConfig:
    def test_defaults(self):
        cfg = DataConfig()
        assert cfg.data_dir == Path("data")
        assert cfg.train_split == "train"

    def test_custom(self):
        cfg = DataConfig(data_dir=Path("/tmp/test"))
        assert cfg.data_dir == Path("/tmp/test")


class TestTrainConfig:
    def test_defaults(self):
        cfg = TrainConfig()
        assert cfg.learning_rate == 1e-5
        assert cfg.batch_size == 4
        assert cfg.epochs == 50
        assert cfg.device == "auto"
        assert cfg.smoke_test is False

    def test_smoke_test(self):
        cfg = TrainConfig(smoke_test=True)
        assert cfg.smoke_test_samples == 10
        assert cfg.smoke_test_epochs == 2


class TestPseudoLabelConfig:
    def test_defaults(self):
        cfg = PseudoLabelConfig()
        assert cfg.confidence_threshold == 0.7
        assert cfg.min_mask_area == 100


class TestAugmentConfig:
    def test_defaults(self):
        cfg = AugmentConfig()
        assert cfg.num_synthetic == 1000
        assert cfg.objects_per_image_mean == 5.0
        assert cfg.scale_range == (0.7, 1.3)
        assert cfg.seed == 42
