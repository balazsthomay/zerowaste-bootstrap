"""Configuration via Pydantic settings."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


ZEROWASTE_CLASSES = {
    1: "rigid_plastic",
    2: "cardboard",
    3: "metal",
    4: "soft_plastic",
}

LABEL2ID = {v: k for k, v in ZEROWASTE_CLASSES.items()}
NUM_CLASSES = len(ZEROWASTE_CLASSES)


class DataConfig(BaseSettings):
    """Data-related configuration."""

    model_config = {"env_prefix": "ZEROWASTE_"}

    data_dir: Path = Field(default=Path("data"), description="Root data directory")
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"


class TrainConfig(BaseSettings):
    """Training configuration."""

    model_config = {"env_prefix": "ZEROWASTE_TRAIN_"}

    learning_rate: float = 1e-5
    batch_size: int = 8
    epochs: int = 50
    device: str = "auto"
    output_dir: Path = Field(
        default=Path("output"), description="Directory for checkpoints and logs"
    )
    smoke_test: bool = False
    smoke_test_samples: int = 10
    smoke_test_epochs: int = 2
    num_workers: int = 2
    fp16: bool = True
    bf16: bool = False
    save_total_limit: int = 3
    eval_strategy: str = "epoch"
    logging_steps: int = 10


class PseudoLabelConfig(BaseSettings):
    """Pseudo-labeling configuration."""

    model_config = {"env_prefix": "ZEROWASTE_PSEUDO_"}

    confidence_threshold: float = 0.7
    min_mask_area: int = 100
    batch_size: int = 4


class AugmentConfig(BaseSettings):
    """Cut-paste augmentation configuration."""

    model_config = {"env_prefix": "ZEROWASTE_AUGMENT_"}

    num_synthetic: int = 1000
    objects_per_image_mean: float = 5.0
    scale_range: tuple[float, float] = (0.7, 1.3)
    rotation_range: tuple[float, float] = (-15.0, 15.0)
    hflip_prob: float = 0.5
    brightness_range: tuple[float, float] = (0.9, 1.1)
    edge_sigma: float = 2.0
    seed: int = 42
