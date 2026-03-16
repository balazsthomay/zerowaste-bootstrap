"""Training loop for Mask2Former on ZeroWaste."""

import json
import logging
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments

from zerowaste_bootstrap.config import DataConfig, TrainConfig
from zerowaste_bootstrap.modeling.model import get_device, load_model, load_processor

logger = logging.getLogger(__name__)


def _build_training_args(config: TrainConfig, device: str) -> TrainingArguments:
    """Build HuggingFace TrainingArguments from our config."""
    epochs = config.smoke_test_epochs if config.smoke_test else config.epochs

    # Determine mixed precision
    fp16 = False
    bf16 = False
    if device == "cuda":
        fp16 = config.fp16 or True  # default to fp16 on CUDA
    elif device == "mps":
        # MPS supports bf16 but not fp16 in newer torch
        bf16 = config.bf16

    import os

    os.environ.setdefault(
        "TENSORBOARD_LOGGING_DIR", str(config.output_dir / "logs")
    )

    return TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        fp16=fp16,
        bf16=bf16,
        eval_strategy=config.eval_strategy,
        save_strategy=config.eval_strategy,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=config.logging_steps,
        dataloader_num_workers=config.num_workers,
        remove_unused_columns=False,
        report_to="tensorboard",
    )


class Mask2FormerCollator:
    """Custom data collator for Mask2Former that handles variable-size masks."""

    def __call__(self, batch: list[dict]) -> dict:
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        pixel_mask = torch.stack([item["pixel_mask"] for item in batch])

        mask_labels = [item["mask_labels"] for item in batch]
        class_labels = [item["class_labels"] for item in batch]

        return {
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask,
            "mask_labels": mask_labels,
            "class_labels": class_labels,
        }


def train(
    train_config: TrainConfig,
    data_config: DataConfig,
    pseudo_labels_path: Path | None = None,
    synthetic_data_path: Path | None = None,
) -> Path:
    """Train Mask2Former on ZeroWaste data.

    Returns the path to the best checkpoint directory.
    """
    device = get_device(train_config.device)
    logger.info("Using device: %s", device)

    # Load model and processor
    processor = load_processor()
    model = load_model()

    # Import here to avoid circular imports
    from zerowaste_bootstrap.data.dataset import ZeroWasteDataset
    from zerowaste_bootstrap.data.dataset import merge_coco_jsons

    # Build dataset paths
    labeled_dir = data_config.data_dir / "zerowaste-f"

    # Determine training data sources
    train_json_paths = [labeled_dir / data_config.train_split / "labels.json"]
    train_image_dirs = [labeled_dir / data_config.train_split / "data"]

    if pseudo_labels_path is not None:
        logger.info("Including pseudo-labels from %s", pseudo_labels_path)
        train_json_paths.append(pseudo_labels_path)
        # Pseudo-label images come from zerowaste-s
        train_image_dirs.append(data_config.data_dir / "zerowaste-s" / "data")

    if synthetic_data_path is not None:
        logger.info("Including synthetic data from %s", synthetic_data_path)
        train_json_paths.append(synthetic_data_path / "annotations.json")
        train_image_dirs.append(synthetic_data_path / "images")

    # If merging multiple sources, create a merged JSON
    if len(train_json_paths) > 1:
        merged = merge_coco_jsons(train_json_paths)
        merged_path = train_config.output_dir / "merged_train.json"
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        with open(merged_path, "w") as f:
            json.dump(merged, f)
        # For merged datasets, we need a combined image lookup
        train_dataset = ZeroWasteDataset.from_merged(
            coco_json=merged_path,
            image_dirs=train_image_dirs,
            processor=processor,
        )
    else:
        train_dir = labeled_dir / data_config.train_split
        train_dataset = ZeroWasteDataset(root_dir=train_dir, processor=processor)

    val_dir = labeled_dir / data_config.val_split
    val_dataset = ZeroWasteDataset(root_dir=val_dir, processor=processor)

    # Smoke test: limit dataset size
    if train_config.smoke_test:
        n = train_config.smoke_test_samples
        logger.info("Smoke test: limiting to %d samples", n)
        train_dataset = torch.utils.data.Subset(train_dataset, range(min(n, len(train_dataset))))
        val_dataset = torch.utils.data.Subset(val_dataset, range(min(n, len(val_dataset))))

    logger.info("Train: %d samples, Val: %d samples", len(train_dataset), len(val_dataset))

    # Training arguments
    training_args = _build_training_args(train_config, device)
    train_config.output_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=Mask2FormerCollator(),
    )

    logger.info("Starting training for %d epochs", training_args.num_train_epochs)
    trainer.train()

    # Save best model
    best_dir = train_config.output_dir / "best"
    trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))
    logger.info("Best model saved to %s", best_dir)

    return best_dir
