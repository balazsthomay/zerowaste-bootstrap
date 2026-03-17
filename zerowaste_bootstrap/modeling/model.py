"""Model loading utilities."""

import logging

from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)

from zerowaste_bootstrap.config import NUM_CLASSES, ZEROWASTE_CLASSES

logger = logging.getLogger(__name__)

MODEL_ID = "facebook/mask2former-swin-tiny-coco-instance"


def load_model(
    num_classes: int = NUM_CLASSES,
    pretrained: str = MODEL_ID,
) -> Mask2FormerForUniversalSegmentation:
    """Load Mask2Former with ZeroWaste class head.

    Loads COCO-pretrained weights and replaces the classification head
    to match the ZeroWaste category count.
    """
    logger.info("Loading model %s with %d classes", pretrained, num_classes)

    # Model uses 0-indexed class labels; ZEROWASTE_CLASSES uses COCO IDs (1-4)
    class_names = list(ZEROWASTE_CLASSES.values())
    id2label = {i: name for i, name in enumerate(class_names)}
    label2id = {name: i for i, name in enumerate(class_names)}

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    return model


def load_processor(pretrained: str = MODEL_ID) -> Mask2FormerImageProcessor:
    """Load the Mask2Former image processor."""
    return Mask2FormerImageProcessor.from_pretrained(pretrained)


def get_device(device_str: str = "auto") -> str:
    """Resolve device string to actual device.

    'auto' → cuda if available, else mps if available, else cpu.
    """
    if device_str != "auto":
        return device_str

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
