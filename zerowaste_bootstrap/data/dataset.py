"""COCO dataset adapter for ZeroWaste instance segmentation."""

import copy
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from transformers import Mask2FormerImageProcessor


class ZeroWasteDataset(Dataset):
    """PyTorch dataset wrapping a COCO-format ZeroWaste split.

    Each item is preprocessed by a ``Mask2FormerImageProcessor`` so it can be
    fed directly into a Mask2Former model.

    Args:
        root_dir: Directory containing ``data/`` (images) and ``labels.json``.
        processor: A ``Mask2FormerImageProcessor`` instance used to convert
            raw images + segmentation maps into model-ready tensors.
    """

    def __init__(self, root_dir: Path, processor: Mask2FormerImageProcessor) -> None:
        self.root_dir = Path(root_dir)
        self.processor = processor

        annotation_path = self.root_dir / "labels.json"
        self.coco = COCO(str(annotation_path))
        self.image_ids: list[int] = sorted(self.coco.getImgIds())

    @classmethod
    def from_merged(
        cls,
        coco_json: Path,
        image_dirs: list[Path],
        processor: Mask2FormerImageProcessor,
    ) -> "ZeroWasteDataset":
        """Create a dataset from a merged COCO JSON with multiple image dirs.

        Images are looked up in each directory in order until found.
        """
        instance = cls.__new__(cls)
        instance.processor = processor
        instance.root_dir = coco_json.parent
        instance.coco = COCO(str(coco_json))
        instance.image_ids = sorted(instance.coco.getImgIds())
        instance._image_dirs = [Path(d) for d in image_dirs]
        return instance

    def _resolve_image_path(self, file_name: str) -> Path:
        """Find image file across configured image directories."""
        if hasattr(self, "_image_dirs"):
            for d in self._image_dirs:
                candidate = d / file_name
                if candidate.exists():
                    return candidate
        return self.root_dir / "data" / file_name

    def load_image(self, idx: int) -> Image.Image:
        """Load raw PIL image by dataset index (no preprocessing)."""
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = self._resolve_image_path(image_info["file_name"])
        return Image.open(image_path).convert("RGB")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]

        # 1. Load the PIL image
        image_path = self._resolve_image_path(image_info["file_name"])
        image = Image.open(image_path).convert("RGB")

        # 2. Get all annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=[image_id])
        annotations = self.coco.loadAnns(ann_ids)

        # 3. Build instance segmentation map (H, W) where each pixel holds a
        #    unique instance ID.  Background is 0.
        height, width = image_info["height"], image_info["width"]
        instance_seg_map = np.zeros((height, width), dtype=np.int32)

        # 4. Build instance_id -> semantic (category) id mapping.
        #    Instance 0 is background -- map it to the processor's ignore_index
        #    so the model learns to ignore it.
        ignore_index = self.processor.ignore_index if self.processor.ignore_index is not None else 255
        instance_id_to_semantic_id: dict[int, int] = {0: ignore_index}

        for instance_id_offset, ann in enumerate(annotations, start=1):
            mask = self.coco.annToMask(ann)  # binary (H, W)
            instance_seg_map[mask == 1] = instance_id_offset
            instance_id_to_semantic_id[instance_id_offset] = ann["category_id"]

        # 5. Preprocess with the Mask2Former processor
        inputs = self.processor(
            images=[image],
            segmentation_maps=[instance_seg_map],
            instance_id_to_semantic_id=[instance_id_to_semantic_id],
            return_tensors="pt",
        )

        # 6. Squeeze the batch dimension added by the processor
        result: dict[str, torch.Tensor] = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                result[key] = value.squeeze(0)
            elif isinstance(value, list) and len(value) == 1:
                result[key] = value[0]
            else:
                result[key] = value
        return result


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    """Custom collate for Mask2Former batches.

    ``pixel_values`` are padded to the same spatial size and stacked.
    ``mask_labels`` and ``class_labels`` are kept as lists because
    Mask2Former expects variable-length labels per image.
    """
    # Pad pixel_values to the largest spatial dimensions in the batch
    pixel_values = [item["pixel_values"] for item in batch]
    max_h = max(pv.shape[-2] for pv in pixel_values)
    max_w = max(pv.shape[-1] for pv in pixel_values)

    padded_pixel_values: list[torch.Tensor] = []
    padded_pixel_masks: list[torch.Tensor] = []

    for pv in pixel_values:
        c, h, w = pv.shape
        padded = torch.zeros(c, max_h, max_w, dtype=pv.dtype)
        padded[:, :h, :w] = pv
        padded_pixel_values.append(padded)

        # Pixel mask: 1 where real image, 0 where padding
        mask = torch.zeros(max_h, max_w, dtype=torch.int64)
        mask[:h, :w] = 1
        padded_pixel_masks.append(mask)

    return {
        "pixel_values": torch.stack(padded_pixel_values),
        "pixel_mask": torch.stack(padded_pixel_masks),
        "mask_labels": [item["mask_labels"] for item in batch],
        "class_labels": [item["class_labels"] for item in batch],
    }


def get_dataloaders(
    data_dir: Path,
    processor: Mask2FormerImageProcessor,
    batch_size: int,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """Create train / val / test ``DataLoader`` instances.

    Expected directory layout::

        data_dir/
          zerowaste-f/
            train/
              data/
              labels.json
            val/
              data/
              labels.json
            test/
              data/
              labels.json

    Args:
        data_dir: Root directory containing the ``zerowaste-f`` folder.
        processor: Image processor for Mask2Former.
        batch_size: Batch size for all loaders.
        num_workers: Number of data-loading workers.

    Returns:
        Dict mapping split name to its ``DataLoader``.
    """
    base = Path(data_dir) / "zerowaste-f"

    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        split_dir = base / split
        dataset = ZeroWasteDataset(root_dir=split_dir, processor=processor)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
    return loaders


def merge_coco_jsons(json_paths: list[Path]) -> dict:
    """Merge multiple COCO JSON files into one.

    Image and annotation IDs are remapped to avoid collisions.  Category
    definitions are taken from the first JSON and assumed to be consistent
    across all files.

    Args:
        json_paths: Paths to COCO JSON files.

    Returns:
        A single merged COCO-format dict.

    Raises:
        ValueError: If ``json_paths`` is empty.
    """
    if not json_paths:
        raise ValueError("json_paths must not be empty")

    merged_images: list[dict] = []
    merged_annotations: list[dict] = []
    categories: list[dict] | None = None

    next_image_id = 0
    next_ann_id = 0

    for path in json_paths:
        with open(path) as f:
            coco_data = json.load(f)

        if categories is None:
            categories = copy.deepcopy(coco_data["categories"])

        # Build old -> new image ID mapping
        old_to_new_img_id: dict[int, int] = {}
        for img in coco_data["images"]:
            new_id = next_image_id
            old_to_new_img_id[img["id"]] = new_id

            remapped_img = copy.deepcopy(img)
            remapped_img["id"] = new_id
            merged_images.append(remapped_img)
            next_image_id += 1

        for ann in coco_data["annotations"]:
            remapped_ann = copy.deepcopy(ann)
            remapped_ann["id"] = next_ann_id
            remapped_ann["image_id"] = old_to_new_img_id[ann["image_id"]]
            merged_annotations.append(remapped_ann)
            next_ann_id += 1

    return {
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": categories,
    }
