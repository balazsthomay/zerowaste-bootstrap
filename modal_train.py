"""Modal app for training Mask2Former on ZeroWaste data.

Usage:
    modal run modal_train.py --action download       # download data (~$0.01)
    modal run modal_train.py --action smoke-test     # verify pipeline (~$0.05)
    modal run modal_train.py --action train          # baseline training (~$2)
"""

import modal

# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------
volume = modal.Volume.from_name("zerowaste-data", create_if_missing=True)
DATA_DIR = "/data"

ZENODO_BASE = "https://zenodo.org/api/records/6412647/files"
ZIP_PASSWORD = "UP#1VuX409z4"

# Labeled data: single zip, extracts to splits_final_deblurred/
LABELED_ZIP = "zerowaste-f-final.zip"
# Unlabeled data: multi-part zip (must download all parts, extract from .zip)
UNLABELED_PARTS = [
    "zerowaste-s-parts.z01",
    "zerowaste-s-parts.z02",
    "zerowaste-s-parts.z03",
    "zerowaste-s-parts.zip",  # final part, also the extraction entry point
]

# The labeled zip extracts to this directory name
LABELED_EXTRACTED_NAME = "splits_final_deblurred"
# We symlink/rename to this for our pipeline
LABELED_DIR_NAME = "zerowaste-f"

# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("unzip", "wget", "curl", "p7zip-full")
)

training_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .uv_pip_install(
        "torch>=2.10.0",
        "torchvision>=0.25.0",
        "transformers>=5.3.0",
        "accelerate>=1.13.0",
        "pycocotools>=2.0.11",
        "pydantic>=2.12.5",
        "pydantic-settings>=2.13.1",
        "numpy>=2.4.3",
        "pillow>=12.1.1",
        "scipy>=1.17.1",
        "tqdm>=4.67.3",
        "matplotlib>=3.10.8",
        "requests>=2.32.5",
        "typer>=0.24.1",
    )
    .add_local_python_source("zerowaste_bootstrap")
)

app = modal.App("zerowaste-bootstrap")

# ---------------------------------------------------------------------------
# Data download (CPU-only, ~$0.05/hr)
# ---------------------------------------------------------------------------
@app.function(
    image=download_image,
    volumes={DATA_DIR: volume},
    timeout=60 * 60 * 3,
)
def download():
    """Download ZeroWaste dataset from Zenodo into Modal Volume."""
    import os
    import pathlib
    import shutil
    import subprocess

    marker = pathlib.Path(f"{DATA_DIR}/.download_complete")
    if marker.exists():
        print("Dataset already downloaded. Skipping.")
        _print_data_summary()
        return

    # --- Labeled data (single zip) ---
    labeled_dest = pathlib.Path(DATA_DIR) / LABELED_DIR_NAME
    if not labeled_dest.exists():
        # Check if already extracted under original name
        extracted = pathlib.Path(DATA_DIR) / LABELED_EXTRACTED_NAME
        if extracted.exists():
            print(f"Renaming {LABELED_EXTRACTED_NAME} -> {LABELED_DIR_NAME}")
            shutil.copytree(extracted, labeled_dest, dirs_exist_ok=True)
            shutil.rmtree(extracted)
        else:
            tmp_zip = pathlib.Path(f"/tmp/{LABELED_ZIP}")
            url = f"{ZENODO_BASE}/{LABELED_ZIP}/content"

            print(f"Downloading {LABELED_ZIP}...")
            subprocess.run(
                ["wget", "--progress=dot:giga", url, "-O", str(tmp_zip)],
                check=True,
            )
            print(f"Downloaded ({tmp_zip.stat().st_size / 1e9:.1f} GB)")

            tmp_extract = pathlib.Path("/tmp/extracted_labeled")
            tmp_extract.mkdir(exist_ok=True)
            print("Extracting labeled data...")
            subprocess.run(
                ["unzip", "-o", "-P", ZIP_PASSWORD, str(tmp_zip), "-d", str(tmp_extract)],
                check=True,
            )
            tmp_zip.unlink()

            # The zip extracts to splits_final_deblurred/, rename to zerowaste-f/
            src = tmp_extract / LABELED_EXTRACTED_NAME
            if src.exists():
                shutil.copytree(src, labeled_dest, dirs_exist_ok=True)
            else:
                # Fallback: copy whatever was extracted
                for item in tmp_extract.iterdir():
                    dest = labeled_dest / item.name if item.is_dir() else labeled_dest
                    if item.is_dir():
                        shutil.copytree(item, dest, dirs_exist_ok=True)
                    else:
                        labeled_dest.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(item, dest)

            shutil.rmtree(tmp_extract)

        volume.commit()
        print("Labeled data ready.")
    else:
        print("Labeled data already present.")

    # --- Unlabeled data (multi-part zip) ---
    unlabeled_dest = pathlib.Path(DATA_DIR) / "zerowaste-s"
    if not unlabeled_dest.exists():
        print("Downloading unlabeled data (multi-part zip)...")
        for part in UNLABELED_PARTS:
            tmp_part = pathlib.Path(f"/tmp/{part}")
            if tmp_part.exists():
                print(f"  {part} already cached")
                continue
            url = f"{ZENODO_BASE}/{part}/content"
            print(f"  Downloading {part}...")
            subprocess.run(
                ["wget", "--progress=dot:giga", url, "-O", str(tmp_part)],
                check=True,
            )

        # Extract multi-part zip (use the .zip file as entry point)
        tmp_extract = pathlib.Path("/tmp/extracted_unlabeled")
        tmp_extract.mkdir(exist_ok=True)
        zip_entry = pathlib.Path(f"/tmp/{UNLABELED_PARTS[-1]}")  # the .zip file
        print("Extracting unlabeled data...")
        subprocess.run(
            ["unzip", "-o", "-P", ZIP_PASSWORD, str(zip_entry), "-d", str(tmp_extract)],
            check=True,
        )

        # Clean up zip parts
        for part in UNLABELED_PARTS:
            pathlib.Path(f"/tmp/{part}").unlink(missing_ok=True)

        # Copy to volume
        unlabeled_dest.mkdir(parents=True, exist_ok=True)
        for item in tmp_extract.iterdir():
            dest = unlabeled_dest / item.name if item.is_dir() else unlabeled_dest
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)

        shutil.rmtree(tmp_extract)
        volume.commit()
        print("Unlabeled data ready.")
    else:
        print("Unlabeled data already present.")

    marker.touch()
    volume.commit()
    print("Download complete!")
    _print_data_summary()


def _print_data_summary():
    """Print dataset structure summary."""
    import os

    for split in ("train", "val", "test"):
        split_dir = f"{DATA_DIR}/{LABELED_DIR_NAME}/{split}"
        if os.path.isdir(split_dir):
            data_dir = f"{split_dir}/data"
            n_images = len(os.listdir(data_dir)) if os.path.isdir(data_dir) else 0
            has_labels = os.path.isfile(f"{split_dir}/labels.json")
            print(f"  {split}: {n_images} images, labels.json={'yes' if has_labels else 'MISSING'}")

    unlabeled = f"{DATA_DIR}/zerowaste-s"
    if os.path.isdir(unlabeled):
        n = 0
        for _, _, files in os.walk(unlabeled):
            n += sum(1 for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg')))
        print(f"  unlabeled (zerowaste-s): {n} images")


# ---------------------------------------------------------------------------
# Training (L4 GPU, ~$0.80/hr)
# ---------------------------------------------------------------------------
@app.function(
    image=training_image,
    gpu="L4",
    volumes={DATA_DIR: volume},
    timeout=60 * 60 * 12,
)
def train(
    experiment: str = "baseline",
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-5,
    smoke_test: bool = False,
):
    """Train Mask2Former on ZeroWaste."""
    import torch

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    from zerowaste_bootstrap.config import DataConfig, TrainConfig
    from zerowaste_bootstrap.modeling.trainer import train as run_training

    data_config = DataConfig(data_dir=DATA_DIR)
    train_config = TrainConfig(
        learning_rate=lr,
        batch_size=batch_size,
        epochs=epochs,
        device="cuda",
        output_dir=f"{DATA_DIR}/output/{experiment}",
        smoke_test=smoke_test,
        fp16=True,
        num_workers=2,
    )

    best_dir = run_training(
        train_config=train_config,
        data_config=data_config,
    )

    volume.commit()
    print(f"Training complete! Best model at: {best_dir}")
    return str(best_dir)


# ---------------------------------------------------------------------------
# Evaluation (L4 GPU)
# ---------------------------------------------------------------------------
@app.function(
    image=training_image,
    gpu="L4",
    volumes={DATA_DIR: volume},
    timeout=60 * 60 * 2,
)
def evaluate(
    experiment: str = "baseline",
    split: str = "test",
):
    """Evaluate a trained model."""
    from pathlib import Path

    from zerowaste_bootstrap.evaluation.metrics import evaluate_model_cli

    checkpoint = Path(f"{DATA_DIR}/output/{experiment}/best")
    output_dir = Path(f"{DATA_DIR}/output/{experiment}/eval")

    metrics = evaluate_model_cli(
        checkpoint=checkpoint,
        data_dir=Path(DATA_DIR),
        split=split,
        output_dir=output_dir,
        device="cuda",
        visualize=True,
        num_vis=20,
    )

    volume.commit()
    print(f"Evaluation complete: {metrics}")
    return metrics


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    action: str = "train",
    experiment: str = "baseline",
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-5,
    smoke_test: bool = False,
    split: str = "test",
):
    """Orchestrate download, training, and evaluation.

    Actions: download, train, evaluate, smoke-test
    """
    if action == "download":
        download.remote()

    elif action == "train":
        result = train.remote(
            experiment=experiment,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            smoke_test=smoke_test,
        )
        print(f"Best model: {result}")

    elif action == "evaluate":
        metrics = evaluate.remote(experiment=experiment, split=split)
        print(f"Metrics: {metrics}")

    elif action == "smoke-test":
        print("Running smoke test (2 epochs, 10 samples)...")
        result = train.remote(
            experiment="smoke-test",
            epochs=2,
            batch_size=4,
            lr=lr,
            smoke_test=True,
        )
        print(f"Smoke test complete: {result}")

    else:
        print(f"Unknown action: {action}. Use: download, train, evaluate, smoke-test")
