"""CLI entry point using Typer."""

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    name="zerowaste-bootstrap",
    help="Bootstrap waste instance segmentation at new MRF sites.",
)


@app.command()
def download(
    data_dir: Annotated[
        Path, typer.Option(help="Directory to download data into")
    ] = Path("data"),
) -> None:
    """Download the ZeroWaste dataset from Zenodo."""
    from zerowaste_bootstrap.data.download import download_zerowaste

    download_zerowaste(data_dir)


@app.command()
def train(
    data_dir: Annotated[Path, typer.Option(help="Data directory")] = Path("data"),
    output_dir: Annotated[Path, typer.Option(help="Output directory")] = Path(
        "output/baseline"
    ),
    epochs: Annotated[int, typer.Option(help="Number of training epochs")] = 50,
    batch_size: Annotated[int, typer.Option(help="Training batch size")] = 4,
    lr: Annotated[float, typer.Option(help="Learning rate")] = 1e-5,
    device: Annotated[str, typer.Option(help="Device: auto, cuda, mps, cpu")] = "auto",
    smoke_test: Annotated[
        bool, typer.Option("--smoke-test", help="Run minimal smoke test")
    ] = False,
    experiment: Annotated[
        str, typer.Option(help="Experiment name: baseline, pseudo, augment, both")
    ] = "baseline",
    pseudo_labels: Annotated[
        Path | None, typer.Option(help="Path to pseudo-label COCO JSON")
    ] = None,
    synthetic_data: Annotated[
        Path | None, typer.Option(help="Path to synthetic data directory")
    ] = None,
) -> None:
    """Train a Mask2Former model on ZeroWaste data."""
    from zerowaste_bootstrap.config import DataConfig, TrainConfig
    from zerowaste_bootstrap.modeling.trainer import train as run_training

    data_config = DataConfig(data_dir=data_dir)
    train_config = TrainConfig(
        learning_rate=lr,
        batch_size=batch_size,
        epochs=epochs,
        device=device,
        output_dir=output_dir / experiment,
        smoke_test=smoke_test,
    )
    run_training(
        train_config=train_config,
        data_config=data_config,
        pseudo_labels_path=pseudo_labels,
        synthetic_data_path=synthetic_data,
    )


@app.command()
def evaluate(
    checkpoint: Annotated[Path, typer.Option(help="Model checkpoint path")],
    data_dir: Annotated[Path, typer.Option(help="Data directory")] = Path("data"),
    split: Annotated[str, typer.Option(help="Dataset split to evaluate")] = "test",
    output_dir: Annotated[Path, typer.Option(help="Output directory")] = Path(
        "output/eval"
    ),
    device: Annotated[str, typer.Option(help="Device")] = "auto",
    visualize: Annotated[
        bool, typer.Option("--visualize", help="Generate visualizations")
    ] = False,
    num_vis: Annotated[
        int, typer.Option(help="Number of images to visualize")
    ] = 10,
) -> None:
    """Evaluate a trained model on a dataset split."""
    from zerowaste_bootstrap.evaluation.metrics import evaluate_model_cli

    evaluate_model_cli(
        checkpoint=checkpoint,
        data_dir=data_dir,
        split=split,
        output_dir=output_dir,
        device=device,
        visualize=visualize,
        num_vis=num_vis,
    )


@app.command()
def pseudo_label(
    checkpoint: Annotated[Path, typer.Option(help="Model checkpoint path")],
    data_dir: Annotated[Path, typer.Option(help="Unlabeled images directory")] = Path(
        "data/zerowaste-s"
    ),
    output_dir: Annotated[Path, typer.Option(help="Output directory")] = Path(
        "output/pseudo_labels"
    ),
    threshold: Annotated[
        float, typer.Option(help="Confidence threshold")
    ] = 0.7,
    min_area: Annotated[int, typer.Option(help="Minimum mask area")] = 100,
    device: Annotated[str, typer.Option(help="Device")] = "auto",
    batch_size: Annotated[int, typer.Option(help="Batch size for inference")] = 4,
) -> None:
    """Generate pseudo-labels on unlabeled data."""
    from zerowaste_bootstrap.pseudo_label.generate import generate_pseudo_labels
    from zerowaste_bootstrap.pseudo_label.filter import filter_pseudo_labels

    raw_json = output_dir / "raw.json"
    filtered_json = output_dir / "filtered.json"

    generate_pseudo_labels(
        model_path=checkpoint,
        image_dir=data_dir,
        output_json=raw_json,
        device=device,
        batch_size=batch_size,
    )
    filter_pseudo_labels(
        raw_json=raw_json,
        output_json=filtered_json,
        confidence_threshold=threshold,
        min_mask_area=min_area,
    )


@app.command()
def augment(
    object_sources: Annotated[
        Path, typer.Option(help="COCO JSON or directory with labeled data")
    ] = Path("data/zerowaste-f/train"),
    output_dir: Annotated[Path, typer.Option(help="Output directory")] = Path(
        "output/synthetic"
    ),
    num_synthetic: Annotated[
        int, typer.Option(help="Number of synthetic images")
    ] = 1000,
    visualize: Annotated[
        bool, typer.Option("--visualize", help="Visualize samples")
    ] = False,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
) -> None:
    """Generate synthetic training data via cut-paste augmentation."""
    from zerowaste_bootstrap.data.augmentation import (
        build_object_bank,
        generate_synthetic_images,
    )

    object_bank_dir = output_dir / "object_bank"
    build_object_bank(
        coco_json=object_sources / "labels.json",
        image_dir=object_sources / "data",
        output_dir=object_bank_dir,
    )
    generate_synthetic_images(
        object_bank_dir=object_bank_dir,
        background_dir=object_sources / "data",
        output_dir=output_dir,
        num_images=num_synthetic,
        seed=seed,
        visualize=visualize,
    )


@app.command()
def compare(
    experiments: Annotated[
        str, typer.Option(help="Comma-separated experiment names")
    ] = "baseline,pseudo,augment,both",
    output_dir: Annotated[Path, typer.Option(help="Output directory")] = Path(
        "output"
    ),
) -> None:
    """Compare results across experiments."""
    from zerowaste_bootstrap.evaluation.compare import compare_experiments_cli

    experiment_names = [e.strip() for e in experiments.split(",")]
    compare_experiments_cli(
        experiment_names=experiment_names,
        output_dir=output_dir,
    )


@app.command()
def visualize(
    annotations: Annotated[Path, typer.Option(help="COCO JSON annotations file")],
    data_dir: Annotated[Path, typer.Option(help="Images directory")],
    output_dir: Annotated[Path, typer.Option(help="Output directory")] = Path(
        "output/vis"
    ),
    num: Annotated[int, typer.Option(help="Number of images to visualize")] = 20,
) -> None:
    """Visualize annotations on images."""
    from zerowaste_bootstrap.evaluation.visualize import visualize_annotations

    visualize_annotations(
        annotations_path=annotations,
        image_dir=data_dir,
        output_dir=output_dir,
        num_images=num,
    )


def main() -> None:
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
