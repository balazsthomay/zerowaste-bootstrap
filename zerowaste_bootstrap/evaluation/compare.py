"""Compare experiment results."""

import json
import logging
from pathlib import Path

from zerowaste_bootstrap.config import ZEROWASTE_CLASSES

logger = logging.getLogger(__name__)


def compare_experiments(results: dict[str, dict]) -> str:
    """Generate a markdown comparison table from experiment results.

    Args:
        results: Mapping of experiment name to metrics dict.

    Returns:
        Markdown-formatted table string.
    """
    if not results:
        return "No results to compare."

    # Determine baseline for delta computation
    baseline_name = next(iter(results))
    baseline = results[baseline_name]

    # Build header
    metric_keys = ["AP", "AP50", "AP75", "AR"] + [
        f"AP_{name}" for name in ZEROWASTE_CLASSES.values()
    ]

    lines = []
    lines.append("# Experiment Comparison\n")

    # Table header
    header = "| Metric |"
    separator = "|--------|"
    for exp_name in results:
        header += f" {exp_name} |"
        separator += "--------|"
        if exp_name != baseline_name:
            header += " delta |"
            separator += "--------|"

    lines.append(header)
    lines.append(separator)

    # Table rows
    for key in metric_keys:
        row = f"| {key} |"
        base_val = baseline.get(key, 0.0)
        for exp_name, metrics in results.items():
            val = metrics.get(key, 0.0)
            row += f" {val:.4f} |"
            if exp_name != baseline_name:
                delta = val - base_val
                sign = "+" if delta >= 0 else ""
                row += f" {sign}{delta:.4f} |"
        lines.append(row)

    return "\n".join(lines)


def compare_experiments_cli(
    experiment_names: list[str],
    output_dir: Path,
) -> None:
    """CLI entry point for experiment comparison."""
    results = {}
    for name in experiment_names:
        metrics_path = output_dir / name / "metrics_test.json"
        if not metrics_path.exists():
            # Try eval subdirectory
            metrics_path = output_dir / "eval" / f"metrics_test_{name}.json"
        if not metrics_path.exists():
            logger.warning("No metrics found for experiment '%s'", name)
            continue
        with open(metrics_path) as f:
            results[name] = json.load(f)

    if not results:
        logger.error("No experiment results found")
        return

    table = compare_experiments(results)

    # Save markdown
    md_path = output_dir / "comparison.md"
    with open(md_path, "w") as f:
        f.write(table)
    logger.info("Comparison table saved to %s", md_path)

    # Save raw JSON
    json_path = output_dir / "comparison.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Raw results saved to %s", json_path)

    print(table)
