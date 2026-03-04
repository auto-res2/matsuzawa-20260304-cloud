"""
Evaluation script for aggregating results across runs.
Fetches data from WandB and generates comparison plots.
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any
import wandb
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate and compare experimental runs"
    )
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Results directory"
    )
    parser.add_argument(
        "--run_ids", type=str, required=True, help="JSON list of run IDs to compare"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default="airas", help="WandB entity"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="20260304-matsuzawa-cloud",
        help="WandB project",
    )
    return parser.parse_args()


def fetch_run_data(
    api: wandb.Api, entity: str, project: str, run_id: str
) -> Dict[str, Any]:
    """
    Fetch run data from WandB by display name.

    Args:
        api: WandB API client
        entity: WandB entity
        project: WandB project
        run_id: Run display name

    Returns:
        Dictionary with run data
    """
    # Query runs by display name
    runs = api.runs(
        f"{entity}/{project}", filters={"display_name": run_id}, order="-created_at"
    )

    if len(runs) == 0:
        print(f"WARNING: No runs found for display_name={run_id}", file=sys.stderr)
        return None

    # Get most recent run
    run = runs[0]

    # Fetch history (step-by-step metrics)
    history = run.history()

    # Get summary metrics
    summary = dict(run.summary)

    # Get config
    config = dict(run.config)

    return {
        "run_id": run_id,
        "wandb_run_id": run.id,
        "history": history,
        "summary": summary,
        "config": config,
        "url": run.url,
    }


def export_per_run_metrics(run_data: Dict, results_dir: str):
    """
    Export per-run metrics to JSON.

    Args:
        run_data: Run data dictionary
        results_dir: Results directory
    """
    run_id = run_data["run_id"]
    run_dir = os.path.join(results_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Export summary metrics
    metrics = {
        "run_id": run_id,
        "summary": run_data["summary"],
        "config": run_data["config"],
        "wandb_url": run_data["url"],
    }

    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Exported metrics: {metrics_path}")

    return metrics_path


def create_per_run_figures(run_data: Dict, results_dir: str) -> List[str]:
    """
    Create per-run visualization figures.

    Args:
        run_data: Run data dictionary
        results_dir: Results directory

    Returns:
        List of created figure paths
    """
    run_id = run_data["run_id"]
    run_dir = os.path.join(results_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    history = run_data["history"]
    figure_paths = []

    # Plot accuracy over examples
    if "accuracy" in history.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(history["example_id"], history["accuracy"])
        plt.xlabel("Example ID")
        plt.ylabel("Cumulative Accuracy")
        plt.title(f"Accuracy over Examples - {run_id}")
        plt.grid(True, alpha=0.3)

        fig_path = os.path.join(run_dir, "accuracy_over_examples.pdf")
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        figure_paths.append(fig_path)
        print(f"Created figure: {fig_path}")

    # Plot arithmetic faithfulness for JAM-CoT runs
    if "arith_faithfulness_rate" in history.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(history["example_id"], history["arith_faithfulness_rate"])
        plt.xlabel("Example ID")
        plt.ylabel("Arithmetic Faithfulness Rate")
        plt.title(f"Arithmetic Faithfulness - {run_id}")
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)

        fig_path = os.path.join(run_dir, "faithfulness_over_examples.pdf")
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        figure_paths.append(fig_path)
        print(f"Created figure: {fig_path}")

    return figure_paths


def create_comparison_figures(all_run_data: List[Dict], results_dir: str) -> List[str]:
    """
    Create comparison figures across all runs.

    Args:
        all_run_data: List of run data dictionaries
        results_dir: Results directory

    Returns:
        List of created figure paths
    """
    comparison_dir = os.path.join(results_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    figure_paths = []

    # Compare accuracy trajectories
    plt.figure(figsize=(12, 7))
    for run_data in all_run_data:
        history = run_data["history"]
        if "accuracy" in history.columns:
            plt.plot(
                history["example_id"],
                history["accuracy"],
                label=run_data["run_id"],
                linewidth=2,
            )

    plt.xlabel("Example ID", fontsize=12)
    plt.ylabel("Cumulative Accuracy", fontsize=12)
    plt.title("Accuracy Comparison Across Methods", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    fig_path = os.path.join(comparison_dir, "comparison_accuracy.pdf")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    figure_paths.append(fig_path)
    print(f"Created comparison figure: {fig_path}")

    # Compare final accuracy as bar chart
    plt.figure(figsize=(10, 6))
    run_ids = [r["run_id"] for r in all_run_data]
    accuracies = [r["summary"].get("accuracy", 0) for r in all_run_data]

    colors = ["#2ecc71" if "proposed" in rid.lower() else "#3498db" for rid in run_ids]
    bars = plt.bar(range(len(run_ids)), accuracies, color=colors)

    plt.xlabel("Method", fontsize=12)
    plt.ylabel("Final Accuracy", fontsize=12)
    plt.title("Final Accuracy Comparison", fontsize=14)
    plt.xticks(range(len(run_ids)), run_ids, rotation=45, ha="right")
    plt.ylim(0, 1.0)
    plt.grid(True, axis="y", alpha=0.3)

    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig_path = os.path.join(comparison_dir, "comparison_final_accuracy.pdf")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    figure_paths.append(fig_path)
    print(f"Created comparison figure: {fig_path}")

    return figure_paths


def compute_aggregated_metrics(all_run_data: List[Dict]) -> Dict[str, Any]:
    """
    Compute aggregated comparison metrics.

    Args:
        all_run_data: List of run data dictionaries

    Returns:
        Aggregated metrics dictionary
    """
    metrics_by_run = {}

    for run_data in all_run_data:
        run_id = run_data["run_id"]
        summary = run_data["summary"]

        metrics_by_run[run_id] = {
            "accuracy": summary.get("accuracy", 0.0),
            "num_correct": summary.get("num_correct", 0),
            "num_total": summary.get("num_total", 0),
        }

        # Add JAM-CoT specific metrics
        if "avg_faithfulness_rate" in summary:
            metrics_by_run[run_id]["avg_faithfulness_rate"] = summary[
                "avg_faithfulness_rate"
            ]

    # Identify proposed and baseline runs
    proposed_runs = [rid for rid in metrics_by_run.keys() if "proposed" in rid.lower()]
    baseline_runs = [
        rid
        for rid in metrics_by_run.keys()
        if "comparative" in rid.lower() or "baseline" in rid.lower()
    ]

    # Find best of each
    best_proposed_acc = (
        max([metrics_by_run[r]["accuracy"] for r in proposed_runs])
        if proposed_runs
        else 0.0
    )
    best_baseline_acc = (
        max([metrics_by_run[r]["accuracy"] for r in baseline_runs])
        if baseline_runs
        else 0.0
    )

    best_proposed = [
        r for r in proposed_runs if metrics_by_run[r]["accuracy"] == best_proposed_acc
    ]
    best_baseline = [
        r for r in baseline_runs if metrics_by_run[r]["accuracy"] == best_baseline_acc
    ]

    gap = best_proposed_acc - best_baseline_acc

    aggregated = {
        "primary_metric": "accuracy",
        "metrics_by_run": metrics_by_run,
        "best_proposed": best_proposed[0] if best_proposed else None,
        "best_proposed_accuracy": best_proposed_acc,
        "best_baseline": best_baseline[0] if best_baseline else None,
        "best_baseline_accuracy": best_baseline_acc,
        "accuracy_gap": gap,
        "relative_improvement": (gap / best_baseline_acc * 100)
        if best_baseline_acc > 0
        else 0.0,
    }

    return aggregated


def main():
    """Main evaluation entry point."""
    args = parse_args()

    # Parse run IDs
    run_ids = json.loads(args.run_ids)
    print(f"Evaluating {len(run_ids)} runs: {run_ids}")

    # Initialize WandB API
    api = wandb.Api()

    # Fetch data for all runs
    all_run_data = []
    for run_id in run_ids:
        print(f"\nFetching data for {run_id}...")
        run_data = fetch_run_data(api, args.wandb_entity, args.wandb_project, run_id)

        if run_data is None:
            print(f"Skipping {run_id} due to missing data")
            continue

        all_run_data.append(run_data)

        # Export per-run metrics
        export_per_run_metrics(run_data, args.results_dir)

        # Create per-run figures
        create_per_run_figures(run_data, args.results_dir)

    if len(all_run_data) == 0:
        print("ERROR: No run data available for evaluation", file=sys.stderr)
        sys.exit(1)

    # Create comparison figures
    print("\nCreating comparison figures...")
    comparison_figures = create_comparison_figures(all_run_data, args.results_dir)

    # Compute aggregated metrics
    print("\nComputing aggregated metrics...")
    aggregated_metrics = compute_aggregated_metrics(all_run_data)

    # Export aggregated metrics
    comparison_dir = os.path.join(args.results_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    agg_metrics_path = os.path.join(comparison_dir, "aggregated_metrics.json")
    with open(agg_metrics_path, "w") as f:
        json.dump(aggregated_metrics, f, indent=2)

    print(f"\nExported aggregated metrics: {agg_metrics_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Primary Metric: {aggregated_metrics['primary_metric']}")
    print(
        f"Best Proposed: {aggregated_metrics['best_proposed']} ({aggregated_metrics['best_proposed_accuracy']:.4f})"
    )
    print(
        f"Best Baseline: {aggregated_metrics['best_baseline']} ({aggregated_metrics['best_baseline_accuracy']:.4f})"
    )
    print(
        f"Accuracy Gap: {aggregated_metrics['accuracy_gap']:.4f} ({aggregated_metrics['relative_improvement']:.2f}% relative)"
    )
    print("=" * 80)

    # Print all generated files
    print("\nGenerated files:")
    for run_data in all_run_data:
        run_dir = os.path.join(args.results_dir, run_data["run_id"])
        print(f"  {run_dir}/metrics.json")
        if os.path.exists(os.path.join(run_dir, "accuracy_over_examples.pdf")):
            print(f"  {run_dir}/accuracy_over_examples.pdf")
        if os.path.exists(os.path.join(run_dir, "faithfulness_over_examples.pdf")):
            print(f"  {run_dir}/faithfulness_over_examples.pdf")

    print(f"  {agg_metrics_path}")
    for fig_path in comparison_figures:
        print(f"  {fig_path}")


if __name__ == "__main__":
    main()
