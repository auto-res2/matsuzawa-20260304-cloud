"""
Main orchestrator for prompt tuning experiments.
Handles configuration and invokes inference script.
"""

import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """
    Main entry point for experiment execution.

    Args:
        cfg: Hydra configuration
    """
    # Print configuration
    print("=" * 80)
    print("Experiment Configuration")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Validate required fields
    if "run" not in cfg or cfg.run is None:
        print(
            "ERROR: No run configuration specified. Use run=<run_id>", file=sys.stderr
        )
        sys.exit(1)

    # Apply mode-specific overrides
    apply_mode_overrides(cfg)

    # Determine task type and execute
    # This is an inference-only task (prompt tuning)
    print(f"\nStarting inference for run: {cfg.run.run_id}")
    print(f"Mode: {cfg.mode}")
    print(f"Method: {cfg.run.method.name}")
    print(f"Model: {cfg.run.model.name}")
    print(f"Dataset: {cfg.run.dataset.name}")
    print()

    # Import and run inference
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from inference import run_inference

    run_inference(cfg)

    print(f"\nExperiment completed: {cfg.run.run_id}")


def apply_mode_overrides(cfg: DictConfig):
    """
    Apply mode-specific configuration overrides.

    Args:
        cfg: Configuration to modify in-place
    """
    if cfg.mode == "sanity_check":
        print(f"Applying sanity_check mode overrides...")
        # Use smaller sample size
        # Already handled in inference.py via num_samples_sanity

        # Ensure WandB is online for sanity checks
        cfg.wandb.mode = "online"

        print(f"  - Using {cfg.run.dataset.num_samples_sanity} samples")
        print(f"  - WandB mode: {cfg.wandb.mode}")
        print(f"  - WandB project: {cfg.wandb.project}-sanity")

    elif cfg.mode == "main":
        # Full execution
        cfg.wandb.mode = "online"
        print(f"Running in main mode with {cfg.run.dataset.num_samples} samples")

    elif cfg.mode == "pilot":
        print(f"Pilot mode: using sanity_check configuration")
        cfg.wandb.mode = "online"


if __name__ == "__main__":
    main()
