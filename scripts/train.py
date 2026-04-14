"""
MINDI 1.5 Vision-Coder — Training Launch Script

Entry point for starting LoRA fine-tuning.
Loads config, initializes model + dataset, and runs training.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    """Parse args and launch training."""
    parser = argparse.ArgumentParser(description="MINDI 1.5 — Launch LoRA Training")
    parser.add_argument(
        "--config", type=str, default="./configs/training_config.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--local", action="store_true", default=True,
        help="Use local GPU overrides (RTX 4060 mode)",
    )
    parser.add_argument(
        "--cloud", action="store_true",
        help="Use cloud GPU settings (MI300X mode)",
    )
    args = parser.parse_args()

    local_mode = not args.cloud
    config_path = Path(args.config)

    print(f"[MINDI Training] Config: {config_path}")
    print(f"[MINDI Training] Mode: {'local (RTX 4060)' if local_mode else 'cloud (MI300X)'}")
    print("[MINDI Training] Pipeline will be wired after Phase 3 setup.")


if __name__ == "__main__":
    main()
