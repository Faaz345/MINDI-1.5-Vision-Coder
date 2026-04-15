#!/usr/bin/env python3
"""
MINDI 1.5 Vision-Coder — Master Training Script

Usage:
    python scripts/train.py --phase 1              # Run phase 1 only
    python scripts/train.py --phase all             # Run all 3 phases
    python scripts/train.py --phase 2 --resume checkpoints/training/phase1_lora_step5000
    python scripts/train.py --dry_run               # Test 10 steps only
    python scripts/train.py --push_to_hub           # Upload after training

Handles Ctrl+C gracefully: saves checkpoint before exit.
"""

from __future__ import annotations

import argparse
import signal
import sys
import traceback
from pathlib import Path

# Resolve project root (scripts/ is one level deep)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MINDI 1.5 Vision-Coder — Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase", type=str, default="all",
        choices=["1", "2", "3", "all"],
        help="Which phase(s) to run: 1, 2, 3, or all (default: all)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint directory to resume from",
    )
    parser.add_argument(
        "--config", type=str,
        default=str(PROJECT_ROOT / "configs" / "training_config.yaml"),
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Test run: only 10 steps per phase",
    )
    parser.add_argument(
        "--push_to_hub", action="store_true",
        help="Push checkpoints to HuggingFace after each phase",
    )
    parser.add_argument(
        "--no_wandb", action="store_true",
        help="Disable WandB logging",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load and return the training config YAML."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_training_config(raw: dict, dry_run: bool = False):
    """Build TrainingConfig from parsed YAML."""
    from src.training.mindi_trainer import PhaseConfig, TrainingConfig

    training = raw.get("training", {})
    data = raw.get("data", {})
    output = raw.get("output", {})
    logging_cfg = raw.get("logging", {})
    model_cfg = raw.get("model", {})

    # Build phase configs from YAML
    phases = []
    phase_defs = [
        ("phase1", "phase1_lora", True, False, False, "text"),
        ("phase2", "phase2_vision_bridge", False, True, True, "vision"),
        ("phase3", "phase3_all", True, True, True, "mixed"),
    ]
    cumulative_step = 0
    for key, name, lora, vision, fusion, data_type in phase_defs:
        pcfg = training.get(key, {})
        steps = pcfg.get("steps", 2500)
        if dry_run:
            steps = 10
        start = cumulative_step
        end = cumulative_step + steps
        phases.append(PhaseConfig(
            name=name,
            start_step=start,
            end_step=end,
            learning_rate=float(pcfg.get("lr", 2e-4)),
            batch_size=pcfg.get("batch_size", 8),
            gradient_accumulation_steps=training.get("grad_accumulation", 4),
            lora=lora,
            vision_projection=vision,
            fusion=fusion,
            data_type=data_type,
        ))
        cumulative_step = end

    config = TrainingConfig(
        train_file=PROJECT_ROOT / data.get("train_file", "data/processed/train.jsonl"),
        val_file=PROJECT_ROOT / data.get("val_file", "data/processed/val.jsonl"),
        vision_train_file=PROJECT_ROOT / data.get("vision_train_file", "data/websight/train.jsonl"),
        vision_val_file=PROJECT_ROOT / data.get("vision_val_file", "data/websight/val.jsonl"),
        output_dir=PROJECT_ROOT / output.get("checkpoint_dir", "checkpoints/training"),
        log_dir=PROJECT_ROOT / logging_cfg.get("log_dir", "logs/training"),
        max_seq_length=data.get("max_length", 4096),
        use_compile=model_cfg.get("use_compile", False),
        gradient_checkpointing=model_cfg.get("gradient_checkpointing", True),
        dtype=model_cfg.get("dtype", "bf16"),
        num_workers=data.get("num_workers", 4),
        pin_memory=True,
        prefetch_factor=2,
        weight_decay=0.01,
        warmup_ratio=0.03,
        max_grad_norm=float(training.get("max_grad_norm", 1.0)),
        seed=42,
        log_every_n_steps=logging_cfg.get("log_every", 10),
        eval_every_n_steps=training.get("eval_every", 250),
        save_every_n_steps=training.get("save_every", 500),
        phases=phases,
    )

    if dry_run:
        config.eval_every_n_steps = 5
        config.save_every_n_steps = 10
        config.log_every_n_steps = 1

    return config


def init_wandb(raw_config: dict, phase: str, disabled: bool = False):
    """Initialize WandB logging."""
    if disabled:
        return None
    try:
        import wandb
        logging_cfg = raw_config.get("logging", {})
        run = wandb.init(
            project=logging_cfg.get("wandb_project", "mindi-1.5-vision-coder"),
            entity=logging_cfg.get("wandb_entity", "mindigenous"),
            name=f"mindi15-{phase}",
            config=raw_config,
            tags=["mindi-1.5", "training", f"phase-{phase}"],
            reinit=True,
        )
        print(f"[train.py] WandB initialized: {run.url}")
        return run
    except ImportError:
        print("[train.py] WandB not installed — logging disabled")
        return None
    except Exception as e:
        print(f"[train.py] WandB init failed: {e} — continuing without logging")
        return None


def push_checkpoint_to_hub(checkpoint_dir: Path, raw_config: dict) -> None:
    """Push a checkpoint to HuggingFace Hub."""
    output = raw_config.get("output", {})
    repo_id = output.get("hf_repo", "Mindigenous/MINDI-1.5-Vision-Coder")

    try:
        from huggingface_hub import HfApi
        import os
        api = HfApi(token=os.environ.get("HF_TOKEN"))

        print(f"[train.py] Pushing checkpoint to {repo_id} ...")
        api.upload_folder(
            folder_path=str(checkpoint_dir),
            repo_id=repo_id,
            path_in_repo=f"checkpoints/{checkpoint_dir.name}",
            repo_type="model",
        )
        print(f"[train.py] Pushed to https://huggingface.co/{repo_id}")
    except ImportError:
        print("[train.py] huggingface_hub not installed — skipping push")
    except Exception as e:
        print(f"[train.py] Push to hub failed: {e}")


def log_wandb_phase_complete(wandb_run, summary: dict) -> None:
    """Log phase completion to WandB."""
    if wandb_run is None:
        return
    try:
        import wandb
        wandb.log({
            "phase_complete": True,
            "phase": summary.get("phase", "unknown"),
            "total_steps": summary.get("total_steps", 0),
            "best_val_loss": summary.get("best_val_loss", 0),
            "elapsed_minutes": summary.get("elapsed_minutes", 0),
        })
    except Exception:
        pass


def main() -> None:
    args = parse_args()

    print()
    print("=" * 60)
    print("  MINDI 1.5 Vision-Coder — Training Launch")
    print("  MINDIGENOUS.AI")
    print("=" * 60)
    print()
    print(f"  Phase:       {args.phase}")
    print(f"  Config:      {args.config}")
    print(f"  Resume:      {args.resume or 'None'}")
    print(f"  Dry run:     {args.dry_run}")
    print(f"  Push to hub: {args.push_to_hub}")
    print(f"  Device:      {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"  GPU:         {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  VRAM:        {vram_gb:.1f} GB")
    print()

    # Load config
    raw_config = load_config(args.config)
    config = build_training_config(raw_config, dry_run=args.dry_run)

    # Filter phases based on --phase arg
    if args.phase != "all":
        phase_idx = int(args.phase) - 1
        if phase_idx < 0 or phase_idx >= len(config.phases):
            print(f"ERROR: Invalid phase {args.phase}. Available: 1-{len(config.phases)}")
            sys.exit(1)
        selected_phase = config.phases[phase_idx]
        # Adjust to start from 0 for single-phase run
        step_count = selected_phase.end_step - selected_phase.start_step
        selected_phase.start_step = 0
        selected_phase.end_step = step_count
        config.phases = [selected_phase]

    # Initialize model
    print("[train.py] Initializing MINDI 1.5 model ...")
    from src.model.mindi_model import MINDI15
    model_cfg = raw_config.get("model", {})
    vision_cfg = raw_config.get("vision", {})

    model = MINDI15(
        model_name=model_cfg.get("name", "Qwen/Qwen2.5-Coder-7B-Instruct"),
        clip_model=vision_cfg.get("clip_model", "openai/clip-vit-large-patch14"),
        hidden_size=model_cfg.get("hidden_size", 3584),
        num_visual_tokens=vision_cfg.get("visual_tokens", 256),
        torch_dtype=config.torch_dtype,
    )

    # Initialize trainer
    from src.training.mindi_trainer import MINDITrainer
    trainer = MINDITrainer(model=model, config=config)

    # Resume from checkpoint
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = PROJECT_ROOT / resume_path
        trainer.resume_from_checkpoint(resume_path)

    # Initialize WandB
    wandb_run = init_wandb(raw_config, args.phase, disabled=args.no_wandb)

    # Graceful Ctrl+C handler
    interrupted = False

    def signal_handler(sig, frame):
        nonlocal interrupted
        if interrupted:
            print("\n[train.py] Forced exit!")
            sys.exit(1)
        interrupted = True
        print("\n[train.py] Ctrl+C received — saving checkpoint before exit ...")
        try:
            emergency_dir = config.output_dir / "emergency_checkpoint"
            emergency_dir.mkdir(parents=True, exist_ok=True)
            model.save(emergency_dir)
            print(f"[train.py] Emergency checkpoint saved: {emergency_dir}")
        except Exception as e:
            print(f"[train.py] Emergency save failed: {e}")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Run training
    try:
        if args.phase == "all":
            summary = trainer.train()

            final_dir = config.output_dir / "final"
            if args.push_to_hub:
                push_checkpoint_to_hub(final_dir, raw_config)
            log_wandb_phase_complete(wandb_run, summary)

        else:
            phase = config.phases[0]
            summary = trainer.train_phase(phase)

            ckpt_dir = config.output_dir / f"{phase.name}_step{phase.end_step}"
            if args.push_to_hub:
                push_checkpoint_to_hub(ckpt_dir, raw_config)
            log_wandb_phase_complete(wandb_run, summary)

    except KeyboardInterrupt:
        signal_handler(None, None)
    except Exception as e:
        print(f"\n[train.py] ERROR: {e}")
        traceback.print_exc()
        try:
            crash_dir = config.output_dir / "crash_checkpoint"
            crash_dir.mkdir(parents=True, exist_ok=True)
            model.save(crash_dir)
            print(f"[train.py] Crash checkpoint saved: {crash_dir}")
        except Exception:
            pass
        sys.exit(1)
    finally:
        if wandb_run is not None:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass

    # Final summary
    hf_repo = raw_config.get("output", {}).get("hf_repo", "Mindigenous/MINDI-1.5-Vision-Coder")
    print()
    print("=" * 60)
    print("  Training complete!")
    print(f"  Best val loss:  {trainer.best_val_loss:.4f}")
    print(f"  Checkpoint at:  {config.output_dir}")
    if args.push_to_hub:
        print(f"  HuggingFace:    https://huggingface.co/{hf_repo}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
