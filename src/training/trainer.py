"""
MINDI 1.5 Vision-Coder — Training Pipeline

LoRA fine-tuning pipeline using Hugging Face Transformers + PEFT.
Designed to run on AMD MI300X (192GB) cloud GPU for full training,
with RTX 4060 (8GB) local overrides for development testing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from transformers import TrainingArguments


class TrainingPipeline:
    """Manages the LoRA fine-tuning pipeline for MINDI 1.5."""

    def __init__(
        self,
        config_path: Optional[Path] = None,
        local_mode: bool = True,
    ) -> None:
        self.config_path = config_path or Path("./configs/training_config.yaml")
        self.local_mode = local_mode
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """Load training configuration from YAML."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Training config not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            full_config = yaml.safe_load(f)

        config = full_config.get("training", {})

        # Apply local overrides if running on RTX 4060
        if self.local_mode and "local_overrides" in config:
            overrides = config.pop("local_overrides")
            config.update(overrides)
            print("[TrainingPipeline] Applied local GPU overrides (RTX 4060 mode)")

        return config

    def build_training_args(self, output_dir: Optional[Path] = None) -> TrainingArguments:
        """Build HuggingFace TrainingArguments from config."""
        output = output_dir or Path("./checkpoints/finetuned")
        output.mkdir(parents=True, exist_ok=True)

        return TrainingArguments(
            output_dir=str(output),
            num_train_epochs=self.config.get("epochs", 3),
            per_device_train_batch_size=self.config.get("batch_size", 1),
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 16),
            learning_rate=self.config.get("learning_rate", 2e-4),
            weight_decay=self.config.get("weight_decay", 0.01),
            warmup_ratio=self.config.get("warmup_ratio", 0.03),
            lr_scheduler_type=self.config.get("lr_scheduler", "cosine"),
            max_grad_norm=self.config.get("max_grad_norm", 1.0),
            bf16=self.config.get("precision", "bf16") == "bf16",
            logging_steps=self.config.get("logging_steps", 10),
            save_strategy=self.config.get("save_strategy", "steps"),
            save_steps=self.config.get("save_steps", 500),
            save_total_limit=self.config.get("save_total_limit", 5),
            eval_strategy=self.config.get("eval_strategy", "steps"),
            eval_steps=self.config.get("eval_steps", 250),
            report_to=self.config.get("report_to", "wandb"),
            gradient_checkpointing=self.config.get("gradient_checkpointing", True),
            optim=self.config.get("optim", "adamw_torch"),
            dataloader_num_workers=2,
            remove_unused_columns=False,
        )
