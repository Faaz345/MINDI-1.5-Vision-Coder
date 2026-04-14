"""
MINDI 1.5 Vision-Coder — Dataset Loader

Loads and preprocesses training data from JSONL files into
tokenized format for LoRA fine-tuning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import yaml
from torch.utils.data import Dataset


class MindiDataset(Dataset):
    """Dataset for MINDI 1.5 fine-tuning data (JSONL format)."""

    def __init__(
        self,
        data_dir: Path,
        tokenizer: Any,
        max_length: int = 8192,
        split: str = "train",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.examples: list[dict[str, Any]] = []
        self._load_data()

    def _load_data(self) -> None:
        """Load all JSONL files from the data directory."""
        data_path = self.data_dir / f"{self.split}.jsonl"
        if not data_path.exists():
            print(f"[MindiDataset] No data file at {data_path} — dataset is empty")
            return

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

        print(f"[MindiDataset] Loaded {len(self.examples)} examples ({self.split})")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Tokenize and return a single training example."""
        example = self.examples[idx]

        # Expected format: {"prompt": "...", "completion": "..."}
        prompt = example.get("prompt", "")
        completion = example.get("completion", "")
        full_text = f"{prompt}\n{completion}"

        encoded = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": encoded["input_ids"].squeeze(0),
        }


def load_data_config(config_path: Optional[Path] = None) -> dict:
    """Load data configuration from YAML."""
    path = config_path or Path("./configs/data_config.yaml")
    if not path.exists():
        raise FileNotFoundError(f"Data config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f).get("dataset", {})
