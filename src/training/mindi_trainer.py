"""
MINDI 1.5 Vision-Coder — Trainer

Production-ready 3-phase training loop optimized for AMD MI300X (192GB VRAM).
Streams training data from disk (4.18GB train.jsonl) to avoid RAM exhaustion.

Phases:
    Phase 1 (steps 0–5000):     LoRA only,           LR 2e-4, batch 16
    Phase 2 (steps 5000–7500):  Vision bridge only,   LR 1e-5, batch 8
    Phase 3 (steps 7500–10000): All trainable,        LR 5e-5, batch 12

MI300X specifics:
    - ROCm presents as CUDA to PyTorch (torch.cuda.* works)
    - bf16 (NOT fp16) for AMD stability
    - torch.compile() optional (works on ROCm)
    - Gradient checkpointing enabled
    - DataLoader: num_workers=4, pin_memory=True, prefetch_factor=2
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, IterableDataset

# ── Configuration ─────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class PhaseConfig:
    """Configuration for a single training phase."""
    name: str
    start_step: int
    end_step: int
    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int = 4
    # Component toggles
    lora: bool = False
    vision_projection: bool = False
    fusion: bool = False


@dataclass
class TrainingConfig:
    """Full training configuration."""

    # Data paths
    train_file: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed" / "train.jsonl")
    val_file: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed" / "val.jsonl")

    # Output
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "checkpoints" / "training")
    log_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "logs" / "training")

    # Model
    max_seq_length: int = 8192
    use_compile: bool = False
    gradient_checkpointing: bool = True

    # Hardware (MI300X defaults)
    dtype: str = "bf16"
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2

    # Training
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    seed: int = 42

    # Logging
    log_every_n_steps: int = 10
    eval_every_n_steps: int = 250
    save_every_n_steps: int = 500

    # Phases
    phases: list[PhaseConfig] = field(default_factory=lambda: [
        PhaseConfig(
            name="phase1_lora",
            start_step=0, end_step=5000,
            learning_rate=2e-4, batch_size=16,
            lora=True, vision_projection=False, fusion=False,
        ),
        PhaseConfig(
            name="phase2_vision_bridge",
            start_step=5000, end_step=7500,
            learning_rate=1e-5, batch_size=8,
            lora=False, vision_projection=True, fusion=True,
        ),
        PhaseConfig(
            name="phase3_all",
            start_step=7500, end_step=10000,
            learning_rate=5e-5, batch_size=12,
            lora=True, vision_projection=True, fusion=True,
        ),
    ])

    @property
    def total_steps(self) -> int:
        return self.phases[-1].end_step if self.phases else 0

    @property
    def torch_dtype(self) -> torch.dtype:
        return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[self.dtype]


# ── Streaming Dataset ─────────────────────────────────────────────────

class StreamingJSONLDataset(IterableDataset):
    """
    Streams JSONL training data from disk line by line.
    Tokenizes on-the-fly to avoid loading 4+ GB into RAM.

    Expected JSONL format:
        {"id": "...", "type": "...", "source": "...",
         "messages": [{"role": "system", "content": "..."},
                      {"role": "user", "content": "..."},
                      {"role": "assistant", "content": "..."}],
         "metadata": {...}}
    """

    def __init__(
        self,
        file_path: Path,
        tokenizer: Any,
        max_length: int = 8192,
        shuffle_buffer: int = 10000,
        seed: int = 42,
    ) -> None:
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

        if not self.file_path.exists():
            raise FileNotFoundError(f"Training data not found: {self.file_path}")

    def _format_messages(self, messages: list[dict[str, str]]) -> str:
        """Format chat messages into a single training string."""
        # Use the tokenizer's chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        # Fallback: simple concatenation
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|{role}|>\n{content}")
        return "\n".join(parts)

    def _tokenize(self, text: str) -> Optional[dict[str, torch.Tensor]]:
        """Tokenize text and create training labels."""
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Labels = input_ids, with padding tokens masked as -100
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _line_iterator(self) -> Iterator[dict]:
        """Iterate over JSONL file line by line."""
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def _shuffled_iterator(self) -> Iterator[dict]:
        """Reservoir-style shuffle buffer for streaming data."""
        import random
        rng = random.Random(self.seed)
        buffer: list[dict] = []

        for item in self._line_iterator():
            buffer.append(item)
            if len(buffer) >= self.shuffle_buffer:
                rng.shuffle(buffer)
                yield from buffer
                buffer.clear()

        # Flush remaining items
        if buffer:
            rng.shuffle(buffer)
            yield from buffer

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for example in self._shuffled_iterator():
            messages = example.get("messages", [])
            if not messages:
                continue
            text = self._format_messages(messages)
            tokenized = self._tokenize(text)
            if tokenized is not None:
                yield tokenized

    def count_lines(self) -> int:
        """Count total lines (for progress estimation). Reads file once."""
        count = 0
        with open(self.file_path, "r", encoding="utf-8") as f:
            for _ in f:
                count += 1
        return count


# ── Trainer ───────────────────────────────────────────────────────────

class MINDITrainer:
    """
    3-phase trainer for MINDI 1.5 Vision-Coder.

    Optimized for AMD MI300X 192GB:
        - bf16 mixed precision
        - Gradient checkpointing
        - Streaming data from disk
        - Optional torch.compile()
        - Phase-based component freezing/unfreezing
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
    ) -> None:
        """
        Initialize the trainer.

        Args:
            model: MINDI15 model instance (already initialized).
            config: Training configuration.
        """
        self.model = model
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Create output directories
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.log_dir.mkdir(parents=True, exist_ok=True)

        # Gradient checkpointing
        if config.gradient_checkpointing:
            base_model = self.model.architecture.get_model()
            if hasattr(base_model, "gradient_checkpointing_enable"):
                base_model.gradient_checkpointing_enable()
                print("[MINDITrainer] Gradient checkpointing enabled")

        # Optional torch.compile (works on ROCm)
        if config.use_compile:
            print("[MINDITrainer] Compiling model with torch.compile() ...")
            self.model.architecture.peft_model = torch.compile(
                self.model.architecture.peft_model
            )
            print("[MINDITrainer] Compilation complete")

        # Mixed precision scaler (bf16 doesn't need GradScaler, but keep structure)
        self.use_amp = config.dtype in ("bf16", "fp16")
        self.amp_dtype = config.torch_dtype

        # Training log
        self.log_file = config.log_dir / "training_log.jsonl"
        self.metrics_history: list[dict] = []

        print(f"[MINDITrainer] Device: {self.device}")
        print(f"[MINDITrainer] Dtype: {config.dtype}")
        print(f"[MINDITrainer] Total steps: {config.total_steps}")
        print(f"[MINDITrainer] Phases: {len(config.phases)}")

    def _build_optimizer(self, phase: PhaseConfig) -> AdamW:
        """Build optimizer for the current phase (only trainable params)."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError(f"No trainable parameters in phase '{phase.name}'")
        return AdamW(
            params,
            lr=phase.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
        )

    def _build_scheduler(
        self, optimizer: AdamW, phase: PhaseConfig
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Build LR scheduler: linear warmup + cosine decay."""
        phase_steps = phase.end_step - phase.start_step
        warmup_steps = max(1, int(phase_steps * self.config.warmup_ratio))
        decay_steps = max(1, phase_steps - warmup_steps)

        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=decay_steps,
            eta_min=phase.learning_rate * 0.1,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )

    def _build_dataloader(
        self, file_path: Path, batch_size: int, shuffle_buffer: int = 10000
    ) -> DataLoader:
        """Build a streaming DataLoader."""
        dataset = StreamingJSONLDataset(
            file_path=file_path,
            tokenizer=self.model.tokenizer,
            max_length=self.config.max_seq_length,
            shuffle_buffer=shuffle_buffer,
            seed=self.config.seed,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor if self.config.num_workers > 0 else None,
            drop_last=True,
        )

    def _log_metrics(self, metrics: dict) -> None:
        """Append metrics to log file and history."""
        self.metrics_history.append(metrics)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, max_batches: int = 50) -> float:
        """
        Run validation and return average loss.

        Args:
            val_loader: Validation DataLoader.
            max_batches: Maximum batches to evaluate (for speed).

        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        count = 0

        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                result = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

            if result["loss"] is not None:
                total_loss += result["loss"].item()
                count += 1

        self.model.train()
        return total_loss / max(count, 1)

    def _save_checkpoint(self, phase_name: str, step: int, val_loss: float) -> Path:
        """Save a training checkpoint."""
        ckpt_dir = self.config.output_dir / f"{phase_name}_step{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        self.model.save(ckpt_dir)

        # Save trainer state
        state = {
            "global_step": self.global_step,
            "phase": phase_name,
            "step_in_phase": step,
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
        }
        torch.save(state, ckpt_dir / "trainer_state.pt")

        print(f"[MINDITrainer] Checkpoint saved: {ckpt_dir}")
        return ckpt_dir

    def train_phase(self, phase: PhaseConfig) -> dict:
        """
        Execute a single training phase.

        Args:
            phase: Phase configuration.

        Returns:
            Dict with phase training metrics.
        """
        print()
        print("=" * 60)
        print(f"  Phase: {phase.name}")
        print(f"  Steps: {phase.start_step} → {phase.end_step}")
        print(f"  LR: {phase.learning_rate}  |  Batch: {phase.batch_size}")
        print(f"  Components: LoRA={phase.lora}, Vision={phase.vision_projection}, "
              f"Fusion={phase.fusion}")
        print("=" * 60)

        # Set trainable components
        self.model.set_trainable_components(
            lora=phase.lora,
            vision_projection=phase.vision_projection,
            fusion=phase.fusion,
        )

        # Build optimizer and scheduler for this phase
        optimizer = self._build_optimizer(phase)
        scheduler = self._build_scheduler(optimizer, phase)

        # Build data loaders
        train_loader = self._build_dataloader(
            self.config.train_file, phase.batch_size
        )
        val_loader = self._build_dataloader(
            self.config.val_file, batch_size=max(phase.batch_size // 2, 1),
            shuffle_buffer=1000,
        )

        self.model.train()
        phase_steps = phase.end_step - phase.start_step
        step_in_phase = 0
        accum_loss = 0.0
        accum_count = 0
        phase_start_time = time.time()

        train_iter = iter(train_loader)

        while step_in_phase < phase_steps:
            # Get next batch (restart iterator if exhausted = new epoch)
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass with mixed precision
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                result = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = result["loss"]

                if loss is None:
                    continue

                # Scale loss for gradient accumulation
                loss = loss / phase.gradient_accumulation_steps

            # Backward pass
            loss.backward()
            accum_loss += loss.item() * phase.gradient_accumulation_steps
            accum_count += 1

            # Optimizer step (every gradient_accumulation_steps)
            if accum_count % phase.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.config.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                step_in_phase += 1
                self.global_step += 1
                avg_loss = accum_loss / phase.gradient_accumulation_steps
                accum_loss = 0.0

                # Logging
                if step_in_phase % self.config.log_every_n_steps == 0:
                    elapsed = time.time() - phase_start_time
                    steps_per_sec = step_in_phase / elapsed if elapsed > 0 else 0.0
                    eta_sec = (phase_steps - step_in_phase) / steps_per_sec if steps_per_sec > 0 else 0.0

                    metrics = {
                        "phase": phase.name,
                        "global_step": self.global_step,
                        "step_in_phase": step_in_phase,
                        "loss": round(avg_loss, 4),
                        "lr": optimizer.param_groups[0]["lr"],
                        "steps_per_sec": round(steps_per_sec, 3),
                        "eta_minutes": round(eta_sec / 60, 1),
                        "elapsed_minutes": round(elapsed / 60, 1),
                    }
                    self._log_metrics(metrics)
                    print(f"  [{phase.name}] step {step_in_phase}/{phase_steps} | "
                          f"loss={avg_loss:.4f} | "
                          f"lr={optimizer.param_groups[0]['lr']:.2e} | "
                          f"speed={steps_per_sec:.2f} steps/s | "
                          f"ETA={eta_sec / 60:.1f}min")

                # Evaluation
                if step_in_phase % self.config.eval_every_n_steps == 0:
                    val_loss = self.evaluate(val_loader)
                    print(f"  [{phase.name}] EVAL step {step_in_phase} | val_loss={val_loss:.4f}")
                    self._log_metrics({
                        "phase": phase.name,
                        "global_step": self.global_step,
                        "val_loss": round(val_loss, 4),
                        "type": "eval",
                    })

                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint(phase.name, step_in_phase, val_loss)
                        print(f"  [{phase.name}] New best val_loss: {val_loss:.4f}")

                # Periodic save
                if step_in_phase % self.config.save_every_n_steps == 0:
                    self._save_checkpoint(phase.name, step_in_phase, self.best_val_loss)

        # End-of-phase save
        phase_elapsed = time.time() - phase_start_time
        self._save_checkpoint(phase.name, step_in_phase, self.best_val_loss)

        phase_summary = {
            "phase": phase.name,
            "total_steps": step_in_phase,
            "elapsed_minutes": round(phase_elapsed / 60, 1),
            "best_val_loss": round(self.best_val_loss, 4),
            "type": "phase_complete",
        }
        self._log_metrics(phase_summary)
        print(f"\n  [{phase.name}] Complete — {step_in_phase} steps in "
              f"{phase_elapsed / 60:.1f} min")

        return phase_summary

    def train(self) -> dict:
        """
        Run all 3 training phases sequentially.

        Returns:
            Dict with complete training summary.
        """
        print()
        print("=" * 60)
        print("  MINDI 1.5 — Training Start")
        print(f"  Total phases: {len(self.config.phases)}")
        print(f"  Total steps:  {self.config.total_steps}")
        print(f"  Device:       {self.device}")
        print(f"  Dtype:        {self.config.dtype}")
        print(f"  Output:       {self.config.output_dir}")
        print("=" * 60)

        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        training_start = time.time()
        phase_summaries = []

        for phase in self.config.phases:
            summary = self.train_phase(phase)
            phase_summaries.append(summary)

        total_elapsed = time.time() - training_start

        # Final save
        final_dir = self.config.output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(final_dir)

        training_summary = {
            "total_steps": self.global_step,
            "total_minutes": round(total_elapsed / 60, 1),
            "best_val_loss": round(self.best_val_loss, 4),
            "phases": phase_summaries,
            "type": "training_complete",
        }
        self._log_metrics(training_summary)

        print()
        print("=" * 60)
        print("  MINDI 1.5 — Training Complete")
        print(f"  Total steps:     {self.global_step}")
        print(f"  Total time:      {total_elapsed / 60:.1f} minutes")
        print(f"  Best val loss:   {self.best_val_loss:.4f}")
        print(f"  Final saved to:  {final_dir}")
        print("=" * 60)

        return training_summary

    def resume_from_checkpoint(self, checkpoint_dir: Path) -> None:
        """
        Resume training from a checkpoint.

        Args:
            checkpoint_dir: Directory containing saved checkpoint.
        """
        checkpoint_dir = Path(checkpoint_dir)
        state_file = checkpoint_dir / "trainer_state.pt"

        if not state_file.exists():
            raise FileNotFoundError(f"Trainer state not found: {state_file}")

        # Load model weights
        self.model.load(checkpoint_dir)

        # Load trainer state
        state = torch.load(state_file, map_location=self.device, weights_only=True)
        self.global_step = state["global_step"]
        self.best_val_loss = state["best_val_loss"]

        print(f"[MINDITrainer] Resumed from step {self.global_step} "
              f"(val_loss={self.best_val_loss:.4f})")


# ── Test block ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  MINDI 1.5 — Trainer Test")
    print("=" * 60)
    print()

    # ── Test 1: Config defaults ──────────────────────────────────
    print("  Test 1: TrainingConfig defaults")
    config = TrainingConfig()
    assert config.total_steps == 10000
    assert config.dtype == "bf16"
    assert config.torch_dtype == torch.bfloat16
    assert len(config.phases) == 3
    assert config.gradient_checkpointing is True
    assert config.num_workers == 4
    assert config.pin_memory is True
    assert config.prefetch_factor == 2
    print(f"    Total steps: {config.total_steps}")
    print(f"    Dtype: {config.dtype}")
    print(f"    Phases: {[p.name for p in config.phases]}")
    print("    ✓ Config defaults correct")

    # ── Test 2: Phase configs ────────────────────────────────────
    print("\n  Test 2: Phase configurations")
    p1, p2, p3 = config.phases
    assert p1.name == "phase1_lora"
    assert p1.batch_size == 16
    assert p1.learning_rate == 2e-4
    assert p1.lora is True and p1.vision_projection is False and p1.fusion is False

    assert p2.name == "phase2_vision_bridge"
    assert p2.batch_size == 8
    assert p2.learning_rate == 1e-5
    assert p2.lora is False and p2.vision_projection is True and p2.fusion is True

    assert p3.name == "phase3_all"
    assert p3.batch_size == 12
    assert p3.learning_rate == 5e-5
    assert p3.lora is True and p3.vision_projection is True and p3.fusion is True
    print("    Phase 1: LoRA only, batch=16, lr=2e-4 ✓")
    print("    Phase 2: Vision bridge, batch=8, lr=1e-5 ✓")
    print("    Phase 3: All, batch=12, lr=5e-5 ✓")

    # ── Test 3: Streaming dataset (if data exists) ───────────────
    print("\n  Test 3: StreamingJSONLDataset")
    train_path = config.train_file
    if train_path.exists():
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            str(PROJECT_ROOT / "data" / "tokenizer" / "mindi_tokenizer"),
            trust_remote_code=True,
        )
        dataset = StreamingJSONLDataset(
            file_path=train_path,
            tokenizer=tok,
            max_length=512,  # small for test
            shuffle_buffer=100,
        )
        count = 0
        for item in dataset:
            assert "input_ids" in item
            assert "attention_mask" in item
            assert "labels" in item
            assert item["input_ids"].shape[0] == 512
            count += 1
            if count >= 5:
                break
        print(f"    Streamed {count} examples, shape={item['input_ids'].shape} ✓")
    else:
        print(f"    [SKIP] Train file not found: {train_path}")

    # ── Test 4: PhaseConfig step ranges ──────────────────────────
    print("\n  Test 4: Phase step continuity")
    for i in range(1, len(config.phases)):
        prev = config.phases[i - 1]
        curr = config.phases[i]
        assert prev.end_step == curr.start_step, \
            f"Gap between {prev.name} and {curr.name}"
    print("    All phases are contiguous ✓")

    # ── Test 5: Gradient accumulation ────────────────────────────
    print("\n  Test 5: Gradient accumulation steps")
    for phase in config.phases:
        assert phase.gradient_accumulation_steps == 4
    print("    All phases: grad_accum=4 ✓")

    print("\n  ✓ All trainer tests passed!")
    print("=" * 60)
