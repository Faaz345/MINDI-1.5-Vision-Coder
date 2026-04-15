"""
MINDI 1.5 Vision-Coder — Model Architecture

Loads Qwen/Qwen2.5-Coder-7B-Instruct with LoRA adapters.
Handles model initialization, LoRA application, save/load,
and parameter counting for the base LLM component.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


class MINDIArchitecture:
    """Qwen2.5-Coder-7B-Instruct with LoRA for MINDI 1.5 fine-tuning."""

    DEFAULT_TARGET_MODULES: list[str] = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """
        Initialize the architecture wrapper.

        Args:
            model_name: HuggingFace model identifier.
            device: Target device ('cuda', 'cpu', or None for auto).
            cache_dir: Local directory for model weight cache.
            torch_dtype: Data type for model weights.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./checkpoints/base")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.torch_dtype = torch_dtype

        self.model: Optional[AutoModelForCausalLM] = None
        self.peft_model: Optional[PeftModel] = None
        self.tokenizer: Optional[AutoTokenizer] = None

        self._load_model()

    def _load_model(self) -> None:
        """Load the base model and tokenizer from HuggingFace or cache."""
        print(f"[MINDIArchitecture] Loading {self.model_name} ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=str(self.cache_dir),
            torch_dtype=self.torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=str(self.cache_dir),
            trust_remote_code=True,
        )
        print(f"[MINDIArchitecture] Loaded on {self.device} "
              f"({self._fmt_params(self._total_params())} params)")

    def apply_lora(
        self,
        r: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        target_modules: Optional[list[str]] = None,
    ) -> PeftModel:
        """
        Apply LoRA adapters to the base model.

        Args:
            r: LoRA rank.
            lora_alpha: LoRA scaling factor.
            lora_dropout: Dropout probability for LoRA layers.
            target_modules: List of module names to apply LoRA to.

        Returns:
            The PEFT-wrapped model.
        """
        if self.model is None:
            raise RuntimeError("Base model not loaded.")

        if target_modules is None:
            target_modules = self.DEFAULT_TARGET_MODULES

        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.peft_model = get_peft_model(self.model, lora_config)

        info = self.get_trainable_params()
        print(f"[MINDIArchitecture] LoRA applied (r={r}, alpha={lora_alpha})")
        print(f"  Trainable:  {info['trainable']:>14,}  ({info['trainable_pct']:.2f}%)")
        print(f"  Frozen:     {info['frozen']:>14,}")
        print(f"  Total:      {info['total']:>14,}")

        return self.peft_model

    def get_trainable_params(self) -> dict:
        """
        Count trainable, frozen, and total parameters.

        Returns:
            Dictionary with 'trainable', 'frozen', 'total', 'trainable_pct'.
        """
        model = self.peft_model or self.model
        if model is None:
            return {"trainable": 0, "frozen": 0, "total": 0, "trainable_pct": 0.0}

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        frozen = total - trainable
        pct = 100.0 * trainable / total if total > 0 else 0.0

        return {
            "trainable": trainable,
            "frozen": frozen,
            "total": total,
            "trainable_pct": round(pct, 4),
        }

    def print_model_info(self) -> None:
        """Print detailed model architecture and parameter information."""
        model = self.peft_model or self.model
        if model is None:
            print("[MINDIArchitecture] No model loaded.")
            return

        info = self.get_trainable_params()
        print()
        print("=" * 60)
        print("  MINDI 1.5 — Model Architecture Info")
        print("=" * 60)
        print(f"  Base model:     {self.model_name}")
        print(f"  Device:         {self.device}")
        print(f"  Dtype:          {self.torch_dtype}")
        print(f"  LoRA active:    {self.peft_model is not None}")
        print(f"  Total params:   {self._fmt_params(info['total'])}")
        print(f"  Trainable:      {self._fmt_params(info['trainable'])} "
              f"({info['trainable_pct']:.2f}%)")
        print(f"  Frozen:         {self._fmt_params(info['frozen'])}")

        if self.peft_model is not None:
            config = self.peft_model.peft_config.get("default")
            if config is not None:
                print(f"  LoRA rank:      {config.r}")
                print(f"  LoRA alpha:     {config.lora_alpha}")
                print(f"  LoRA dropout:   {config.lora_dropout}")
                print(f"  Target modules: {config.target_modules}")
        print("=" * 60)
        print()

    def save_lora(self, path: Optional[Path] = None) -> Path:
        """
        Save LoRA adapter weights to disk.

        Args:
            path: Directory to save to. Defaults to checkpoints/lora.

        Returns:
            Path where weights were saved.
        """
        if self.peft_model is None:
            raise RuntimeError("No LoRA adapter to save. Call apply_lora() first.")

        save_path = Path(path) if path else Path("./checkpoints/lora")
        save_path.mkdir(parents=True, exist_ok=True)
        self.peft_model.save_pretrained(str(save_path))
        print(f"[MINDIArchitecture] LoRA saved to {save_path}")
        return save_path

    def load_lora(self, path: Path) -> PeftModel:
        """
        Load LoRA adapter weights from disk.

        Args:
            path: Directory containing saved adapter weights.

        Returns:
            The PEFT-wrapped model with loaded adapter.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"LoRA adapter not found: {path}")
        if self.model is None:
            raise RuntimeError("Base model not loaded.")

        self.peft_model = PeftModel.from_pretrained(
            self.model, str(path)
        )
        print(f"[MINDIArchitecture] LoRA loaded from {path}")
        return self.peft_model

    def resize_embeddings(self, new_vocab_size: int) -> None:
        """Resize model embeddings for new special tokens."""
        model = self.peft_model or self.model
        if model is None:
            raise RuntimeError("No model loaded.")
        old_size = model.get_input_embeddings().weight.shape[0]
        if new_vocab_size != old_size:
            model.resize_token_embeddings(new_vocab_size)
            print(f"[MINDIArchitecture] Resized embeddings: {old_size} → {new_vocab_size}")

    def get_model(self) -> AutoModelForCausalLM | PeftModel:
        """Return the active model (PEFT if LoRA applied, else base)."""
        model = self.peft_model or self.model
        if model is None:
            raise RuntimeError("No model loaded.")
        return model

    # ── helpers ───────────────────────────────────────────────────
    def _total_params(self) -> int:
        model = self.peft_model or self.model
        if model is None:
            return 0
        return sum(p.numel() for p in model.parameters())

    @staticmethod
    def _fmt_params(n: int) -> str:
        if n >= 1_000_000_000:
            return f"{n / 1_000_000_000:.2f}B"
        if n >= 1_000_000:
            return f"{n / 1_000_000:.2f}M"
        if n >= 1_000:
            return f"{n / 1_000:.1f}K"
        return str(n)


# ── Test block ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  MINDI 1.5 — Architecture Test")
    print("=" * 60)
    print()

    # 1. Load base model
    arch = MINDIArchitecture(
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
    )

    # 2. Apply LoRA
    peft_model = arch.apply_lora(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
    )

    # 3. Print full info
    arch.print_model_info()

    # 4. Verify trainable params
    info = arch.get_trainable_params()
    assert info["trainable"] > 0, "No trainable parameters!"
    assert info["frozen"] > info["trainable"], "More trainable than frozen — LoRA may not be applied!"

    # 5. Verify LoRA modules exist
    lora_modules = [name for name, _ in peft_model.named_parameters() if "lora_" in name]
    print(f"  LoRA modules found: {len(lora_modules)}")
    assert len(lora_modules) > 0, "No LoRA modules found!"

    # 6. Quick forward pass test (small input)
    print("\n  Running forward pass test ...")
    test_input = arch.tokenizer("Hello MINDI!", return_tensors="pt")
    test_input = {k: v.to(arch.device) for k, v in test_input.items()}
    with torch.no_grad():
        output = peft_model(**test_input)
    print(f"  Output logits shape: {output.logits.shape}")
    print(f"  Loss: {output.loss}")

    print("\n  ✓ All architecture tests passed!")
    print("=" * 60)
