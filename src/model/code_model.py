"""
MINDI 1.5 Vision-Coder — Code Generation Model

Loads the base coding model with LoRA adapters for fine-tuning
on Next.js + Tailwind + TypeScript code generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


class MindiCodeModel:
    """Base coding model with LoRA for MINDI 1.5 fine-tuning."""

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        load_in_4bit: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir or Path("./checkpoints/base")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.load_in_4bit = load_in_4bit
        self.model: Optional[AutoModelForCausalLM] = None
        self.peft_model: Optional[PeftModel] = None

    def load_base_model(self) -> AutoModelForCausalLM:
        """Load the base model with optional 4-bit quantization."""
        quantization_config = None
        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=str(self.cache_dir),
            torch_dtype=torch.bfloat16,
            device_map="auto" if self.device == "cuda" else None,
            quantization_config=quantization_config,
            trust_remote_code=True,
        )
        return self.model

    def apply_lora(
        self,
        rank: int = 64,
        alpha: int = 128,
        dropout: float = 0.05,
        target_modules: Optional[list[str]] = None,
    ) -> PeftModel:
        """Apply LoRA adapters to the base model for efficient fine-tuning."""
        if self.model is None:
            raise RuntimeError("Base model not loaded. Call load_base_model() first.")

        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.peft_model = get_peft_model(self.model, lora_config)
        trainable, total = self._count_parameters()
        print(f"[MindiCodeModel] LoRA applied — trainable: {trainable:,} / {total:,} "
              f"({100 * trainable / total:.2f}%)")
        return self.peft_model

    def _count_parameters(self) -> tuple[int, int]:
        """Count trainable and total parameters."""
        model = self.peft_model or self.model
        if model is None:
            return 0, 0
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        return trainable, total

    def save_adapter(self, output_dir: Optional[Path] = None) -> Path:
        """Save the LoRA adapter weights."""
        if self.peft_model is None:
            raise RuntimeError("No LoRA adapter to save. Call apply_lora() first.")
        save_path = output_dir or Path("./checkpoints/finetuned")
        save_path.mkdir(parents=True, exist_ok=True)
        self.peft_model.save_pretrained(str(save_path))
        return save_path

    def load_adapter(self, adapter_dir: Path) -> PeftModel:
        """Load a saved LoRA adapter onto the base model."""
        if self.model is None:
            self.load_base_model()
        self.peft_model = PeftModel.from_pretrained(
            self.model, str(adapter_dir)
        )
        return self.peft_model

    def resize_embeddings(self, new_vocab_size: int) -> None:
        """Resize model embeddings to accommodate new special tokens."""
        model = self.peft_model or self.model
        if model is None:
            raise RuntimeError("No model loaded.")
        model.resize_token_embeddings(new_vocab_size)
