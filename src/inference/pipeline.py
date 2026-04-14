"""
MINDI 1.5 Vision-Coder — Inference Pipeline

End-to-end inference: takes a user prompt, runs through the agent
pipeline, and returns generated Next.js + Tailwind + TypeScript code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer


class InferencePipeline:
    """Inference pipeline for MINDI 1.5 code generation."""

    def __init__(
        self,
        model: Optional[object] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        device: Optional[str] = None,
        max_new_tokens: int = 4096,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
    ) -> str:
        """Generate code from a user prompt."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before inference.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=False)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: Path,
        base_model_name: str = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    ) -> "InferencePipeline":
        """Load an inference pipeline from a saved checkpoint."""
        from src.model.code_model import MindiCodeModel

        model_wrapper = MindiCodeModel(model_name=base_model_name)
        model_wrapper.load_base_model()
        model_wrapper.load_adapter(checkpoint_dir)

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, trust_remote_code=True
        )

        return cls(
            model=model_wrapper.peft_model,
            tokenizer=tokenizer,
        )
