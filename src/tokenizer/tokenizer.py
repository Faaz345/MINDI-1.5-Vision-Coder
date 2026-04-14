"""
MINDI 1.5 Vision-Coder — Tokenizer Wrapper

Wraps the base model tokenizer with MINDI-specific special tokens
and encoding utilities for code generation tasks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer, PreTrainedTokenizerFast


# Special tokens for MINDI's structured output format
SPECIAL_TOKENS: dict[str, str] = {
    "code_start": "<|code_start|>",
    "code_end": "<|code_end|>",
    "file_start": "<|file_start|>",
    "file_end": "<|file_end|>",
    "critique_start": "<|critique_start|>",
    "critique_end": "<|critique_end|>",
    "search_start": "<|search_start|>",
    "search_end": "<|search_end|>",
    "fix_start": "<|fix_start|>",
    "fix_end": "<|fix_end|>",
}


class MindiTokenizer:
    """Tokenizer wrapper with MINDI-specific special tokens."""

    def __init__(self, model_name: str, cache_dir: Optional[Path] = None) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir or Path("./data/tokenizer")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(self.cache_dir),
            trust_remote_code=True,
        )
        self._add_special_tokens()

    def _add_special_tokens(self) -> None:
        """Register MINDI special tokens with the tokenizer."""
        new_tokens = list(SPECIAL_TOKENS.values())
        num_added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": new_tokens}
        )
        if num_added > 0:
            print(f"[MindiTokenizer] Added {num_added} special tokens")

    @property
    def vocab_size(self) -> int:
        """Return the full vocabulary size including special tokens."""
        return len(self.tokenizer)

    def encode(self, text: str, max_length: int = 8192) -> list[int]:
        """Encode text to token IDs with truncation."""
        return self.tokenizer.encode(
            text, max_length=max_length, truncation=True
        )

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def save(self, output_dir: Optional[Path] = None) -> Path:
        """Save the tokenizer to disk."""
        save_path = output_dir or self.cache_dir / "mindi_tokenizer"
        save_path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(str(save_path))
        return save_path
