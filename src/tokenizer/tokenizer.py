"""
MINDI 1.5 Vision-Coder — Tokenizer Wrapper

Wraps the MINDI tokenizer (Qwen2.5-Coder base + 22 special tokens)
with encoding utilities for code generation, conversation formatting,
and special-token-aware operations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer, PreTrainedTokenizerFast


# All 22 MINDI special tokens (pairs)
MINDI_SPECIAL_TOKENS: dict[str, str] = {
    "mindi_start": "<|mindi_start|>",
    "mindi_end": "<|mindi_end|>",
    "code_start": "<|code_start|>",
    "code_end": "<|code_end|>",
    "vision_start": "<|vision_start|>",
    "vision_end": "<|vision_end|>",
    "critique_start": "<|critique_start|>",
    "critique_end": "<|critique_end|>",
    "suggest_start": "<|suggest_start|>",
    "suggest_end": "<|suggest_end|>",
    "think_start": "<|think_start|>",
    "think_end": "<|think_end|>",
    "file_start": "<|file_start|>",
    "file_end": "<|file_end|>",
    "search_start": "<|search_start|>",
    "search_end": "<|search_end|>",
    "sandbox_start": "<|sandbox_start|>",
    "sandbox_end": "<|sandbox_end|>",
    "error_start": "<|error_start|>",
    "error_end": "<|error_end|>",
    "fix_start": "<|fix_start|>",
    "fix_end": "<|fix_end|>",
}

# Default tokenizer path (pre-built with special tokens already added)
DEFAULT_TOKENIZER_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "tokenizer" / "mindi_tokenizer"


class MindiTokenizer:
    """Tokenizer wrapper with MINDI-specific special tokens and conversation formatting."""

    def __init__(
        self,
        tokenizer_path: Optional[Path] = None,
        max_length: int = 32768,
    ) -> None:
        self.tokenizer_path = tokenizer_path or DEFAULT_TOKENIZER_PATH
        self.max_length = max_length

        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            str(self.tokenizer_path),
            trust_remote_code=True,
        )

        # Cache special token IDs for fast lookup
        self._special_token_ids: dict[str, int] = {
            name: self.tokenizer.convert_tokens_to_ids(token)
            for name, token in MINDI_SPECIAL_TOKENS.items()
        }

    # ── Core API ──────────────────────────────────────────────────────

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        max_length: Optional[int] = None,
    ) -> list[int]:
        return self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            max_length=max_length or self.max_length,
            truncation=True,
        )

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def encode_conversation(
        self,
        messages: list[dict[str, str]],
        wrap_mindi: bool = True,
    ) -> list[int]:
        """Encode a list of messages [{"role": ..., "content": ...}] into token IDs.

        Uses Qwen's im_start/im_end chat template with optional mindi_start/end wrapper.
        """
        parts: list[str] = []
        if wrap_mindi:
            parts.append("<|mindi_start|>\n")

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

        if wrap_mindi:
            parts.append("<|mindi_end|>")

        full_text = "".join(parts)
        return self.encode(full_text, add_special_tokens=False)

    def encode_with_special_tokens(self, text: str) -> list[int]:
        """Encode text that contains MINDI special tokens, preserving them as single tokens."""
        return self.encode(text, add_special_tokens=False)

    # ── Introspection ─────────────────────────────────────────────────

    def get_vocab_size(self) -> int:
        return len(self.tokenizer)

    def get_special_token_ids(self) -> dict[str, int]:
        return dict(self._special_token_ids)

    def get_special_token_id(self, name: str) -> int:
        return self._special_token_ids[name]

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, output_dir: Optional[Path] = None) -> Path:
        save_path = output_dir or self.tokenizer_path
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(str(save_path))
        return save_path
