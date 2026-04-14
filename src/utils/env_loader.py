"""
MINDI 1.5 Vision-Coder — Environment Variable Loader

Loads secrets from .env, validates required keys, and provides
typed access to environment configuration.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass
class EnvValidationResult:
    """Result of environment variable validation."""
    valid: bool
    missing: list[str]
    warnings: list[str]


class EnvLoader:
    """
    Loads and validates environment variables from .env files.

    Usage:
        env = EnvLoader()
        env.load()
        env.validate()
        key = env.get("TAVILY_API_KEY")
    """

    REQUIRED_KEYS = [
        "HUGGINGFACE_TOKEN",
        "TAVILY_API_KEY",
        "WANDB_API_KEY",
        "E2B_API_KEY",
    ]

    OPTIONAL_KEYS = [
        "HUGGINGFACE_REPO",
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "MODEL_NAME",
        "BASE_MODEL_PATH",
        "FINETUNED_MODEL_PATH",
        "API_HOST",
        "API_PORT",
        "API_WORKERS",
        "DEVICE",
        "MIXED_PRECISION",
        "MAX_SEQ_LENGTH",
        "TRAINING_OUTPUT_DIR",
        "LOG_DIR",
        "DATA_DIR",
        "CHECKPOINT_DIR",
        "SANDBOX_TYPE",
        "MAX_SEARCH_RESULTS",
        "SEARCH_TIMEOUT",
        "CLOUD_GPU_HOST",
        "CLOUD_GPU_USER",
        "CLOUD_GPU_SSH_KEY",
        "PROJECT_NAME",
        "STARTUP_NAME",
        "HF_USERNAME",
    ]

    KEY_PREFIXES = {
        "HUGGINGFACE_TOKEN": "hf_",
        "TAVILY_API_KEY": "tvly-",
        "E2B_API_KEY": "e2b_",
    }

    def __init__(self, env_path: Optional[Path] = None) -> None:
        self.env_path = env_path or Path(__file__).resolve().parents[2] / ".env"
        self._loaded = False

    def load(self, override: bool = False) -> None:
        """Load environment variables from .env file."""
        if not self.env_path.exists():
            raise FileNotFoundError(
                f".env file not found at {self.env_path}\n"
                f"Copy .env.example to .env and fill in your API keys."
            )
        load_dotenv(self.env_path, override=override)
        self._loaded = True

    def validate(self) -> EnvValidationResult:
        """Validate that all required environment variables are set and well-formed."""
        if not self._loaded:
            self.load()

        missing: list[str] = []
        warnings: list[str] = []

        for key in self.REQUIRED_KEYS:
            value = os.environ.get(key, "").strip()
            if not value:
                missing.append(key)
                continue

            # Check prefix format
            expected_prefix = self.KEY_PREFIXES.get(key)
            if expected_prefix and not value.startswith(expected_prefix):
                warnings.append(
                    f"{key} doesn't start with expected prefix '{expected_prefix}'"
                )

        return EnvValidationResult(
            valid=len(missing) == 0,
            missing=missing,
            warnings=warnings,
        )

    def get(self, key: str, default: Optional[str] = None) -> str:
        """Get an environment variable with optional default."""
        if not self._loaded:
            self.load()
        return os.environ.get(key, default or "")

    def get_int(self, key: str, default: int = 0) -> int:
        """Get an environment variable as an integer."""
        value = self.get(key)
        if not value:
            return default
        return int(value)

    def get_path(self, key: str, default: str = ".") -> Path:
        """Get an environment variable as a Path."""
        return Path(self.get(key, default))

    # ── Convenience properties ──

    @property
    def huggingface_token(self) -> str:
        return self.get("HUGGINGFACE_TOKEN")

    @property
    def tavily_api_key(self) -> str:
        return self.get("TAVILY_API_KEY")

    @property
    def wandb_api_key(self) -> str:
        return self.get("WANDB_API_KEY")

    @property
    def e2b_api_key(self) -> str:
        return self.get("E2B_API_KEY")

    @property
    def model_name(self) -> str:
        return self.get("MODEL_NAME", "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct")

    @property
    def device(self) -> str:
        return self.get("DEVICE", "cuda")

    @property
    def mixed_precision(self) -> str:
        return self.get("MIXED_PRECISION", "bf16")

    @property
    def sandbox_type(self) -> str:
        return self.get("SANDBOX_TYPE", "e2b")

    def print_status(self) -> None:
        """Print a summary of environment variable status."""
        result = self.validate()

        print("\n╔══════════════════════════════════════════╗")
        print("║    MINDI 1.5 — Environment Status        ║")
        print("╠══════════════════════════════════════════╣")

        for key in self.REQUIRED_KEYS:
            value = os.environ.get(key, "")
            if value:
                masked = value[:8] + "..." + value[-4:]
                print(f"  ✅ {key:<25} = {masked}")
            else:
                print(f"  ❌ {key:<25} = NOT SET")

        print("╠──────────────────────────────────────────╣")

        for key in self.OPTIONAL_KEYS:
            value = os.environ.get(key, "")
            if value:
                display = value if len(value) <= 40 else value[:37] + "..."
                print(f"  ✅ {key:<25} = {display}")
            else:
                print(f"  ⚪ {key:<25} = (not set)")

        print("╠══════════════════════════════════════════╣")

        if result.valid:
            print("  ✅ All required keys are set!")
        else:
            print(f"  ❌ Missing {len(result.missing)} required key(s): {', '.join(result.missing)}")

        for w in result.warnings:
            print(f"  ⚠️  {w}")

        print("╚══════════════════════════════════════════╝\n")


if __name__ == "__main__":
    env = EnvLoader()
    env.load()
    env.print_status()
    result = env.validate()
    sys.exit(0 if result.valid else 1)
