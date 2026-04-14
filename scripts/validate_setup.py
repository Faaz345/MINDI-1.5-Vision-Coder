"""
MINDI 1.5 Vision-Coder — Setup Validation Script

Comprehensive readiness check: environment, configs, directories,
API keys, GPU, and package imports.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def header(title: str) -> None:
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")


def check(label: str, passed: bool, detail: str = "") -> bool:
    icon = "✅" if passed else "❌"
    msg = f"  {icon} {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return passed


def validate_directories() -> int:
    header("1. Directory Structure")
    required_dirs = [
        "configs", "src", "src/model", "src/agents", "src/search",
        "src/sandbox", "src/training", "src/inference", "src/evaluation",
        "src/tokenizer", "src/utils", "api", "api/routes", "api/middleware",
        "scripts", "data", "data/raw", "data/processed", "data/knowledge_base",
        "checkpoints", "logs", "tests", "docs", "frontend",
    ]
    failures = 0
    for d in required_dirs:
        path = PROJECT_ROOT / d
        if not check(d, path.is_dir()):
            failures += 1
    return failures


def validate_files() -> int:
    header("2. Key Files")
    required_files = [
        ".env", ".env.example", ".gitignore", "README.md",
        "requirements.txt", "setup.py",
        "configs/model_config.yaml", "configs/training_config.yaml",
        "configs/data_config.yaml", "configs/search_config.yaml",
        "src/__init__.py", "src/utils/__init__.py",
        "src/utils/env_loader.py", "src/utils/config_loader.py",
        "src/model/vision_encoder.py", "src/model/code_model.py",
        "src/agents/orchestrator.py", "src/agents/ui_critic.py",
        "src/agents/error_fixer.py", "src/search/search_agent.py",
        "src/sandbox/sandbox_runner.py", "src/training/trainer.py",
        "src/training/dataset.py", "src/inference/pipeline.py",
        "src/evaluation/evaluator.py",
        "api/main.py", "api/routes/generate.py", "api/middleware/auth.py",
        "scripts/health_check.py", "scripts/train.py",
    ]
    failures = 0
    for f in required_files:
        path = PROJECT_ROOT / f
        if not check(f, path.is_file()):
            failures += 1
    return failures


def validate_env() -> int:
    header("3. Environment Variables")
    from src.utils.env_loader import EnvLoader

    env = EnvLoader()
    env.load()
    result = env.validate()

    failures = 0
    required = ["HUGGINGFACE_TOKEN", "TAVILY_API_KEY", "WANDB_API_KEY", "E2B_API_KEY"]
    for key in required:
        value = os.environ.get(key, "")
        if value:
            masked = value[:8] + "..." + value[-4:]
            check(key, True, masked)
        else:
            check(key, False, "NOT SET")
            failures += 1

    for w in result.warnings:
        print(f"  ⚠️  {w}")

    return failures


def validate_configs() -> int:
    header("4. YAML Configurations")
    from src.utils.config_loader import ConfigLoader

    loader = ConfigLoader()
    failures = 0

    try:
        m = loader.model
        check("model_config.yaml", True, f"{m.name} v{m.version}")
    except Exception as e:
        check("model_config.yaml", False, str(e))
        failures += 1

    try:
        t = loader.training
        check("training_config.yaml", True, f"{t.epochs} epochs, lr={t.learning_rate}")
    except Exception as e:
        check("training_config.yaml", False, str(e))
        failures += 1

    try:
        d = loader.data
        check("data_config.yaml", True, f"{d.target_size:,} target samples")
    except Exception as e:
        check("data_config.yaml", False, str(e))
        failures += 1

    try:
        s = loader.search
        check("search_config.yaml", True, f"provider={s.provider}")
    except Exception as e:
        check("search_config.yaml", False, str(e))
        failures += 1

    return failures


def validate_packages() -> int:
    header("5. Critical Package Imports")
    packages = [
        ("torch", "PyTorch"),
        ("transformers", "HuggingFace Transformers"),
        ("peft", "PEFT (LoRA)"),
        ("datasets", "HuggingFace Datasets"),
        ("wandb", "Weights & Biases"),
        ("fastapi", "FastAPI"),
        ("httpx", "HTTPX"),
        ("PIL", "Pillow"),
        ("yaml", "PyYAML"),
        ("dotenv", "python-dotenv"),
        ("pydantic", "Pydantic"),
        ("playwright", "Playwright"),
    ]
    failures = 0
    for module, label in packages:
        try:
            importlib.import_module(module)
            check(label, True)
        except ImportError:
            check(label, False, "not installed")
            failures += 1
    return failures


def validate_gpu() -> int:
    header("6. GPU / CUDA")
    failures = 0
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        check("CUDA available", cuda_available)
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            check("GPU", True, f"{gpu_name} ({vram:.1f} GB)")
            check("PyTorch CUDA version", True, torch.version.cuda or "N/A")
        else:
            failures += 1
    except Exception as e:
        check("GPU check", False, str(e))
        failures += 1
    return failures


def validate_gitignore() -> int:
    header("7. Security Check")
    gitignore = PROJECT_ROOT / ".gitignore"
    failures = 0
    if gitignore.is_file():
        content = gitignore.read_text(encoding="utf-8")
        check(".env in .gitignore", ".env" in content)
        check("venv/ in .gitignore", "venv" in content)
        if ".env" not in content:
            failures += 1
    else:
        check(".gitignore exists", False)
        failures += 1
    return failures


def main() -> None:
    print("\n╔══════════════════════════════════════════════════╗")
    print("║  MINDI 1.5 Vision-Coder — Full Setup Validation  ║")
    print("╚══════════════════════════════════════════════════╝")

    total_failures = 0
    total_failures += validate_directories()
    total_failures += validate_files()
    total_failures += validate_env()
    total_failures += validate_configs()
    total_failures += validate_packages()
    total_failures += validate_gpu()
    total_failures += validate_gitignore()

    header("RESULT")
    if total_failures == 0:
        print("  ✅ ALL CHECKS PASSED — MINDI 1.5 is ready!")
    else:
        print(f"  ❌ {total_failures} check(s) failed — review above")

    sys.exit(0 if total_failures == 0 else 1)


if __name__ == "__main__":
    main()
