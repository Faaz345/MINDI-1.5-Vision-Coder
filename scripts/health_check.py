"""
MINDI 1.5 Vision-Coder — System Health Check Script

Verifies that all dependencies, configs, and environment variables
are correctly set up before starting development or training.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def check_python() -> bool:
    """Verify Python version."""
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 10
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] Python {v.major}.{v.minor}.{v.micro}")
    return ok


def check_env_vars() -> bool:
    """Check that required environment variables are set."""
    required = ["HUGGINGFACE_TOKEN", "TAVILY_API_KEY", "WANDB_API_KEY", "E2B_API_KEY"]
    all_ok = True
    for var in required:
        value = os.environ.get(var, "")
        if value:
            print(f"  [OK] {var} = {value[:8]}...")
        else:
            print(f"  [MISSING] {var}")
            all_ok = False
    return all_ok


def check_directories() -> bool:
    """Verify project directory structure exists."""
    project_root = Path(__file__).resolve().parent.parent
    required_dirs = [
        "configs", "data/raw", "data/processed", "data/tokenizer",
        "data/knowledge_base", "src/model", "src/agents", "src/search",
        "src/sandbox", "src/training", "src/inference", "src/evaluation",
        "api/routes", "api/middleware", "scripts", "tests",
        "checkpoints", "logs", "docs",
    ]
    all_ok = True
    for d in required_dirs:
        path = project_root / d
        if path.exists():
            print(f"  [OK] {d}/")
        else:
            print(f"  [MISSING] {d}/")
            all_ok = False
    return all_ok


def check_configs() -> bool:
    """Verify config files exist."""
    project_root = Path(__file__).resolve().parent.parent
    configs = [
        "configs/model_config.yaml",
        "configs/training_config.yaml",
        "configs/data_config.yaml",
        "configs/search_config.yaml",
    ]
    all_ok = True
    for c in configs:
        path = project_root / c
        if path.exists():
            print(f"  [OK] {c}")
        else:
            print(f"  [MISSING] {c}")
            all_ok = False
    return all_ok


def check_gpu() -> bool:
    """Check CUDA GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            print(f"  [OK] GPU: {name} ({vram:.1f} GB VRAM)")
            return True
        else:
            print("  [WARN] No CUDA GPU detected (CPU mode)")
            return False
    except ImportError:
        print("  [WARN] PyTorch not installed yet")
        return False


def main() -> None:
    """Run all health checks."""
    print("=" * 55)
    print("  MINDI 1.5 Vision-Coder — System Health Check")
    print("=" * 55)

    print("\n[1] Python Version:")
    check_python()

    print("\n[2] GPU:")
    check_gpu()

    print("\n[3] Environment Variables:")
    check_env_vars()

    print("\n[4] Directory Structure:")
    check_directories()

    print("\n[5] Config Files:")
    check_configs()

    print("\n" + "=" * 55)
    print("  Health check complete.")
    print("=" * 55)


if __name__ == "__main__":
    main()
