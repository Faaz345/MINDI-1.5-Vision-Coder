"""
MINDI 1.5 Vision-Coder — Smoke Test

Basic tests to verify the project structure and imports work correctly.
"""

from __future__ import annotations

from pathlib import Path


def test_project_structure_exists() -> None:
    """Verify all critical directories exist."""
    root = Path(__file__).resolve().parent.parent
    required = [
        "configs", "src", "api", "scripts", "tests",
        "data", "checkpoints", "logs", "docs",
    ]
    for d in required:
        assert (root / d).exists(), f"Missing directory: {d}"


def test_config_files_exist() -> None:
    """Verify config YAML files are present."""
    root = Path(__file__).resolve().parent.parent
    configs = [
        "configs/model_config.yaml",
        "configs/training_config.yaml",
        "configs/data_config.yaml",
        "configs/search_config.yaml",
    ]
    for c in configs:
        assert (root / c).exists(), f"Missing config: {c}"


def test_src_packages_importable() -> None:
    """Verify src __init__.py files exist (importability test)."""
    root = Path(__file__).resolve().parent.parent
    packages = [
        "src", "src/model", "src/agents", "src/search",
        "src/sandbox", "src/training", "src/inference", "src/evaluation",
    ]
    for pkg in packages:
        init_file = root / pkg / "__init__.py"
        assert init_file.exists(), f"Missing __init__.py in {pkg}"
