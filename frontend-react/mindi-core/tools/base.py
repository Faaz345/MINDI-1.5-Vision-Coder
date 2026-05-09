from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ToolResult:
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def safe_repo_path(root: Path, requested: str) -> Path:
    base = root.resolve()
    target = (base / requested).resolve()
    if base != target and base not in target.parents:
        raise ValueError(f"Path escapes workspace: {requested}")
    return target


def detect_stack(files: list[str]) -> dict[str, bool]:
    names = set(files)
    return {
        "react": any(name.endswith((".jsx", ".tsx")) for name in names) or "package.json" in names,
        "vite": "vite.config.js" in names or "vite.config.ts" in names,
        "tailwind": "tailwind.config.js" in names or "tailwind.config.ts" in names,
        "python": any(name.endswith(".py") for name in names),
        "fastapi": any("fastapi" in name.lower() for name in names),
    }
