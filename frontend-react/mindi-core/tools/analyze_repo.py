from __future__ import annotations

from pathlib import Path

from tools.base import ToolResult, detect_stack


IGNORED_DIRS = {".git", "node_modules", "dist", "build", ".venv", "__pycache__"}


class AnalyzeRepoTool:
    name = "analyze_repo"

    async def run(self, root: Path, max_files: int = 500) -> ToolResult:
        try:
            files: list[str] = []
            for path in root.rglob("*"):
                if any(part in IGNORED_DIRS for part in path.parts):
                    continue
                if path.is_file():
                    files.append(path.relative_to(root).as_posix())
                    if len(files) >= max_files:
                        break

            key_files = [
                name for name in files
                if name in {"package.json", "vite.config.js", "pyproject.toml", "README.md", ".env.example"}
            ]
            return ToolResult(ok=True, data={"files": files, "key_files": key_files, "stack": detect_stack(files)})
        except Exception as exc:
            return ToolResult(ok=False, error=str(exc))
