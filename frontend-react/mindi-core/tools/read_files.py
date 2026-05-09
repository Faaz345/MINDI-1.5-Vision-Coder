from __future__ import annotations

from pathlib import Path

from tools.base import ToolResult, safe_repo_path


class ReadFilesTool:
    name = "read_files"

    async def run(self, root: Path, paths: list[str], max_bytes: int = 120_000) -> ToolResult:
        files: dict[str, str] = {}
        try:
            for path in paths:
                target = safe_repo_path(root, path)
                if not target.is_file():
                    files[path] = ""
                    continue
                files[path] = target.read_text(encoding="utf-8", errors="replace")[:max_bytes]
            return ToolResult(ok=True, data={"files": files})
        except Exception as exc:
            return ToolResult(ok=False, error=str(exc))
