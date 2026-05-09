from __future__ import annotations

from pathlib import Path

from tools.base import ToolResult, safe_repo_path


class WriteFilesTool:
    name = "write_files"

    async def run(self, root: Path, files: dict[str, str]) -> ToolResult:
        written: list[str] = []
        try:
            for path, content in files.items():
                target = safe_repo_path(root, path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                written.append(path)
            return ToolResult(ok=True, data={"written": written})
        except Exception as exc:
            return ToolResult(ok=False, error=str(exc))
