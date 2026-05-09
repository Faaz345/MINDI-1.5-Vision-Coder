from __future__ import annotations

import asyncio
from pathlib import Path

from tools.base import ToolResult


class LintProjectTool:
    name = "lint_project"

    async def run(self, root: Path, command: list[str] | None = None, timeout: int = 90) -> ToolResult:
        command = command or ["npm", "run", "lint"]
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            return ToolResult(
                ok=process.returncode == 0,
                data={
                    "command": command,
                    "returncode": process.returncode,
                    "stdout": stdout.decode(errors="replace")[-12_000:],
                    "stderr": stderr.decode(errors="replace")[-12_000:],
                },
                error=None if process.returncode == 0 else "Lint command failed",
            )
        except FileNotFoundError:
            return ToolResult(ok=False, error="Lint command is not available", data={"command": command})
        except Exception as exc:
            return ToolResult(ok=False, error=str(exc), data={"command": command})
