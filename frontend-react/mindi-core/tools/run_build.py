from __future__ import annotations

import asyncio
from pathlib import Path

from tools.base import ToolResult


class RunBuildTool:
    name = "run_build"

    async def run(self, root: Path, command: list[str] | None = None, timeout: int = 120) -> ToolResult:
        command = command or ["npm", "run", "build"]
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
                error=None if process.returncode == 0 else "Build command failed",
            )
        except Exception as exc:
            return ToolResult(ok=False, error=str(exc), data={"command": command})
