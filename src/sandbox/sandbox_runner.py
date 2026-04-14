"""
MINDI 1.5 Vision-Coder — Sandbox Runner

Executes generated code in an isolated environment (E2B cloud sandbox
or local Docker container) to test for errors and capture screenshots.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SandboxResult:
    """Result from running code in the sandbox."""
    success: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    screenshot_path: Optional[Path] = None
    execution_time_ms: float = 0.0
    errors: list[str] = field(default_factory=list)


class SandboxRunner:
    """
    Isolated code execution environment.

    Supports two backends:
    - E2B (cloud): For production — real browser rendering + screenshots
    - Docker (local): For development/testing
    """

    def __init__(
        self,
        backend: str = "e2b",
        e2b_api_key: Optional[str] = None,
        screenshot_dir: Optional[Path] = None,
    ) -> None:
        self.backend = backend
        self.e2b_api_key = e2b_api_key or os.environ.get("E2B_API_KEY", "")
        self.screenshot_dir = screenshot_dir or Path("./logs/screenshots")
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

    async def run_code(
        self,
        code: str,
        filename: str = "page.tsx",
        capture_screenshot: bool = True,
    ) -> SandboxResult:
        """
        Execute code in the sandbox and optionally capture a screenshot.

        The screenshot is used by the VisionCritic to evaluate UI quality.
        """
        if self.backend == "e2b":
            return await self._run_e2b(code, filename, capture_screenshot)
        elif self.backend == "docker":
            return await self._run_docker(code, filename, capture_screenshot)
        else:
            return SandboxResult(
                success=False,
                stderr=f"Unknown sandbox backend: {self.backend}",
                errors=[f"Unknown backend: {self.backend}"],
            )

    async def _run_e2b(
        self, code: str, filename: str, capture_screenshot: bool
    ) -> SandboxResult:
        """Execute in E2B cloud sandbox."""
        if not self.e2b_api_key:
            return SandboxResult(
                success=False,
                stderr="E2B_API_KEY not set",
                errors=["E2B_API_KEY not configured"],
            )

        # Will be implemented with e2b-code-interpreter SDK
        return SandboxResult(success=False, stderr="E2B integration pending")

    async def _run_docker(
        self, code: str, filename: str, capture_screenshot: bool
    ) -> SandboxResult:
        """Execute in local Docker container."""
        # Will be implemented with Docker SDK
        return SandboxResult(success=False, stderr="Docker integration pending")
