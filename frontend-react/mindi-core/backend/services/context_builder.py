from __future__ import annotations

from pathlib import Path


class ContextBuilder:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(__file__).resolve().parents[2]

    async def build(
        self,
        *,
        prompt: str,
        workflow_path: Path,
        files: dict[str, str],
        memory: dict[str, str],
        plan: list,
    ) -> str:
        system_path = self.root / "prompts" / "system" / "mindi_system.md"
        system_prompt = system_path.read_text(encoding="utf-8") if system_path.exists() else ""
        workflow = workflow_path.read_text(encoding="utf-8") if workflow_path.exists() else ""
        file_context = "\n".join(f"### {path}\n{content[:4000]}" for path, content in list(files.items())[:12])
        plan_text = "\n".join(f"- {step.title}" for step in plan)
        return (
            f"{system_prompt}\n\n"
            f"## Workflow\n{workflow}\n\n"
            f"## Memory\n{memory.get('summary', '')}\n\n"
            f"## Plan\n{plan_text}\n\n"
            f"## Active Files\n{file_context}\n\n"
            f"## User Task\n{prompt}"
        )
