from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backend.api.schemas import IntentName


@dataclass(slots=True)
class WorkflowRoute:
    workflow_path: Path
    tools: list[str]


class WorkflowRouter:
    def __init__(self, workflow_root: Path | None = None) -> None:
        self.workflow_root = workflow_root or Path(__file__).resolve().parents[1] / "workflows"

    async def route(self, intent: IntentName) -> WorkflowRoute:
        mapping: dict[IntentName, tuple[str, list[str]]] = {
            "build_ui": ("generate_ui.md", ["analyze_repo", "write_files", "lint_project"]),
            "debug": ("debug_error.md", ["read_files", "run_build", "lint_project"]),
            "refactor": ("refactor_code.md", ["read_files", "write_files", "lint_project"]),
            "search_web": ("web_search.md", ["web_search"]),
            "analyze_repo": ("analyze_repo.md", ["analyze_repo", "read_files"]),
            "deploy": ("deploy_project.md", ["run_build"]),
            "explain": ("analyze_repo.md", ["read_files"]),
        }
        filename, tools = mapping[intent]
        return WorkflowRoute(workflow_path=self.workflow_root / filename, tools=tools)
