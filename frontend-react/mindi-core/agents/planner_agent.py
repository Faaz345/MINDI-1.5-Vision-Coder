from __future__ import annotations

from dataclasses import dataclass

from backend.api.schemas import IntentName


@dataclass(slots=True)
class PlanStep:
    id: str
    title: str
    tool: str | None = None


class PlannerAgent:
    async def plan(self, intent: IntentName, prompt: str) -> list[PlanStep]:
        base = [
            PlanStep("understand", "Clarify the task and constraints"),
            PlanStep("context", "Build project context"),
        ]
        workflow_steps = {
            "build_ui": [PlanStep("generate", "Generate UI files"), PlanStep("validate", "Validate generated code", "lint_project")],
            "debug": [PlanStep("inspect", "Inspect relevant files", "read_files"), PlanStep("validate", "Run checks", "run_build")],
            "refactor": [PlanStep("edit", "Prepare targeted code edits"), PlanStep("validate", "Run lint/build checks", "lint_project")],
            "deploy": [PlanStep("build", "Build production artifact", "run_build")],
            "search_web": [PlanStep("search", "Retrieve external references", "web_search")],
            "analyze_repo": [PlanStep("analyze", "Analyze repository structure", "analyze_repo")],
            "explain": [PlanStep("answer", "Explain with current context")],
        }
        return [*base, *workflow_steps.get(intent, workflow_steps["explain"])]
