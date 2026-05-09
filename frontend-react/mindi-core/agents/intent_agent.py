from __future__ import annotations

from dataclasses import dataclass

from backend.api.schemas import IntentName


@dataclass(slots=True)
class IntentResult:
    intent: IntentName
    confidence: float
    rationale: str


class IntentAgent:
    async def classify(self, prompt: str) -> IntentResult:
        text = prompt.lower()
        rules: list[tuple[IntentName, tuple[str, ...]]] = [
            ("search_web", ("search", "look up", "latest", "docs", "documentation")),
            ("debug", ("bug", "error", "fix", "broken", "stack trace", "not working")),
            ("refactor", ("refactor", "clean up", "restructure", "optimize code")),
            ("deploy", ("deploy", "publish", "ship", "vercel", "production")),
            ("analyze_repo", ("analyze", "inspect repo", "project structure", "architecture")),
            ("build_ui", ("build", "create", "generate", "make", "design", "ui", "website", "app")),
            ("explain", ("explain", "what is", "how does", "summarize")),
        ]
        for intent, needles in rules:
            if any(needle in text for needle in needles):
                return IntentResult(intent=intent, confidence=0.78, rationale=f"Matched {intent} keywords.")
        return IntentResult(intent="explain", confidence=0.45, rationale="No strong workflow keyword matched.")
