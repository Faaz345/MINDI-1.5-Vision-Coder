from __future__ import annotations

import asyncio
from pathlib import Path

from agents.workflow_router import WorkflowRouter
from backend.api.schemas import IntentName
from tools.read_files import ReadFilesTool


def test_workflow_router_covers_all_intents():
    router = WorkflowRouter()
    intents: list[IntentName] = [
        "build_ui",
        "debug",
        "explain",
        "search_web",
        "refactor",
        "deploy",
        "analyze_repo",
    ]

    for intent in intents:
        route = asyncio.run(router.route(intent))
        assert route.workflow_path.name.endswith(".md")
        assert route.tools


def test_read_files_rejects_path_traversal(tmp_path: Path):
    result = asyncio.run(ReadFilesTool().run(tmp_path, ["../secret.txt"]))

    assert result.ok is False
    assert "escapes workspace" in (result.error or "")
