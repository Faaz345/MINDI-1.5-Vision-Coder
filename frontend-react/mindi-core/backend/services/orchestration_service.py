from __future__ import annotations

import asyncio
import html
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from agents.intent_agent import IntentAgent
from agents.memory_agent import MemoryAgent
from agents.planner_agent import PlannerAgent
from agents.validation_agent import ValidationAgent
from agents.workflow_router import WorkflowRouter
from backend.api.schemas import ChatRequest, WorkflowRequest
from backend.inference.inference_client import InferenceClient
from backend.services.context_builder import ContextBuilder
from backend.services.nlp_enhancer import NlpEnhancer
from tools.analyze_repo import AnalyzeRepoTool


class OrchestrationService:
    def __init__(self) -> None:
        self.inference = InferenceClient()
        self.intent_agent = IntentAgent()
        self.planner_agent = PlannerAgent()
        self.router = WorkflowRouter()
        self.validator = ValidationAgent()
        self.memory_agent = MemoryAgent()
        self.context_builder = ContextBuilder()
        self.nlp_enhancer = NlpEnhancer()
        self.repo_root = Path(__file__).resolve().parents[3]

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[dict[str, Any]]:
        yield {"event": "meta", "data": {"type": "chat", "project_id": request.project_id}}
        messages = [message.model_dump() for message in request.messages]
        async for event in self.inference.stream_chat(messages):
            yield self._inference_to_sse(event.type, event.data)
        await self.memory_agent.remember(request.project_id, {"kind": "chat", "message_count": len(messages)})

    async def stream_workflow(self, request: WorkflowRequest) -> AsyncIterator[dict[str, Any]]:
        prompt = request.prompt.strip()
        yield {"event": "meta", "data": {"type": "workflow", "project_id": request.project_id, "mode": request.mode}}

        intent = await self.intent_agent.classify(prompt)
        yield {"event": "log", "data": {"stage": "intent", "message": intent.rationale, "intent": intent.intent}}

        enhanced_prompt = await self.nlp_enhancer.enhance(prompt, intent.intent, request.design_settings)
        route = await self.router.route(intent.intent)
        yield {
            "event": "log",
            "data": {
                "stage": "router",
                "message": f"Selected {route.workflow_path.name}",
                "tools": route.tools,
            },
        }

        plan = await self.planner_agent.plan(intent.intent, enhanced_prompt)
        yield {"event": "log", "data": {"stage": "planner", "message": "Execution plan prepared", "steps": [step.title for step in plan]}}

        memory = await self.memory_agent.recall(request.project_id, prompt)
        repo_data: dict[str, Any] = {}
        if "analyze_repo" in route.tools:
            yield {"event": "tool_start", "data": {"tool": "analyze_repo"}}
            repo_result = await AnalyzeRepoTool().run(self.repo_root)
            repo_data = repo_result.data
            yield {"event": "tool_result", "data": {"tool": "analyze_repo", "ok": repo_result.ok, "data": repo_data, "error": repo_result.error}}

        context = await self.context_builder.build(
            prompt=enhanced_prompt,
            workflow_path=route.workflow_path,
            files=request.files,
            memory=memory,
            plan=plan,
        )

        generated_files: dict[str, str] = {}
        if intent.intent == "build_ui":
            generated_files = self._starter_ui_files(prompt, request.design_settings)
            for path, content in generated_files.items():
                yield {"event": "file_delta", "data": {"path": path, "content": content, "operation": "upsert"}}
                await asyncio.sleep(0)

        messages = [{"role": "user", "content": enhanced_prompt}]
        token_buffer: list[str] = []
        async for event in self.inference.stream_chat(messages, system_prompt=context):
            if event.type == "token":
                token_buffer.append(event.data.get("text", ""))
            yield self._inference_to_sse(event.type, event.data)

        validation = await self.validator.validate_files({**request.files, **generated_files})
        yield {"event": "log", "data": {"stage": "validation", "message": "Validation completed", "warnings": validation.warnings}}
        summary = "".join(token_buffer).strip() or f"Workflow completed for {intent.intent}."
        await self.memory_agent.remember(
            request.project_id,
            {"kind": "workflow", "intent": intent.intent, "prompt": prompt, "summary": summary[:1000], "repo": repo_data},
        )
        await self.memory_agent.memory_service.save_project_summary(request.project_id, summary[:1500])
        yield {"event": "done", "data": {"summary": summary, "intent": intent.intent, "files": list(generated_files)}}

    def _inference_to_sse(self, event_type: str, data: dict[str, Any]) -> dict[str, Any]:
        if event_type == "token":
            return {"event": "token", "data": data}
        if event_type == "tool_call":
            return {"event": "log", "data": {"stage": "tool_call", **data}}
        if event_type == "error":
            return {"event": "error", "data": data}
        if event_type == "done":
            return {"event": "log", "data": {"stage": "inference", "message": "MINDI response completed"}}
        return {"event": "log", "data": data}

    def _starter_ui_files(self, prompt: str, design_settings: dict[str, Any]) -> dict[str, str]:
        title = html.escape((prompt.split(".")[0] or "MINDIGENOUS build")[:70])
        accent = design_settings.get("accent") or "#22c55e"
        return {
            "index.html": f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{title}</title>
    <link rel="stylesheet" href="styles.css" />
  </head>
  <body>
    <main class="shell">
      <section class="hero">
        <p class="eyebrow">MINDIGENOUS workspace</p>
        <h1>{title}</h1>
        <p class="summary">{html.escape(prompt)}</p>
        <button id="primaryAction">Open workflow</button>
      </section>
    </main>
    <script src="script.js"></script>
  </body>
</html>
""",
            "styles.css": f""":root {{
  color-scheme: dark;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #080d0c;
  color: #f8fafc;
}}

* {{ box-sizing: border-box; }}
body {{ margin: 0; min-height: 100vh; background: radial-gradient(circle at top right, {accent}33, transparent 34rem), #080d0c; }}
.shell {{ min-height: 100vh; display: grid; place-items: center; padding: 4rem 1.5rem; }}
.hero {{ width: min(920px, 100%); border: 1px solid rgba(255,255,255,.14); border-radius: 18px; padding: clamp(2rem, 6vw, 5rem); background: rgba(10,15,14,.78); backdrop-filter: blur(18px); }}
.eyebrow {{ margin: 0 0 .8rem; color: {accent}; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; }}
h1 {{ margin: 0; font-size: clamp(2.4rem, 7vw, 5.8rem); line-height: .95; letter-spacing: 0; }}
.summary {{ max-width: 68ch; color: #cbd5e1; font-size: 1.05rem; line-height: 1.7; }}
button {{ border: 0; border-radius: 999px; padding: .9rem 1.2rem; background: {accent}; color: #04100b; font-weight: 800; cursor: pointer; }}
""",
            "script.js": """document.getElementById("primaryAction")?.addEventListener("click", () => {
  document.body.classList.toggle("workflow-open");
});
""",
        }
