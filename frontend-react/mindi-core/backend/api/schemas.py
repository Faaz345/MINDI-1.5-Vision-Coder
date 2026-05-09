from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


MessageRole = Literal["system", "user", "assistant", "tool"]
IntentName = Literal["build_ui", "debug", "explain", "search_web", "refactor", "deploy", "analyze_repo"]
WorkflowMode = Literal["chat", "workspace"]
FileOperation = Literal["upsert", "delete"]
SearchSource = Literal["web", "docs", "github", "stackoverflow"]


class ChatMessage(BaseModel):
    role: MessageRole
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    project_id: str | None = None
    files: dict[str, str] = Field(default_factory=dict)
    stream: bool = True


class WorkflowRequest(BaseModel):
    prompt: str
    project_id: str | None = None
    files: dict[str, str] = Field(default_factory=dict)
    design_settings: dict[str, Any] = Field(default_factory=dict)
    mode: WorkflowMode = "chat"


class WebSearchRequest(BaseModel):
    query: str
    sources: list[SearchSource] = Field(default_factory=lambda: ["web"])
    max_results: int = Field(default=5, ge=1, le=10)


class FileDelta(BaseModel):
    path: str
    content: str | None = None
    operation: FileOperation = "upsert"


class StreamEvent(BaseModel):
    event: str
    data: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str = ""
    source: SearchSource | str = "web"


class WebSearchResponse(BaseModel):
    query: str
    summary: str
    results: list[SearchResult]
