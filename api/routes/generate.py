"""
MINDI 1.5 Vision-Coder — Code Generation Route

Accepts user prompts and returns generated Next.js + Tailwind + TypeScript code
via the agent orchestration pipeline.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()


class GenerateRequest(BaseModel):
    """Request body for code generation."""
    prompt: str = Field(..., min_length=1, max_length=10000, description="User's code generation prompt")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(4096, ge=1, le=8192)
    use_search: bool = Field(True, description="Enable web search for context")
    use_sandbox: bool = Field(True, description="Enable sandbox testing")
    use_vision: bool = Field(True, description="Enable vision-based UI critique")


class GenerateResponse(BaseModel):
    """Response body for code generation."""
    code: str
    language: str = "typescript"
    file_path: str = "page.tsx"
    critique: Optional[str] = None
    search_sources: list[str] = []
    iterations: int = 1
    success: bool = True


@router.post("/generate", response_model=GenerateResponse)
async def generate_code(request: GenerateRequest) -> GenerateResponse:
    """Generate code from a user prompt using the MINDI agent pipeline."""
    # Will be wired to AgentOrchestrator in later phases
    raise HTTPException(
        status_code=503,
        detail="Model not loaded yet. Complete training pipeline first.",
    )
