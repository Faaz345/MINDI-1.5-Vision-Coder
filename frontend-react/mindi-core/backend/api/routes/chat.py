from __future__ import annotations

from fastapi import APIRouter

from backend.api.schemas import ChatRequest
from backend.api.streaming import stream_response
from backend.services.orchestration_service import OrchestrationService

router = APIRouter()


@router.post("/chat")
async def chat(request: ChatRequest):
    service = OrchestrationService()
    return stream_response(service.stream_chat(request))
