from __future__ import annotations

from fastapi import APIRouter

from backend.api.schemas import WorkflowRequest
from backend.api.streaming import stream_response
from backend.services.orchestration_service import OrchestrationService

router = APIRouter()


@router.post("/workflow")
async def workflow(request: WorkflowRequest):
    service = OrchestrationService()
    return stream_response(service.stream_workflow(request))
