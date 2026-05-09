from __future__ import annotations

from fastapi import APIRouter

from backend.api.schemas import WebSearchRequest, WebSearchResponse
from backend.services.search_service import SearchService

router = APIRouter()


@router.post("/web-search", response_model=WebSearchResponse)
async def web_search(request: WebSearchRequest) -> WebSearchResponse:
    service = SearchService()
    return await service.search(request)
