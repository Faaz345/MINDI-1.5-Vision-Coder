from __future__ import annotations

from backend.api.schemas import WebSearchRequest, WebSearchResponse
from backend.services.search_service import SearchService


class WebSearchAgent:
    def __init__(self, search_service: SearchService | None = None) -> None:
        self.search_service = search_service or SearchService()

    async def search(self, request: WebSearchRequest) -> WebSearchResponse:
        return await self.search_service.search(request)
