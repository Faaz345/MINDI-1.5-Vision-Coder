from __future__ import annotations

from backend.api.schemas import SearchResult, WebSearchRequest, WebSearchResponse
from tools.web_search import WebSearchTool


class SearchService:
    def __init__(self, tool: WebSearchTool | None = None) -> None:
        self.tool = tool or WebSearchTool()

    async def search(self, request: WebSearchRequest) -> WebSearchResponse:
        result = await self.tool.run(request.query, request.sources, request.max_results)
        raw_results = result.data.get("results", []) if result.ok else []
        results = [SearchResult(**item) for item in raw_results]
        summary = self._summarize(request.query, results)
        return WebSearchResponse(query=request.query, summary=summary, results=results)

    def _summarize(self, query: str, results: list[SearchResult]) -> str:
        if not results:
            return f"No search results returned for: {query}"
        if results[0].title == "Search provider not configured":
            return results[0].snippet
        titles = "; ".join(result.title for result in results[:3])
        return f"Found {len(results)} result(s) for '{query}': {titles}"
