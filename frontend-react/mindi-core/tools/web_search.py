from __future__ import annotations

import os
from typing import Any

import httpx

from backend.api.schemas import SearchResult, SearchSource
from tools.base import ToolResult


class WebSearchTool:
    name = "web_search"

    async def run(self, query: str, sources: list[SearchSource], max_results: int = 5) -> ToolResult:
        provider = os.getenv("SEARCH_PROVIDER", "tavily").lower()
        if provider == "tavily" and os.getenv("TAVILY_API_KEY"):
            results = await self._tavily(query, max_results)
        elif provider == "serper" and os.getenv("SERPER_API_KEY"):
            results = await self._serper(query, max_results)
        else:
            results = self._fallback(query, sources, max_results)
        return ToolResult(ok=True, data={"results": [result.model_dump() for result in results]})

    async def _tavily(self, query: str, max_results: int) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": os.getenv("TAVILY_API_KEY"),
                    "query": query,
                    "max_results": max_results,
                    "include_answer": False,
                },
            )
            response.raise_for_status()
            payload = response.json()
        return [
            SearchResult(
                title=item.get("title", "Untitled"),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
                source="web",
            )
            for item in payload.get("results", [])[:max_results]
        ]

    async def _serper(self, query: str, max_results: int) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": os.getenv("SERPER_API_KEY", "")},
                json={"q": query, "num": max_results},
            )
            response.raise_for_status()
            payload = response.json()
        items: list[dict[str, Any]] = payload.get("organic", [])
        return [
            SearchResult(
                title=item.get("title", "Untitled"),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                source="web",
            )
            for item in items[:max_results]
        ]

    def _fallback(self, query: str, sources: list[SearchSource], max_results: int) -> list[SearchResult]:
        source_label = ", ".join(sources)
        return [
            SearchResult(
                title="Search provider not configured",
                url="",
                snippet=(
                    f"Set TAVILY_API_KEY or SERPER_API_KEY to enable live {source_label} search. "
                    f"Query received: {query}"
                ),
                source="web",
            )
        ][:max_results]
