"""
MINDI 1.5 Vision-Coder — Web Search Agent

Uses Tavily API to search for latest documentation, packages,
and code examples to ground the model's code generation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
import yaml


@dataclass
class SearchResult:
    """A single search result from Tavily."""
    title: str
    url: str
    content: str
    score: float


@dataclass
class SearchResponse:
    """Aggregated search response."""
    query: str
    results: list[SearchResult] = field(default_factory=list)
    context: str = ""  # Concatenated relevant content for the model


class SearchAgent:
    """Web search agent powered by Tavily for documentation lookup."""

    def __init__(
        self,
        config_path: Optional[Path] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.config_path = config_path or Path("./configs/search_config.yaml")
        self.config = self._load_config()

        self.api_key = api_key or os.environ.get("TAVILY_API_KEY", "")
        if not self.api_key:
            print("[SearchAgent] WARNING: TAVILY_API_KEY not set")

        self.base_url = "https://api.tavily.com"

    def _load_config(self) -> dict:
        """Load search configuration from YAML."""
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f).get("search", {})
        return {}

    async def search(self, query: str, max_results: int = 5) -> SearchResponse:
        """Execute a web search via Tavily API."""
        if not self.api_key:
            return SearchResponse(query=query, context="Search unavailable — no API key.")

        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": self.config.get("search_depth", "advanced"),
            "max_results": max_results,
            "include_domains": self.config.get("include_domains", []),
            "exclude_domains": self.config.get("exclude_domains", []),
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{self.base_url}/search", json=payload)
            response.raise_for_status()
            data = response.json()

        results = [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                content=r.get("content", ""),
                score=r.get("score", 0.0),
            )
            for r in data.get("results", [])
        ]

        # Build concatenated context for the model
        context_parts = [f"### {r.title}\n{r.content}" for r in results]
        context = "\n\n".join(context_parts)

        return SearchResponse(query=query, results=results, context=context)

    async def search_docs(self, topic: str) -> SearchResponse:
        """Search specifically for framework documentation."""
        query = f"{topic} documentation latest Next.js 14 Tailwind TypeScript"
        return await self.search(query)

    async def search_package(self, package_name: str) -> SearchResponse:
        """Search for an npm package's usage and API."""
        query = f"npm {package_name} usage example TypeScript"
        return await self.search(query)
