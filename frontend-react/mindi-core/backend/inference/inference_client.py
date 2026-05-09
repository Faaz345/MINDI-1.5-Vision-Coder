from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(slots=True)
class InferenceEvent:
    type: str
    data: dict[str, Any]


class InferenceClient:
    """OpenAI-compatible client for the AMD-hosted MINDI endpoint."""

    def __init__(self) -> None:
        self.base_url = os.getenv("MINDI_API_URL", "").rstrip("/")
        self.api_key = os.getenv("MINDI_API_KEY", "")
        self.model = os.getenv("MINDI_MODEL", "mindi-1.5")
        self.timeout = float(os.getenv("MINDI_REQUEST_TIMEOUT_SECONDS", "120"))
        self._semaphore = asyncio.Semaphore(int(os.getenv("MINDI_MAX_CONCURRENT_REQUESTS", "4")))

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key)

    @property
    def chat_completions_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/v1/chat/completions"

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
    ) -> AsyncIterator[InferenceEvent]:
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}, *messages]

        if not self.configured:
            async for event in self._fallback_stream(messages):
                yield event
            return

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools

        async with self._semaphore:
            async for event in self._request_with_retries(payload):
                yield event

    async def _request_with_retries(self, payload: dict[str, Any]) -> AsyncIterator[InferenceEvent]:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    async with client.stream(
                        "POST",
                        self.chat_completions_url,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    ) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            data = line.removeprefix("data:").strip()
                            if data == "[DONE]":
                                yield InferenceEvent("done", {})
                                return
                            event = self._parse_openai_chunk(data)
                            if event:
                                yield event
                        yield InferenceEvent("done", {})
                        return
            except (httpx.HTTPError, httpx.TimeoutException) as exc:
                last_error = exc
                await asyncio.sleep(0.6 * (2**attempt))

        yield InferenceEvent("error", {"message": f"MINDI inference request failed: {last_error}"})

    def _parse_openai_chunk(self, raw: str) -> InferenceEvent | None:
        chunk = json.loads(raw)
        choice = (chunk.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if content:
            return InferenceEvent("token", {"text": content})
        tool_calls = delta.get("tool_calls")
        if tool_calls:
            return InferenceEvent("tool_call", {"tool_calls": tool_calls})
        return None

    async def _fallback_stream(self, messages: list[dict[str, str]]) -> AsyncIterator[InferenceEvent]:
        latest = next((message["content"] for message in reversed(messages) if message["role"] == "user"), "")
        text = (
            "MINDI backend is running in local fallback mode. "
            "Set MINDI_API_URL and MINDI_API_KEY to stream from the AMD GPU endpoint. "
            f"Received task: {latest[:220]}"
        )
        for token in text.split(" "):
            await asyncio.sleep(0.01)
            yield InferenceEvent("token", {"text": f"{token} "})
        yield InferenceEvent("done", {})
