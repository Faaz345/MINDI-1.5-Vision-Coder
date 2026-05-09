from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi.responses import StreamingResponse


def sse_encode(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


async def sse_stream(events: AsyncIterator[dict[str, Any]]) -> AsyncIterator[str]:
    async for item in events:
        yield sse_encode(item.get("event", "message"), item.get("data", {}))


def stream_response(events: AsyncIterator[dict[str, Any]]) -> StreamingResponse:
    return StreamingResponse(
        sse_stream(events),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
