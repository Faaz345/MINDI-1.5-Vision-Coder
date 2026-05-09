from __future__ import annotations

from backend.services.memory_service import MemoryService


class MemoryAgent:
    def __init__(self, memory_service: MemoryService | None = None) -> None:
        self.memory_service = memory_service or MemoryService()

    async def recall(self, project_id: str | None, prompt: str) -> dict[str, str]:
        return await self.memory_service.recall(project_id, prompt)

    async def remember(self, project_id: str | None, event: dict) -> None:
        await self.memory_service.append_session_event(project_id, event)
