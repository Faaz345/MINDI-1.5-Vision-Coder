from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class MemoryService:
    def __init__(self, root: Path | None = None) -> None:
        env_root = os.getenv("MINDI_MEMORY_ROOT")
        self.root = root or (Path(env_root) if env_root else Path(__file__).resolve().parents[2] / "memory")
        self.session_dir = self.root / "session_memory"
        self.project_dir = self.root / "project_memory"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.project_dir.mkdir(parents=True, exist_ok=True)

    async def recall(self, project_id: str | None, prompt: str) -> dict[str, str]:
        project_key = project_id or "local"
        summary_path = self.project_dir / f"{project_key}.json"
        if not summary_path.exists():
            return {"summary": "", "prompt_hint": prompt[:240]}
        return await asyncio.to_thread(self._read_json, summary_path)

    async def append_session_event(self, project_id: str | None, event: dict[str, Any]) -> None:
        project_key = project_id or "local"
        payload = {"ts": datetime.now(timezone.utc).isoformat(), "project_id": project_key, **event}
        path = self.session_dir / f"{project_key}.jsonl"
        await asyncio.to_thread(self._append_jsonl, path, payload)

    async def save_project_summary(self, project_id: str | None, summary: str) -> None:
        project_key = project_id or "local"
        path = self.project_dir / f"{project_key}.json"
        await asyncio.to_thread(self._write_json, path, {"summary": summary})

    def _read_json(self, path: Path) -> dict[str, str]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _write_json(self, path: Path, payload: dict[str, str]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
