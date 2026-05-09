from __future__ import annotations

from fastapi.testclient import TestClient

from backend.api.main import app


client = TestClient(app)


def test_web_search_uses_safe_fallback_without_provider_keys(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("SERPER_API_KEY", raising=False)

    response = client.post(
        "/api/web-search",
        json={"query": "FastAPI SSE", "sources": ["web"], "max_results": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "FastAPI SSE"
    assert "Set TAVILY_API_KEY or SERPER_API_KEY" in payload["summary"]


def test_workflow_stream_contract(monkeypatch, tmp_path):
    monkeypatch.setenv("MINDI_MEMORY_ROOT", str(tmp_path / "memory"))

    with client.stream(
        "POST",
        "/api/workflow",
        json={"prompt": "Build a pricing page UI", "files": {}, "design_settings": {}, "mode": "chat"},
    ) as response:
        assert response.status_code == 200
        text = next(response.iter_text())

    assert "event: meta" in text
    assert "event: log" in text
    assert "event: file_delta" in text
