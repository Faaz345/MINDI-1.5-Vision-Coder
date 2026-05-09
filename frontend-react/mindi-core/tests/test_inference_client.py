from __future__ import annotations

from backend.inference.inference_client import InferenceClient


def test_vllm_base_url_adds_v1_chat_path(monkeypatch):
    monkeypatch.setenv("MINDI_API_URL", "http://165.245.141.245:8000")
    monkeypatch.setenv("MINDI_API_KEY", "test-key")

    client = InferenceClient()

    assert client.chat_completions_url == "http://165.245.141.245:8000/v1/chat/completions"


def test_v1_url_keeps_openai_path(monkeypatch):
    monkeypatch.setenv("MINDI_API_URL", "http://127.0.0.1:9000/v1")
    monkeypatch.setenv("MINDI_API_KEY", "test-key")

    client = InferenceClient()

    assert client.chat_completions_url == "http://127.0.0.1:9000/v1/chat/completions"
