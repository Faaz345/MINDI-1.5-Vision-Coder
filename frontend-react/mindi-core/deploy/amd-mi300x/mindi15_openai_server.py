import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

try:
    from peft import PeftModel
except Exception:  # pragma: no cover - deployment guard
    PeftModel = None


API_KEY = os.getenv("MINDI_API_KEY", "")
MODEL_NAME = os.getenv("MINDI_SERVED_MODEL_NAME", "mindi-1.5")
REPO_ID = os.getenv("MINDI_REPO_ID", "Mindigenous/MINDI-1.5-Vision-Coder")
BASE_MODEL_ID = os.getenv("MINDI_BASE_MODEL_ID", "Qwen/Qwen2.5-Coder-7B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or None
CHECKPOINT_CACHE = Path(os.getenv("MINDI_CHECKPOINT_CACHE", "/shared-docker/models/mindi-1.5-cache"))
CHECKPOINT_DIR = os.getenv("MINDI_CHECKPOINT_DIR", "")
REQUIRE_CHECKPOINT = os.getenv("MINDI_REQUIRE_CHECKPOINT", "true").lower() != "false"
ALLOW_BASE_FALLBACK = os.getenv("MINDI_ALLOW_BASE_FALLBACK", "false").lower() == "true"
DTYPE = torch.bfloat16 if os.getenv("MINDI_DTYPE", "bfloat16") == "bfloat16" else torch.float16
MAX_MODEL_LEN = int(os.getenv("MINDI_MAX_MODEL_LEN", "8192"))


SYSTEM_PROMPT = (
    "You are MINDI 1.5 Vision-Coder, an AI coding assistant created by MINDIGENOUS.AI. "
    "You are built for production web and app development. Return complete, usable code "
    "when asked to build or edit software."
)


app = FastAPI(title="MINDI 1.5 OpenAI-compatible API")
_state: dict[str, Any] = {"tokenizer": None, "model": None, "loaded": False, "source": None}
_load_lock = threading.Lock()
_generation_lock = threading.Lock()


def _check_auth(request: Request) -> None:
    if not API_KEY:
        return
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")


def _download_checkpoint() -> Path:
    if CHECKPOINT_DIR:
        path = Path(CHECKPOINT_DIR)
        if not path.exists():
            raise RuntimeError(f"MINDI_CHECKPOINT_DIR does not exist: {path}")
        return path

    CHECKPOINT_CACHE.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        REPO_ID,
        repo_type="model",
        local_dir=str(CHECKPOINT_CACHE),
        token=HF_TOKEN,
        allow_patterns=[
            "checkpoints/phase3_final/**",
            "data/tokenizer/**",
        ],
    )
    return CHECKPOINT_CACHE


def _assert_checkpoint_available() -> None:
    api = HfApi(token=HF_TOKEN)
    files = api.list_repo_files(REPO_ID, repo_type="model")
    has_phase3 = any(path.startswith("checkpoints/phase3_final/") for path in files)
    if not has_phase3 and REQUIRE_CHECKPOINT and not ALLOW_BASE_FALLBACK:
        raise RuntimeError(
            "MINDI 1.5 checkpoint files are not visible in the Hugging Face repo. "
            "Set HF_TOKEN for the private checkpoint, set MINDI_CHECKPOINT_DIR to a local "
            "checkpoint path, or export a merged vLLM-compatible model first."
        )


def _load_model() -> None:
    if _state["loaded"]:
        return

    with _load_lock:
        if _state["loaded"]:
            return

        _assert_checkpoint_available()
        ckpt_dir = _download_checkpoint()

        tokenizer_path = ckpt_dir / "data" / "tokenizer" / "mindi_tokenizer"
        tokenizer_source = str(tokenizer_path) if tokenizer_path.exists() else BASE_MODEL_ID
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=DTYPE,
            device_map="auto",
            trust_remote_code=True,
        )
        model.resize_token_embeddings(len(tokenizer))

        lora_path = ckpt_dir / "checkpoints" / "phase3_final" / "lora"
        if lora_path.exists():
            if PeftModel is None:
                raise RuntimeError("peft is required to load the MINDI 1.5 LoRA checkpoint.")
            model = PeftModel.from_pretrained(model, str(lora_path))
            source = f"{REPO_ID}:checkpoints/phase3_final/lora"
        elif REQUIRE_CHECKPOINT and not ALLOW_BASE_FALLBACK:
            raise RuntimeError(f"MINDI LoRA checkpoint missing: {lora_path}")
        else:
            source = BASE_MODEL_ID

        model.eval()
        _state.update({"tokenizer": tokenizer, "model": model, "loaded": True, "source": source})


def _format_messages(messages: list[dict[str, str]]) -> str:
    tokenizer = _state["tokenizer"]
    normalized = [m for m in messages if m.get("role") in {"system", "user", "assistant"}]
    if not normalized or normalized[0].get("role") != "system":
        normalized.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(normalized, tokenize=False, add_generation_prompt=True)

    rendered = []
    for msg in normalized:
        rendered.append(f"{msg['role']}: {msg.get('content', '')}")
    rendered.append("assistant:")
    return "\n".join(rendered)


def _chunk_payload(request_id: str, token: str, finish_reason: str | None = None) -> str:
    payload = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "delta": {"content": token} if token else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def _completion_payload(request_id: str, content: str) -> dict[str, Any]:
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def _generate_tokens(messages: list[dict[str, str]], temperature: float, max_tokens: int):
    _load_model()
    tokenizer = _state["tokenizer"]
    model = _state["model"]
    prompt = _format_messages(messages)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_MODEL_LEN)
    first_device = next(model.parameters()).device
    inputs = {key: value.to(first_device) for key, value in inputs.items()}

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_tokens,
        "temperature": max(float(temperature), 0.01),
        "do_sample": float(temperature) > 0,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }

    with _generation_lock:
        worker = threading.Thread(target=model.generate, kwargs=kwargs)
        worker.start()
        for text in streamer:
            if text:
                yield text
        worker.join()


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "model": MODEL_NAME, "loaded": _state["loaded"], "source": _state["source"]}


@app.get("/v1/models")
async def models(request: Request) -> dict[str, Any]:
    _check_auth(request)
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "mindigenous"}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    _check_auth(request)
    body = await request.json()
    messages = body.get("messages") or []
    if not messages:
        raise HTTPException(status_code=400, detail="messages is required")

    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    temperature = float(body.get("temperature", 0.7))
    max_tokens = int(body.get("max_tokens", body.get("max_completion_tokens", 1024)))
    stream = bool(body.get("stream", True))

    if stream:
        def event_stream():
            try:
                for token in _generate_tokens(messages, temperature, max_tokens):
                    yield _chunk_payload(request_id, token)
                yield _chunk_payload(request_id, "", "stop")
                yield "data: [DONE]\n\n"
            except Exception as exc:
                payload = {"error": {"message": str(exc), "type": "mindi_server_error"}}
                yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    try:
        content = "".join(_generate_tokens(messages, temperature, max_tokens))
    except Exception as exc:
        return JSONResponse(status_code=503, content={"error": {"message": str(exc), "type": "mindi_server_error"}})
    return _completion_payload(request_id, content)


if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv("MINDI_HOST", "0.0.0.0"), port=int(os.getenv("MINDI_PORT", "8000")))
