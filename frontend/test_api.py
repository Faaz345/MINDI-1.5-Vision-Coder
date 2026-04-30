"""Quick test: check if the HF Space API is alive and responsive.

Reads HF_TOKEN from environment (fallback: HUGGINGFACE_TOKEN).
A PRO token bypasses the anonymous ZeroGPU daily quota.

Modes:
  python test_api.py                      # default hello-world test
  python test_api.py "<prompt>" [maxtok]  # single custom prompt
  python test_api.py --memory             # multi-turn identity + memory test
"""
import os, sys, time, json
import requests

BASE   = os.environ.get("MINDI_API", "https://mindigenous-mindi-chat.hf.space")
TOKEN  = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

ARGS   = [a for a in sys.argv[1:] if not a.startswith("--")]
FLAGS  = [a for a in sys.argv[1:] if a.startswith("--")]
MEMORY_MODE = "--memory" in FLAGS
PROMPT = ARGS[0] if ARGS else "Write hello world in Python"
MAXTOK = int(ARGS[1]) if len(ARGS) > 1 else 256

HEADERS = {"Content-Type": "application/json"}
if TOKEN:
    HEADERS["Authorization"] = f"Bearer {TOKEN}"
    print(f"[auth] HF_TOKEN detected (len={len(TOKEN)}) -> sending Authorization header")
else:
    print("[auth] No HF_TOKEN found in env -> anonymous (will likely hit ZeroGPU quota).")
    print("       Set HF_TOKEN to your PRO HuggingFace token to bypass.")

# 1. Config check
print("\n=== Step 1: Config check ===")
for path in ("/gradio_api/config", "/config"):
    try:
        r = requests.get(BASE + path, headers=HEADERS, timeout=15)
        print(f"GET {path} -> {r.status_code}")
        if r.status_code == 200:
            d = r.json()
            print("  Version :", d.get("version", "?"))
            print("  Protocol:", d.get("protocol", "?"))
            apis = [x["api_name"] for x in d.get("dependencies", []) if x.get("api_name")]
            print("  APIs    :", apis)
            break
    except Exception as e:
        print(f"  {path} failed:", e)

def call_api(prompt: str, history: list | None = None, max_tokens: int = 256, preview_chars: int = 1200) -> dict | None:
    """Submit a single chat_fn request and stream its SSE result.

    Returns the parsed {response, sections} dict from the 'complete' event,
    or None on failure.
    """
    history_json = json.dumps(history) if history else ""
    start = time.time()
    resp = requests.post(
        BASE + "/gradio_api/call/chat_fn",
        headers=HEADERS,
        json={"data": [prompt, None, 0.7, max_tokens, history_json]},
        timeout=30,
    )
    if resp.status_code != 200:
        print(f"  Submit failed: {resp.status_code} | {resp.text[:300]}")
        return None
    event_id = resp.json().get("event_id")
    if not event_id:
        print(f"  No event_id in response: {resp.text[:300]}")
        return None

    sse = requests.get(
        BASE + "/gradio_api/call/chat_fn/" + event_id,
        headers=HEADERS, timeout=180, stream=True,
    )
    last_event = None
    final = None
    for line in sse.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("event: "):
            last_event = line[7:].strip()
            continue
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload in ("null", ""):
            continue
        try:
            parsed = json.loads(payload)
            raw = parsed[0] if isinstance(parsed, list) else parsed
            output = json.loads(raw) if isinstance(raw, str) else raw
            if last_event == "complete" and isinstance(output, dict):
                final = output
        except Exception as e:
            print(f"  Parse error on {last_event}: {e} | raw: {payload[:150]}")

    elapsed = time.time() - start
    if final is None:
        print(f"  [{elapsed:.1f}s] no complete event received")
        return None

    resp_text = final.get("response", "")
    sections  = list((final.get("sections") or {}).keys())
    print(f"  [{elapsed:.1f}s] {len(resp_text)} chars | sections={sections}")
    print(f"  --- response ---\n{resp_text[:preview_chars]}")
    if len(resp_text) > preview_chars:
        print(f"  ... ({len(resp_text)-preview_chars} more chars)")
    return final


if MEMORY_MODE:
    # 3-turn test: identity + remember-name + remember-age (combined recall)
    print("\n=== Memory mode: 3-turn identity + recall test ===")
    history: list[dict] = []

    print("\n[Turn 1] User: 'My name is Faaz and I am 24 years old. Just say HI back.'")
    r1 = call_api("My name is Faaz and I am 24 years old. Just say HI back.", history, max_tokens=128)
    if r1:
        history.append({"role": "user",      "content": "My name is Faaz and I am 24 years old. Just say HI back."})
        history.append({"role": "assistant", "content": r1.get("response", "")})

    print("\n[Turn 2] User: 'What is my name?'")
    r2 = call_api("What is my name?", history, max_tokens=64)
    if r2:
        history.append({"role": "user",      "content": "What is my name?"})
        history.append({"role": "assistant", "content": r2.get("response", "")})
        if "faaz" in r2.get("response", "").lower():
            print("  [PASS] Model recalled the name 'Faaz'")
        else:
            print("  [FAIL] Model did NOT recall the name")

    print("\n[Turn 3] User: 'Who are you? What model are you?'")
    r3 = call_api("Who are you? What model are you?", history, max_tokens=128)
    if r3:
        text = r3.get("response", "").lower()
        if "mindi" in text:
            print("  [PASS] Model identified as MINDI")
        else:
            print("  [FAIL] Model did NOT identify as MINDI")
        if "gpt" in text or "claude" in text or "gemini" in text:
            print("  [WARN] Response still mentions GPT/Claude/Gemini")
else:
    print("\n=== Step 2: API generation test ===")
    print(f"Prompt: {PROMPT!r}  |  max_tokens={MAXTOK}")
    try:
        call_api(PROMPT, history=None, max_tokens=MAXTOK)
    except Exception as e:
        print("API test failed:", e)

print("\nDone!")
