"""Quick test: check if the HF Space API is alive and responsive.

Reads HF_TOKEN from environment (fallback: HUGGINGFACE_TOKEN).
A PRO token bypasses the anonymous ZeroGPU daily quota.
"""
import os, sys, time, json
import requests

BASE   = os.environ.get("MINDI_API", "https://mindigenous-mindi-chat.hf.space")
TOKEN  = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
PROMPT = sys.argv[1] if len(sys.argv) > 1 else "Write hello world in Python"
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 256

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

# 2. Quick generation test
print("\n=== Step 2: API generation test ===")
print(f"Prompt: {PROMPT!r}  |  max_tokens={MAXTOK}")
try:
    start = time.time()
    resp = requests.post(
        BASE + "/gradio_api/call/chat_fn",
        headers=HEADERS,
        json={"data": [PROMPT, None, 0.7, MAXTOK]},
        timeout=30,
    )
    print("Submit status:", resp.status_code)
    if resp.status_code != 200:
        print("Error body:", resp.text[:400])
        sys.exit(1)

    event_id = resp.json().get("event_id")
    print("Event ID:", event_id)
    if not event_id:
        print("No event_id returned:", resp.text[:300])
        sys.exit(1)

    sse = requests.get(
        BASE + "/gradio_api/call/chat_fn/" + event_id,
        headers=HEADERS, timeout=180, stream=True,
    )
    print("SSE status :", sse.status_code)

    last_event = None
    got_complete = False
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
            resp_text = output.get("response", "") if isinstance(output, dict) else str(output)
            sections  = list(output.get("sections", {}).keys()) if isinstance(output, dict) else []
            elapsed = time.time() - start
            print(f"\n--- {last_event or 'data'} ({elapsed:.1f}s) ---")
            print("Length  :", len(resp_text), "chars")
            print("Sections:", sections)
            print("Preview :")
            print(resp_text[:1200])
            if len(resp_text) > 1200:
                print(f"... ({len(resp_text)-1200} more chars)")
            if last_event == "complete":
                got_complete = True
        except Exception as e:
            print(f"Parse error on {last_event}: {e} | raw: {payload[:200]}")

    if not got_complete:
        print("\n[!] No 'complete' event received.")
except Exception as e:
    print("API test failed:", e)

print("\nDone!")
