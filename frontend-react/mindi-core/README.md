# MINDIGENOUS Core

`mindi-core` is the backend orchestration layer for MINDIGENOUS.

Architecture:

```txt
Workflows -> Agents -> Tools -> MINDI 1.5
```

Agents produce structured reasoning and plans. Tools execute deterministic work such as repo reads, lint/build commands, and web search provider calls.

## Run

```bash
cd mindi-core
python -m venv .venv
.venv\Scripts\activate
pip install -e .
uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000
```

Copy `.env.example` to `.env` and set `MINDI_API_URL` and `MINDI_API_KEY` for the AMD GPU-hosted MINDI endpoint.

For the AMD Developer Cloud MI300X vLLM image, use the deployment helper in:

```txt
deploy/amd-mi300x/
```

Recommended development connection is an SSH tunnel:

```bash
ssh -L 9000:127.0.0.1:8000 root@YOUR_DROPLET_PUBLIC_IP
```

Then set:

```env
MINDI_API_URL=http://127.0.0.1:9000/v1
MINDI_API_KEY=the_key_printed_by_setup_script
MINDI_MODEL=mindi-1.5
```

## API

- `POST /api/chat`
- `POST /api/workflow`
- `POST /api/web-search`

Streaming endpoints use SSE frames over a `fetch()` POST response.
