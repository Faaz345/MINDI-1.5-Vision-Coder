#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run as root: sudo HF_TOKEN=your_hf_token ./setup_mindi15_openai_server.sh"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_SOURCE="${SCRIPT_DIR}/mindi15_openai_server.py"
if [[ ! -f "${SERVER_SOURCE}" ]]; then
  echo "Missing ${SERVER_SOURCE}"
  exit 1
fi

VLLM_CONTAINER="${VLLM_CONTAINER:-rocm}"
MINDI_REPO_ID="${MINDI_REPO_ID:-Mindigenous/MINDI-1.5-Vision-Coder}"
MINDI_BASE_MODEL_ID="${MINDI_BASE_MODEL_ID:-Qwen/Qwen2.5-Coder-7B-Instruct}"
MINDI_SERVED_MODEL_NAME="${MINDI_SERVED_MODEL_NAME:-mindi-1.5}"
MINDI_CHECKPOINT_CACHE="${MINDI_CHECKPOINT_CACHE:-/shared-docker/models/mindi-1.5-cache}"
MINDI_CHECKPOINT_DIR="${MINDI_CHECKPOINT_DIR:-}"
MINDI_REQUIRE_CHECKPOINT="${MINDI_REQUIRE_CHECKPOINT:-true}"
MINDI_ALLOW_BASE_FALLBACK="${MINDI_ALLOW_BASE_FALLBACK:-false}"
MINDI_PORT="${MINDI_PORT:-8000}"
MINDI_API_KEY="${MINDI_API_KEY:-$(openssl rand -hex 32)}"
HF_TOKEN="${HF_TOKEN:-}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required on the AMD Developer Cloud image."
  exit 1
fi

if ! docker exec -i "${VLLM_CONTAINER}" python3 - <<'PY' >/dev/null 2>&1
import fastapi, huggingface_hub, peft, torch, transformers, uvicorn
PY
then
  echo "The '${VLLM_CONTAINER}' container is missing required Python packages."
  echo "Expected: fastapi, uvicorn, torch, transformers, peft, huggingface_hub."
  exit 1
fi

install -d -m 0755 /etc/mindigenous
install -d -m 0755 /shared-docker/mindigenous
install -d -m 0755 /shared-docker/models
install -m 0644 "${SERVER_SOURCE}" /shared-docker/mindigenous/mindi15_openai_server.py

cat >/etc/mindigenous/mindi-openai.env <<EOF
MINDI_API_KEY=${MINDI_API_KEY}
MINDI_PORT=${MINDI_PORT}
MINDI_SERVED_MODEL_NAME=${MINDI_SERVED_MODEL_NAME}
MINDI_REPO_ID=${MINDI_REPO_ID}
MINDI_BASE_MODEL_ID=${MINDI_BASE_MODEL_ID}
MINDI_CHECKPOINT_CACHE=${MINDI_CHECKPOINT_CACHE}
MINDI_CHECKPOINT_DIR=${MINDI_CHECKPOINT_DIR}
MINDI_REQUIRE_CHECKPOINT=${MINDI_REQUIRE_CHECKPOINT}
MINDI_ALLOW_BASE_FALLBACK=${MINDI_ALLOW_BASE_FALLBACK}
MINDI_DTYPE=bfloat16
MINDI_MAX_MODEL_LEN=8192
HF_TOKEN=${HF_TOKEN}
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
VLLM_CONTAINER=${VLLM_CONTAINER}
EOF
chmod 0600 /etc/mindigenous/mindi-openai.env

if [[ "${MINDI_REQUIRE_CHECKPOINT}" == "true" && "${MINDI_ALLOW_BASE_FALLBACK}" != "true" ]]; then
  echo "Checking for MINDI 1.5 phase3 checkpoint visibility..."
  if ! docker exec -i \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e MINDI_REPO_ID="${MINDI_REPO_ID}" \
    "${VLLM_CONTAINER}" python3 - <<'PY'
import os
import sys
from huggingface_hub import HfApi

repo = os.environ["MINDI_REPO_ID"]
token = os.environ.get("HF_TOKEN") or None
files = HfApi(token=token).list_repo_files(repo, repo_type="model")
if not any(path.startswith("checkpoints/phase3_final/") for path in files):
    print(f"Missing checkpoints/phase3_final in {repo}.")
    print("Set HF_TOKEN with access to the private checkpoint files or set MINDI_CHECKPOINT_DIR.")
    sys.exit(2)
print("MINDI 1.5 checkpoint files are visible.")
PY
  then
    echo "Preflight failed. Service was installed but not started."
    echo "Env file: /etc/mindigenous/mindi-openai.env"
    exit 2
  fi
fi

cat >/usr/local/bin/mindi-openai-runner <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
source /etc/mindigenous/mindi-openai.env

exec docker exec \
  -e MINDI_API_KEY="${MINDI_API_KEY}" \
  -e MINDI_PORT="${MINDI_PORT}" \
  -e MINDI_SERVED_MODEL_NAME="${MINDI_SERVED_MODEL_NAME}" \
  -e MINDI_REPO_ID="${MINDI_REPO_ID}" \
  -e MINDI_BASE_MODEL_ID="${MINDI_BASE_MODEL_ID}" \
  -e MINDI_CHECKPOINT_CACHE="${MINDI_CHECKPOINT_CACHE}" \
  -e MINDI_CHECKPOINT_DIR="${MINDI_CHECKPOINT_DIR}" \
  -e MINDI_REQUIRE_CHECKPOINT="${MINDI_REQUIRE_CHECKPOINT}" \
  -e MINDI_ALLOW_BASE_FALLBACK="${MINDI_ALLOW_BASE_FALLBACK}" \
  -e MINDI_DTYPE="${MINDI_DTYPE}" \
  -e MINDI_MAX_MODEL_LEN="${MINDI_MAX_MODEL_LEN}" \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}" \
  "${VLLM_CONTAINER}" \
  python3 /shared-docker/mindigenous/mindi15_openai_server.py
EOF
chmod 0755 /usr/local/bin/mindi-openai-runner

cat >/etc/systemd/system/mindi-openai.service <<'EOF'
[Unit]
Description=MINDI 1.5 OpenAI-compatible API
Requires=docker.service
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=simple
EnvironmentFile=/etc/mindigenous/mindi-openai.env
ExecStart=/usr/local/bin/mindi-openai-runner
Restart=on-failure
RestartSec=8
TimeoutStartSec=1200
LimitNOFILE=1048576

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable mindi-openai
systemctl restart mindi-openai

echo "MINDI OpenAI-compatible service started."
echo "Service logs: journalctl -u mindi-openai -f"
echo "Local API URL on droplet: http://127.0.0.1:${MINDI_PORT}/v1"
echo "Public API URL if firewall allows it: http://$(curl -fsS ifconfig.me || hostname -I | awk '{print $1}'):${MINDI_PORT}/v1"
echo "MINDI_API_KEY=${MINDI_API_KEY}"
