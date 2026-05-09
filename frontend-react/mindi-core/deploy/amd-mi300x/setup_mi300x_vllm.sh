#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run as root: sudo MINDI_MODEL_PATH=/shared-docker/models/mindi-1.5 ./setup_mi300x_vllm.sh"
  exit 1
fi

MINDI_MODEL_PATH="${MINDI_MODEL_PATH:-}"
if [[ -z "${MINDI_MODEL_PATH}" ]]; then
  echo "MINDI_MODEL_PATH is required. Use /shared-docker/models/mindi-1.5 or a model repo id."
  exit 1
fi

MINDI_SERVED_MODEL_NAME="${MINDI_SERVED_MODEL_NAME:-mindi-1.5}"
MINDI_PORT="${MINDI_PORT:-8000}"
MINDI_API_KEY="${MINDI_API_KEY:-$(openssl rand -hex 32)}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---trust-remote-code}"
VLLM_CONTAINER="${VLLM_CONTAINER:-rocm}"
HF_TOKEN="${HF_TOKEN:-}"

VLLM_BIN="$(command -v vllm || true)"
VLLM_RUN_MODE="${VLLM_RUN_MODE:-host}"
if [[ -z "${VLLM_BIN}" ]]; then
  if command -v docker >/dev/null 2>&1 && docker exec "${VLLM_CONTAINER}" bash -lc 'command -v vllm' >/dev/null 2>&1; then
    VLLM_RUN_MODE="docker-exec"
    VLLM_BIN="vllm"
  else
    echo "vllm was not found on host or inside container '${VLLM_CONTAINER}'."
    exit 1
  fi
fi

if command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi || true
else
  echo "rocm-smi not found. ROCm may not be initialized correctly."
fi

install -d -m 0755 /etc/mindigenous
install -d -m 0755 /var/log/mindigenous

cat >/etc/mindigenous/mindi-vllm.env <<EOF
MINDI_MODEL_PATH=${MINDI_MODEL_PATH}
MINDI_SERVED_MODEL_NAME=${MINDI_SERVED_MODEL_NAME}
MINDI_API_KEY=${MINDI_API_KEY}
MINDI_PORT=${MINDI_PORT}
VLLM_RUN_MODE=${VLLM_RUN_MODE}
VLLM_CONTAINER=${VLLM_CONTAINER}
VLLM_BIN=${VLLM_BIN}
VLLM_DTYPE=${VLLM_DTYPE}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN}
VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS}
HF_TOKEN=${HF_TOKEN}
EOF
chmod 0600 /etc/mindigenous/mindi-vllm.env

cat >/usr/local/bin/mindi-vllm-runner <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
source /etc/mindigenous/mindi-vllm.env

ARGS=(
  serve "$MINDI_MODEL_PATH"
  --host 0.0.0.0
  --port "$MINDI_PORT"
  --served-model-name "$MINDI_SERVED_MODEL_NAME"
  --api-key "$MINDI_API_KEY"
  --dtype "$VLLM_DTYPE"
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
  --max-model-len "$VLLM_MAX_MODEL_LEN"
)

if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
  read -r -a EXTRA <<< "$VLLM_EXTRA_ARGS"
  ARGS+=("${EXTRA[@]}")
fi

if [[ "${VLLM_RUN_MODE}" == "docker-exec" ]]; then
  exec docker exec \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
    "${VLLM_CONTAINER}" \
    bash -lc "exec vllm ${ARGS[*]}"
fi

exec "$VLLM_BIN" "${ARGS[@]}"
EOF
chmod 0755 /usr/local/bin/mindi-vllm-runner

cat >/etc/systemd/system/mindi-vllm.service <<'EOF'
[Unit]
Description=MINDI 1.5 vLLM OpenAI-compatible API
Requires=docker.service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
EnvironmentFile=/etc/mindigenous/mindi-vllm.env
WorkingDirectory=/root
ExecStart=/usr/local/bin/mindi-vllm-runner
Restart=on-failure
RestartSec=8
TimeoutStartSec=900
LimitNOFILE=1048576

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable mindi-vllm
systemctl restart mindi-vllm

echo "MINDI vLLM service started."
echo "Service logs: journalctl -u mindi-vllm -f"
echo "Local API URL on droplet: http://127.0.0.1:${MINDI_PORT}/v1"
echo "Public API URL if firewall allows it: http://$(curl -fsS ifconfig.me || hostname -I | awk '{print $1}'):${MINDI_PORT}/v1"
echo "MINDI_API_KEY=${MINDI_API_KEY}"
