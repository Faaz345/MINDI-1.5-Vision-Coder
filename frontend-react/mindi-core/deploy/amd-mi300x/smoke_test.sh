#!/usr/bin/env bash
set -euo pipefail

API_URL="${1:-http://127.0.0.1:8000/v1}"
API_KEY="${2:-${MINDI_API_KEY:-}}"
MODEL="${3:-mindi-1.5}"

if [[ -z "${API_KEY}" ]]; then
  echo "Usage: ./smoke_test.sh http://127.0.0.1:8000/v1 YOUR_API_KEY [model]"
  exit 1
fi

curl -fsS "${API_URL}/models" \
  -H "Authorization: Bearer ${API_KEY}" \
  | python3 -m json.tool

curl -fsS "${API_URL}/chat/completions" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Reply with MINDI API ready.\"}],
    \"stream\": false,
    \"max_tokens\": 32
  }" \
  | python3 -m json.tool
