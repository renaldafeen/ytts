#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

: "${YTTS_TOKEN:?YTTS_TOKEN is required}"
export CBOX_PORT="${CBOX_PORT:-9126}"

args=()
if [[ -n "${YTTS_URL:-}" ]]; then
  args+=(--url "$YTTS_URL")
fi

exec python3 client/ytts_client.py "${args[@]}"
