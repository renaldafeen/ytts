#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

: "${CBOX_VENV:?CBOX_VENV is required}"
: "${CBOX_PORT:?CBOX_PORT is required}"
: "${YTTS_TOKEN:?YTTS_TOKEN is required}"
YTTS_HOST="${YTTS_HOST:-127.0.0.1}"

PIDFILE="ytts_server.pid"
if [[ -f "$PIDFILE" ]]; then
  oldpid="$(cat "$PIDFILE")"
  if [[ -n "$oldpid" ]] && kill -0 "$oldpid" 2>/dev/null; then
    echo "ytts server already running (pid $oldpid)" >&2
    exit 1
  fi
  rm -f "$PIDFILE"
fi

if [[ -f ytts_server.log ]] && [[ "$(stat -c%s ytts_server.log)" -gt 1048576 ]]; then
  mv ytts_server.log ytts_server.log.1
fi

# shellcheck disable=SC1091
source "$CBOX_VENV/bin/activate"

nohup python3 ./server/ytts_server.py \
  --server \
  --host "$YTTS_HOST" \
  --port "$CBOX_PORT" \
  --vmap voices/vmap.json \
  --aref voices/ytts_ref_1.wav \
  --text "Your personal Y-TTS server is now running, enabled and ready for operation!" \
  --exaggeration 0.35 \
  --pacing 0.8 \
  --volume 0.7 \
  >> ytts_server.log 2>&1 &
echo $! > "$PIDFILE"
