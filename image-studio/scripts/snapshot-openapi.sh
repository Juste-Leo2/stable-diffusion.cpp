#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

HOST="${OPENAPI_SNAPSHOT_HOST:-127.0.0.1}"
PORT="${OPENAPI_SNAPSHOT_PORT:-8011}"
OUT="tokens/api/openapi.json"
TMP="${OUT}.tmp"
LOG="${TMP}.log"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

.venv/bin/python -m uvicorn backend.server:app --host "$HOST" --port "$PORT" --lifespan off >"$LOG" 2>&1 &
SERVER_PID=$!

for _ in {1..50}; do
  if curl --fail --silent --show-error "http://${HOST}:${PORT}/openapi.json" -o "$TMP"; then
    mv "$TMP" "$OUT"
    rm -f "$LOG"
    echo "Wrote $OUT"
    exit 0
  fi
  sleep 0.2
done

echo "Failed to snapshot OpenAPI schema." >&2
echo "--- uvicorn log ---" >&2
cat "$LOG" >&2
exit 1
