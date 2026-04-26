#!/usr/bin/env bash
set -euo pipefail

API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-5173}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY="${PYTHON_BIN}"
elif [[ -f .venv/bin/python ]]; then
  PY=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
elif command -v python >/dev/null 2>&1; then
  PY="python"
else
  echo "[dev] error: python not found" >&2; exit 1
fi

"${PY}" -c "import uvicorn" 2>/dev/null || {
  echo "[dev] error: uvicorn missing — run: pip install -e ." >&2; exit 1
}

free_port() {
  "${PY}" -c "
import socket, sys
p = int(sys.argv[1])
for c in range(p, p+100):
    try:
        s = socket.socket(); s.bind(('0.0.0.0', c)); s.close(); print(c); break
    except OSError:
        continue
else:
    sys.exit(1)
" "$1"
}

API_PORT="$(free_port "${API_PORT}")"
UI_PORT="$(free_port "${UI_PORT}")"

cleanup() { kill "${API_PID:-}" "${UI_PID:-}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

"${PY}" -m uvicorn patch2prod.server:app --host 0.0.0.0 --port "${API_PORT}" &
API_PID=$!

"${PY}" -m http.server "${UI_PORT}" --directory demo-ui &
UI_PID=$!

echo "[dev] API  http://localhost:${API_PORT}"
echo "[dev] UI   http://localhost:${UI_PORT}"

while kill -0 "${API_PID}" 2>/dev/null && kill -0 "${UI_PID}" 2>/dev/null; do
  sleep 1
done
