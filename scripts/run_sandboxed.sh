#!/bin/bash
# Start SEVERANCE inside the macOS kernel sandbox (config/severance.sb):
# network limited to this machine's loopback, writes limited to this
# project and temp folders, personal folders unreadable.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "$PROJECT_DIR"

# 127.0.0.1, not "localhost": the sandbox blocks name resolution entirely.
export OLLAMA_API_BASE="${OLLAMA_API_BASE:-http://127.0.0.1:11434}"

exec /usr/bin/sandbox-exec \
  -D PROJECT_DIR="$PROJECT_DIR" \
  -D HOME_DIR="$(cd "$HOME" && pwd -P)" \
  -f "$PROJECT_DIR/config/severance.sb" \
  "$PROJECT_DIR/.venv/bin/python" -m uvicorn api.main:app --host 127.0.0.1 --port "${PORT:-8000}"
