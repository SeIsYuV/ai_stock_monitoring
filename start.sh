#!/usr/bin/env bash
set -euo pipefail

# Docker 和本地都统一走这个入口，避免启动参数分叉。
export ASM_HOST="${ASM_HOST:-0.0.0.0}"
export ASM_PORT="${ASM_PORT:-1217}"
export ASM_DB_PATH="${ASM_DB_PATH:-/app/data/stock_monitor.db}"

mkdir -p "$(dirname "$ASM_DB_PATH")"

if [[ -x ".venv/bin/python" ]]; then
  exec ./.venv/bin/python main.py
fi

exec python main.py
