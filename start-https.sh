#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

./prepare_env.sh

echo "[https] starting app + https proxy profile"
COMPOSE_PROFILES=https docker compose up -d --build
