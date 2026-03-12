#!/usr/bin/env bash
set -euo pipefail

# 升级流程：
# 1. 备份当前 SQLite 数据
# 2. 尝试更新基础镜像与本地构建缓存
# 3. 无损重建并重启容器

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$PROJECT_DIR/data"
DB_FILE="$DATA_DIR/stock_monitor.db"
BACKUP_DIR="$DATA_DIR/backups"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$DATA_DIR" "$BACKUP_DIR"

if [[ -f "$DB_FILE" ]]; then
  cp "$DB_FILE" "$BACKUP_DIR/stock_monitor_${TIMESTAMP}.db"
  echo "[upgrade] database backup created: $BACKUP_DIR/stock_monitor_${TIMESTAMP}.db"
fi

cd "$PROJECT_DIR"

./prepare_env.sh

# `build --pull` 会尽量获取更新过的 Python 基础镜像，
# 即使项目源码没变，也能通过这个命令拿到更近的安全补丁。
docker compose build --pull
docker compose up -d --force-recreate --remove-orphans

echo "[upgrade] finished"
docker compose ps
