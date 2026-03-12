#!/usr/bin/env bash
set -euo pipefail

cat >/etc/docker/daemon.json <<'JSON'
{
  "registry-mirrors": [
    "https://dockerproxy.com",
    "https://hub-mirror.c.163.com",
    "https://mirror.baidubce.com"
  ]
}
JSON

systemctl restart docker

echo "[docker] registry mirrors configured"
