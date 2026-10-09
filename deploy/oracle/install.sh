#!/usr/bin/env bash
# AI Market Predictor - always-on backend on Oracle Cloud Free (Ubuntu 22.04/24.04, Ampere A1 or AMD).
#
#   curl -fsSL https://raw.githubusercontent.com/8gv457b4ht-arch/ai-market-predictor/main/deploy/oracle/install.sh -o install.sh
#   sudo bash install.sh
#
# What it does (idempotent, safe to run again for updates):
#   1. Docker Engine + compose plugin, enabled at boot; swap on small machines
#   2. code in /opt/ai-market-predictor (git clone / git pull)
#   3. continues from the published state (database, models, backups from the `data` branch)
#   4. .env with BACKEND_MODE=continuous; asks for the GitHub publishing token (stored only in .env, mode 600)
#   5. docker compose up: collector (persistent WebSocket), predictor, forecaster, learner, news, backup, publisher
#      - every service restarts after a crash and after a reboot; the API listens on localhost only
# Read-only research system: no exchange keys, no orders.
set -euo pipefail
REPO_URL="${REPO_URL:-https://github.com/8gv457b4ht-arch/ai-market-predictor.git}"
REPO_SLUG="${REPO_SLUG:-8gv457b4ht-arch/ai-market-predictor}"
DIR="${DIR:-/opt/ai-market-predictor}"
ask() { local prompt="$1" var; if [ -t 0 ] || [ -r /dev/tty ]; then read -r -p "$prompt" var </dev/tty || true; fi; echo "${var:-}"; }
asks() { local prompt="$1" var; if [ -r /dev/tty ]; then read -r -s -p "$prompt" var </dev/tty || true; echo >/dev/tty; fi; echo "${var:-}"; }

[ "$(id -u)" = 0 ] || { echo "Run with sudo: sudo bash install.sh"; exit 1; }
echo "== 1/5 Docker"
if ! command -v docker >/dev/null; then curl -fsSL https://get.docker.com | sh; fi
systemctl enable --now docker containerd
apt-get install -y -q git ca-certificates curl >/dev/null
MEM_MB=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)
if [ "$MEM_MB" -lt 3000 ] && ! swapon --show | grep -q swapfile; then
  echo "   small machine (${MEM_MB} MB): adding 4 GB swap"
  fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile >/dev/null && swapon /swapfile
  grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

echo "== 2/5 Code"
if [ -d "$DIR/.git" ]; then git -C "$DIR" pull -q --ff-only; else git clone -q "$REPO_URL" "$DIR"; fi
cd "$DIR"
mkdir -p data models backups control

echo "== 3/5 State"
if [ ! -f data/market.sqlite3 ]; then
  T=$(mktemp -d)
  if git -C "$T" init -q && git -C "$T" fetch -q --depth 1 "$REPO_URL" data 2>/dev/null; then
    git -C "$T" archive FETCH_HEAD state | tar -x -C "$T"
    [ -f "$T/state/market.sqlite3.gz" ] && gunzip -c "$T/state/market.sqlite3.gz" > data/market.sqlite3
    cp -n "$T"/state/models/* models/ 2>/dev/null || true
    cp -n "$T"/state/backups/* backups/ 2>/dev/null || true
    echo "   continuing from the published state ($(du -h data/market.sqlite3 | cut -f1) database)"
  else
    echo "   no published state found: starting empty"
  fi
  rm -rf "$T"
fi
chown -R 10001:10001 data models backups   # the containers run as the unprivileged user 10001

echo "== 4/5 Settings"
if [ ! -f .env ]; then
  cp .env.example .env
  set_env() { if grep -q "^$1=" .env; then sed -i "s|^$1=.*|$1=$2|" .env; else echo "$1=$2" >> .env; fi; }
  set_env BACKEND_MODE continuous
  set_env PRIMARY_EXCHANGE binance
  set_env BIND_ADDRESS 127.0.0.1
  set_env PUBLISH_REPO "$REPO_SLUG"
  set_env FORECAST_TRAIN_BUDGET_SEC 900
  set_env LOG_FORMAT json
  echo "   GitHub token so the server can publish data for the website"
  echo "   (fine-grained token, only this repository, permission Contents: Read and write)."
  TOKEN=$(asks "   Paste the token (input hidden, Enter to skip): ")
  [ -n "$TOKEN" ] && set_env PUBLISH_TOKEN "$TOKEN"
  NTFY=$(ask "   ntfy topic for phone alerts (optional, Enter to skip): ")
  [ -n "$NTFY" ] && set_env NTFY_TOPIC "$NTFY"
fi
chmod 600 .env

echo "== 5/5 Start"
docker compose up -d --build
sleep 20
docker compose ps
echo
echo "Done. Health:  curl -s http://127.0.0.1:8000/api/health ; logs: docker compose logs -f --tail 50 forecaster"
echo "Last step on GitHub: Settings -> Secrets and variables -> Actions -> Variables -> BACKEND_MODE = external"
echo "Update later: sudo bash $DIR/deploy/oracle/install.sh"
