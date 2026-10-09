#!/usr/bin/env bash
# One-command start: ./scripts/start.sh            (add --skip-tests to skip the in-image test run)
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi
if ! grep -qE '^API_KEY=.+' .env; then
  KEY=$(python3 -c "import secrets;print(secrets.token_urlsafe(32))" 2>/dev/null || openssl rand -base64 32 | tr -d '/+=')
  sed -i "s|^API_KEY=.*|API_KEY=${KEY}|" .env
  echo "Generated API_KEY in .env"
fi
chmod 600 .env

mkdir -p data models backups
# containers run as uid 10001: give it the persistent directories
if [ "$(stat -c %u data)" != "10001" ]; then
  chown -R 10001:10001 data models backups 2>/dev/null || sudo chown -R 10001:10001 data models backups
fi

if [ "${1:-}" != "--skip-tests" ]; then
  echo "Building and running the test suite inside the image..."
  docker build --target test -t ai-market-predictor:test .
fi
docker compose up -d --build

echo "Waiting for the API..."
for i in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:$(grep -E '^API_PORT=' .env | cut -d= -f2 || echo 8000)/api/health" >/dev/null 2>&1; then break; fi
  sleep 2
done
docker compose ps
PORT=$(grep -E '^API_PORT=' .env | cut -d= -f2); PORT=${PORT:-8000}
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo
echo "Dashboard:  http://${IP:-<server-ip>}:${PORT}/"
echo "API key:    $(grep -E '^API_KEY=' .env | cut -d= -f2)"
echo "First models are trained automatically once history is downloaded (usually 5-15 minutes)."
echo "For HTTPS set DOMAIN in .env and run: docker compose --profile https up -d"
