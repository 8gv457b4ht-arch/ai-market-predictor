#!/usr/bin/env bash
# Shows container state and the API's own component status.
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose ps
PORT=$(grep -E '^API_PORT=' .env | cut -d= -f2); PORT=${PORT:-8000}
KEY=$(grep -E '^API_KEY=' .env | cut -d= -f2)
echo; curl -fsS "http://127.0.0.1:${PORT}/api/ready" || true
echo; curl -fsS -H "X-API-Key: ${KEY}" "http://127.0.0.1:${PORT}/api/status" | python3 -c "
import json,sys
s=json.load(sys.stdin)
print('websocket :', s['websocket']['state'])
print('market    :', s['market_data']['state'])
print('news      :', s['news']['state'], s['news'].get('analyzer'))
print('model     :', s['model']['state'])
print('learning  :', s['learning']['state'])
print('database  :', s['database']['state'])
print('services  :', {k: v['alive'] for k, v in s['services'].items()})
print('last backup:', (s.get('last_backup') or {}).get('database'))"
