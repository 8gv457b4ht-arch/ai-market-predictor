"""Container healthcheck.

    python scripts/healthcheck.py api                 -> GET /api/health must return 200
    python scripts/healthcheck.py service <name> [s]  -> worker heartbeat in the DB must be younger than s seconds

Exit code 0 = healthy, 1 = unhealthy (Docker then restarts/flags the container).
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_api() -> int:
    url = os.getenv("HEALTH_URL", "http://127.0.0.1:8000/api/health")
    try:
        with urllib.request.urlopen(url, timeout=6) as r:
            ok = r.status == 200 and json.loads(r.read()).get("database") is True
    except Exception as exc:  # noqa: BLE001
        print(f"api unhealthy: {exc}")
        return 1
    print("api healthy" if ok else "api unhealthy")
    return 0 if ok else 1


def check_service(name: str, max_age: float) -> int:
    url = os.getenv("DATABASE_URL", f"sqlite:///{ROOT / 'data' / 'market.sqlite3'}")
    try:
        if url.startswith("sqlite:///"):
            con = sqlite3.connect(f"file:{url[len('sqlite:///'):]}?mode=ro", uri=True, timeout=10)
            row = con.execute("SELECT value_json FROM system_state WHERE key=?", (f"heartbeat:{name}",)).fetchone()
            con.close()
        else:  # pragma: no cover - PostgreSQL
            import psycopg  # type: ignore
            with psycopg.connect(url) as con:
                row = con.execute("SELECT value_json FROM system_state WHERE key=%s", (f"heartbeat:{name}",)).fetchone()
        if not row:
            print(f"{name}: no heartbeat yet")
            return 1
        age = time.time() - json.loads(row[0])["ts_ms"] / 1000
        print(f"{name}: heartbeat {age:.0f}s ago")
        return 0 if age <= max_age else 1
    except Exception as exc:  # noqa: BLE001
        print(f"{name} unhealthy: {exc}")
        return 1


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["api"]:
        sys.exit(check_api())
    if len(a) >= 2 and a[0] == "service":
        sys.exit(check_service(a[1], float(a[2]) if len(a) > 2 else 120))
    print(__doc__)
    sys.exit(2)
