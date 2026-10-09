"""Run one backend cycle (used by .github/workflows/backend.yml; also works from cron/systemd).

    python scripts/cloud_cycle.py --state state --public public

State (SQLite + models + backups) lives in --state and must be kept between runs.
Exit code is 0 even when an exchange is down: failures are recorded in public/state.json.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="state")
    ap.add_argument("--public", default="public")
    ap.add_argument("--no-learning", action="store_true")
    ap.add_argument("--no-news", action="store_true")
    a = ap.parse_args()
    state = Path(a.state).resolve()
    (state / "models").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("DATABASE_URL", f"sqlite:///{state / 'market.sqlite3'}")
    os.environ.setdefault("MODEL_DIR", str(state / "models"))
    os.environ.setdefault("BACKUP_DIR", str(state / "backups"))
    os.environ.setdefault("PUBLIC_DIR", str(Path(a.public).resolve()))
    os.environ.setdefault("PRIMARY_EXCHANGE", "auto")
    from backend.app.cloud.cycle import run_cycle
    from backend.app.config import get_settings
    from backend.app.db import get_db
    from backend.app.workers.runner import setup_logging
    setup_logging("cycle")
    report = run_cycle(get_db(), get_settings(), learn=not a.no_learning, news=not a.no_news)
    summary = {k: report.get(k) for k in ("ok", "duration_sec", "primary_exchange", "timings_sec", "errors")}
    summary["exchanges"] = {ex: {"rest_ok": v.get("ok"), "rest_kind": v.get("kind"), "host": v.get("host")}
                            for ex, v in (report.get("rest") or {}).items()}
    summary["ws"] = {ex: {"verified_live": v.get("verified_live"), "kind": v.get("kind"), "url": v.get("url"),
                          "error": v.get("error")} for ex, v in ((report.get("ws") or {}).get("streams") or {}).items()}
    summary["predictions"] = report.get("predictions")
    summary["learning"] = report.get("learning")
    print(json.dumps(summary, indent=2, default=str))
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write("## Backend cycle\n\n```json\n" + json.dumps(summary, indent=2, default=str)[:60000] + "\n```\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
