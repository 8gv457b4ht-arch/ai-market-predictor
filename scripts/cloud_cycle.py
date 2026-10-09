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
    # GitHub starts scheduled runs late; wait (at most 16 min) for the next 15m close and predict on time
    os.environ.setdefault("WAIT_CLOSE_MAX_SEC", "960")
    os.environ.setdefault("CONTROL_DIR", str(ROOT / "control"))
    # the whole state is pushed to git after every run: fewer short-horizon rows and 1-second bars than a server keeps
    os.environ.setdefault("FORECAST_REFRESH_SCALE", "4")
    os.environ.setdefault("SHORT_FORECAST_RETENTION_DAYS", "3")
    os.environ.setdefault("SECOND_BARS_RETENTION_HOURS", "24")
    from backend.app.cloud.cycle import run_cycle
    from backend.app.cloud.health import ensure_database, restore_latest_good, sqlite_check
    from backend.app.config import get_settings
    from backend.app import control
    from backend.app.db import get_db
    from backend.app.workers.runner import setup_logging
    setup_logging("cycle")
    settings = get_settings()
    db_file = state / "market.sqlite3"
    # 1) before opening: a damaged database is replaced by the newest backup that passes integrity_check
    pre = ensure_database(db_file, settings.backup_dir)
    rid = control.pending_restore(settings)
    if rid:  # operator asked for a restore (control/commands.json)
        pre = {"status": "restore_requested", "command": rid, **restore_latest_good(db_file, settings.backup_dir)}
        control.mark_done(settings, rid)
    db = get_db()
    if pre.get("status") not in ("ok", "new"):
        db.log_event("database_recovery", pre)
        db.set_state("last_recovery", {"ts_ms": int(__import__("time").time() * 1000), **pre})
    report = run_cycle(db, settings, learn=not a.no_learning, news=not a.no_news)
    report["database_precheck"] = pre
    # 2) after the run: never hand a damaged database to the save step
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    post = sqlite_check(db_file, full=True)
    if post != "ok":
        db.conn.close()
        report["database_postcheck"] = {"problem": post, **restore_latest_good(db_file, settings.backup_dir)}
    else:
        report["database_postcheck"] = {"status": "ok"}
    summary = {k: report.get(k) for k in ("ok", "duration_sec", "primary_exchange", "timings_sec", "errors",
                                          "database_precheck", "database_postcheck", "wait_for_close")}
    summary["exchanges"] = {ex: {"rest_ok": v.get("ok"), "rest_kind": v.get("kind"), "host": v.get("host")}
                            for ex, v in (report.get("rest") or {}).items()}
    summary["ws"] = {ex: {"verified_live": v.get("verified_live"), "kind": v.get("kind"), "url": v.get("url"),
                          "error": v.get("error")} for ex, v in ((report.get("ws") or {}).get("streams") or {}).items()}
    summary["predictions"] = report.get("predictions")
    summary["predictions_at_close"] = report.get("predictions_at_close")
    summary["learning"] = report.get("learning")
    print(json.dumps(summary, indent=2, default=str))
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write("## Backend cycle\n\n```json\n" + json.dumps(summary, indent=2, default=str)[:60000] + "\n```\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
