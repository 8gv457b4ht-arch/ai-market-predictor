"""Operator commands without secrets or a server: a JSON file in the repository (control/commands.json).

    {"commands": [{"id": "2026-10-10-rollback-btc", "action": "rollback", "model_key": "BTC/USDT|15m|h4"}]}

Each command id is executed once (ids are remembered in the database). Supported actions:
  rollback        -> previous working model of `model_key` back to production (optionally `to_version`)
  restore_backup  -> marks the database for restore from the newest good backup at the next start
  backup          -> make a verified backup now
Notification preferences live in control/notify.json (no secrets: tokens stay in GitHub Secrets).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .config import Settings
from .db import Database, now_ms

log = logging.getLogger("control")
NOTIFY_KEYS = {"enabled": "notify_enabled", "events": "notify_events", "min_confidence": "notify_min_confidence",
               "timeframes": "notify_timeframes", "symbols": "notify_symbols", "max_age_sec": "notify_max_age_sec",
               "min_quality": "notify_min_quality", "language": "notify_lang"}


def _read(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.error("cannot read %s: %s", path, exc)
        return {"_error": str(exc)}


def apply_notify_file(settings: Settings) -> dict:
    """Overlay control/notify.json on the settings (environment variables stay the default)."""
    cfg = _read(Path(settings.control_dir) / "notify.json")
    if not cfg:
        return {"source": "environment"}
    if "_error" in cfg:
        return {"source": "environment", "error": cfg["_error"]}
    applied = {}
    for k, attr in NOTIFY_KEYS.items():
        if k in cfg:
            v = cfg[k]
            if attr in ("notify_timeframes", "notify_events"):
                v = [str(x) for x in (v if isinstance(v, list) else str(v).split(","))]
            elif attr == "notify_symbols":
                v = [str(x).upper() for x in (v if isinstance(v, list) else str(v).split(","))]
            elif attr in ("notify_min_confidence", "notify_max_age_sec", "notify_min_quality"):
                v = float(v)
            elif attr == "notify_enabled":
                v = bool(v)
            elif attr == "notify_lang":
                v = str(v).lower() if str(v).lower() in ("ru", "uk", "en") else "ru"
            setattr(settings, attr, v)
            applied[k] = v
    return {"source": "control/notify.json", "applied": applied}


def _done_path(settings: Settings) -> Path:
    # kept next to the database, not inside it: a restored database must not forget executed commands
    return Path(settings.model_dir).parent / "control_done.json"


def _commands(settings: Settings) -> list[dict]:
    cfg = _read(Path(settings.control_dir) / "commands.json") or {}
    return [c for c in cfg.get("commands", []) if isinstance(c, dict)] if isinstance(cfg, dict) else []


def _load_done(settings: Settings) -> set[str]:
    d = _read(_done_path(settings))
    return set(d.get("done", [])) if isinstance(d, dict) else set()


def _save_done(settings: Settings, done: set[str]) -> None:
    p = _done_path(settings)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"done": sorted(done)}), encoding="utf-8")
    tmp.replace(p)


def pending_restore(settings: Settings) -> str | None:
    """Id of a not-yet-executed restore_backup command (handled before the database is opened)."""
    done = _load_done(settings)
    for c in _commands(settings):
        if c.get("action") == "restore_backup" and str(c.get("id") or "") and str(c["id"]) not in done:
            return str(c["id"])
    return None


def mark_done(settings: Settings, cid: str) -> None:
    _save_done(settings, _load_done(settings) | {cid})


def run_commands(db: Database, settings: Settings) -> list[dict]:
    from .learning import registry
    from .workers.backup import run_backup_once
    done = _load_done(settings) | set(db.get_state("control_done", []) or [])
    results = []
    for c in _commands(settings):
        cid = str(c.get("id") or "")
        if not cid or cid in done:
            continue
        action = c.get("action")
        try:
            if action == "rollback":
                r = registry.rollback(db, c["model_key"], settings.model_dir, f"operator command {cid}", c.get("to_version"))
            elif action == "restore_backup":
                r = {"status": "handled before the database was opened"}
            elif action == "backup":
                r = run_backup_once(db, settings)
            else:
                r = {"status": "unknown action"}
        except Exception as exc:  # noqa: BLE001
            r = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        done.add(cid)
        results.append({"id": cid, "action": action, "result": r})
        db.log_event("operator_command", {"id": cid, "action": action, "result": r}, c.get("model_key"))
    if results:
        db.set_state("control_done", sorted(done))
        _save_done(settings, done)
    return results
