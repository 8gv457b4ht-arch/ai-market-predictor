"""Shared test helpers: every test gets an isolated temporary database and model
directory, so tests never touch real data."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def make_env(tmp: Path, **overrides) -> dict:
    env = {
        "DATABASE_URL": f"sqlite:///{tmp}/test.sqlite3",
        "MODEL_DIR": str(tmp / "models"),
        "BACKUP_DIR": str(tmp / "backups"),
        "SYMBOLS": "BTC/USDT",
        "EXCHANGES": "binance,bybit,okx",
        "PRIMARY_EXCHANGE": "binance",
        "TIMEFRAMES": "15m,1h,4h,1d",
        "PREDICT_TIMEFRAMES": "15m",
        "HORIZON_BARS": "4",
        "API_KEY": "",
        "NEWS_FEEDS": "https://example.invalid/feed.xml",
        "ANTHROPIC_API_KEY": "",
        "OPENAI_API_KEY": "",
        "CONTROL_DIR": str(tmp / "control"),  # never the repository's real control files
        "WAIT_CLOSE_MAX_SEC": "0",
    }
    env.update({k: str(v) for k, v in overrides.items()})
    return env


def fresh_settings(monkeypatch, tmp: Path, **overrides):
    """Apply env overrides, reset cached settings/DB and return (settings, db)."""
    from backend.app import config, db as dbmod
    for k, v in make_env(tmp, **overrides).items():
        monkeypatch.setenv(k, v)
    config.reset_settings()
    dbmod._db_cache.clear()
    return config.get_settings(), dbmod.get_db()
