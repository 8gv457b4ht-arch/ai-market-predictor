"""Self-checks around a cycle: database integrity with automatic restore from the newest good backup,
production-model artifacts with automatic rollback, and the model-vs-naive-baseline test.

Nothing here invents data: when something cannot be repaired it is reported, and predictions that
depend on it are blocked with an explicit reason.
"""
from __future__ import annotations

import gzip
import logging
import shutil
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import TIMEFRAME_MS, Settings
from ..db import Database, now_ms

log = logging.getLogger("health")


# ------------------------------------------------------------------ database
def sqlite_check(path: str | Path, full: bool = False) -> str:
    """'ok' or the first problem reported by SQLite (also catches 'file is not a database')."""
    try:
        c = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
        try:
            r = c.execute("PRAGMA integrity_check" if full else "PRAGMA quick_check").fetchone()[0]
            c.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
            return str(r)
        finally:
            c.close()
    except sqlite3.Error as exc:
        return f"{type(exc).__name__}: {exc}"


def _backups(backup_dir: Path) -> list[Path]:
    return sorted(Path(backup_dir).glob("market_*.sqlite3.gz"), key=lambda p: p.name, reverse=True)


def restore_latest_good(db_path: str | Path, backup_dir: Path) -> dict:
    """Restore the newest backup that passes a full integrity check. The damaged file is kept aside."""
    db_path = Path(db_path)
    tried = []
    for b in _backups(backup_dir):
        tmp = db_path.with_name(db_path.name + ".restore")
        try:
            with gzip.open(b, "rb") as fi, open(tmp, "wb") as fo:
                shutil.copyfileobj(fi, fo)
        except (OSError, EOFError) as exc:
            tried.append({"backup": b.name, "result": f"unreadable: {exc}"})
            tmp.unlink(missing_ok=True)
            continue
        res = sqlite_check(tmp, full=True)
        tried.append({"backup": b.name, "result": res})
        if res != "ok":
            tmp.unlink(missing_ok=True)
            continue
        if db_path.exists():
            db_path.replace(db_path.with_name(f"{db_path.name}.damaged-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"))
        for suffix in ("-wal", "-shm"):
            Path(str(db_path) + suffix).unlink(missing_ok=True)
        tmp.replace(db_path)
        return {"restored": True, "backup": b.name, "tried": tried}
    return {"restored": False, "tried": tried}


def ensure_database(db_path: str | Path, backup_dir: Path) -> dict:
    """Run before the database is opened. A missing file is fine (first run); a damaged one is restored."""
    db_path = Path(db_path)
    if not db_path.exists():
        return {"status": "new"}
    res = sqlite_check(db_path)
    if res == "ok":
        return {"status": "ok"}
    log.error("database check failed: %s", res)
    out = {"status": "damaged", "problem": res, **restore_latest_good(db_path, backup_dir)}
    if not out["restored"]:
        # keep the damaged file for inspection and start empty rather than crash every run
        db_path.replace(db_path.with_name(f"{db_path.name}.damaged-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"))
        out["status"] = "damaged_no_backup"
    return out


# ------------------------------------------------------------------ models
def verify_models(db: Database, settings: Settings) -> dict:
    """Every production model must load; otherwise roll back to the previous working version."""
    from ..learning import registry
    out = {}
    for sym in settings.symbols:
        for tf in settings.predict_timeframes:
            key = registry.model_key(sym, tf, settings.horizon_bars)
            prod = registry.get_production(db, key)
            if prod is None:
                out[key] = {"status": "not_ready"}
                continue
            ok, why = registry.artifact_ok(prod["artifact_path"], settings.model_dir)
            if ok:
                out[key] = {"status": "ok", "version": prod["version"]}
                continue
            log.error("production model %s unusable (%s): rolling back", prod["version"], why)
            out[key] = {"status": "broken", "version": prod["version"], "problem": why,
                        "rollback": registry.rollback(db, key, settings.model_dir, f"artifact {why}")}
    return out


def naive_for_oos(db: Database, settings: Settings, symbol: str, tf: str, oos: pd.DataFrame) -> pd.DataFrame | None:
    """Rebuild the naive (training base-rate) forecast for an OOS file written before it stored one:
    for every fold, class frequencies of the labelled rows before that fold minus the purge gap."""
    from ..ml.dataset import build_training_frame
    frame, _ = build_training_frame(db, settings, symbol, tf)
    if frame.empty:
        return None
    lab = frame.dropna(subset=["label"]).reset_index(drop=True)
    ts = lab["open_ts"].to_numpy("int64")
    y = lab["label"].astype(int).to_numpy()
    out = oos.copy()
    for fold, part in oos.groupby("fold"):
        s = int(np.searchsorted(ts, int(part["open_ts"].iloc[0])))
        ytr = y[: max(0, s - settings.horizon_bars)]
        if len(ytr) < 50:
            return None
        prior = np.clip(np.bincount(ytr, minlength=3) / len(ytr), 1e-6, 1)
        prior = prior / prior.sum()
        out.loc[part.index, ["p_naive_down", "p_naive_flat", "p_naive_up"]] = prior
    return out


def baseline_tests(db: Database, settings: Settings) -> dict:
    """Make sure every production model has an out-of-sample comparison with the naive baseline.
    Computed once per version and cached in the state table."""
    from ..learning import registry
    from ..ml.evaluation import NAIVE_COLS, baseline_test
    out = {}
    for sym in settings.symbols:
        for tf in settings.predict_timeframes:
            key = registry.model_key(sym, tf, settings.horizon_bars)
            prod = registry.get_production(db, key)
            if prod is None:
                continue
            cache_key = f"baseline_test:{prod['version']}"
            res = (prod.get("metrics") or {}).get("baseline_test") or db.get_state(cache_key)
            if res is None:
                p = registry.resolve_path((prod.get("params") or {}).get("oos_path"), settings.model_dir)
                if p is not None and p.exists():
                    oos = pd.read_csv(p)
                    if not set(NAIVE_COLS) <= set(oos.columns):
                        oos = naive_for_oos(db, settings, sym, tf, oos)
                    res = baseline_test(oos, settings.horizon_bars, settings.baseline_p_better) if oos is not None else None
                if res is not None:
                    db.set_state(cache_key, res)
            if res is not None:
                res = {**res, "passed": bool(res["gain"] > 0 and res["p_better"] >= settings.baseline_p_better),
                       "p_required": settings.baseline_p_better}
            out[key] = res or {"status": "unavailable"}
    db.set_state("baseline_tests", out)
    return out


# ------------------------------------------------------------------ candles
def candle_gaps(db: Database, exchange: str, symbol: str, tf: str, lookback: int) -> list[tuple[int, int]]:
    """Missing bars inside the most recent `lookback` bars: [(first_missing_open_ts, n_missing)]."""
    rows = db.query("SELECT open_ts FROM candles WHERE exchange=? AND symbol=? AND timeframe=? "
                    "ORDER BY open_ts DESC LIMIT ?", (exchange, symbol, tf, lookback))
    ts = sorted(int(r["open_ts"]) for r in rows)
    bar = TIMEFRAME_MS[tf]
    return [(a + bar, (b - a) // bar - 1) for a, b in zip(ts, ts[1:]) if b - a > bar]


def misaligned_candles(db: Database, exchange: str, symbol: str, tf: str) -> int:
    bar = TIMEFRAME_MS[tf]
    return int(db.scalar("SELECT COUNT(*) FROM candles WHERE exchange=? AND symbol=? AND timeframe=? AND open_ts % ? != 0",
                         (exchange, symbol, tf, bar)) or 0)


def db_summary(db: Database, settings: Settings) -> dict:
    res = sqlite_check(db.path) if db.dialect == "sqlite" and db.path != ":memory:" else "ok"
    return {"check": res, "ok": res == "ok", "ts_ms": now_ms(),
            "predictions": db.scalar("SELECT COUNT(*) FROM predictions"),
            "candles": db.scalar("SELECT COUNT(*) FROM candles"),
            "models": db.scalar("SELECT COUNT(*) FROM model_registry")}
