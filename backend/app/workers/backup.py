"""Consistent online backups (SQLite backup API, not a file copy) + rotation,
model-artifact archive, and data-retention maintenance."""
from __future__ import annotations

import gzip
import logging
import os
import shutil
import sqlite3
import subprocess
import tarfile
import time
from pathlib import Path

from ..config import Settings
from ..db import Database, now_ms

log = logging.getLogger("backup")


def _rotate(directory: Path, pattern: str, keep: int) -> list[str]:
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    removed = []
    for f in files[keep:]:
        f.unlink(missing_ok=True)
        removed.append(f.name)
    return removed


def backup_sqlite(db_path: str, out_dir: Path) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    tmp = out_dir / f".market_{stamp}.sqlite3"
    src = sqlite3.connect(db_path, timeout=60)
    dst = sqlite3.connect(tmp)
    try:
        src.backup(dst, pages=2048, sleep=0.05)  # online, transactionally consistent copy
    finally:
        dst.close()
        src.close()
    chk = sqlite3.connect(tmp)
    try:
        if chk.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("backup integrity_check failed")
    finally:
        chk.close()
    final = out_dir / f"market_{stamp}.sqlite3.gz"
    with open(tmp, "rb") as fi, gzip.open(final, "wb", compresslevel=6) as fo:
        shutil.copyfileobj(fi, fo)
    tmp.unlink(missing_ok=True)
    return final


def backup_models(model_dir: Path, out_dir: Path) -> Path | None:
    if not model_dir.exists() or not any(model_dir.glob("*.joblib")):
        return None
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    final = out_dir / f"models_{stamp}.tar.gz"
    with tarfile.open(final, "w:gz") as tar:
        for f in model_dir.iterdir():
            if f.is_file() and f.suffix in (".joblib", ".csv"):
                tar.add(f, arcname=f"models/{f.name}")
    return final


def run_backup_once(db: Database, settings: Settings) -> dict:
    out_dir = settings.backup_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = now_ms()
    result: dict = {"ts_ms": t0}
    try:
        result["retention"] = db.apply_retention(settings)
    except Exception as exc:  # noqa: BLE001
        result["retention_error"] = str(exc)
    if db.dialect == "sqlite":
        path = backup_sqlite(db.path, out_dir)
        result["database"] = path.name
        result["database_bytes"] = path.stat().st_size
        result["rotated"] = _rotate(out_dir, "market_*.sqlite3.gz", settings.backup_keep)
    else:  # pragma: no cover - PostgreSQL path, requires pg_dump in the image
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        path = out_dir / f"market_{stamp}.pgdump"
        subprocess.run(["pg_dump", "--format=custom", f"--file={path}", settings.database_url], check=True,
                       env={**os.environ})
        result["database"] = path.name
        result["rotated"] = _rotate(out_dir, "market_*.pgdump", settings.backup_keep)
    m = backup_models(settings.model_dir, out_dir)
    if m:
        result["models"] = m.name
        _rotate(out_dir, "models_*.tar.gz", settings.backup_keep)
    result["duration_ms"] = now_ms() - t0
    db.set_state("last_backup", result)
    log.info("backup done: %s", result)
    return result


def restore_sqlite(backup_file: Path, db_path: str) -> None:
    """Restore helper used by scripts/restore_backup.py (services must be stopped)."""
    tmp = Path(db_path).with_suffix(".restore")
    with gzip.open(backup_file, "rb") as fi, open(tmp, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    for suffix in ("-wal", "-shm"):
        Path(db_path + suffix).unlink(missing_ok=True)
    tmp.replace(db_path)
