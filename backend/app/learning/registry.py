"""Model registry: production / candidate / rejected / archived versions with
their metrics, feature version, training window and artifact path."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import joblib

from ..db import Database, now_ms


def model_key(symbol: str, timeframe: str, horizon: int) -> str:
    return f"{symbol}|{timeframe}|h{horizon}"


def new_version(key: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", key).strip("-").lower()
    return f"{slug}-{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{int(time.time() * 1000) % 1000:03d}"


def save_artifact(model_dir: Path, version: str, payload: dict) -> str:
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"{version}.joblib"
    tmp = path.with_suffix(".tmp")
    joblib.dump(payload, tmp, compress=3)  # several models per run go into the state that is pushed to git
    tmp.replace(path)  # atomic: readers never see a half-written file
    return str(path)


def resolve_path(path: str | None, model_dir: Path | None = None) -> Path | None:
    """Artifacts move with the state directory (e.g. a new CI workspace): fall back to model_dir/<name>."""
    if not path:
        return None
    p = Path(path)
    if not p.exists() and model_dir is not None and (Path(model_dir) / p.name).exists():
        return Path(model_dir) / p.name
    return p


def load_artifact(path: str, model_dir: Path | None = None) -> dict:
    return joblib.load(resolve_path(path, model_dir))


def register(db: Database, *, version: str, key: str, status: str, train_start_ts: int | None,
             train_end_ts: int | None, n_train: int, n_validation: int, feature_version: str,
             features: list[str], params: dict, metrics: dict, comparison: dict | None,
             artifact_path: str | None, parent_version: str | None, reason: str) -> None:
    db.execute(
        "INSERT INTO model_registry(version,model_key,status,created_ms,promoted_ms,train_start_ts,train_end_ts,"
        "n_train,n_validation,feature_version,features_json,params_json,metrics_json,comparison_json,"
        "artifact_path,parent_version,reason) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (version, key, status, now_ms(), now_ms() if status == "production" else None, train_start_ts,
         train_end_ts, n_train, n_validation, feature_version, json.dumps(features), json.dumps(params, default=str),
         json.dumps(metrics, default=str), json.dumps(comparison, default=str) if comparison else None,
         artifact_path, parent_version, reason))


def promote(db: Database, key: str, version: str, reason: str) -> None:
    with db.transaction():
        db.execute("UPDATE model_registry SET status='archived' WHERE model_key=? AND status='production'", (key,))
        db.execute("UPDATE model_registry SET status='production', promoted_ms=?, reason=? WHERE version=?",
                   (now_ms(), reason, version))


def get_production(db: Database, key: str) -> dict | None:
    row = db.query_one("SELECT * FROM model_registry WHERE model_key=? AND status='production' "
                       "ORDER BY promoted_ms DESC LIMIT 1", (key,))
    return _decode(row) if row else None


def list_versions(db: Database, key: str | None = None, limit: int = 50) -> list[dict]:
    if key:
        rows = db.query("SELECT * FROM model_registry WHERE model_key=? ORDER BY created_ms DESC LIMIT ?", (key, limit))
    else:
        rows = db.query("SELECT * FROM model_registry ORDER BY created_ms DESC LIMIT ?", (limit,))
    return [_decode(r) for r in rows]


def _decode(row: dict) -> dict:
    out = dict(row)
    for k in ("features_json", "params_json", "metrics_json", "comparison_json"):
        v = out.pop(k, None)
        out[k.replace("_json", "")] = json.loads(v) if v else None
    return out


def artifact_ok(path: str | None, model_dir: Path | None = None) -> tuple[bool, str | None]:
    """The artifact exists and loads (a truncated or corrupt file fails here, not during prediction)."""
    p = resolve_path(path, model_dir)
    if p is None or not p.exists():
        return False, "missing"
    try:
        art = joblib.load(p)
        if "model" not in art or "features" not in art:
            return False, "incomplete"
    except Exception as exc:  # noqa: BLE001
        return False, f"unreadable: {type(exc).__name__}"
    return True, None


def _summary(v: dict | None) -> dict | None:
    if not v:
        return None
    m = v.get("metrics") or {}
    bt = m.get("baseline_test") or {}
    return {"version": v["version"], "log_loss": m.get("log_loss"), "brier": m.get("brier"), "accuracy": m.get("accuracy"),
            "naive_log_loss": (m.get("baseline_prior") or {}).get("log_loss"), "baseline_p_better": bt.get("p_better")}


def rollback(db: Database, key: str, model_dir: Path | None, reason: str, to_version: str | None = None) -> dict:
    """Return the previous working model to production. The replaced version is kept (status rolled_back)."""
    cur = get_production(db, key)
    rows = db.query("SELECT * FROM model_registry WHERE model_key=? AND status='archived' "
                    "ORDER BY COALESCE(promoted_ms, created_ms) DESC", (key,))
    for r in rows:
        if to_version and r["version"] != to_version:
            continue
        ok, _ = artifact_ok(r["artifact_path"], model_dir)
        if not ok:
            continue
        with db.transaction():
            if cur:
                db.execute("UPDATE model_registry SET status='rolled_back' WHERE version=?", (cur["version"],))
            db.execute("UPDATE model_registry SET status='production', promoted_ms=?, reason=? WHERE version=?",
                       (now_ms(), f"rollback: {reason}", r["version"]))
        new = get_production(db, key)
        payload = {"from": _summary(cur), "to": _summary(new), "reason": reason}
        db.log_event("model_rollback", payload, key)
        return {"status": "rolled_back", **payload}
    return {"status": "no_previous_model", "key": key, "reason": reason}
