"""Training, validation and compute scheduling of the per-horizon forecast models.

Validation protocol (EVAL_PROTOCOL, fixed):
  * first model of a symbol x horizon: trained on the first 70 % of the rows, judged on the last 30 %
    (after a purge gap of `steps` bars); the judged model is exactly the deployed model
  * every later candidate is trained on data before a FRESH window that no earlier decision has used;
    candidate and production are compared on that window; the window is then marked as used
  * a model counts as validated only with >= MIN_INDEPENDENT non-overlapping holdout outcomes
  * metrics: Brier / log loss vs the naive baseline (block bootstrap, block = horizon), MAE of the median
    vs random walk, direction hit rate, 10-90 % range coverage, signals and their result after costs
Compute priority decides only WHEN something is (re)trained, never how it is judged.
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd

from ..config import Settings
from ..db import Database, now_ms
from ..learning import registry
from ..ml.evaluation import _block_bootstrap, gate_array
from .features import GRID_FEATURES, SECOND_FEATURES, grid_training, seconds_training
from .horizons import EVAL_PROTOCOL, HORIZONS, Horizon, model_key
from .models import Baseline, ForecastModel, labels

log = logging.getLogger("forecast.train")
MIN_INDEPENDENT = 100      # non-overlapping holdout outcomes needed to call a model validated
FEATURE_VERSION = "ff1"


def dataset(db: Database, settings: Settings, symbol: str, h: Horizon) -> tuple[pd.DataFrame, list[str]]:
    ex = settings.primary_exchange
    if h.grid == "1s":
        f = seconds_training(db, ex, symbol, h.seconds)
        feats = SECOND_FEATURES
    else:
        f = grid_training(db, ex, symbol, h.grid, h.steps)
        feats = GRID_FEATURES
    if f.empty:
        return f, feats
    return f.dropna(subset=["fwd_bps"]).reset_index(drop=True), feats


def independent(n: int, h: Horizon) -> int:
    """Rows overlap when the horizon is longer than the row spacing."""
    spacing = 5 if h.grid == "1s" else 1
    return int(n * spacing // max(1, h.steps))


def evaluate(fwd: np.ndarray, P: np.ndarray, Q: np.ndarray, base: Baseline, cost: float, h: Horizon,
             threshold: float, min_edge: float) -> dict:
    y = labels(fwd, cost)
    n = len(y)
    onehot = np.eye(3)[y]
    Pb = base.proba(n)
    ll = -np.log(np.clip(P[np.arange(n), y], 1e-9, 1))
    llb = -np.log(np.clip(Pb[np.arange(n), y], 1e-9, 1))
    block = max(1, h.steps if h.grid != "1s" else h.seconds // 5)
    boots = _block_bootstrap(llb - ll, block) if n >= 20 else np.array([0.0])
    nz = fwd != 0
    sig = gate_array(P, threshold, min_edge)
    act = np.where(sig >= 0)[0]
    taken, i_last = [], -10 ** 9
    for i in act:  # non-overlapping signals only (one position per horizon)
        if i - i_last >= block:
            taken.append(i)
            i_last = i
    taken = np.array(taken, dtype=int)
    side = np.where(sig[taken] == 2, 1.0, -1.0) if len(taken) else np.array([])
    net = side * fwd[taken] - cost if len(taken) else np.array([])
    return {
        "protocol": EVAL_PROTOCOL, "n": int(n), "n_independent": independent(n, h),
        "brier": float(np.mean(np.sum((P - onehot) ** 2, axis=1))),
        "brier_base": float(np.mean(np.sum((Pb - onehot) ** 2, axis=1))),
        "log_loss": float(ll.mean()), "log_loss_base": float(llb.mean()),
        "gain": float((llb - ll).mean()), "gain_ci95": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
        "p_better": float((boots > 0).mean()),
        "ece": _ece(P, y),
        "direction_hit": float(np.mean(np.sign(Q[nz, 1]) == np.sign(fwd[nz]))) if nz.any() else None,
        "mae_bps": float(np.mean(np.abs(Q[:, 1] - fwd))), "mae_rw_bps": float(np.mean(np.abs(fwd))),
        "mae_base_bps": float(np.mean(np.abs(base.q[0.5] - fwd))),
        "coverage_10_90": float(np.mean((fwd >= Q[:, 0]) & (fwd <= Q[:, 2]))),
        "label_share": {"DOWN": float(np.mean(y == 0)), "FLAT": float(np.mean(y == 1)), "UP": float(np.mean(y == 2))},
        "signals": int(len(taken)), "signal_hit": float(np.mean(side * fwd[taken] > 0)) if len(taken) else None,
        "avg_net_bps": float(net.mean()) if len(taken) else None,
    }


def _ece(P: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf = P.max(axis=1)
    pred = P.argmax(axis=1)
    e = 0.0
    for b in range(bins):
        m = (conf > b / bins) & (conf <= (b + 1) / bins)
        if m.any():
            e += m.mean() * abs(np.mean(pred[m] == y[m]) - conf[m].mean())
    return float(e)


def verdict(m: dict, settings: Settings) -> dict:
    """Fixed rules turning holdout metrics into the gate's inputs."""
    validated = m["n_independent"] >= MIN_INDEPENDENT
    better = validated and m["gain"] > 0 and m["p_better"] >= settings.baseline_p_better
    cost_edge = (m["signals"] or 0) >= settings.min_oos_signals and (m["avg_net_bps"] or 0) > 0
    return {"validated": validated, "better_than_baseline": better, "edge_after_costs": cost_edge}


def _edges(X: np.ndarray) -> list:
    """Decile edges of every training feature: reference for the drift check (PSI) on live inputs."""
    out = []
    for j in range(X.shape[1]):
        col = X[:, j][np.isfinite(X[:, j])]
        out.append(np.unique(np.quantile(col, np.linspace(0.1, 0.9, 9))).tolist() if len(col) > 50 else None)
    return out


def _payload(model, base, feats, cost, h, version, X=None) -> dict:
    return {"model": model, "baseline": base, "features": feats, "cost_bps": cost, "horizon_sec": h.seconds,
            "grid": h.grid, "version": version, "feature_version": FEATURE_VERSION,
            "feature_edges": _edges(X) if X is not None else None}


def state_key(key: str) -> str:
    return f"fstate:{key}"


def train_one(db: Database, settings: Settings, symbol: str, h: Horizon) -> dict:
    key = model_key(symbol, h)
    st = db.get_state(state_key(key)) or {}
    t0 = time.monotonic()
    d, feats = dataset(db, settings, symbol, h)
    cost = settings.round_trip_cost_bps
    gap = h.steps if h.grid != "1s" else max(1, h.seconds // 5)
    st.update({"n_rows": int(len(d)), "n_independent_total": independent(len(d), h), "ts_ms": now_ms()})
    min_rows = 400 if h.grid == "1s" else 600
    if len(d) < min_rows or independent(int(len(d) * 0.3), h) < 20:
        st.update({"status": "collecting", "reason": "not enough data for training and an independent check"})
        return _finish(db, key, st, t0, h)
    prod = registry.get_production(db, key)
    X = d[feats].to_numpy(float)
    fwd = d["fwd_bps"].to_numpy(float)
    ts = d["decision_ts"].to_numpy("int64")
    if prod is None:
        cut = int(len(d) * 0.7)
        tr, te = slice(0, cut), slice(cut + gap, len(d))
        model = ForecastModel().fit(X[tr], fwd[tr], cost, feats)
        base = Baseline(fwd[tr], cost)
        m = evaluate(fwd[te], model.proba(X[te]), model.quantiles(X[te]), base, cost, h,
                     settings.confidence_threshold, settings.min_edge)
        version = registry.new_version(key)
        path = registry.save_artifact(settings.model_dir, version, _payload(model, base, feats, cost, h, version, X[tr]))
        v = verdict(m, settings)
        registry.register(db, version=version, key=key, status="production", train_start_ts=int(ts[0]),
                          train_end_ts=int(ts[cut - 1]), n_train=cut, n_validation=m["n"], feature_version=FEATURE_VERSION,
                          features=feats, params={**model.describe(), "grid": h.grid, "horizon_sec": h.seconds,
                                                  "holdout_start": int(ts[cut + gap]) if cut + gap < len(ts) else None,
                                                  "holdout_end": int(ts[-1])},
                          metrics={**m, **v}, comparison=None, artifact_path=path, parent_version=None,
                          reason="first model: trained on the first 70 %, judged on the last 30 %")
        db.log_event("forecast_model_trained", {"version": version, "holdout": m, **v}, key)
        st.update({"status": _status(v), "production": version, "holdout": m, **v, "used_until_ts": int(ts[-1]),
                   "reason": "first validation", "moves_beyond_costs_rare": bool(model.degenerate)})
        return _finish(db, key, st, t0, h)
    # fresh window, never used by an earlier decision
    used_until = int(st.get("used_until_ts") or (prod.get("params") or {}).get("holdout_end") or prod["train_end_ts"])
    fresh = np.where(ts > used_until)[0]
    if independent(len(fresh), h) < MIN_INDEPENDENT:
        st.update({"status": st.get("status", "validated"), "waiting_for_fresh": int(len(fresh)),
                   "reason": f"waiting for a fresh window ({independent(len(fresh), h)}/{MIN_INDEPENDENT} independent outcomes)"})
        return _finish(db, key, st, t0, h)
    w0 = int(fresh[0])
    tr = slice(0, max(0, w0 - gap))
    art = registry.load_artifact(prod["artifact_path"], settings.model_dir)
    pm, pbase = art["model"], art["baseline"]
    cand = ForecastModel().fit(X[tr], fwd[tr], cost, feats)
    cbase = Baseline(fwd[tr], cost)
    Xw, fw = X[w0:], fwd[w0:]
    m_prod = evaluate(fw, pm.proba(Xw), pm.quantiles(Xw), pbase, cost, h, settings.confidence_threshold, settings.min_edge)
    m_cand = evaluate(fw, cand.proba(Xw), cand.quantiles(Xw), cbase, cost, h, settings.confidence_threshold, settings.min_edge)
    y = labels(fw, cost)
    llp = -np.log(np.clip(pm.proba(Xw)[np.arange(len(y)), y], 1e-9, 1))
    llc = -np.log(np.clip(cand.proba(Xw)[np.arange(len(y)), y], 1e-9, 1))
    block = max(1, h.steps if h.grid != "1s" else h.seconds // 5)
    boots = _block_bootstrap(llp - llc, block)
    half = len(y) // 2
    cmp_ = {"n": int(len(y)), "gain": float((llp - llc).mean()), "p_better": float((boots > 0).mean()),
            "halves": [float((llp - llc)[:half].mean()), float((llp - llc)[half:].mean())],
            "mae_prod": m_prod["mae_bps"], "mae_cand": m_cand["mae_bps"]}
    promote = cmp_["gain"] > 0 and cmp_["p_better"] >= 0.9 and all(x > 0 for x in cmp_["halves"]) \
        and m_cand["mae_bps"] <= m_prod["mae_bps"] * 1.01
    version = registry.new_version(key)
    vc = verdict(m_cand, settings)
    if promote:
        path = registry.save_artifact(settings.model_dir, version, _payload(cand, cbase, feats, cost, h, version, X[tr]))
        registry.register(db, version=version, key=key, status="candidate", train_start_ts=int(ts[0]),
                          train_end_ts=int(ts[max(0, w0 - gap - 1)]), n_train=w0 - gap, n_validation=m_cand["n"],
                          feature_version=FEATURE_VERSION, features=feats,
                          params={**cand.describe(), "grid": h.grid, "horizon_sec": h.seconds, "holdout_start": int(ts[w0]),
                                  "holdout_end": int(ts[-1])},
                          metrics={**m_cand, **vc}, comparison=cmp_, artifact_path=path, parent_version=prod["version"],
                          reason="challenger: better than production on a fresh window")
        registry.promote(db, key, version, f"promoted: log-loss gain {cmp_['gain']:.4f}, P(better) {cmp_['p_better']:.2f}")
        db.log_event("forecast_model_promoted", {"version": version, "replaced": prod["version"], "comparison": cmp_}, key)
        st.update({"status": _status(vc), "production": version, "holdout": m_cand, **vc, "reason": "challenger promoted",
                   "moves_beyond_costs_rare": bool(cand.degenerate)})
    else:
        registry.register(db, version=version, key=key, status="rejected", train_start_ts=int(ts[0]),
                          train_end_ts=int(ts[max(0, w0 - gap - 1)]), n_train=w0 - gap, n_validation=m_cand["n"],
                          feature_version=FEATURE_VERSION, features=feats, params={"grid": h.grid, "horizon_sec": h.seconds},
                          metrics={**m_cand, **vc}, comparison=cmp_, artifact_path=None, parent_version=prod["version"],
                          reason="rejected: no confirmed improvement on the fresh window")
        vp = verdict(m_prod, settings)
        # the production model was just checked again on data it never saw: its standing follows that check
        st.update({"status": _status(vp), "production": prod["version"], "holdout": m_prod, **vp,
                   "reason": "challenger rejected; production re-checked on a fresh window"})
        db.log_event("forecast_challenger_rejected", {"version": version, "production": prod["version"], "comparison": cmp_,
                                                      "production_recheck": m_prod}, key)
    st["used_until_ts"] = int(ts[-1])
    st["last_comparison"] = cmp_
    return _finish(db, key, st, t0, h)


def _status(v: dict) -> str:
    if not v["validated"]:
        return "collecting"
    return "validated" if v["better_than_baseline"] else "not_better"


def _finish(db: Database, key: str, st: dict, t0: float, h: Horizon) -> dict:
    st["train_sec"] = round(time.monotonic() - t0, 1)
    st["last_train_ms"] = now_ms()
    st["protocol"] = EVAL_PROTOCOL
    db.set_state(state_key(key), json.loads(json.dumps(st, default=float)))
    return st


# ------------------------------------------------------------------ compute scheduling
RETRAIN_SEC = {"validated": 6 * 3600, "not_better": 12 * 3600, "collecting": 24 * 3600, "degraded": 0}


def priority(st: dict) -> float:
    s = st.get("status")
    if s is None:
        return 90.0
    if s == "degraded":
        return 100.0
    if s == "validated":
        g = (st.get("holdout") or {}).get("gain") or 0.0
        return 50.0 + 10.0 * min(3.0, max(0.0, g / 0.01))
    if s == "not_better":
        return 20.0
    return 10.0 + min(10.0, (st.get("n_independent_total") or 0) / 1000)


def due(st: dict, h: Horizon, now: int) -> bool:
    if not st.get("last_train_ms"):
        return True
    interval = RETRAIN_SEC.get(st.get("status", "collecting"), 24 * 3600)
    if h.grid == "1s" and st.get("status") == "collecting":
        interval = 3 * 3600  # live second data grows quickly
    return now - int(st["last_train_ms"]) >= interval * 1000


def run_training(db: Database, settings: Settings, budget_sec: float, now: int | None = None) -> dict:
    """Train due models, most promising first, until the time budget is spent."""
    now = now or now_ms()
    jobs = []
    for sym in settings.symbols:
        for h in HORIZONS:
            st = db.get_state(state_key(model_key(sym, h))) or {}
            if due(st, h, now):
                jobs.append((priority(st), sym, h))
    jobs.sort(key=lambda j: -j[0])
    t0 = time.monotonic()
    done, skipped = {}, []
    for pr, sym, h in jobs:
        if time.monotonic() - t0 > budget_sec:
            skipped.append(model_key(sym, h))
            continue
        try:
            st = train_one(db, settings, sym, h)
            done[model_key(sym, h)] = {"status": st.get("status"), "priority": pr, "sec": st.get("train_sec")}
        except Exception as exc:  # noqa: BLE001
            log.exception("forecast training failed for %s %s", sym, h.label)
            done[model_key(sym, h)] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:300]}
    return {"trained": done, "postponed": skipped, "budget_sec": budget_sec, "used_sec": round(time.monotonic() - t0, 1)}


def psi(edges: list | None, values: np.ndarray) -> float | None:
    """Population stability index of live values against the training deciles (> 0.25: distribution shifted)."""
    v = values[np.isfinite(values)]
    if not edges or len(v) < 50:
        return None
    bins = np.concatenate([[-np.inf], edges, [np.inf]])
    exp = np.full(len(bins) - 1, 1.0 / (len(bins) - 1))
    act = np.histogram(v, bins=bins)[0] / len(v)
    act = np.clip(act, 1e-4, 1)
    return float(np.sum((act - exp) * np.log(act / exp)))
