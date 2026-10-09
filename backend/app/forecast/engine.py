"""Issue forward-looking forecasts for every horizon, check them when the horizon has passed, and keep
per-horizon statistics.

Rules that keep the ledger honest:
  * a forecast is written before its target time and uses only data with timestamps <= its creation
    time (`data_ts_ms <= created_ms` is asserted for every row);
  * forecasts are append-only (database trigger): the prediction can never be changed, the outcome
    columns are written once;
  * the outcome is the last trade price at the target time (1-second bars) or, for horizons of 15 minutes
    and more, the close of the 1-minute candle ending at most 60 s before the target; otherwise the
    outcome is recorded as missing - never guessed;
  * every forecast stores its exact inputs and model version; `reproduce()` recomputes it.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math

import numpy as np
import pandas as pd

from ..config import Settings
from ..db import Database, now_ms
from ..learning import registry
from ..ml.evaluation import _block_bootstrap, gate
from .features import grid_live_row, seconds_live_row
from .horizons import HORIZONS, BY_SECONDS, Horizon, model_key
from .models import labels
from .train import MIN_INDEPENDENT, state_key

log = logging.getLogger("forecast")
CLASS_NAMES = ["DOWN", "FLAT", "UP"]


def _r(x, sig: int = 8):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return float(f"{float(x):.{sig}g}")


class Forecaster:
    def __init__(self, db: Database, settings: Settings):
        self.db, self.s = db, settings
        self.cache: dict[str, tuple[str, dict]] = {}
        self.next_due: dict[str, int] = {}
        self.missing: dict[str, dict] = {}

    # ---------------------------------------------------------------- inputs
    def live_price(self, symbol: str, now: int) -> tuple[float | None, int | None, str]:
        ex = self.s.primary_exchange
        r = self.db.query_one("SELECT close, last_trade_ms FROM price_seconds WHERE exchange=? AND symbol=? AND ts_sec <= ? "
                              "AND last_trade_ms <= ? ORDER BY ts_sec DESC LIMIT 1", (ex, symbol, now // 1000, now))
        if r and now - r["last_trade_ms"] <= 5_000:
            return float(r["close"]), int(r["last_trade_ms"]), "trade"
        c = self.db.query_one("SELECT open_ts, close FROM candles WHERE exchange=? AND symbol=? AND timeframe='1m' AND closed=1 "
                              "AND open_ts + 60000 <= ? ORDER BY open_ts DESC LIMIT 1", (ex, symbol, now))
        if c:
            return float(c["close"]), int(c["open_ts"]) + 60_000, "kline_1m"
        return None, None, "none"

    def model(self, key: str):
        prod = registry.get_production(self.db, key)
        if prod is None:
            return None, None
        c = self.cache.get(key)
        if not c or c[0] != prod["version"]:
            self.cache[key] = (prod["version"], registry.load_artifact(prod["artifact_path"], self.s.model_dir))
        return prod, self.cache[key][1]

    def refresh_ms(self, h: Horizon) -> int:
        return int(h.refresh_sec * 1000 * max(1.0, getattr(self.s, "forecast_refresh_scale", 1.0)))

    # ---------------------------------------------------------------- issue
    def tick(self, now: int | None = None) -> dict:
        now = now or now_ms()
        made = 0
        for sym in self.s.symbols:
            for h in HORIZONS:
                key = model_key(sym, h)
                if key not in self.next_due:
                    last = self.db.scalar("SELECT MAX(created_ms) FROM forecasts WHERE symbol=? AND horizon_sec=?", (sym, h.seconds))
                    self.next_due[key] = (int(last) + self.refresh_ms(h)) if last else 0
                if now < self.next_due[key]:
                    continue
                self.next_due[key] = now + self.refresh_ms(h)
                try:
                    if self.issue(sym, h, now):
                        made += 1
                except Exception:  # noqa: BLE001 - one horizon must not stop the others
                    log.exception("forecast %s %s failed", sym, h.label)
        return {"made": made}

    def _miss(self, key: str, why: str, now: int) -> None:
        m = self.missing.setdefault(key, {})
        m[why] = m.get(why, 0) + 1
        m["last"] = why
        m["last_ms"] = now

    def issue(self, symbol: str, h: Horizon, now: int) -> str | None:
        row = self.compute(symbol, h, now)
        if row is None:
            return None
        self.db.execute(f"INSERT INTO forecasts({','.join(row)}) VALUES({','.join('?' * len(row))})", tuple(row.values()))
        return row["forecast_id"]

    def compute(self, symbol: str, h: Horizon, now: int) -> dict | None:
        """The forecast for `symbol` and horizon `h` made at `now`, from data available at `now` only."""
        key = model_key(symbol, h)
        ex = self.s.primary_exchange
        prod, art = self.model(key)
        if art is None:
            self._miss(key, "model_not_trained", now)
            return None
        ref, ref_ts, ref_src = self.live_price(symbol, now)
        if ref is None:
            self._miss(key, "no_price", now)
            return None
        if h.grid == "1s":
            row, data_ts = seconds_live_row(self.db, ex, symbol, now)
        else:
            row, data_ts = grid_live_row(self.db, ex, symbol, h.grid, now)
        if row is None:
            self._miss(key, "insufficient_data", now)
            return None
        data_ts = max(int(data_ts), int(ref_ts))
        assert data_ts <= now, "a forecast may only use data that existed when it was made"
        feats = art["features"]
        x = np.array([[_r(row.get(f)) if _r(row.get(f)) is not None else np.nan for f in feats]], dtype=float)
        m, base = art["model"], art["baseline"]
        P = m.proba(x)[0]
        Q = m.quantiles(x)[0]
        st = self.db.get_state(state_key(key)) or {}
        status = st.get("status", "collecting")
        age_sec = (now - data_ts) / 1000
        reasons = []
        if age_sec > h.max_data_age_sec:
            reasons.append("data_stale")
        if status in ("collecting", "degraded"):
            reasons.append("model_not_validated")
        elif status == "not_better":
            reasons.append("model_not_better_than_baseline")
        if status in ("validated", "not_better") and not st.get("edge_after_costs"):
            reasons.append("no_edge_after_costs")
        sig, model_dir, conf = gate(float(P[0]), float(P[1]), float(P[2]), self.s.confidence_threshold, self.s.min_edge)
        if sig == "NO TRADE":
            reasons.append("low_confidence")
        decision = "NO TRADE" if reasons else ("LONG" if sig == "UP" else "SHORT")
        ent = float(-(P * np.log(np.clip(P, 1e-12, 1))).sum() / np.log(3))
        fvals = [None if not np.isfinite(v) else float(v) for v in x[0]]
        created = now
        payload = json.dumps({"v": prod["version"], "f": fvals, "ref": ref, "t": created, "h": h.seconds}, sort_keys=True)
        ihash = hashlib.sha256(payload.encode()).hexdigest()[:24]
        fid = f"{symbol.replace('/', '')}-{h.label}-{created}-{ihash[:6]}"
        detail = {"entropy": _r(ent, 4), "data_age_sec": _r(age_sec, 4), "grid": h.grid, "status": status,
                  "model_direction": model_dir, "confidence": _r(conf, 6),
                  "range_price": [_r(ref * (1 + Q[0] / 1e4), 10), _r(ref * (1 + Q[2] / 1e4), 10)]}
        return {"forecast_id": fid, "created_ms": created, "data_ts_ms": data_ts, "symbol": symbol, "exchange": ex,
                "horizon_sec": h.seconds, "target_ts": created + h.seconds * 1000, "ref_price": ref, "ref_source": ref_src,
                "p_up": float(P[2]), "p_down": float(P[0]), "p_flat": float(P[1]), "cost_bps": float(art["cost_bps"]),
                "q10_bps": float(Q[0]), "q50_bps": float(Q[1]), "q90_bps": float(Q[2]), "uncertainty": float(Q[2] - Q[0]),
                "decision": decision, "reasons": json.dumps(reasons), "model_version": prod["version"], "model_status": status,
                "base_p_up": float(base.prior[2]), "base_p_down": float(base.prior[0]), "base_p_flat": float(base.prior[1]),
                "base_q50_bps": float(base.q[0.5]), "features_json": json.dumps(fvals), "input_hash": ihash,
                "detail_json": json.dumps(detail)}

    # ---------------------------------------------------------------- reproduce
    def reproduce(self, forecast_id: str) -> dict:
        r = self.db.query_one("SELECT * FROM forecasts WHERE forecast_id=?", (forecast_id,))
        prod = self.db.query_one("SELECT artifact_path FROM model_registry WHERE version=?", (r["model_version"],))
        art = registry.load_artifact(prod["artifact_path"], self.s.model_dir)
        x = np.array([[np.nan if v is None else v for v in json.loads(r["features_json"])]], dtype=float)
        P, Q = art["model"].proba(x)[0], art["model"].quantiles(x)[0]
        payload = json.dumps({"v": r["model_version"], "f": json.loads(r["features_json"]), "ref": r["ref_price"],
                              "t": r["created_ms"], "h": r["horizon_sec"]}, sort_keys=True)
        return {"p_up": float(P[2]), "p_down": float(P[0]), "p_flat": float(P[1]), "q10_bps": float(Q[0]),
                "q50_bps": float(Q[1]), "q90_bps": float(Q[2]),
                "input_hash_ok": hashlib.sha256(payload.encode()).hexdigest()[:24] == r["input_hash"],
                "same": bool(np.allclose([P[2], P[0], P[1], Q[0], Q[1], Q[2]],
                                         [r["p_up"], r["p_down"], r["p_flat"], r["q10_bps"], r["q50_bps"], r["q90_bps"]],
                                         rtol=0, atol=1e-9))}

    # ---------------------------------------------------------------- resolve
    def price_at(self, ex: str, symbol: str, target: int, h: Horizon):
        tsec = target // 1000
        # a 1-second bar's close is its last trade: only bars whose last trade is not after the target count
        r = self.db.query_one("SELECT ts_sec, close, last_trade_ms FROM price_seconds WHERE exchange=? AND symbol=? "
                              "AND ts_sec <= ? AND ts_sec >= ? AND last_trade_ms <= ? ORDER BY ts_sec DESC LIMIT 1",
                              (ex, symbol, tsec, tsec - 60, target))
        after = self.db.scalar("SELECT MIN(ts_sec) FROM price_seconds WHERE exchange=? AND symbol=? AND ts_sec > ? AND ts_sec <= ?",
                               (ex, symbol, tsec, tsec + 120))
        if r and after is not None and target - int(r["last_trade_ms"]) <= max(h.resolve_tolerance_ms, 10_000) \
                and int(after) - tsec <= 10:
            # last trade at or before the target, with the stream demonstrably alive around the target
            return float(r["close"]), "trade", target - int(r["last_trade_ms"])
        if h.seconds >= 900:
            o = (target // 60_000) * 60_000 - 60_000
            c = self.db.query_one("SELECT close FROM candles WHERE exchange=? AND symbol=? AND timeframe='1m' AND open_ts=? AND closed=1",
                                  (ex, symbol, o))
            if c:
                return float(c["close"]), "kline_1m", target - (o + 60_000)
        return None, None, None

    def resolve(self, now: int | None = None, limit: int = 5000) -> dict:
        now = now or now_ms()
        rows = self.db.query("SELECT * FROM forecasts WHERE resolved_ms IS NULL AND target_ts + 3000 <= ? ORDER BY target_ts LIMIT ?",
                             (now, limit))
        done = missing = 0
        for f in rows:
            h = BY_SECONDS.get(int(f["horizon_sec"]))
            if h is None:
                continue
            price, src, lag = self.price_at(f["exchange"], f["symbol"], int(f["target_ts"]), h)
            if price is None:
                if now - int(f["target_ts"]) > max(2 * 3600_000, 2 * h.seconds * 1000):
                    self.db.execute("UPDATE forecasts SET resolved_ms=?, resolution_source='missing' WHERE forecast_id=?",
                                    (now, f["forecast_id"]))
                    missing += 1
                continue
            act = (price / float(f["ref_price"]) - 1) * 1e4
            y = int(labels(np.array([act]), float(f["cost_bps"]))[0])
            P = np.array([f["p_down"], f["p_flat"], f["p_up"]], dtype=float)
            B = np.array([f["base_p_down"], f["base_p_flat"], f["base_p_up"]], dtype=float)
            oh = np.eye(3)[y]
            side = 1.0 if f["decision"] == "LONG" else -1.0 if f["decision"] == "SHORT" else 0.0
            self.db.execute(
                "UPDATE forecasts SET resolved_ms=?, resolution_source=?, resolution_lag_ms=?, actual_price=?, actual_bps=?, "
                "actual_class=?, brier=?, brier_base=?, logloss=?, logloss_base=?, abs_err_bps=?, abs_err_rw_bps=?, in_range=?, "
                "net_bps=? WHERE forecast_id=?",
                (now, src, int(lag), price, act, CLASS_NAMES[y], float(((P - oh) ** 2).sum()), float(((B - oh) ** 2).sum()),
                 float(-np.log(max(P[y], 1e-9))), float(-np.log(max(B[y], 1e-9))), abs(float(f["q50_bps"]) - act), abs(act),
                 int(float(f["q10_bps"]) <= act <= float(f["q90_bps"])),
                 (side * act - float(f["cost_bps"])) if side else None, f["forecast_id"]))
            done += 1
        return {"resolved": done, "missing": missing}

    # ---------------------------------------------------------------- statistics
    def stats(self, now: int | None = None, days: float = 30) -> dict:
        now = now or now_ms()
        out = {}
        for sym in self.s.symbols:
            for h in HORIZONS:
                key = model_key(sym, h)
                rows = self.db.query(
                    "SELECT created_ms, decision, reasons, q50_bps, actual_bps, brier, brier_base, abs_err_bps, abs_err_rw_bps, "
                    "in_range, net_bps, resolution_source, resolved_ms FROM forecasts WHERE symbol=? AND horizon_sec=? AND created_ms>=? "
                    "ORDER BY created_ms", (sym, h.seconds, now - int(days * 86400_000)))
                st = self.db.get_state(state_key(key)) or {}
                miss = self.missing.get(key, {})
                d = pd.DataFrame(rows)
                refresh = self.refresh_ms(h) / 1000
                rec = {"horizon": h.label, "seconds": h.seconds, "refresh_sec": refresh, "grid": h.grid,
                       "status": st.get("status", "not_trained"), "training": {k: st.get(k) for k in (
                           "n_rows", "n_independent_total", "reason", "validated", "better_than_baseline", "edge_after_costs",
                           "last_train_ms", "production", "waiting_for_fresh")},
                       "holdout": st.get("holdout"), "not_issued": miss, "issued": int(len(d))}
                if len(d):
                    res = d[d.resolved_ms.notna()]
                    # quality is measured on forecasts made from current data; stale ones are only counted
                    ok = res[res.resolution_source.isin(["trade", "kline_1m"]) & ~res.reasons.str.contains("data_stale")]
                    rec.update({"resolved": int(len(ok)), "missing_outcome": int((res.resolution_source == "missing").sum()),
                                "pending": int(d.resolved_ms.isna().sum()),
                                "stale": int(d.reasons.str.contains("data_stale").sum()),
                                "decisions": {k: int(v) for k, v in d.decision.value_counts().items()}})
                    if len(ok):
                        a, q = ok.actual_bps.astype(float).to_numpy(), ok.q50_bps.astype(float).to_numpy()
                        nz = a != 0
                        gain = (ok.brier_base - ok.brier).astype(float).to_numpy()
                        block = max(1, math.ceil(h.seconds / refresh))
                        boots = _block_bootstrap(gain, block) if len(gain) >= 20 else np.array([np.nan])
                        n_ind = int(len(ok) * refresh // max(refresh, h.seconds)) or (1 if len(ok) else 0)
                        sig = ok[ok.net_bps.notna()]
                        rec["live"] = {
                            "n": int(len(ok)), "n_independent": n_ind,
                            "direction_hit": float(np.mean(np.sign(q[nz]) == np.sign(a[nz]))) if nz.any() else None,
                            "mae_bps": float(ok.abs_err_bps.mean()), "mae_rw_bps": float(ok.abs_err_rw_bps.mean()),
                            "brier": float(ok.brier.mean()), "brier_base": float(ok.brier_base.mean()),
                            "brier_gain": float(gain.mean()),
                            "brier_gain_ci95": [float(np.nanquantile(boots, .025)), float(np.nanquantile(boots, .975))],
                            "coverage_10_90": float(ok.in_range.mean()),
                            "signals": int(len(sig)), "avg_net_bps": float(sig.net_bps.mean()) if len(sig) else None,
                            "sufficient": n_ind >= MIN_INDEPENDENT}
                        # quality got worse on live data -> back to validation (criteria unchanged)
                        if st.get("status") == "validated" and n_ind >= MIN_INDEPENDENT and rec["live"]["brier_gain_ci95"][1] < 0:
                            st["status"] = "degraded"
                            st["reason"] = "live forecasts worse than the baseline (95 % interval below zero)"
                            self.db.set_state(state_key(key), st)
                            self.db.log_event("forecast_model_degraded", {"live": rec["live"]}, key)
                            rec["status"] = "degraded"
                rec["drift"] = self.drift(key)
                out[key] = rec
        return out

    def drift(self, key: str, n: int = 300) -> dict | None:
        """Have the live inputs moved away from what the production model was trained on?"""
        from .train import psi
        prod, art = self.model(key)
        if art is None or not art.get("feature_edges"):
            return None
        sym, h = key.split("|f")
        rows = self.db.query("SELECT features_json FROM forecasts WHERE symbol=? AND horizon_sec=? AND model_version=? "
                             "ORDER BY created_ms DESC LIMIT ?", (sym, int(h), prod["version"], n))
        if len(rows) < 50:
            return None
        X = np.array([[np.nan if v is None else v for v in json.loads(r["features_json"])] for r in rows], dtype=float)
        vals = {f: psi(e, X[:, j]) for j, (f, e) in enumerate(zip(art["features"], art["feature_edges"]))}
        vals = {f: v for f, v in vals.items() if v is not None}
        if not vals:
            return None
        top = sorted(vals.items(), key=lambda kv: -kv[1])[:3]
        return {"max_psi": round(top[0][1], 3), "top": [[f, round(v, 3)] for f, v in top], "shifted": top[0][1] > 0.25,
                "n": len(rows)}

    def latest(self) -> dict:
        out = {}
        for sym in self.s.symbols:
            for h in HORIZONS:
                r = self.db.query_one("SELECT * FROM forecasts WHERE symbol=? AND horizon_sec=? ORDER BY created_ms DESC LIMIT 1",
                                      (sym, h.seconds))
                if r:
                    r = {k: v for k, v in r.items() if k != "features_json"}
                    r["reasons"] = json.loads(r["reasons"])
                    r["detail"] = json.loads(r.pop("detail_json") or "{}")
                out[model_key(sym, h)] = r
        return out
