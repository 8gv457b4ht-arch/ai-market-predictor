"""Prediction worker logic: MarketSnapshot fusion -> model -> gating -> ledger,
and resolution of past predictions once their horizon has passed."""
from __future__ import annotations

import json
import logging
import math

import numpy as np

from ..config import TIMEFRAME_MS, Settings
from ..db import Database, now_ms
from ..market.candles import load_candles
from ..market.quality import QualityTracker, evaluate
from ..ml.dataset import latest_feature_row
from ..ml.evaluation import gate
from ..ml.features import FEATURE_VERSION, direction_of
from ..news.service import news_features
from . import registry

log = logging.getLogger("predictor")

# half-lives (seconds) used to down-weight stale live inputs
FRESHNESS_HALF_LIFE = {"order_book": 10.0, "order_flow": 60.0, "ticker": 120.0}


def freshness(age_sec: float | None, half_life: float) -> float:
    if age_sec is None:
        return 0.0
    return round(0.5 ** (max(0.0, age_sec) / half_life), 4)


def _num(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


class Predictor:
    def __init__(self, db: Database, settings: Settings):
        self.db = db
        self.s = settings
        self.cache: dict[str, tuple[str, dict]] = {}
        self.tracker = QualityTracker()

    # -------------------------------------------------------------- model
    def model_for(self, key: str) -> tuple[dict | None, dict | None]:
        prod = registry.get_production(self.db, key)
        if prod is None:
            return None, None
        cached = self.cache.get(key)
        if cached and cached[0] == prod["version"]:
            return prod, cached[1]
        art = registry.load_artifact(prod["artifact_path"], self.s.model_dir)
        self.cache[key] = (prod["version"], art)
        log.info("loaded production model %s", prod["version"])
        return prod, art

    # ----------------------------------------------------------- snapshot
    def build_snapshot(self, symbol: str, tf: str, row, now: int, quality: dict) -> dict:
        ex = self.s.primary_exchange
        book = self.db.query_one("SELECT * FROM book_state WHERE exchange=? AND symbol=?", (ex, symbol))
        flow = self.db.query_one("SELECT * FROM flow_bars WHERE exchange=? AND symbol=? ORDER BY open_ts DESC LIMIT 1",
                                 (ex, symbol))
        ticker = self.db.query_one("SELECT * FROM market_ticker WHERE exchange=? AND symbol=?", (ex, symbol))
        # only news published before the candle closed: no look-ahead through the news channel
        news = news_features(self.db, symbol, int(row["decision_ts"]), self.s.news_half_life_min)
        book_age = (now - book["recv_ms"]) / 1000 if book else None
        flow_age = (now - flow["updated_ms"]) / 1000 if flow else None
        tick_age = (now - ticker["ts_ms"]) / 1000 if ticker else None
        fr = {"order_book": freshness(book_age, FRESHNESS_HALF_LIFE["order_book"]),
              "order_flow": freshness(flow_age, FRESHNESS_HALF_LIFE["order_flow"]),
              "ticker": freshness(tick_age, FRESHNESS_HALF_LIFE["ticker"])}
        decision_ts = int(row["decision_ts"])
        return {
            "timestamp": decision_ts, "built_ms": now, "exchange": ex, "symbol": symbol, "timeframe": tf,
            "price": float(row["close"]),
            "bid": _num(book["best_bid"]) if book else None, "ask": _num(book["best_ask"]) if book else None,
            "spread_bps": _num(book["spread_bps"]) if book else None,
            "orderbook_imbalance": _num(book["imbalance"]) if book else None,
            "bid_depth": _num(book["bid_depth"]) if book else None, "ask_depth": _num(book["ask_depth"]) if book else None,
            "volume": float(row["volume"]),
            "cvd_live": _num(flow["cvd"]) if flow else None,
            "flow_delta_1m": _num(flow["delta"]) if flow else None,
            "news_impact": news["news_impact"], "news_relevance": news["news_relevance"],
            "news_event_count": news["news_event_count"], "news_top": news["news_top"],
            "regime": row["regime"],
            "quality": quality,
            "freshness": {"weights": fr, "age_sec": {"order_book": book_age, "order_flow": flow_age,
                                                     "ticker": tick_age, "candle": (now - decision_ts) / 1000}},
            "features": {},
        }

    def live_regime(self, base_regime: str, snap: dict, quality_ok: bool) -> str:
        if not quality_ok:
            return "abnormal"
        if base_regime == "abnormal":
            return base_regime
        sp = snap.get("spread_bps")
        # only trust the order book when it is fresh
        if sp is not None and snap["freshness"]["weights"]["order_book"] >= 0.5 and sp > self.s.max_spread_bps:
            return "low_liquidity"
        return base_regime

    # ------------------------------------------------------------ predict
    def predict(self, symbol: str, tf: str, now: int | None = None) -> dict:
        now = now or now_ms()
        key = registry.model_key(symbol, tf, self.s.horizon_bars)
        prod, art = self.model_for(key)
        if prod is None:
            return {"status": "model_not_ready", "key": key}
        feats = art["features"]
        row, frame = latest_feature_row(self.db, self.s, symbol, tf, feats)
        if row is None:
            return {"status": "no_data", "key": key}
        candle_ts = int(row["open_ts"])
        exists = self.db.query_one(
            "SELECT prediction_id FROM predictions WHERE symbol=? AND exchange=? AND timeframe=? AND horizon_bars=? AND candle_ts=?",
            (symbol, self.s.primary_exchange, tf, self.s.horizon_bars, candle_ts))
        if exists:
            return {"status": "already_predicted", "candle_ts": candle_ts}
        base = load_candles(self.db, self.s.primary_exchange, symbol, tf, 400)
        q = evaluate(self.db, self.s, symbol, tf, base, self.tracker, now).to_dict()
        snap = self.build_snapshot(symbol, tf, row, now, q)
        x = np.array([[_num(row.get(f)) if _num(row.get(f)) is not None else np.nan for f in feats]], dtype=float)
        p_down, p_flat, p_up = map(float, art["model"].predict_proba(x)[0])
        signal, model_dir, conf = gate(p_down, p_flat, p_up, self.s.confidence_threshold, self.s.min_edge)
        regime = self.live_regime(str(row["regime"]), snap, q["ok"])
        reasons = []
        if signal == "NO TRADE":
            if conf < self.s.confidence_threshold:
                reasons.append("confidence_below_threshold")
            elif abs(p_up - p_down) < self.s.min_edge:
                reasons.append("edge_below_min_edge")
            else:
                reasons.append("flat_more_likely")
        if not q["ok"]:
            reasons.append("data_quality")
        if regime == "abnormal":
            reasons.append("abnormal_market")
        if regime == "low_liquidity" and self.s.block_low_liquidity:
            reasons.append("low_liquidity")
        late = (now - int(row["decision_ts"])) > 0.25 * self.s.horizon_bars * TIMEFRAME_MS[tf]
        if late:
            reasons.append("late_decision")
        ni, nr = snap["news_impact"], snap["news_relevance"]
        gate_detail = {  # exact numbers behind the decision, shown in the dashboard
            "confidence": conf, "threshold": self.s.confidence_threshold,
            "edge": abs(p_up - p_down), "min_edge": self.s.min_edge, "p_flat": p_flat,
            "model_direction": model_dir, "quality_score": q["score"], "quality_ok": q["ok"],
            "quality_issues": [f"{i['code']}: {i['detail']}" for i in q["issues"]],
            "regime": regime, "news_impact": ni, "news_relevance": nr,
            "decision_delay_sec": round((now - int(row["decision_ts"])) / 1000, 1),
            # decided after more than a quarter of the horizon had passed (e.g. a delayed scheduled run):
            # still causal, but not usable as a timely signal
            "late": late,
        }
        if signal in ("UP", "DOWN") and nr >= 0.5 and ((signal == "UP" and ni <= -0.35) or (signal == "DOWN" and ni >= 0.35)):
            reasons.append("news_conflict")  # news can veto, never create, a signal
        final = signal if not reasons else "NO TRADE"
        thr = float(max(self.s.round_trip_cost_bps / 1e4,
                        self.s.label_atr_mult * float(row["atr_pct"]) * math.sqrt(self.s.horizon_bars)))
        feat_dict = {f: _num(row.get(f)) for f in feats}
        snap["features"] = {k: v for k, v in feat_dict.items() if v is not None}
        snap["regime"] = regime
        bar = TIMEFRAME_MS[tf]
        self.db.execute(
            "INSERT INTO predictions(created_ms,candle_ts,target_ts,symbol,exchange,timeframe,horizon_bars,price,"
            "prediction,model_direction,p_up,p_down,p_flat,confidence,label_threshold,gate_reasons,gate_json,regime,"
            "quality_score,news_impact,news_relevance,features_version,model_version,features_json,snapshot_json) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(symbol,exchange,timeframe,horizon_bars,candle_ts) DO NOTHING",
            (now, candle_ts, candle_ts + self.s.horizon_bars * bar, symbol, self.s.primary_exchange, tf,
             self.s.horizon_bars, float(row["close"]), final, model_dir, p_up, p_down, p_flat, conf, thr,
             json.dumps(reasons), json.dumps(gate_detail, default=float), regime, q["score"], ni, nr,
             FEATURE_VERSION, prod["version"],
             json.dumps(feat_dict), json.dumps(snap, default=str)))
        out = {"status": "predicted", "symbol": symbol, "timeframe": tf, "candle_ts": candle_ts, "prediction": final,
               "p_up": p_up, "p_down": p_down, "p_flat": p_flat, "confidence": conf, "reasons": reasons,
               "regime": regime, "model_version": prod["version"], "gate": gate_detail}
        log.info("%s %s %s up=%.3f down=%.3f flat=%.3f reasons=%s", symbol, tf, final, p_up, p_down, p_flat, reasons)
        return out

    # ------------------------------------------------------------ resolve
    def resolve(self, symbol: str, tf: str, now: int | None = None) -> int:
        now = now or now_ms()
        bar = TIMEFRAME_MS[tf]
        pending = self.db.query(
            "SELECT * FROM predictions WHERE symbol=? AND timeframe=? AND exchange=? AND resolved_ms IS NULL "
            "AND target_ts + ? <= ? ORDER BY candle_ts", (symbol, tf, self.s.primary_exchange, bar, now))
        n = 0
        for p in pending:
            c = self.db.query_one(
                "SELECT open_ts, close FROM candles WHERE exchange=? AND symbol=? AND timeframe=? AND closed=1 "
                "AND open_ts >= ? ORDER BY open_ts LIMIT 1", (p["exchange"], symbol, tf, p["target_ts"]))
            if c is None:
                continue
            late = int(c["open_ts"]) != int(p["target_ts"])
            if late and int(c["open_ts"]) - int(p["target_ts"]) > 3 * bar:
                continue  # data gap too large to resolve fairly; leave pending (quality issue)
            ret = float(c["close"]) / float(p["price"]) - 1
            actual = direction_of(ret, float(p["label_threshold"]))
            probs = {"UP": p["p_up"], "DOWN": p["p_down"], "FLAT": p["p_flat"]}
            err = 1.0 - float(probs[actual])
            pred = p["prediction"]
            if pred in ("UP", "DOWN"):
                result = "correct" if pred == actual else "wrong"
                ecls = "correct" if pred == actual else ("false_move" if actual == "FLAT" else "wrong_direction")
            else:
                result = "no_trade"
                if actual == "FLAT":
                    ecls = "correct_abstain"
                elif p["model_direction"] == actual:
                    ecls = "gated_would_be_correct"
                else:
                    ecls = "missed_move"
            self.db.execute(
                "UPDATE predictions SET resolved_ms=?, actual_price=?, actual_return=?, actual_direction=?, error=?, "
                "result=?, error_class=? WHERE prediction_id=?",
                (now, float(c["close"]), ret, actual, err, result, ecls + ("_late" if late else ""), p["prediction_id"]))
            n += 1
        return n

    def run_once(self) -> dict:
        out = {}
        for symbol in self.s.symbols:
            for tf in self.s.predict_timeframes:
                k = f"{symbol}|{tf}"
                try:
                    res = self.predict(symbol, tf)
                    res["resolved"] = self.resolve(symbol, tf)
                    out[k] = res
                except Exception as exc:  # noqa: BLE001
                    log.exception("prediction failed for %s", k)
                    out[k] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        return out
