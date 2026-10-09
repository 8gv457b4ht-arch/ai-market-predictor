"""Write the public JSON the static dashboard reads (no secrets, no raw model files)."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from ..config import Settings
from ..db import Database, now_ms
from ..learning import registry
from ..learning.engine import ledger_review
from ..market.candles import load_candles
from ..ml.features import compute_indicators
from ..news.service import news_features

EXPORT_VERSION = 1
METRIC_KEYS = ("n", "accuracy", "log_loss", "brier", "ece", "f1_macro", "precision_macro", "recall_macro",
               "signals", "baseline_prior", "label_distribution", "gate_diagnostics", "label_diagnostics",
               "by_regime", "method", "backtest", "confusion_matrix", "folds")


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    return o


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(clean(obj), separators=(",", ":"), default=str))
    os.replace(tmp, path)


def _prediction_row(r: dict) -> dict:
    out = {k: r[k] for k in ("prediction_id", "created_ms", "candle_ts", "target_ts", "symbol", "exchange", "timeframe",
                             "horizon_bars", "price", "prediction", "model_direction", "p_up", "p_down", "p_flat",
                             "confidence", "label_threshold", "regime", "quality_score", "news_impact", "news_relevance",
                             "features_version", "model_version", "resolved_ms", "actual_price", "actual_return",
                             "actual_direction", "error", "result", "error_class")}
    out["gate_reasons"] = json.loads(r.get("gate_reasons") or "[]")
    out["gate"] = json.loads(r.get("gate_json") or "{}")
    return out


def export_public(db: Database, settings: Settings, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    now = now_ms()
    primary = settings.primary_exchange if settings.primary_exchange != "auto" else db.get_state("primary_exchange")
    probes = db.get_state("exchange_probes") or {}
    markets = {}
    for sym in settings.symbols:
        t = db.query_one("SELECT * FROM market_ticker WHERE exchange=? AND symbol=?", (primary, sym)) if primary else None
        book = db.query_one("SELECT best_bid,best_ask,mid,spread_bps,imbalance,bid_depth,ask_depth,recv_ms,levels_json "
                            "FROM book_state WHERE exchange=? AND symbol=?", (primary, sym)) if primary else None
        flow = db.query("SELECT open_ts,buy_volume,sell_volume,delta,cvd,trades,last_price FROM flow_bars "
                        "WHERE exchange=? AND symbol=? ORDER BY open_ts DESC LIMIT 60", (primary, sym)) if primary else []
        if book:
            lv = json.loads(book.pop("levels_json") or "{}")
            book["bids"], book["asks"] = lv.get("bids", [])[:20], lv.get("asks", [])[:20]
        markets[sym] = {"ticker": t, "book": book, "flow_bars": list(reversed(flow)),
                        "news": news_features(db, sym, now, settings.news_half_life_min)}
        for tf in settings.predict_timeframes:
            df = load_candles(db, primary, sym, tf, 450, closed_only=False) if primary else None
            if df is None or df.empty:
                continue
            ind = compute_indicators(df).tail(220)
            cols = ["open_ts", "open", "high", "low", "close", "volume", "closed", "ema_21", "ema_55", "bb_upper",
                    "bb_lower", "rsi_14", "macd_hist", "adx_14", "cvd"]
            _write(out_dir / f"candles_{sym.replace('/', '-')}_{tf}.json",
                   {"symbol": sym, "timeframe": tf, "exchange": primary, "generated_ms": now,
                    "rows": ind[cols].replace({np.nan: None}).to_dict(orient="list")})
    models, latest = {}, {}
    for sym in settings.symbols:
        for tf in settings.predict_timeframes:
            key = registry.model_key(sym, tf, settings.horizon_bars)
            prod = registry.get_production(db, key)
            versions = registry.list_versions(db, key, 15)
            models[key] = {
                "production": prod and {"version": prod["version"], "created_ms": prod["created_ms"],
                                        "promoted_ms": prod["promoted_ms"], "n_train": prod["n_train"],
                                        "train_end_ts": prod["train_end_ts"], "feature_version": prod["feature_version"],
                                        "reason": prod["reason"],
                                        "metrics": {k: (prod["metrics"] or {}).get(k) for k in METRIC_KEYS}},
                "versions": [{"version": v["version"], "status": v["status"], "created_ms": v["created_ms"],
                              "reason": v["reason"], "log_loss": (v["metrics"] or {}).get("log_loss"),
                              "accuracy": (v["metrics"] or {}).get("accuracy"),
                              "comparison": v["comparison"] and {k: v["comparison"].get(k) for k in
                                                                 ("n_holdout", "logloss_gain", "bootstrap_p_better",
                                                                  "logloss_production", "logloss_challenger", "checks", "promote")}}
                             for v in versions],
                "learning_status": db.get_state(f"learning_status:{key}"),
                "ledger_review": ledger_review(db, sym, tf, settings.horizon_bars),
            }
            r = db.query_one("SELECT * FROM predictions WHERE symbol=? AND timeframe=? ORDER BY candle_ts DESC LIMIT 1", (sym, tf))
            latest[key] = r and _prediction_row(r)
    ledger = [_prediction_row(r) for r in db.query("SELECT * FROM predictions ORDER BY candle_ts DESC LIMIT 3000")]
    reasons: dict = {}
    for r in ledger:
        if r["prediction"] == "NO TRADE":
            for x in r["gate_reasons"]:
                reasons[x] = reasons.get(x, 0) + 1
    news = db.query("SELECT published_ms,source,title,url,category,asset,direction,relevance,novelty,confidence,"
                    "horizon_min,affected_assets,analyzer FROM news_events ORDER BY published_ms DESC LIMIT 60")
    for n in news:
        n["affected_assets"] = json.loads(n["affected_assets"] or "[]")
    events = db.query("SELECT ts_ms, model_key, event_type, payload_json FROM learning_events ORDER BY id DESC LIMIT 40")
    for e in events:
        e["payload"] = json.loads(e.pop("payload_json") or "{}")
    last_cycle = db.get_state("last_cycle") or {}
    state = {
        "export_version": EXPORT_VERSION, "generated_ms": now, "mode": "scheduled",
        "schedule_note": "The backend runs as a scheduled job (about every 15 minutes). Live WebSocket data is sampled "
                         "during each run; between runs the dashboard shows the last verified values with their age.",
        "settings": {"symbols": settings.symbols, "predict_timeframes": settings.predict_timeframes,
                     "horizon_bars": settings.horizon_bars, "confidence_threshold": settings.confidence_threshold,
                     "min_edge": settings.min_edge, "exchanges": settings.exchanges,
                     "costs": {"fee_bps": settings.fee_bps, "slippage_bps": settings.slippage_bps,
                               "spread_bps": settings.spread_bps, "latency_ms": settings.latency_ms}},
        "primary_exchange": primary, "probes": probes, "markets": markets, "models": models,
        "latest_predictions": latest,
        "ledger_summary": {"total": db.scalar("SELECT COUNT(*) FROM predictions"),
                           "signals": db.scalar("SELECT COUNT(*) FROM predictions WHERE prediction!='NO TRADE'"),
                           "resolved_signals": db.scalar("SELECT COUNT(*) FROM predictions WHERE prediction!='NO TRADE' AND resolved_ms IS NOT NULL"),
                           "correct_signals": db.scalar("SELECT COUNT(*) FROM predictions WHERE result='correct'"),
                           "no_trade_reasons": reasons},
        "news": {"status": db.get_state("news_status"), "events": news},
        "learning": {"last_run": db.get_state("learning_last_run"), "events": events},
        "last_cycle": {k: last_cycle.get(k) for k in ("started_ms", "duration_sec", "ok", "errors", "timings_sec", "primary_exchange")},
        "last_backup": db.get_state("last_backup"),
        "disclaimer": "Read-only research system. Probabilities are not trading recommendations; no orders are placed.",
    }
    _write(out_dir / "state.json", state)
    _write(out_dir / "ledger.json", {"generated_ms": now, "predictions": ledger})
    return {"files": sorted(p.name for p in out_dir.glob("*.json")), "ledger_rows": len(ledger)}
