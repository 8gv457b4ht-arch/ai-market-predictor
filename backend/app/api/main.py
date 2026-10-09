"""HTTP API + static dashboard (Starlette / ASGI). Read-only market research:
there are no endpoints that place, modify or cancel orders."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from ..config import ROOT, TIMEFRAME_MS, get_settings
from ..db import get_db, now_ms
from ..learning import registry
from ..learning.engine import ledger_review
from ..market.candles import load_candles
from ..market.quality import evaluate
from ..ml.evaluation import backtest
from ..ml.features import compute_indicators
from ..news.service import news_features
from .security import SecurityMiddleware

FRONTEND = ROOT / "frontend"
VERSION = "2.0.0"
SERVICES = ("collector", "news", "predictor", "learner", "backup")


def clean(o):
    """JSON-safe conversion (NaN/inf -> null, numpy -> python)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


def J(content, status: int = 200) -> JSONResponse:
    return JSONResponse(clean(content), status_code=status)


class BadRequest(Exception):
    pass


def _params(req: Request):
    s = get_settings()
    q = req.query_params
    symbol = q.get("symbol", s.symbols[0]).upper()
    tf = q.get("tf", q.get("timeframe", s.predict_timeframes[0]))
    exchange = q.get("exchange", s.primary_exchange).lower()
    if symbol not in s.symbols:
        raise BadRequest(f"symbol must be one of {s.symbols}")
    if tf not in TIMEFRAME_MS or tf not in s.timeframes:
        raise BadRequest(f"timeframe must be one of {s.timeframes}")
    if exchange not in s.exchanges:
        raise BadRequest(f"exchange must be one of {s.exchanges}")
    return s, symbol, tf, exchange


def _int(req: Request, name: str, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(req.query_params.get(name, default))))
    except ValueError:
        raise BadRequest(f"{name} must be an integer") from None


def _float(req: Request, name: str, default: float, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(req.query_params.get(name, default))))
    except ValueError:
        raise BadRequest(f"{name} must be a number") from None


def _age(ts_ms, now=None):
    if not ts_ms:
        return None
    return round(((now or now_ms()) - int(ts_ms)) / 1000, 1)


# ------------------------------------------------------------- monitoring
def health(req: Request):
    db = get_db()
    ok = db.ping()
    return J({"status": "ok" if ok else "degraded", "database": ok, "version": VERSION}, 200 if ok else 503)


def _market_fresh(db, s) -> tuple[bool, float | None]:
    row = db.query_one("SELECT MAX(updated_ms) AS t FROM candles WHERE exchange=?", (s.primary_exchange,))
    age = _age(row["t"]) if row and row["t"] else None
    return (age is not None and age < 300), age


def ready(req: Request):
    db, s = get_db(), get_settings()
    db_ok = db.ping()
    fresh, age = _market_fresh(db, s) if db_ok else (False, None)
    keys = [registry.model_key(sym, tf, s.horizon_bars) for sym in s.symbols for tf in s.predict_timeframes]
    models = sum(1 for k in keys if db_ok and registry.get_production(db, k))
    ok = db_ok and fresh and models > 0
    return J({"ready": ok, "database": db_ok, "market_data_fresh": fresh, "market_data_age_sec": age,
              "models_ready": models, "models_expected": len(keys)}, 200 if ok else 503)


def status(req: Request):
    db, s = get_db(), get_settings()
    now = now_ms()
    db_ok = db.ping()
    size = None
    if db.dialect == "sqlite" and db.path and os.path.exists(db.path):
        size = sum(os.path.getsize(p) for p in (db.path, db.path + "-wal") if os.path.exists(p))
    streams = db.query("SELECT * FROM stream_status ORDER BY exchange, symbol")
    ws = {}
    for r in streams:
        e = ws.setdefault(r["exchange"], {"symbols": {}, "connected": True})
        age = _age(r["last_msg_ms"], now)
        live = r["state"] == "connected" and age is not None and age < s.stale_after_sec
        e["connected"] = e["connected"] and live
        e["symbols"][r["symbol"]] = {"state": r["state"], "last_msg_age_sec": age, "reconnects": r["reconnects"],
                                     "gaps": r["gaps"], "duplicates": r["duplicates"], "invalid": r["invalid"],
                                     "stale_events": r["stale_events"], "clock_skew_ms": r["clock_skew_ms"],
                                     "last_error": r["last_error"]}
    for ex in s.exchanges:
        ws.setdefault(ex, {"symbols": {}, "connected": False})
    n_conn = sum(1 for v in ws.values() if v["connected"])
    ws_state = "CONNECTED" if n_conn == len(s.exchanges) else ("PARTIAL" if n_conn else "DISCONNECTED")
    fresh, mage = _market_fresh(db, s)
    last_tick = db.query_one("SELECT MAX(last_trade_ms) AS t FROM stream_status")
    hb = {}
    for svc in SERVICES:
        v = db.get_state(f"heartbeat:{svc}")
        hb[svc] = {"age_sec": _age(v.get("ts_ms"), now) if v else None,
                   "alive": bool(v and _age(v.get("ts_ms"), now) is not None and _age(v.get("ts_ms"), now) < 120),
                   "status": (v or {}).get("status")}
    news_st = db.get_state("news_status") or {}
    news_age = _age(news_st.get("ts_ms"), now)
    if not news_st:
        news_state = "NOT STARTED"
    elif news_st.get("feeds_ok", 0) == 0:
        news_state = "DOWN"
    elif news_age is not None and news_age < 3 * s.news_interval_sec:
        news_state = "LIVE" if news_st.get("feeds_ok") == news_st.get("feeds_total") else "DEGRADED"
    else:
        news_state = "STALE"
    keys = [registry.model_key(sym, tf, s.horizon_bars) for sym in s.symbols for tf in s.predict_timeframes]
    models = {k: (registry.get_production(db, k) or {}).get("version") for k in keys}
    n_models = sum(1 for v in models.values() if v)
    model_state = "READY" if n_models == len(keys) else ("PARTIAL" if n_models else "NOT TRAINED")
    learn_hb = hb["learner"]
    last_pred = db.query_one("SELECT created_ms, symbol, timeframe, prediction FROM predictions ORDER BY created_ms DESC LIMIT 1")
    backup = db.get_state("last_backup")
    return J({
        "version": VERSION, "time_ms": now, "api": "OK", "auth": "enabled" if s.api_key else "DISABLED (set API_KEY)",
        "database": {"state": "OK" if db_ok else "ERROR", "dialect": db.dialect, "size_bytes": size},
        "websocket": {"state": ws_state, "exchanges": ws},
        "market_data": {"state": "LIVE" if fresh else "STALE", "last_candle_update_age_sec": mage,
                        "last_tick_age_sec": _age(last_tick["t"], now) if last_tick and last_tick["t"] else None},
        "news": {"state": news_state, "analyzer": news_st.get("analyzer"), "llm_enabled": news_st.get("llm_enabled"),
                 "llm_error": news_st.get("llm_error"), "feeds_ok": news_st.get("feeds_ok"),
                 "feeds_total": news_st.get("feeds_total"), "last_run_age_sec": news_age, "feeds": news_st.get("feeds")},
        "model": {"state": model_state, "production": models},
        "learning": {"state": "ACTIVE" if learn_hb["alive"] else "STOPPED", "heartbeat_age_sec": learn_hb["age_sec"],
                     "last_run": db.get_state("learning_last_run")},
        "services": hb,
        "last_prediction": last_pred and {**last_pred, "age_sec": _age(last_pred["created_ms"], now)},
        "last_backup": backup and {**backup, "age_sec": _age(backup.get("ts_ms"), now)},
    })


def config(req: Request):
    s = get_settings()
    return J({"symbols": s.symbols, "timeframes": s.timeframes, "predict_timeframes": s.predict_timeframes,
              "exchanges": s.exchanges, "primary_exchange": s.primary_exchange, "horizon_bars": s.horizon_bars,
              "confidence_threshold": s.confidence_threshold, "min_edge": s.min_edge,
              "auth_required": bool(s.api_key), "version": VERSION,
              "costs": {"fee_bps": s.fee_bps, "slippage_bps": s.slippage_bps, "spread_bps": s.spread_bps,
                        "latency_ms": s.latency_ms}})


# ------------------------------------------------------------- market data
def overview(req: Request):
    s, symbol, tf, exchange = _params(req)
    db = get_db()
    now = now_ms()
    ticker = db.query_one("SELECT * FROM market_ticker WHERE exchange=? AND symbol=?", (s.primary_exchange, symbol))
    book = db.query_one("SELECT best_bid,best_ask,mid,spread_bps,imbalance,bid_depth,ask_depth,recv_ms FROM book_state "
                        "WHERE exchange=? AND symbol=?", (s.primary_exchange, symbol))
    flow = db.query_one("SELECT * FROM flow_bars WHERE exchange=? AND symbol=? ORDER BY open_ts DESC LIMIT 1",
                        (s.primary_exchange, symbol))
    pred = db.query_one("SELECT * FROM predictions WHERE symbol=? AND timeframe=? ORDER BY candle_ts DESC LIMIT 1",
                        (symbol, tf))
    candles = load_candles(db, s.primary_exchange, symbol, tf, 400)
    price, price_src = None, None
    if flow and flow.get("last_price") and _age(flow["updated_ms"], now) < 30:
        price, price_src = flow["last_price"], "trade_stream"
    elif book and book.get("mid") and _age(book["recv_ms"], now) < 30:
        price, price_src = book["mid"], "order_book"
    elif ticker:
        price, price_src = ticker["last"], "rest_ticker"
    elif not candles.empty:
        price, price_src = float(candles.close.iloc[-1]), "last_closed_candle"
    quality = evaluate(db, s, symbol, tf, candles).to_dict()
    if pred:
        pred["gate_reasons"] = json.loads(pred["gate_reasons"] or "[]")
        snap = json.loads(pred.pop("snapshot_json") or "{}")
        pred.pop("features_json", None)
        pred["snapshot"] = {k: snap.get(k) for k in ("spread_bps", "orderbook_imbalance", "cvd_live", "news_impact",
                                                    "news_relevance", "news_top", "freshness", "quality")}
        pred["age_sec"] = _age(pred["created_ms"], now)
    regime = pred["regime"] if pred else None
    return J({"symbol": symbol, "timeframe": tf, "exchange": s.primary_exchange, "price": price,
              "price_source": price_src, "ticker": ticker, "book": book and {**book, "age_sec": _age(book["recv_ms"], now)},
              "cvd_live": flow and flow["cvd"], "regime": regime, "prediction": pred, "quality": quality,
              "horizon_bars": s.horizon_bars, "confidence_threshold": s.confidence_threshold})


def candles_ep(req: Request):
    s, symbol, tf, exchange = _params(req)
    limit = _int(req, "limit", 300, 20, 1500)
    df = load_candles(get_db(), exchange, symbol, tf, limit + 250, closed_only=False)
    if df.empty:
        return J({"symbol": symbol, "timeframe": tf, "exchange": exchange, "candles": []})
    ind = compute_indicators(df).tail(limit)
    cols = ["open_ts", "open", "high", "low", "close", "volume", "closed", "ema_21", "ema_55", "bb_upper",
            "bb_lower", "rsi_14", "macd_hist", "adx_14", "atr_pct", "cvd"]
    rows = ind[cols].replace({np.nan: None}).to_dict(orient="records")
    return J({"symbol": symbol, "timeframe": tf, "exchange": exchange, "candles": rows,
              "cvd_source": "exchange taker-buy volume" if ind["cvd"].notna().any() else None})


def orderbook_ep(req: Request):
    s, symbol, tf, exchange = _params(req)
    row = get_db().query_one("SELECT * FROM book_state WHERE exchange=? AND symbol=?", (exchange, symbol))
    if not row:
        return J({"symbol": symbol, "exchange": exchange, "available": False,
                  "detail": "no live order book yet (collector not connected?)"})
    levels = json.loads(row.pop("levels_json") or "{}")
    return J({"symbol": symbol, "exchange": exchange, "available": True, **row,
              "age_sec": _age(row["recv_ms"]), "bids": levels.get("bids", []), "asks": levels.get("asks", [])})


def orderflow_ep(req: Request):
    s, symbol, tf, exchange = _params(req)
    minutes = _int(req, "minutes", 240, 10, 1440)
    db = get_db()
    since = now_ms() - minutes * 60_000
    bars = db.query("SELECT open_ts,buy_volume,sell_volume,delta,cvd,trades,vwap,last_price FROM flow_bars "
                    "WHERE exchange=? AND symbol=? AND open_ts>=? ORDER BY open_ts", (exchange, symbol, since))
    samples = db.query("SELECT ts_ms,mid,spread_bps,imbalance FROM book_samples WHERE exchange=? AND symbol=? "
                       "AND ts_ms>=? ORDER BY ts_ms", (exchange, symbol, since))
    return J({"symbol": symbol, "exchange": exchange, "flow_bars": bars, "book_samples": samples})


def news_ep(req: Request):
    s, symbol, tf, exchange = _params(req)
    limit = _int(req, "limit", 30, 1, 200)
    db = get_db()
    rows = db.query("SELECT id,published_ms,source,title,url,summary,category,event_type,asset,direction,relevance,"
                    "novelty,confidence,horizon_min,affected_assets,analyzer FROM news_events "
                    "ORDER BY published_ms DESC LIMIT ?", (limit * 3,))
    base = symbol.split("/")[0]
    for r in rows:
        r["affected_assets"] = json.loads(r["affected_assets"] or "[]")
        r["asset_match"] = base in r["affected_assets"] or "CRYPTO_MARKET" in r["affected_assets"]
    rows.sort(key=lambda r: (-(r["relevance"] or 0) * (1 if r["asset_match"] else 0.5), -r["published_ms"]))
    rows = sorted(rows[:limit], key=lambda r: -r["published_ms"])
    return J({"symbol": symbol, "events": rows, "features": news_features(db, symbol, now_ms(), s.news_half_life_min),
              "status": db.get_state("news_status")})


def predictions_ep(req: Request):
    s, symbol, tf, exchange = _params(req)
    limit = _int(req, "limit", 50, 1, 500)
    rows = get_db().query(
        "SELECT prediction_id,created_ms,candle_ts,target_ts,symbol,exchange,timeframe,horizon_bars,price,prediction,"
        "model_direction,p_up,p_down,p_flat,confidence,gate_reasons,regime,quality_score,news_impact,model_version,"
        "features_version,resolved_ms,actual_price,actual_return,actual_direction,error,result,error_class "
        "FROM predictions WHERE symbol=? AND timeframe=? ORDER BY candle_ts DESC LIMIT ?", (symbol, tf, limit))
    for r in rows:
        r["gate_reasons"] = json.loads(r["gate_reasons"] or "[]")
    return J({"symbol": symbol, "timeframe": tf, "predictions": rows})


def model_ep(req: Request):
    s, symbol, tf, exchange = _params(req)
    db = get_db()
    key = registry.model_key(symbol, tf, s.horizon_bars)
    prod = registry.get_production(db, key)
    versions = registry.list_versions(db, key, 20)
    slim = [{k: v.get(k) for k in ("version", "status", "created_ms", "promoted_ms", "n_train", "n_validation",
                                     "reason", "parent_version", "feature_version")}
            | {"log_loss": (v.get("metrics") or {}).get("log_loss"), "accuracy": (v.get("metrics") or {}).get("accuracy"),
               "comparison": v.get("comparison") and {k: v["comparison"].get(k) for k in
                                                      ("logloss_gain", "bootstrap_p_better", "promote", "checks")}}
            for v in versions]
    if prod:
        prod.pop("features", None)
    return J({"model_key": key, "production": prod, "versions": slim,
              "ledger_review": ledger_review(db, symbol, tf, s.horizon_bars),
              "learning_status": db.get_state(f"learning_status:{key}")})


def learning_ep(req: Request):
    db = get_db()
    limit = _int(req, "limit", 50, 1, 500)
    ev = db.query("SELECT ts_ms, model_key, event_type, payload_json FROM learning_events ORDER BY id DESC LIMIT ?", (limit,))
    for e in ev:
        e["payload"] = json.loads(e.pop("payload_json") or "{}")
    return J({"events": ev, "last_run": db.get_state("learning_last_run"),
              "request": db.get_state("learning_request")})


def backtest_ep(req: Request):
    s, symbol, tf, exchange = _params(req)
    db = get_db()
    prod = registry.get_production(db, registry.model_key(symbol, tf, s.horizon_bars))
    if not prod:
        return J({"error": "no production model yet"}, 404)
    oos_path = registry.resolve_path((prod.get("params") or {}).get("oos_path"), s.model_dir)
    if not oos_path or not oos_path.exists():
        return J({"error": "this model version has no stored out-of-sample predictions",
                  "stored_backtest": (prod.get("metrics") or {}).get("backtest")}, 404)
    oos = pd.read_csv(oos_path)
    p = dict(threshold=_float(req, "threshold", s.confidence_threshold, 0.34, 0.99),
             min_edge=_float(req, "min_edge", s.min_edge, 0.0, 0.9),
             fee_bps=_float(req, "fee_bps", s.fee_bps, 0, 100), slippage_bps=_float(req, "slippage_bps", s.slippage_bps, 0, 100),
             spread_bps=_float(req, "spread_bps", s.spread_bps, 0, 100), latency_ms=_float(req, "latency_ms", s.latency_ms, 0, 3_600_000))
    res = backtest(oos, tf, int(prod["params"].get("horizon", s.horizon_bars)), **p)
    return J({"model_version": prod["version"], "params": p, "result": res,
              "oos_rows": len(oos), "method": (prod.get("metrics") or {}).get("method")})


def _request(key: str, label: str):
    def handler(req: Request):
        db = get_db()
        prev = db.get_state(key)
        if prev and not prev.get("handled") and now_ms() - prev.get("requested_ms", 0) < 600_000:
            return J({"status": "already_queued", "requested_ms": prev["requested_ms"]}, 202)
        db.set_state(key, {"requested_ms": now_ms(), "handled": False})
        return J({"status": "queued", "detail": f"{label} will start within a few seconds"}, 202)
    return handler


def home(req: Request):
    return FileResponse(FRONTEND / "index.html", headers={"cache-control": "no-cache"})


async def bad_request(req: Request, exc: BadRequest):
    return J({"error": "bad_request", "detail": str(exc)}, 400)


async def server_error(req: Request, exc: Exception):
    return J({"error": "internal_error", "detail": f"{type(exc).__name__}: {exc}"[:300]}, 500)


def create_app() -> Starlette:
    s = get_settings()
    get_db()  # create schema on startup
    routes = [
        Route("/", home),
        Route("/api/health", health), Route("/api/ready", ready), Route("/api/status", status),
        Route("/api/config", config), Route("/api/overview", overview), Route("/api/candles", candles_ep),
        Route("/api/orderbook", orderbook_ep), Route("/api/orderflow", orderflow_ep), Route("/api/news", news_ep),
        Route("/api/predictions", predictions_ep), Route("/api/model", model_ep), Route("/api/learning", learning_ep),
        Route("/api/backtest", backtest_ep),
        Route("/api/learning/run", _request("learning_request", "learning cycle"), methods=["POST"]),
        Route("/api/news/refresh", _request("news_refresh_request", "news refresh"), methods=["POST"]),
        Route("/api/backup/run", _request("backup_request", "backup"), methods=["POST"]),
        Mount("/static", app=StaticFiles(directory=str(FRONTEND)), name="static"),
    ]
    middleware = [Middleware(SecurityMiddleware, api_key=s.api_key, per_minute=s.rate_limit_per_min,
                             trust_proxy=s.trust_proxy_headers)]
    if s.cors_origins:
        middleware.insert(0, Middleware(CORSMiddleware, allow_origins=s.cors_origins, allow_methods=["GET", "POST"],
                                        allow_headers=["X-API-Key", "Authorization", "Content-Type"]))
    return Starlette(routes=routes, middleware=middleware,
                     exception_handlers={BadRequest: bad_request, Exception: server_error})

