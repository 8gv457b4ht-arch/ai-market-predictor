"""One complete backend cycle, designed for a scheduler (GitHub Actions cron, systemd timer, ...).

    REST candles + tickers (all exchanges, with diagnostics)
 -> live WebSocket sample (trades + order books, timestamp-verified)
 -> predictions on the just-closed candles (while the stream is live) + resolution of old ones
 -> guarded learning -> news -> notifications -> backup -> public JSON export

State lives in DATABASE_URL + MODEL_DIR, so the next run continues where this one stopped.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback

from ..config import TIMEFRAME_MS, Settings
from ..db import Database, now_ms
from ..exchanges import rest
from ..exchanges.ws import build_protocol
from ..market.aggregator import Aggregator
from ..market.candles import upsert_candles
from ..market.collector import ExchangeStream

log = logging.getLogger("cycle")
PREFERRED_PRIMARY = ("binance", "okx", "bybit")


def _err(exc: BaseException) -> dict:
    return {"ok": False, "error": str(exc)[:300], "kind": getattr(exc, "kind", type(exc).__name__)}


def history_bars(settings: Settings, exchange: str, primary: str | None, tf: str) -> int:
    if exchange == primary:
        return settings.history_bars if tf in settings.predict_timeframes else settings.context_history_bars
    return 300


def sync_candles(db: Database, settings: Settings, exchange: str, symbol: str, tf: str, want: int) -> dict:
    """Fetch only what is missing: full history on first run, the tail afterwards."""
    t0 = time.monotonic()
    row = db.query_one("SELECT COUNT(*) AS n, MAX(open_ts) AS last FROM candles WHERE exchange=? AND symbol=? AND timeframe=?",
                       (exchange, symbol, tf))
    have, last = int(row["n"] or 0), row["last"]
    missing = (now_ms() - int(last)) // TIMEFRAME_MS[tf] + 2 if last else want
    if have >= want * 0.9 and missing <= 900:
        rows = rest.fetch_klines(exchange, symbol, tf, int(min(1000, max(5, missing))))
    else:
        rows = rest.fetch_history(exchange, symbol, tf, want)
    upsert_candles(db, exchange, symbol, tf, rows)
    closed = [r for r in rows if r["closed"]]
    return {"ok": bool(rows), "rows": len(rows), "latency_ms": int((time.monotonic() - t0) * 1000),
            "last_closed_ts": closed[-1]["open_ts"] if closed else None,
            "last_close_age_sec": round((now_ms() - closed[-1]["open_ts"] - TIMEFRAME_MS[tf]) / 1000, 1) if closed else None,
            "host": rest._working_host.get(exchange)}


def probe_rest(db: Database, settings: Settings, primary_hint: str | None) -> dict:
    """Candles for every exchange/symbol/timeframe + 24h tickers. One failing exchange never stops the others."""
    out: dict = {}
    for ex in settings.exchanges:
        res = {"candles": {}, "tickers": {}}
        for sym in settings.symbols:
            tfs = settings.timeframes if ex == primary_hint else [tf for tf in settings.timeframes if tf in ("1m", "15m", "1h")]
            for tf in tfs:
                try:
                    res["candles"][f"{sym} {tf}"] = sync_candles(db, settings, ex, sym, tf, history_bars(settings, ex, primary_hint, tf))
                except Exception as exc:  # noqa: BLE001
                    res["candles"][f"{sym} {tf}"] = _err(exc)
            try:
                t = rest.fetch_ticker(ex, sym)
                db.execute(
                    "INSERT INTO market_ticker(exchange,symbol,last,change_24h_pct,volume_24h,quote_volume_24h,high_24h,low_24h,ts_ms) "
                    "VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(exchange,symbol) DO UPDATE SET last=excluded.last,"
                    "change_24h_pct=excluded.change_24h_pct,volume_24h=excluded.volume_24h,quote_volume_24h=excluded.quote_volume_24h,"
                    "high_24h=excluded.high_24h,low_24h=excluded.low_24h,ts_ms=excluded.ts_ms",
                    (ex, sym, t["last"], t["change_24h_pct"], t["volume_24h"], t["quote_volume_24h"], t["high_24h"], t["low_24h"], now_ms()))
                res["tickers"][sym] = {"ok": True, "last": t["last"], "exchange_ts": t["ts_ms"]}
            except Exception as exc:  # noqa: BLE001
                res["tickers"][sym] = _err(exc)
        cands = list(res["candles"].values())
        good = [c for c in cands if c.get("ok")]
        bad = [c for c in cands if not c.get("ok")]
        res["ok"] = bool(good) and not bad
        res["partial"] = bool(good) and bool(bad)
        res["error"] = bad[0]["error"] if bad else None
        res["kind"] = bad[0]["kind"] if bad else None
        res["host"] = rest._working_host.get(ex)
        out[ex] = res
    return out


def choose_primary(db: Database, settings: Settings, rest_probe: dict) -> str | None:
    if settings.primary_exchange != "auto":
        return settings.primary_exchange
    prev = db.get_state("primary_exchange")
    fails = db.get_state("primary_failures", 0) or 0
    if prev and rest_probe.get(prev, {}).get("ok"):
        db.set_state("primary_failures", 0)
        return prev
    if prev and fails < 3:  # do not abandon the exchange the models were trained on after one bad run
        db.set_state("primary_failures", fails + 1)
        return prev
    for ex in PREFERRED_PRIMARY:
        if ex in settings.exchanges and rest_probe.get(ex, {}).get("ok"):
            db.set_state("primary_exchange", ex)
            db.set_state("primary_failures", 0)
            if prev and prev != ex:
                db.log_event("primary_changed", {"from": prev, "to": ex})
            return ex
    return prev


async def sample_streams(db: Database, settings: Settings, seconds: float, during=None, connect=None) -> dict:
    """Run all exchange streams for up to `seconds`, call `during()` while they are still live, then stop."""
    agg = Aggregator(db, store_raw_trades=False)
    for ex in settings.exchanges:
        for sym in settings.symbols:
            agg.state(ex, sym)
    streams = {ex: ExchangeStream(build_protocol(ex, settings.symbols, settings), settings.symbols, agg, settings, connect)
               for ex in settings.exchanges}
    stop = asyncio.Event()
    tasks = [asyncio.create_task(s.run_forever(stop)) for s in streams.values()]
    t0 = time.monotonic()
    min_wait = min(15.0, seconds)
    while time.monotonic() - t0 < seconds:
        await asyncio.sleep(1)
        done = all(st.stats.last_trade_ms and st.book.initialized for st in agg.states.values())
        if done and time.monotonic() - t0 >= min_wait:
            break
    await asyncio.to_thread(agg.flush)
    during_result = await asyncio.to_thread(during) if during else None
    await asyncio.to_thread(agg.flush)
    out = {}
    now = now_ms()
    for ex, s in streams.items():
        syms = {}
        for sym in settings.symbols:
            st = agg.state(ex, sym)
            m = st.book.metrics(10) if st.book.initialized else {}
            syms[sym] = {"trade_received": st.stats.last_trade_ms is not None, "book_received": st.book.initialized,
                         "last_verified_age_sec": round((now - st.stats.last_msg_ms) / 1000, 1) if st.stats.last_msg_ms else None,
                         "clock_skew_ms": st.stats.clock_skew_ms, "gaps": st.stats.gaps, "duplicates": st.stats.duplicates,
                         "invalid": st.stats.invalid, "spread_bps": m.get("spread_bps"), "mid": m.get("mid")}
        good = [sym for sym, v in syms.items() if v["trade_received"] and v["book_received"]]
        verified = len(good) == len(syms)
        out[ex] = {"ok": verified, "verified_live": verified, "verified_symbols": good, "url": getattr(s, "current_url", s.p.url).split("?")[0],
                   "sessions": s.sessions, "failures": s.failures, "events": s.events,
                   "first_event_after_ms": (s.first_event_ms - s.connected_ms) if s.first_event_ms and s.connected_ms else None,
                   "error": None if verified else (agg.state(ex, settings.symbols[0]).stats.last_error or "no verified data"),
                   "kind": None if verified else (s.last_error_kind or "no_data"), "symbols": syms}
    stop.set()
    await asyncio.gather(*tasks, return_exceptions=True)
    return {"streams": out, "during": during_result, "seconds": round(time.monotonic() - t0, 1)}


def prune_models(db: Database, settings: Settings, keep_archived: int = 2) -> dict:
    """Delete model files that are no longer needed (metrics stay in the registry)."""
    from ..learning.registry import resolve_path
    removed = []
    for key in {r["model_key"] for r in db.query("SELECT DISTINCT model_key FROM model_registry")}:
        old = db.query("SELECT version, artifact_path, params_json FROM model_registry WHERE model_key=? "
                       "AND status IN ('archived','rejected') ORDER BY created_ms DESC", (key,))
        for r in old[keep_archived:]:
            paths = [r["artifact_path"], (json.loads(r["params_json"] or "{}")).get("oos_path")]
            for p in paths:
                rp = resolve_path(p, settings.model_dir)
                if rp is not None and rp.exists():
                    rp.unlink()
                    removed.append(rp.name)
            db.execute("UPDATE model_registry SET artifact_path=NULL WHERE version=?", (r["version"],))
    return {"removed": removed}


def run_cycle(db: Database, settings: Settings, connect=None, learn: bool = True, news: bool = True) -> dict:
    from ..learning.engine import run_learning_once
    from ..learning.predictor import Predictor
    from ..news.service import run_news_once
    from ..notify import Notifier
    from ..workers.backup import run_backup_once
    from .export import export_public

    t0, started = time.monotonic(), now_ms()
    report: dict = {"started_ms": started, "errors": []}

    def step(name, fn):
        s0 = time.monotonic()
        try:
            r = fn()
            report[name] = r
            return r
        except Exception as exc:  # noqa: BLE001 - every step is isolated
            log.exception("step %s failed", name)
            report["errors"].append({"step": name, "error": f"{type(exc).__name__}: {exc}"[:400],
                                     "trace": traceback.format_exc()[-1500:]})
            return None
        finally:
            report.setdefault("timings_sec", {})[name] = round(time.monotonic() - s0, 1)

    hint = db.get_state("primary_exchange") if settings.primary_exchange == "auto" else settings.primary_exchange
    rest_probe = step("rest", lambda: probe_rest(db, settings, hint or "binance")) or {}
    primary = choose_primary(db, settings, rest_probe)
    if primary and primary != hint:  # first run on this primary: download its full history now
        rest_probe = step("rest_primary", lambda: probe_rest(db, settings, primary)) or rest_probe
    report["primary_exchange"] = primary
    if primary:
        settings.primary_exchange = primary
    predictor = Predictor(db, settings) if primary else None
    ws = step("ws", lambda: asyncio.run(sample_streams(db, settings, settings.ws_sample_sec,
                                                       during=predictor.run_once if predictor else None, connect=connect))) or {}
    report["predictions"] = ws.get("during")
    probes = {ex: {"rest": rest_probe.get(ex, {}), "ws": (ws.get("streams") or {}).get(ex, {})} for ex in settings.exchanges}
    db.set_state("exchange_probes", {"ts_ms": now_ms(), "primary": primary, "probes": probes})
    if learn and primary:
        report["learning"] = step("learning", lambda: run_learning_once(db, settings))
    if news:
        report["news"] = step("news", lambda: run_news_once(db, settings))
    notifier = Notifier(db, settings)
    if notifier.enabled:
        step("notify", lambda: (notifier.outages(probes), notifier.predictions(), notifier.promotions(),
                                {"sent": len(notifier.sent), "errors": notifier.errors})[-1])
    step("prune_models", lambda: prune_models(db, settings))
    last_backup = (db.get_state("last_backup") or {}).get("ts_ms", 0)
    if now_ms() - last_backup > settings.backup_interval_sec * 1000:
        step("backup", lambda: run_backup_once(db, settings))
    report["duration_sec"] = round(time.monotonic() - t0, 1)
    report["ok"] = not report["errors"]
    db.set_state("last_cycle", json.loads(json.dumps(report, default=str)))
    db.heartbeat("cycle", {"ok": report["ok"], "duration_sec": report["duration_sec"]})
    step("export", lambda: export_public(db, settings, settings.public_dir))
    return report
