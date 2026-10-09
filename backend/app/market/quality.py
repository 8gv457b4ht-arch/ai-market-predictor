"""Data Quality Engine. Bad data -> NO TRADE.

Checks: stale candles, missing candles, invalid prices/volumes, timestamp
problems (future bars, clock skew), WebSocket state and gaps/duplicates since
the previous check, stale order book, wide spread, and cross-exchange price
disagreement.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import pandas as pd

from ..config import TIMEFRAME_MS, Settings
from ..db import Database, now_ms
from .candles import missing_bars

CRITICAL = "critical"
WARNING = "warning"


@dataclass
class QualityReport:
    score: float = 1.0
    ok: bool = True
    issues: list[dict] = field(default_factory=list)
    checks: dict = field(default_factory=dict)

    def add(self, code: str, severity: str, penalty: float, detail: str) -> None:
        self.issues.append({"code": code, "severity": severity, "detail": detail})
        self.score = max(0.0, self.score - penalty)

    def finalize(self, min_score: float) -> "QualityReport":
        self.score = round(self.score, 3)
        self.ok = self.score >= min_score and not any(i["severity"] == CRITICAL for i in self.issues)
        return self

    def to_dict(self) -> dict:
        return {"score": self.score, "ok": self.ok, "issues": self.issues, "checks": self.checks}


class QualityTracker:
    """Remembers stream counters between checks to detect *new* gaps/duplicates."""

    def __init__(self):
        self.prev: dict[tuple, dict] = {}

    def delta(self, exchange: str, symbol: str, row: dict) -> dict:
        key = (exchange, symbol)
        cur = {k: int(row.get(k) or 0) for k in ("gaps", "duplicates", "invalid", "reconnects", "stale_events")}
        prev = self.prev.get(key, cur)
        self.prev[key] = cur
        return {k: max(0, cur[k] - prev.get(k, 0)) for k in cur}


def evaluate(db: Database, settings: Settings, symbol: str, timeframe: str, candles: pd.DataFrame,
             tracker: QualityTracker | None = None, now: int | None = None) -> QualityReport:
    now = now or now_ms()
    r = QualityReport()
    tf_ms = TIMEFRAME_MS[timeframe]
    primary = settings.primary_exchange

    # 1. candle freshness / continuity / validity
    if candles is None or candles.empty:
        r.add("no_candles", CRITICAL, 1.0, f"no closed {timeframe} candles for {symbol}")
        return r.finalize(settings.min_quality_score)
    last_close_ts = int(candles.open_ts.iloc[-1]) + tf_ms
    age = now - last_close_ts
    r.checks["last_candle_age_sec"] = round(age / 1000, 1)
    if age > tf_ms + 180_000:
        r.add("stale_candles", CRITICAL, 0.6, f"last closed {timeframe} candle ended {age/1000:.0f}s ago")
    recent = candles.tail(300)
    miss = missing_bars(recent, timeframe)
    r.checks["missing_bars_recent"] = miss
    if miss > 0:
        r.add("missing_bars", CRITICAL if miss > 10 else WARNING, min(0.4, 0.02 * miss), f"{miss} missing {timeframe} bars in last 300")
    bad = int(((recent[["open", "high", "low", "close"]] <= 0).any(axis=1) | (recent.volume < 0)
               | (recent.high < recent[["open", "close"]].max(axis=1)) | (recent.low > recent[["open", "close"]].min(axis=1))).sum())
    r.checks["invalid_candles"] = bad
    if bad:
        r.add("invalid_candles", CRITICAL, 0.5, f"{bad} candles with invalid prices/volumes")
    zero_vol = int((recent.tail(20).volume == 0).sum())
    if zero_vol >= 5:
        r.add("zero_volume", WARNING, 0.15, f"{zero_vol} of last 20 bars have zero volume")
    if int(candles.open_ts.iloc[-1]) > now:
        r.add("future_candle", CRITICAL, 0.5, "candle timestamp is in the future")
    dups = int(candles.open_ts.duplicated().sum())
    if dups:
        r.add("duplicate_candles", WARNING, 0.1, f"{dups} duplicate candle timestamps")

    # 2. live stream state (primary exchange)
    st = db.query_one("SELECT * FROM stream_status WHERE exchange=? AND symbol=?", (primary, symbol))
    if st is None:
        r.add("no_stream", WARNING, 0.2, f"no WebSocket status for {primary}")
    else:
        last_msg_age = (now - st["last_msg_ms"]) / 1000 if st.get("last_msg_ms") else None
        r.checks["ws_state"] = st["state"]
        r.checks["ws_last_msg_age_sec"] = None if last_msg_age is None else round(last_msg_age, 1)
        if st["state"] != "connected" or last_msg_age is None or last_msg_age > settings.stale_after_sec:
            r.add("ws_down", WARNING, 0.2, f"{primary} WebSocket {st['state']}, last msg age {last_msg_age}")
        if st.get("clock_skew_ms") is not None and abs(st["clock_skew_ms"]) > 5000:
            r.add("clock_skew", WARNING, 0.1, f"exchange/local clock skew {st['clock_skew_ms']} ms")
        if tracker is not None:
            d = tracker.delta(primary, symbol, st)
            r.checks["stream_deltas"] = d
            if d["gaps"]:
                r.add("ws_gaps", WARNING, min(0.3, 0.05 * d["gaps"]), f"{d['gaps']} new stream gaps")
            if d["duplicates"] > 50:
                r.add("duplicate_ticks", WARNING, 0.05, f"{d['duplicates']} duplicate ticks")
            if d["invalid"]:
                r.add("invalid_ticks", WARNING, min(0.3, 0.05 * d["invalid"]), f"{d['invalid']} invalid ticks")

    # 3. order book freshness and spread
    book = db.query_one("SELECT * FROM book_state WHERE exchange=? AND symbol=?", (primary, symbol))
    if book is not None:
        b_age = (now - book["recv_ms"]) / 1000
        r.checks["book_age_sec"] = round(b_age, 1)
        r.checks["spread_bps"] = book["spread_bps"]
        if b_age > settings.stale_after_sec:
            r.add("stale_book", WARNING, 0.15, f"order book {b_age:.0f}s old")
        elif book["spread_bps"] is not None and book["spread_bps"] > settings.max_spread_bps:
            r.add("wide_spread", WARNING, 0.2, f"spread {book['spread_bps']:.1f} bps > {settings.max_spread_bps}")

    # 4. exchange disagreement (fresh mids / last prices across exchanges)
    prices = {}
    for row in db.query("SELECT exchange, mid, recv_ms FROM book_state WHERE symbol=?", (symbol,)):
        if row["mid"] and now - row["recv_ms"] < 60_000:
            prices[row["exchange"]] = row["mid"]
    if len(prices) < 2:
        for row in db.query("SELECT exchange, last, ts_ms FROM market_ticker WHERE symbol=?", (symbol,)):
            if row["last"] and now - row["ts_ms"] < 120_000:
                prices.setdefault(row["exchange"], row["last"])
    r.checks["exchange_prices"] = prices
    if len(prices) >= 2:
        med = statistics.median(prices.values())
        dev = {ex: abs(p / med - 1) * 1e4 for ex, p in prices.items()}
        worst = max(dev, key=dev.get)
        r.checks["max_divergence_bps"] = round(dev[worst], 2)
        if dev[worst] > settings.max_exchange_divergence_bps:
            sev = CRITICAL if worst == primary else WARNING
            r.add("exchange_disagreement", sev, 0.3, f"{worst} deviates {dev[worst]:.1f} bps from cross-exchange median")
    return r.finalize(settings.min_quality_score)
