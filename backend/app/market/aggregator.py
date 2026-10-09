"""In-memory aggregation of live WebSocket events and persistence to the DB.

Keeps, per (exchange, symbol):
  * 1-minute order-flow bars built from trades (buy/sell volume, delta, CVD, VWAP)
  * the local order book and its metrics
  * stream health counters (gaps, duplicates, invalid ticks, clock skew)
Flushing is batched (one transaction per FLUSH_INTERVAL) so the DB is not hit
once per tick.
"""
from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field

from ..db import Database, now_ms
from ..exchanges.orderbook import LocalOrderBook, book_metrics

log = logging.getLogger(__name__)
MINUTE = 60_000
BOOK_SAMPLE_MS = 10_000
MAX_FUTURE_SKEW_MS = 5_000
MAX_PAST_AGE_MS = 300_000


@dataclass
class FlowBar:
    open_ts: int
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    trades: int = 0
    notional: float = 0.0
    high: float | None = None
    low: float | None = None
    last_price: float | None = None
    cvd_close: float = 0.0
    dirty: bool = True


@dataclass
class StreamStats:
    state: str = "init"
    connected_since_ms: int | None = None
    last_msg_ms: int | None = None
    last_trade_ms: int | None = None
    last_book_ms: int | None = None
    last_exchange_ts_ms: int | None = None
    reconnects: int = 0
    gaps: int = 0
    duplicates: int = 0
    invalid: int = 0
    stale_events: int = 0
    clock_skew_ms: int | None = None
    last_error: str | None = None


@dataclass
class SymbolState:
    exchange: str
    symbol: str
    cvd: float = 0.0
    bars: dict[int, FlowBar] = field(default_factory=dict)
    book: LocalOrderBook = field(default_factory=LocalOrderBook)
    book_dirty: bool = False
    last_book_sample_ms: int = 0
    last_trade_id: int | None = None
    recent_ids: deque = field(default_factory=lambda: deque(maxlen=5000))
    recent_id_set: set = field(default_factory=set)
    raw_trades: list = field(default_factory=list)
    stats: StreamStats = field(default_factory=StreamStats)
    # 1-second bars by exchange timestamp: sec -> [open, high, low, close, buy_vol, sell_vol, trades, last_trade_ms]
    sec_bars: dict = field(default_factory=dict)
    last_price: float | None = None
    last_trade_ts: int | None = None


class Aggregator:
    def __init__(self, db: Database | None = None, store_raw_trades: bool = True,
                 second_bar_exchanges: set[str] | None = None):
        self.db = db
        self.store_raw_trades = store_raw_trades
        # exchanges whose 1-second bars are stored (None = all); the forecaster needs the primary one
        self.second_bar_exchanges = second_bar_exchanges
        self.states: dict[tuple[str, str], SymbolState] = {}
        self.exchange_stats: dict[str, StreamStats] = {}

    # ---------------------------------------------------------------- state
    def state(self, exchange: str, symbol: str) -> SymbolState:
        key = (exchange, symbol)
        st = self.states.get(key)
        if st is None:
            st = SymbolState(exchange, symbol)
            if self.db is not None:  # continue CVD across restarts
                row = self.db.query_one(
                    "SELECT cvd FROM flow_bars WHERE exchange=? AND symbol=? ORDER BY open_ts DESC LIMIT 1",
                    (exchange, symbol))
                if row:
                    st.cvd = float(row["cvd"])
            self.states[key] = st
        return st

    def symbols_of(self, exchange: str) -> list[SymbolState]:
        return [s for (ex, _), s in self.states.items() if ex == exchange]

    def set_connection_state(self, exchange: str, symbols: list[str], state: str, error: str | None = None,
                             reconnect: bool = False, stale: bool = False) -> None:
        t = now_ms()
        for sym in symbols:
            st = self.state(exchange, sym).stats
            st.state = state
            if state == "connected":
                st.connected_since_ms = t
            if error:
                st.last_error = error[:500]
            if reconnect:
                st.reconnects += 1
            if stale:
                st.stale_events += 1

    # ------------------------------------------------------------- events
    def on_event(self, ev: dict, recv_ms: int | None = None) -> str:
        """Apply one normalized event. Returns a short status string (for tests/metrics)."""
        recv_ms = recv_ms if recv_ms is not None else now_ms()
        kind = ev.get("kind")
        if kind not in ("trade", "book"):
            return "ignored"
        st = self.state(ev["exchange"], ev["symbol"])
        ts = ev.get("ts_ms") or recv_ms
        if ts > recv_ms + MAX_FUTURE_SKEW_MS:
            st.stats.invalid += 1
            st.stats.last_error = f"timestamp {ts} is in the future (recv {recv_ms})"
            return "invalid_ts"
        if ts < recv_ms - MAX_PAST_AGE_MS:
            st.stats.invalid += 1
            st.stats.last_error = f"event timestamp {ts} is {(recv_ms - ts) // 1000}s old"
            return "invalid_ts"
        # only events with a plausible exchange timestamp count as live data
        st.stats.last_msg_ms = recv_ms
        st.stats.clock_skew_ms = recv_ms - ts
        st.stats.last_exchange_ts_ms = ts
        return self._on_trade(st, ev, ts, recv_ms) if kind == "trade" else self._on_book(st, ev, ts, recv_ms)

    def _on_trade(self, st: SymbolState, ev: dict, ts: int, recv_ms: int) -> str:
        price, qty, side = ev["price"], ev["qty"], ev["side"]
        if not (price > 0) or not (qty > 0) or side not in ("buy", "sell"):
            st.stats.invalid += 1
            return "invalid"
        tid = ev.get("trade_id")
        status = "ok"
        if isinstance(tid, int):
            if st.last_trade_id is not None:
                if tid <= st.last_trade_id and tid in st.recent_id_set:
                    st.stats.duplicates += 1
                    return "duplicate"
                # Binance trade ids are strictly consecutive per symbol -> a jump is a missed-data gap.
                if ev["exchange"] == "binance" and tid > st.last_trade_id + 1:
                    st.stats.gaps += 1
                    status = "gap"
            st.last_trade_id = tid if st.last_trade_id is None else max(st.last_trade_id, tid)
        elif tid is not None and tid in st.recent_id_set:
            st.stats.duplicates += 1
            return "duplicate"
        if tid is not None:
            if len(st.recent_ids) == st.recent_ids.maxlen:
                st.recent_id_set.discard(st.recent_ids[0])
            st.recent_ids.append(tid)
            st.recent_id_set.add(tid)

        bar_ts = ts - ts % MINUTE
        bar = st.bars.get(bar_ts)
        if bar is None:
            bar = st.bars[bar_ts] = FlowBar(bar_ts)
        signed = qty if side == "buy" else -qty
        if side == "buy":
            bar.buy_volume += qty
        else:
            bar.sell_volume += qty
        st.cvd += signed
        bar.cvd_close = st.cvd
        bar.trades += 1
        bar.notional += price * qty
        bar.high = price if bar.high is None else max(bar.high, price)
        bar.low = price if bar.low is None else min(bar.low, price)
        bar.last_price = price
        bar.dirty = True
        st.stats.last_trade_ms = recv_ms
        if st.last_trade_ts is None or ts >= st.last_trade_ts:
            st.last_price, st.last_trade_ts = price, ts
        # a trade arriving > 4 s late would overwrite an already written second with a partial bar: skip it there
        if (self.second_bar_exchanges is None or st.exchange in self.second_bar_exchanges) and ts // 1000 >= recv_ms // 1000 - 4:
            sec = ts // 1000
            sb = st.sec_bars.get(sec)
            if sb is None:
                st.sec_bars[sec] = [price, price, price, price, qty if side == "buy" else 0.0,
                                    qty if side == "sell" else 0.0, 1, ts]
            else:
                sb[1], sb[2] = max(sb[1], price), min(sb[2], price)
                if ts >= sb[7]:
                    sb[3], sb[7] = price, ts
                sb[4 if side == "buy" else 5] += qty
                sb[6] += 1
        if self.store_raw_trades:
            st.raw_trades.append((st.exchange, st.symbol, None if tid is None else str(tid), ts, price, qty, side, recv_ms))
        return status

    def _on_book(self, st: SymbolState, ev: dict, ts: int, recv_ms: int) -> str:
        book = st.book
        if ev["type"] == "snapshot":
            if book.last_update_id is not None and ev.get("update_id") is not None \
                    and ev["update_id"] < book.last_update_id and ev["exchange"] != "bybit":
                st.stats.duplicates += 1
                return "stale"
            book.apply_snapshot(ev["bids"], ev["asks"], ev.get("update_id"), ev.get("seq"), ts)
            res = "ok"
        else:
            res = book.apply_delta(ev["bids"], ev["asks"], ev.get("update_id"), ev.get("seq"), ts)
            if res == "gap":
                st.stats.gaps += 1
                return "resync"
            if res == "stale":
                st.stats.duplicates += 1
                return "stale"
        m = book.metrics(20)
        if m["best_bid"] is not None and m["best_ask"] is not None and m["best_bid"] >= m["best_ask"]:
            st.stats.invalid += 1
            book.initialized = False
            return "resync"
        st.book_dirty = True
        st.stats.last_book_ms = recv_ms
        return res

    def needs_resync(self, exchange: str) -> bool:
        return any(s.book.gaps and not s.book.initialized for s in self.symbols_of(exchange))

    # ---------------------------------------------------------------- flush
    def flush(self, now: int | None = None) -> dict:
        if self.db is None:
            return {}
        now = now or now_ms()
        bars, raws, books, samples, statuses, secs = [], [], [], [], [], []
        for st in self.states.values():
            for sec, b in list(st.sec_bars.items()):
                secs.append((st.exchange, st.symbol, sec, *b))
                if sec < now // 1000 - 5:  # complete and written -> drop from memory
                    del st.sec_bars[sec]
            for ts, b in list(st.bars.items()):
                if b.dirty:
                    vwap = b.notional / (b.buy_volume + b.sell_volume) if (b.buy_volume + b.sell_volume) else None
                    bars.append((st.exchange, st.symbol, ts, b.buy_volume, b.sell_volume,
                                 b.buy_volume - b.sell_volume, b.cvd_close, b.trades, vwap, b.high, b.low,
                                 b.last_price, now))
                    b.dirty = False
                if ts < now - 3 * MINUTE:  # finished and persisted -> drop from memory
                    del st.bars[ts]
            if st.raw_trades:
                raws.extend(st.raw_trades)
                st.raw_trades = []
            if st.book_dirty and st.book.initialized:
                bids, asks = st.book.top(20)
                m = book_metrics(bids, asks)
                books.append((st.exchange, st.symbol, st.book.ts_ms or now, now, m["best_bid"], m["best_ask"],
                              m["mid"], m["spread_bps"], m["bid_depth"], m["ask_depth"], m["imbalance"],
                              json.dumps({"bids": bids, "asks": asks})))
                if now - st.last_book_sample_ms >= BOOK_SAMPLE_MS:
                    samples.append((st.exchange, st.symbol, now - now % BOOK_SAMPLE_MS, m["mid"], m["spread_bps"],
                                    m["imbalance"], m["bid_depth"], m["ask_depth"]))
                    st.last_book_sample_ms = now
                st.book_dirty = False
            s = st.stats
            statuses.append((st.exchange, st.symbol, s.state, s.connected_since_ms, s.last_msg_ms, s.last_trade_ms,
                             s.last_book_ms, s.last_exchange_ts_ms, s.reconnects, s.gaps, s.duplicates, s.invalid,
                             s.stale_events, s.clock_skew_ms, s.last_error, now))
        with self.db.transaction():
            self.db.executemany(
                "INSERT INTO flow_bars(exchange,symbol,open_ts,buy_volume,sell_volume,delta,cvd,trades,vwap,high,low,last_price,updated_ms) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(exchange,symbol,open_ts) DO UPDATE SET "
                "buy_volume=excluded.buy_volume,sell_volume=excluded.sell_volume,delta=excluded.delta,cvd=excluded.cvd,"
                "trades=excluded.trades,vwap=excluded.vwap,high=excluded.high,low=excluded.low,"
                "last_price=excluded.last_price,updated_ms=excluded.updated_ms", bars)
            self.db.executemany(
                "INSERT INTO trades_raw(exchange,symbol,trade_id,ts_ms,price,qty,side,recv_ms) VALUES(?,?,?,?,?,?,?,?)", raws)
            self.db.executemany(
                "INSERT INTO price_seconds(exchange,symbol,ts_sec,open,high,low,close,buy_volume,sell_volume,trades,last_trade_ms) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(exchange,symbol,ts_sec) DO UPDATE SET high=excluded.high,"
                "low=excluded.low,close=excluded.close,buy_volume=excluded.buy_volume,sell_volume=excluded.sell_volume,"
                "trades=excluded.trades,last_trade_ms=excluded.last_trade_ms", secs)
            self.db.executemany(
                "INSERT INTO book_state(exchange,symbol,ts_ms,recv_ms,best_bid,best_ask,mid,spread_bps,bid_depth,ask_depth,imbalance,levels_json) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(exchange,symbol) DO UPDATE SET ts_ms=excluded.ts_ms,"
                "recv_ms=excluded.recv_ms,best_bid=excluded.best_bid,best_ask=excluded.best_ask,mid=excluded.mid,"
                "spread_bps=excluded.spread_bps,bid_depth=excluded.bid_depth,ask_depth=excluded.ask_depth,"
                "imbalance=excluded.imbalance,levels_json=excluded.levels_json", books)
            self.db.executemany(
                "INSERT INTO book_samples(exchange,symbol,ts_ms,mid,spread_bps,imbalance,bid_depth,ask_depth) "
                "VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(exchange,symbol,ts_ms) DO NOTHING", samples)
            self.db.executemany(
                "INSERT INTO stream_status(exchange,symbol,state,connected_since_ms,last_msg_ms,last_trade_ms,last_book_ms,"
                "last_exchange_ts_ms,reconnects,gaps,duplicates,invalid,stale_events,clock_skew_ms,last_error,updated_ms) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(exchange,symbol) DO UPDATE SET state=excluded.state,"
                "connected_since_ms=excluded.connected_since_ms,last_msg_ms=excluded.last_msg_ms,"
                "last_trade_ms=excluded.last_trade_ms,last_book_ms=excluded.last_book_ms,"
                "last_exchange_ts_ms=excluded.last_exchange_ts_ms,reconnects=excluded.reconnects,gaps=excluded.gaps,"
                "duplicates=excluded.duplicates,invalid=excluded.invalid,stale_events=excluded.stale_events,"
                "clock_skew_ms=excluded.clock_skew_ms,last_error=excluded.last_error,updated_ms=excluded.updated_ms",
                statuses)
        return {"bars": len(bars), "trades": len(raws), "books": len(books), "samples": len(samples), "seconds": len(secs)}

