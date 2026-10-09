"""Live collector service.

* One supervised WebSocket task per exchange (trades + order book for every
  configured symbol). A failing exchange only affects itself: it reconnects
  with exponential back-off + jitter while the others keep streaming.
* Stale detection: no message within STALE_AFTER_SEC -> connection is dropped
  and re-established. Book gaps/crossed books -> forced re-subscribe.
* REST candle sync: history backfill on start, then incremental refresh of the
  latest closed candles for every timeframe, plus 24h tickers.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time

from ..config import Settings
from ..db import Database, now_ms
from ..exchanges import rest
from ..exchanges.ws import Protocol, build_protocol
from .aggregator import Aggregator
from .candles import upsert_candles

log = logging.getLogger("collector")


class StaleStream(Exception):
    pass


class ResyncNeeded(Exception):
    pass


def classify_ws_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if isinstance(exc, StaleStream):
        return "stale"
    if isinstance(exc, ResyncNeeded):
        return "gap"
    if "451" in text or "403" in text or "restricted" in text:
        return "region_blocked"
    if "429" in text or "too many" in text:
        return "rate_limited"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "exchange error" in text:
        return "exchange_error"
    return "network"


def default_connect():
    try:
        from websockets.asyncio.client import connect  # websockets >= 13
    except ImportError:  # pragma: no cover
        from websockets import connect  # type: ignore
    return connect


class ExchangeStream:
    def __init__(self, protocol: Protocol, symbols: list[str], agg: Aggregator, settings: Settings,
                 connect=None, sleep=asyncio.sleep):
        self.p = protocol
        self.symbols = symbols
        self.agg = agg
        self.s = settings
        self._connect = connect
        self._sleep = sleep
        self.failures = 0
        self.sessions = 0
        self.url_index = 0
        self.got_data = False
        self.events = 0
        self.first_event_ms: int | None = None
        self.connected_ms: int | None = None
        self.last_error_kind: str | None = None

    def _backoff(self) -> float:
        base = min(self.s.reconnect_max_sec, 1.0 * (2 ** min(self.failures, 10)))
        return base * (0.5 + random.random() / 2)

    async def run_forever(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            started = time.monotonic()
            self.agg.set_connection_state(self.p.exchange, self.symbols, "connecting")
            try:
                connect = self._connect or default_connect()
                await self._session(connect, stop)
                if stop.is_set():
                    break
                raise ConnectionError("server closed the connection")
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - every failure means reconnect
                stale = isinstance(exc, StaleStream)
                self.last_error_kind = classify_ws_error(exc)
                if not self.got_data:  # endpoint never delivered data: try the alternative next time
                    self.url_index += 1
                if time.monotonic() - started > 60:
                    self.failures = 0  # the session was healthy for a while: reset back-off
                self.failures += 1
                delay = self._backoff()
                msg = f"{type(exc).__name__}: {exc}"
                log.warning("%s stream down (%s); reconnect in %.1fs", self.p.exchange, msg, delay)
                self.agg.set_connection_state(self.p.exchange, self.symbols, "reconnecting", error=msg,
                                              reconnect=True, stale=stale)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
        self.agg.set_connection_state(self.p.exchange, self.symbols, "stopped")

    async def _keepalive(self, ws) -> None:
        while True:
            await self._sleep(self.p.keepalive_sec)
            await ws.send(self.p.keepalive_message)

    async def _session(self, connect, stop: asyncio.Event) -> None:
        urls = self.p.urls()
        url = urls[self.url_index % len(urls)]
        self.current_url = url
        self.got_data = False
        async with connect(url, ping_interval=20, ping_timeout=20, max_size=8 * 1024 * 1024,
                           open_timeout=15, close_timeout=5) as ws:
            self.sessions += 1
            for m in self.p.subscribe_messages:
                await ws.send(m)
            self.connected_ms = now_ms()
            # "connected" = socket open and subscribed; the API reports LIVE only once
            # timestamp-checked events have actually arrived (see Aggregator).
            self.agg.set_connection_state(self.p.exchange, self.symbols, "connected")
            log.info("%s connected to %s (%s)", self.p.exchange, url.split("?")[0], ", ".join(self.symbols))
            ka = asyncio.create_task(self._keepalive(ws)) if self.p.keepalive_message else None
            stop_wait = asyncio.create_task(stop.wait())
            try:
                while True:
                    recv = asyncio.ensure_future(ws.recv())
                    done, _ = await asyncio.wait({recv, stop_wait}, timeout=self.s.stale_after_sec,
                                                 return_when=asyncio.FIRST_COMPLETED)
                    if stop_wait in done:  # shutdown requested: do not wait for the next message
                        recv.cancel()
                        return
                    if recv not in done:
                        recv.cancel()
                        raise StaleStream(f"no data for {self.s.stale_after_sec:.0f}s")
                    raw = recv.result()
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", "replace")
                    try:
                        events = self.p.parse(raw)
                    except (ValueError, KeyError, TypeError) as exc:
                        for sym in self.symbols:
                            self.agg.state(self.p.exchange, sym).stats.invalid += 1
                        log.debug("%s unparsable message: %s (%s)", self.p.exchange, raw[:200], exc)
                        continue
                    recv = now_ms()
                    if any(e["kind"] in ("trade", "book") for e in events):
                        self.got_data = True
                        self.events += 1
                        if self.first_event_ms is None:
                            self.first_event_ms = recv
                    for ev in events:
                        if ev["kind"] == "error":
                            raise ConnectionError(f"exchange error: {ev['detail']}")
                        if ev["kind"] in ("trade", "book") and ev["symbol"] not in self.symbols:
                            continue
                        if self.agg.on_event(ev, recv) == "resync":
                            raise ResyncNeeded(f"order book gap on {ev['symbol']}")
            finally:
                stop_wait.cancel()
                if ka:
                    ka.cancel()


class CandleSync:
    def __init__(self, db: Database, settings: Settings):
        self.db = db
        self.s = settings
        self.last_sync: dict[tuple, float] = {}
        self.errors: dict[str, str] = {}

    def _bars_for(self, exchange: str, tf: str) -> int:
        if exchange == self.s.primary_exchange and tf == "1m" and self.s.forecast_enabled:
            return self.s.forecast_1m_bars  # minute-horizon forecasts learn on 30 days of 1-minute candles
        if exchange == self.s.primary_exchange:
            return self.s.history_bars if tf in self.s.predict_timeframes else self.s.context_history_bars
        return 500

    def backfill(self, exchange: str, symbol: str, tf: str) -> int:
        have = self.db.scalar(
            "SELECT COUNT(*) FROM candles WHERE exchange=? AND symbol=? AND timeframe=?", (exchange, symbol, tf)) or 0
        want = self._bars_for(exchange, tf)
        if have >= want * 0.95:
            rows = rest.fetch_klines(exchange, symbol, tf, 1000)  # just refresh the tail
        else:
            rows = rest.fetch_history(exchange, symbol, tf, want)
        return upsert_candles(self.db, exchange, symbol, tf, rows)

    def refresh(self, exchange: str, symbol: str, tf: str) -> int:
        return upsert_candles(self.db, exchange, symbol, tf, rest.fetch_klines(exchange, symbol, tf, 5))

    def ticker(self, exchange: str, symbol: str) -> None:
        t = rest.fetch_ticker(exchange, symbol)
        self.db.execute(
            "INSERT INTO market_ticker(exchange,symbol,last,change_24h_pct,volume_24h,quote_volume_24h,high_24h,low_24h,ts_ms) "
            "VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(exchange,symbol) DO UPDATE SET last=excluded.last,"
            "change_24h_pct=excluded.change_24h_pct,volume_24h=excluded.volume_24h,"
            "quote_volume_24h=excluded.quote_volume_24h,high_24h=excluded.high_24h,low_24h=excluded.low_24h,"
            "ts_ms=excluded.ts_ms",
            (exchange, symbol, t["last"], t["change_24h_pct"], t["volume_24h"], t["quote_volume_24h"],
             t["high_24h"], t["low_24h"], t["ts_ms"]))

    async def run_exchange(self, exchange: str, stop: asyncio.Event) -> None:
        """Independent loop per exchange so a REST outage on one does not block others."""
        backfilled: set = set()
        while not stop.is_set():
            for symbol in self.s.symbols:
                for tf in self.s.timeframes:
                    key = (exchange, symbol, tf)
                    try:
                        if key not in backfilled:
                            n = await asyncio.to_thread(self.backfill, exchange, symbol, tf)
                            backfilled.add(key)
                            log.info("backfilled %s %s %s: %d candles", exchange, symbol, tf, n)
                        else:
                            period = self.s.candle_sync_sec if tf == "1m" else max(self.s.candle_sync_sec, 60)
                            if time.monotonic() - self.last_sync.get(key, 0) >= period:
                                await asyncio.to_thread(self.refresh, exchange, symbol, tf)
                        self.last_sync[key] = time.monotonic()
                        self.errors.pop(exchange, None)
                    except Exception as exc:  # noqa: BLE001
                        self.errors[exchange] = f"{type(exc).__name__}: {exc}"[:300]
                        log.warning("candle sync %s %s %s failed: %s", exchange, symbol, tf, exc)
                    if stop.is_set():
                        return
                try:
                    await asyncio.to_thread(self.ticker, exchange, symbol)
                except Exception as exc:  # noqa: BLE001
                    self.errors[exchange] = f"ticker: {exc}"[:300]
            try:
                await asyncio.wait_for(stop.wait(), timeout=min(10.0, self.s.candle_sync_sec))
            except asyncio.TimeoutError:
                pass


async def run_collector(db: Database, settings: Settings, stop: asyncio.Event, connect=None) -> None:
    primary = settings.primary_exchange
    # 1-second bars for the forecaster: the model exchange only (all exchanges while it is still "auto")
    agg = Aggregator(db, settings.store_raw_trades, second_bar_exchanges=None if primary == "auto" else {primary})
    for ex in settings.exchanges:
        for sym in settings.symbols:
            agg.state(ex, sym)
    streams = [ExchangeStream(build_protocol(ex, settings.symbols, settings), settings.symbols, agg, settings, connect)
               for ex in settings.exchanges]
    sync = CandleSync(db, settings)
    tasks = [asyncio.create_task(s.run_forever(stop), name=f"ws-{s.p.exchange}") for s in streams]
    tasks += [asyncio.create_task(sync.run_exchange(ex, stop), name=f"rest-{ex}") for ex in settings.exchanges]

    async def flusher():
        while not stop.is_set():
            try:
                stats = await asyncio.to_thread(agg.flush)
                await asyncio.to_thread(db.heartbeat, "collector", {
                    "flushed": stats, "rest_errors": sync.errors,
                    "streams": {s.p.exchange: {"sessions": s.sessions, "failures": s.failures} for s in streams}})
            except Exception as exc:  # noqa: BLE001
                log.exception("flush failed: %s", exc)
            try:
                await asyncio.wait_for(stop.wait(), timeout=settings.flush_interval_sec)
            except asyncio.TimeoutError:
                pass
        await asyncio.to_thread(agg.flush)

    tasks.append(asyncio.create_task(flusher(), name="flusher"))
    try:
        await stop.wait()
    finally:
        for t in tasks:
            if t.get_name() != "flusher":
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

