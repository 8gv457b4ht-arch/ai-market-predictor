"""Order book, live aggregation (CVD, gaps, duplicates, invalid ticks) and the
reconnecting WebSocket stream, exercised with an in-memory fake transport."""
import asyncio
import json
import time
import tempfile

from backend.app.config import Settings
from backend.app.db import Database
from backend.app.exchanges.orderbook import LocalOrderBook, book_metrics
from backend.app.exchanges.ws import Protocol, parse_binance
from backend.app.market.aggregator import Aggregator
from backend.app.market.collector import ExchangeStream


def _db():
    d = Database(f"sqlite:///{tempfile.mkdtemp()}/t.sqlite3")
    d.init_schema()
    return d


def test_local_orderbook_snapshot_delta_gap_and_crossed():
    b = LocalOrderBook()
    assert b.apply_delta([["1", "1"]], [], update_id=1) == "gap"  # delta before snapshot
    b.apply_snapshot([["100", "2"], ["99", "1"]], [["101", "1"], ["102", "4"]], update_id=10)
    assert b.apply_delta([["100", "0"], ["99.5", "3"]], [["101", "2"]], update_id=11) == "ok"
    bids, asks = b.top(5)
    assert bids[0] == [99.5, 3.0] and asks[0] == [101.0, 2.0]
    assert b.apply_delta([["50", "1"]], [], update_id=11) == "stale"  # out of order
    assert b.apply_delta([["103", "1"]], [], update_id=12) == "gap"   # crossed book -> resync
    assert not b.initialized
    m = book_metrics([[100, 3], [99, 1]], [[101, 1], [102, 1]])
    assert m["imbalance"] == (4 - 2) / 6
    assert round(m["spread_bps"], 3) == round(1 / 100.5 * 1e4, 3)


def _trade(tid, price=100.0, qty=1.0, side="buy", ts=60_000, ex="binance"):
    return {"kind": "trade", "exchange": ex, "symbol": "BTC/USDT", "trade_id": tid, "ts_ms": ts,
            "price": price, "qty": qty, "side": side}


def test_aggregator_cvd_flow_bars_duplicates_gaps_invalid():
    db = _db()
    agg = Aggregator(db)
    assert agg.on_event(_trade(1, qty=2), recv_ms=61_000) == "ok"
    assert agg.on_event(_trade(2, qty=0.5, side="sell"), recv_ms=61_000) == "ok"
    assert agg.on_event(_trade(2, qty=0.5, side="sell"), recv_ms=61_000) == "duplicate"
    assert agg.on_event(_trade(5, qty=1, ts=125_000), recv_ms=126_000) == "gap"  # ids 3,4 missing
    assert agg.on_event(_trade(6, price=-1), recv_ms=126_000) == "invalid"
    assert agg.on_event(_trade(7, ts=10_000_000), recv_ms=126_000) == "invalid_ts"  # future timestamp
    st = agg.state("binance", "BTC/USDT")
    assert st.cvd == 2 - 0.5 + 1
    assert st.stats.duplicates == 1 and st.stats.gaps == 1 and st.stats.invalid == 2
    agg.flush(now=130_000)
    bars = db.query("SELECT * FROM flow_bars ORDER BY open_ts")
    assert [b["open_ts"] for b in bars] == [60_000, 120_000]
    assert bars[0]["buy_volume"] == 2 and bars[0]["sell_volume"] == 0.5 and bars[0]["delta"] == 1.5
    assert bars[1]["cvd"] == 2.5
    assert db.scalar("SELECT COUNT(*) FROM trades_raw") == 3
    s = db.query_one("SELECT * FROM stream_status")
    assert s["gaps"] == 1 and s["duplicates"] == 1
    # CVD continues after a restart (new aggregator reads the last stored bar)
    agg2 = Aggregator(db)
    assert agg2.state("binance", "BTC/USDT").cvd == 2.5


def test_aggregator_book_and_resync():
    db = _db()
    agg = Aggregator(db)
    snap = {"kind": "book", "exchange": "bybit", "symbol": "BTC/USDT", "ts_ms": 1000, "type": "snapshot",
            "bids": [["100", "1"]], "asks": [["101", "1"]], "update_id": 5, "seq": 1}
    assert agg.on_event(snap, recv_ms=1100) == "ok"
    agg.flush(now=1200)
    row = db.query_one("SELECT * FROM book_state WHERE exchange='bybit'")
    assert row["best_bid"] == 100 and row["best_ask"] == 101 and json.loads(row["levels_json"])["bids"]
    crossed = {**snap, "type": "delta", "bids": [["102", "1"]], "asks": [], "update_id": 6}
    assert agg.on_event(crossed, recv_ms=1300) == "resync"


class FakeWS:
    """Async context manager / websocket double driven by a script of messages or exceptions."""

    def __init__(self, script, sent):
        self.script = list(script)
        self.sent = sent

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def send(self, m):
        self.sent.append(m)

    async def recv(self):
        if not self.script:
            await asyncio.sleep(3600)  # silence -> stale detection must kick in
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        await asyncio.sleep(0)
        return item


def _settings(**kw):
    s = Settings()
    s.stale_after_sec = kw.get("stale", 0.2)
    s.reconnect_max_sec = 0.05
    return s


def _trade_msg(tid):
    return json.dumps({"stream": "btcusdt@trade", "data": {"e": "trade", "E": 1, "s": "BTCUSDT", "t": tid,
                                                           "p": "100", "q": "1", "T": int(time.time() * 1000), "m": False}})


def test_stream_reconnects_after_errors_and_stale_data():
    sessions, sent = [], []
    scripts = [
        [_trade_msg(1), ConnectionError("network reset")],  # dies -> reconnect
        [_trade_msg(2)],                                       # goes silent -> stale -> reconnect
        [_trade_msg(3), _trade_msg(4)],
    ]

    def connect(url, **kw):
        sessions.append(url)
        return FakeWS(scripts.pop(0) if scripts else [], sent)

    agg = Aggregator(None)
    proto = Protocol("binance", "wss://fake", ["SUB"], None, 0, parse_binance)
    stream = ExchangeStream(proto, ["BTC/USDT"], agg, _settings(), connect=connect)

    async def run():
        stop = asyncio.Event()
        task = asyncio.create_task(stream.run_forever(stop))
        for _ in range(200):
            await asyncio.sleep(0.02)
            if agg.state("binance", "BTC/USDT").last_trade_id == 4:
                break
        stop.set()
        await asyncio.wait_for(task, 3)
    asyncio.run(run())
    st = agg.state("binance", "BTC/USDT")
    assert st.last_trade_id == 4
    assert len(sessions) >= 3 and sent.count("SUB") >= 3  # resubscribed after every reconnect
    assert st.stats.reconnects >= 2 and st.stats.stale_events >= 1
    assert "network reset" in (st.stats.last_error or "") or "no data" in (st.stats.last_error or "")


def test_one_exchange_failure_does_not_stop_others():
    agg = Aggregator(None)
    good_msgs = [_trade_msg(i) for i in range(1, 6)]

    def bad_connect(url, **kw):
        raise OSError("exchange unreachable")

    def good_connect(url, **kw):
        return FakeWS(list(good_msgs), [])

    s = _settings(stale=5)
    bad = ExchangeStream(Protocol("okx", "wss://bad", [], None, 0, parse_binance), ["BTC/USDT"], agg, s, bad_connect)
    good = ExchangeStream(Protocol("binance", "wss://ok", [], None, 0, parse_binance), ["BTC/USDT"], agg, s, good_connect)

    async def run():
        stop = asyncio.Event()
        tasks = [asyncio.create_task(bad.run_forever(stop)), asyncio.create_task(good.run_forever(stop))]
        await asyncio.sleep(0.3)
        stop.set()
        await asyncio.wait_for(asyncio.gather(*tasks), 3)
    asyncio.run(run())
    assert agg.state("binance", "BTC/USDT").last_trade_id == 5
    assert agg.state("okx", "BTC/USDT").stats.reconnects >= 1
    assert "unreachable" in agg.state("okx", "BTC/USDT").stats.last_error


def test_keepalive_is_sent():
    sent = []

    async def fast_sleep(_):
        await asyncio.sleep(0.01)

    def connect(url, **kw):
        return FakeWS([], sent)

    agg = Aggregator(None)
    proto = Protocol("okx", "wss://x", [], "ping", 20, parse_binance)
    stream = ExchangeStream(proto, ["BTC/USDT"], agg, _settings(stale=0.3), connect=connect, sleep=fast_sleep)

    async def run():
        stop = asyncio.Event()
        t = asyncio.create_task(stream.run_forever(stop))
        await asyncio.sleep(0.15)
        stop.set()
        await asyncio.wait_for(t, 3)
    asyncio.run(run())
    assert sent.count("ping") >= 3



def test_old_or_future_timestamps_are_not_live():
    agg = Aggregator(None)
    assert agg.on_event(_trade(1, ts=1_000), recv_ms=1_000_000) == "invalid_ts"
    st = agg.state("binance", "BTC/USDT")
    assert st.stats.last_msg_ms is None and st.stats.invalid == 1  # nothing counted as live
    assert agg.on_event(_trade(2, ts=999_000), recv_ms=1_000_000) == "ok"
    assert st.stats.last_msg_ms == 1_000_000
