"""Exchange adapters: symbol/interval mapping, REST payload parsing, WebSocket message parsing.

Payload shapes follow the public API documentation of each exchange.
"""
import json

import pytest

from backend.app.exchanges import rest
from backend.app.exchanges.symbols import canonical_symbol, exchange_symbol, interval
from backend.app.exchanges.ws import (binance_protocol, bybit_protocol, okx_protocol, parse_binance,
                                      parse_bybit, parse_okx)


def test_symbol_and_interval_mapping():
    assert exchange_symbol("binance", "BTC/USDT") == "BTCUSDT"
    assert exchange_symbol("bybit", "eth/usdt") == "ETHUSDT"
    assert exchange_symbol("okx", "BTC/USDT") == "BTC-USDT"
    assert canonical_symbol("binance", "ETHUSDT") == "ETH/USDT"
    assert canonical_symbol("okx", "SOL-USDT") == "SOL/USDT"
    # regression: the original project mapped Bybit "1m" to "M" (monthly candles)
    assert interval("bybit", "1m") == "1"
    assert interval("bybit", "1h") == "60" and interval("bybit", "4h") == "240" and interval("bybit", "1d") == "D"
    # regression: OKX needs upper-case hour/day bars
    assert interval("okx", "1h") == "1H" and interval("okx", "4h") == "4H" and interval("okx", "1d") == "1Dutc"
    assert interval("binance", "30m") == "30m"
    with pytest.raises(ValueError):
        interval("binance", "2h")
    with pytest.raises(ValueError):
        exchange_symbol("binance", "BTCUSDT")


def test_binance_klines_parse_closed_flag_and_taker_volume():
    now = 1_700_000_900_000
    rows = [[1_700_000_000_000, "100", "110", "95", "105", "12", 1_700_000_899_999, "1260", 50, "7", "735", "0"],
            [1_700_000_900_000, "105", "106", "104", "105.5", "3", 1_700_001_799_999, "316", 9, "1", "105", "0"]]
    out = rest.parse_binance_klines(rows, 900_000, now)
    assert out[0]["closed"] == 1 and out[1]["closed"] == 0
    assert out[0]["taker_buy_volume"] == 7.0 and out[0]["trades"] == 50


def test_bybit_and_okx_klines_parse(monkeypatch):
    now = 1_700_002_000_000
    by = {"retCode": 0, "result": {"list": [["1700001000000", "2", "3", "1", "2.5", "10", "25"],
                                            ["1700000000000", "1", "2", "0.5", "2", "11", "22"]]}}
    out = rest.parse_bybit_klines(by, 900_000, now)
    assert [c["open_ts"] for c in out] == [1700001000000, 1700000000000]
    assert out[0]["closed"] == 1
    okx = {"code": "0", "data": [["1700001000000", "2", "3", "1", "2.5", "10", "25", "25", "0"],
                                 ["1700000100000", "1", "2", "0.5", "2", "11", "22", "22", "1"]]}
    o = rest.parse_okx_candles(okx, 900_000, now)
    assert o[0]["closed"] == 0 and o[1]["closed"] == 1  # OKX 'confirm' flag is respected
    with pytest.raises(rest.ExchangeError):
        rest.parse_okx_candles({"code": "51000", "msg": "bad"}, 900_000, now)


def test_fetch_klines_validates_and_sorts(monkeypatch):
    payload = {"retCode": 0, "result": {"list": [
        ["1700000900000", "2", "3", "1", "2.5", "10", "25"],
        ["1700000000000", "1", "2", "0.5", "2", "11", "22"],
        ["1699999100000", "1", "0.5", "0.9", "1", "1", "1"],  # high < open -> invalid, dropped
    ]}}
    monkeypatch.setattr(rest, "http_get_json", lambda *a, **k: payload)
    out = rest.fetch_klines("bybit", "BTC/USDT", "15m", 3)
    assert [c["open_ts"] for c in out] == [1700000000000, 1700000900000]


def test_fetch_history_paginates_backwards(monkeypatch):
    calls = []

    def fake(exchange, symbol, tf, limit, end_ms=None):
        calls.append(end_ms)
        top = 1_000_000 if end_ms is None else end_ms + 1
        start = top - 3 * 60_000
        return [{"open_ts": t, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1, "closed": 1}
                for t in range(start, top, 60_000)] if top > 600_000 else []
    monkeypatch.setattr(rest, "fetch_klines", fake)
    monkeypatch.setattr(rest.time, "sleep", lambda s: None)
    rows = rest.fetch_history("binance", "BTC/USDT", "1m", 7, pause=0)
    assert len(rows) == 7
    assert rows == sorted(rows, key=lambda r: r["open_ts"])
    assert calls[0] is None and calls[1] < 1_000_000


def test_tickers_and_orderbooks(monkeypatch):
    t = rest.parse_ticker("bybit", {"retCode": 0, "time": 1, "result": {"list": [{
        "lastPrice": "100", "price24hPcnt": "0.025", "volume24h": "5", "turnover24h": "500",
        "highPrice24h": "101", "lowPrice24h": "95"}]}})
    assert t["change_24h_pct"] == pytest.approx(2.5)
    t = rest.parse_ticker("okx", {"code": "0", "data": [{"last": "110", "open24h": "100", "high24h": "111",
                                                          "low24h": "99", "vol24h": "3", "volCcy24h": "330", "ts": "5"}]})
    assert t["change_24h_pct"] == pytest.approx(10.0)
    monkeypatch.setattr(rest, "http_get_json", lambda *a, **k: {"code": "0", "data": [
        {"bids": [["99", "2", "0", "1"], ["98", "1", "0", "1"]], "asks": [["101", "3", "0", "1"]], "ts": "7"}]})
    ob = rest.fetch_orderbook("okx", "BTC/USDT", 5)
    assert ob["bids"][0] == [99.0, 2.0] and ob["asks"][0] == [101.0, 3.0] and ob["ts_ms"] == 7


def test_binance_ws_parsing():
    tr = parse_binance(json.dumps({"stream": "btcusdt@trade", "data": {
        "e": "trade", "E": 2, "s": "BTCUSDT", "t": 42, "p": "100.5", "q": "0.25", "T": 1, "m": True}}))
    assert tr == [{"kind": "trade", "exchange": "binance", "symbol": "BTC/USDT", "trade_id": 42, "ts_ms": 1,
                   "price": 100.5, "qty": 0.25, "side": "sell"}]  # m=True: buyer is maker -> taker sold
    bk = parse_binance(json.dumps({"stream": "btcusdt@depth20@100ms", "data": {
        "lastUpdateId": 9, "bids": [["100", "1"]], "asks": [["101", "2"]]}}))
    assert bk[0]["type"] == "snapshot" and bk[0]["update_id"] == 9 and bk[0]["symbol"] == "BTC/USDT"
    p = binance_protocol(["BTC/USDT", "ETH/USDT"], "wss://x/stream")
    assert "btcusdt@trade" in p.url and "ethusdt@depth20@100ms" in p.url and p.subscribe_messages == []


def test_bybit_ws_parsing():
    tr = parse_bybit(json.dumps({"topic": "publicTrade.BTCUSDT", "type": "snapshot", "ts": 5, "data": [
        {"T": 4, "s": "BTCUSDT", "S": "Buy", "v": "0.1", "p": "100", "i": "abc"}]}))
    assert tr[0]["side"] == "buy" and tr[0]["trade_id"] == "abc"
    snap = parse_bybit(json.dumps({"topic": "orderbook.50.BTCUSDT", "type": "snapshot", "ts": 6, "cts": 5,
                                   "data": {"s": "BTCUSDT", "b": [["100", "1"]], "a": [["101", "1"]], "u": 10, "seq": 1}}))
    assert snap[0]["type"] == "snapshot" and snap[0]["ts_ms"] == 5
    reset = parse_bybit(json.dumps({"topic": "orderbook.50.BTCUSDT", "type": "delta", "ts": 6,
                                    "data": {"s": "BTCUSDT", "b": [], "a": [], "u": 1, "seq": 2}}))
    assert reset[0]["type"] == "snapshot"  # u == 1 means service restart -> treat as snapshot
    err = parse_bybit(json.dumps({"success": False, "ret_msg": "bad topic", "op": "subscribe"}))
    assert err[0]["kind"] == "error"
    p = bybit_protocol([f"C{i}/USDT" for i in range(6)], "wss://x")
    assert len(p.subscribe_messages) == 2  # max 10 args per request
    assert all(len(json.loads(m)["args"]) <= 10 for m in p.subscribe_messages)


def test_okx_ws_parsing():
    tr = parse_okx(json.dumps({"arg": {"channel": "trades", "instId": "BTC-USDT"}, "data": [
        {"instId": "BTC-USDT", "tradeId": "77", "px": "100", "sz": "2", "side": "buy", "ts": "3"}]}))
    assert tr[0]["trade_id"] == 77 and tr[0]["symbol"] == "BTC/USDT"
    bk = parse_okx(json.dumps({"arg": {"channel": "books5", "instId": "BTC-USDT"}, "data": [
        {"asks": [["101", "1", "0", "2"]], "bids": [["100", "1", "0", "3"]], "ts": "4", "seqId": 12}]}))
    assert bk[0]["bids"] == [["100", "1"]] and bk[0]["update_id"] == 12
    assert parse_okx("pong")[0]["kind"] == "control"
    assert parse_okx(json.dumps({"event": "error", "code": "60012", "msg": "x"}))[0]["kind"] == "error"
    p = okx_protocol(["BTC/USDT"], "wss://ws.okx.com/ws/v5/public")
    assert p.keepalive_message == "ping" and p.keepalive_sec < 30
    assert ":8443" not in p.url


def test_error_classification_and_host_fallback(monkeypatch):
    assert rest.classify_http(451, "Service unavailable from a restricted location") == "region_blocked"
    assert rest.classify_http(403, "<html>CloudFront request blocked</html>") == "region_blocked"
    assert rest.classify_http(429, "") == "rate_limited"
    assert rest.classify_http(500, "oops") == "http_error"
    calls = []

    def fake(url, params=None, timeout=15, retries=3):
        calls.append(url)
        if url.startswith("https://api.binance.com"):
            raise rest.ExchangeError("HTTP 451", "region_blocked", 451, "api.binance.com")
        return [[1, "1", "2", "0.5", "1.5", "3", 2, "0", 1, "1", "0", "0"]]
    monkeypatch.setattr(rest, "http_get_json", fake)
    monkeypatch.setattr(rest, "_working_host", {})
    out = rest.fetch_klines("binance", "BTC/USDT", "1m", 1)
    assert len(out) == 1 and calls[-1].startswith("https://data-api.binance.vision")
    rest.fetch_klines("binance", "BTC/USDT", "1m", 1)
    assert calls[-1].startswith("https://data-api.binance.vision")  # working host remembered

    def all_fail(url, params=None, timeout=15, retries=3):
        raise rest.ExchangeError("HTTP 403", "region_blocked", 403, url)
    monkeypatch.setattr(rest, "http_get_json", all_fail)
    monkeypatch.setattr(rest, "_working_host", {})
    try:
        rest.fetch_ticker("bybit", "BTC/USDT")
        raise AssertionError("expected ExchangeError")
    except rest.ExchangeError as exc:
        assert exc.kind == "region_blocked"
