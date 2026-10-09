"""WebSocket protocol definitions: URLs, subscriptions, keep-alives and message
parsers for Binance, Bybit and OKX public spot streams.

Parsers are pure functions (raw text -> list of normalized events) so they are
unit-testable without a network connection.

Normalized events:
  {"kind": "trade", "exchange", "symbol", "trade_id", "ts_ms", "price", "qty", "side"}
  {"kind": "book",  "exchange", "symbol", "ts_ms", "type": "snapshot"|"delta",
                    "bids", "asks", "update_id", "seq"}
  {"kind": "control", "exchange", "detail"}
  {"kind": "error", "exchange", "detail"}
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from .symbols import canonical_symbol, exchange_symbol


@dataclass(frozen=True)
class Protocol:
    exchange: str
    url: str
    subscribe_messages: list[str]
    keepalive_message: str | None
    keepalive_sec: float
    parse: Callable[[str], list[dict]]
    fallback_urls: tuple = ()  # alternative endpoints, tried when the main one gives no data

    def urls(self) -> list[str]:
        return [self.url, *self.fallback_urls]


def _f(x) -> float:
    return float(x)


# ---------------------------------------------------------------- Binance
def parse_binance(raw: str) -> list[dict]:
    msg = json.loads(raw)
    stream = msg.get("stream", "")
    data = msg.get("data", msg)
    if "result" in msg and "id" in msg:
        return [{"kind": "control", "exchange": "binance", "detail": msg}]
    if isinstance(data, dict) and data.get("e") == "trade":
        return [{
            "kind": "trade", "exchange": "binance", "symbol": canonical_symbol("binance", data["s"]),
            "trade_id": int(data["t"]), "ts_ms": int(data["T"]), "price": _f(data["p"]),
            "qty": _f(data["q"]), "side": "sell" if data.get("m") else "buy",
        }]
    if "@depth" in stream and isinstance(data, dict) and "bids" in data:
        sym = stream.split("@", 1)[0].upper()
        return [{
            "kind": "book", "exchange": "binance", "symbol": canonical_symbol("binance", sym),
            "ts_ms": int(data["E"]) if "E" in data else None, "type": "snapshot",
            "bids": data["bids"], "asks": data["asks"], "update_id": int(data["lastUpdateId"]), "seq": None,
        }]
    if isinstance(data, dict) and "code" in data and "msg" in data:
        return [{"kind": "error", "exchange": "binance", "detail": data}]
    return []


def binance_protocol(symbols: list[str], base_url: str) -> Protocol:
    streams = []
    for s in symbols:
        x = exchange_symbol("binance", s).lower()
        streams += [f"{x}@trade", f"{x}@depth20@100ms"]
    # Combined stream URL: no subscribe message needed; Binance sends pings, the client auto-pongs.
    q = f"?streams={'/'.join(streams)}"
    alt = tuple(u + q for u in ("wss://data-stream.binance.vision/stream",) if u != base_url)
    return Protocol("binance", base_url + q, [], None, 0, parse_binance, alt)


# ------------------------------------------------------------------ Bybit
def parse_bybit(raw: str) -> list[dict]:
    msg = json.loads(raw)
    topic = msg.get("topic", "")
    if not topic:
        if msg.get("op") in ("subscribe", "ping", "pong") or "success" in msg:
            if msg.get("success") is False:
                return [{"kind": "error", "exchange": "bybit", "detail": msg}]
            return [{"kind": "control", "exchange": "bybit", "detail": msg}]
        return []
    if topic.startswith("publicTrade."):
        out = []
        for t in msg.get("data", []):
            out.append({
                "kind": "trade", "exchange": "bybit", "symbol": canonical_symbol("bybit", t["s"]),
                "trade_id": str(t.get("i")), "ts_ms": int(t["T"]), "price": _f(t["p"]),
                "qty": _f(t["v"]), "side": "buy" if t["S"] == "Buy" else "sell",
            })
        return out
    if topic.startswith("orderbook."):
        d = msg["data"]
        kind = msg.get("type", "delta")
        u = int(d.get("u", 0))
        return [{
            "kind": "book", "exchange": "bybit", "symbol": canonical_symbol("bybit", d["s"]),
            "ts_ms": int(msg.get("cts") or msg.get("ts")),
            # u == 1 means the server restarted and this is a fresh snapshot (Bybit docs).
            "type": "snapshot" if kind == "snapshot" or u == 1 else "delta",
            "bids": d.get("b", []), "asks": d.get("a", []), "update_id": u, "seq": d.get("seq"),
        }]
    return []


def bybit_protocol(symbols: list[str], url: str) -> Protocol:
    args = []
    for s in symbols:
        x = exchange_symbol("bybit", s)
        args += [f"publicTrade.{x}", f"orderbook.50.{x}"]
    # Bybit spot allows at most 10 args per subscribe request.
    msgs = [json.dumps({"op": "subscribe", "args": args[i:i + 10]}) for i in range(0, len(args), 10)]
    return Protocol("bybit", url, msgs, json.dumps({"op": "ping"}), 20, parse_bybit)


# -------------------------------------------------------------------- OKX
def parse_okx(raw: str) -> list[dict]:
    if raw == "pong":
        return [{"kind": "control", "exchange": "okx", "detail": "pong"}]
    msg = json.loads(raw)
    if "event" in msg:
        kind = "error" if msg["event"] == "error" else "control"
        return [{"kind": kind, "exchange": "okx", "detail": msg}]
    arg = msg.get("arg", {})
    ch = arg.get("channel")
    out = []
    for d in msg.get("data", []):
        if ch == "trades":
            out.append({
                "kind": "trade", "exchange": "okx", "symbol": canonical_symbol("okx", d["instId"]),
                "trade_id": int(d["tradeId"]), "ts_ms": int(d["ts"]), "price": _f(d["px"]),
                "qty": _f(d["sz"]), "side": d["side"],
            })
        elif ch in ("books5", "books"):
            out.append({
                "kind": "book", "exchange": "okx", "symbol": canonical_symbol("okx", arg.get("instId") or d.get("instId")),
                "ts_ms": int(d["ts"]),
                "type": "snapshot" if ch == "books5" or msg.get("action") == "snapshot" else "delta",
                "bids": [x[:2] for x in d.get("bids", [])], "asks": [x[:2] for x in d.get("asks", [])],
                "update_id": int(d["seqId"]) if d.get("seqId") is not None else None, "seq": d.get("prevSeqId"),
            })
    return out


def okx_protocol(symbols: list[str], url: str) -> Protocol:
    args = []
    for s in symbols:
        x = exchange_symbol("okx", s)
        args += [{"channel": "trades", "instId": x}, {"channel": "books5", "instId": x}]
    # OKX closes idle connections after 30s: send text "ping" every 20s.
    alt = tuple(u for u in ("wss://ws.okx.com:8443/ws/v5/public", "wss://wsaws.okx.com:8443/ws/v5/public") if u != url)
    return Protocol("okx", url, [json.dumps({"op": "subscribe", "args": args})], "ping", 20, parse_okx, alt)


def build_protocol(exchange: str, symbols: list[str], settings) -> Protocol:
    if exchange == "binance":
        return binance_protocol(symbols, settings.binance_ws_url)
    if exchange == "bybit":
        return bybit_protocol(symbols, settings.bybit_ws_url)
    if exchange == "okx":
        return okx_protocol(symbols, settings.okx_ws_url)
    raise ValueError(exchange)
