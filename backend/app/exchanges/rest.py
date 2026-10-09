"""Public (read-only, key-less) REST adapters for Binance, Bybit and OKX.

Only market-data endpoints are used. There is no order, account or withdrawal
code anywhere in this project.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import socket
import urllib.parse
import urllib.request
from urllib.parse import urlencode

from ..config import TIMEFRAME_MS, get_settings
from .symbols import exchange_symbol, interval

log = logging.getLogger(__name__)
USER_AGENT = "AI-Market-Predictor/2.0 (read-only research)"


class ExchangeError(RuntimeError):
    """Failure talking to a public exchange endpoint. `kind` is a stable category used by the
    diagnostics: region_blocked, rate_limited, timeout, network, http_error, bad_response."""

    def __init__(self, message: str, kind: str = "http_error", status: int | None = None, host: str | None = None):
        super().__init__(message)
        self.kind = kind
        self.status = status
        self.host = host


def classify_http(code: int, body: str) -> str:
    b = body.lower()
    if code == 451 or "restricted location" in b or "unavailable for legal reasons" in b:
        return "region_blocked"
    if code == 403 and ("cloudfront" in b or "country" in b or "region" in b or "forbidden" in b):
        return "region_blocked"
    if code in (418, 429):
        return "rate_limited"
    return "http_error"


def http_get_json(url: str, params: dict | None = None, timeout: float = 15, retries: int = 3):
    q = urlencode({k: v for k, v in (params or {}).items() if v is not None})
    full = f"{url}?{q}" if q else url
    host = urllib.parse.urlparse(url).netloc
    delay = 1.0
    last: ExchangeError | None = None
    for _attempt in range(retries):
        try:
            req = urllib.request.Request(full, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
            try:
                return json.loads(raw.decode("utf-8"))
            except ValueError as exc:
                raise ExchangeError(f"non-JSON response from {host}", "bad_response", None, host) from exc
        except urllib.error.HTTPError as exc:
            body = exc.read()[:300].decode("utf-8", "replace") if hasattr(exc, "read") else ""
            kind = classify_http(exc.code, body)
            last = ExchangeError(f"HTTP {exc.code} from {host}: {body[:160]}", kind, exc.code, host)
            if kind == "region_blocked" or (400 <= exc.code < 500 and kind != "rate_limited"):
                raise last from exc  # retrying cannot help
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            time.sleep(min(30.0, max(delay, float(retry_after) if retry_after and retry_after.isdigit() else delay)))
            delay *= 2
        except (TimeoutError, socket.timeout) as exc:
            last = ExchangeError(f"timeout after {timeout:.0f}s from {host}", "timeout", None, host)
            time.sleep(delay)
            delay *= 2
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            kind = "timeout" if isinstance(reason, (TimeoutError, socket.timeout)) else "network"
            last = ExchangeError(f"{kind} error for {host}: {reason}", kind, None, host)
            time.sleep(delay)
            delay *= 2
    raise last or ExchangeError(f"GET {url} failed", "network", None, host)


# Public market-data hosts, tried in order. The first host that answers is remembered.
REST_HOSTS = {
    "binance": ["https://api.binance.com", "https://data-api.binance.vision"],
    "bybit": ["https://api.bybit.com", "https://api.bytick.com"],
    "okx": ["https://www.okx.com", "https://aws.okx.com"],
}
_working_host: dict[str, str] = {}


def _hosts(exchange: str) -> list[str]:
    s = get_settings()
    configured = {"binance": s.binance_rest_url, "bybit": s.bybit_rest_url, "okx": s.okx_rest_url}[exchange]
    hosts = [configured] + [h for h in REST_HOSTS[exchange] if h != configured]
    pref = _working_host.get(exchange)
    return ([pref] + [h for h in hosts if h != pref]) if pref else hosts


def get_with_fallback(exchange: str, path: str, params: dict, timeout: float = 15):
    """GET path on the first reachable host of `exchange`; raises the most informative error."""
    errors: list[ExchangeError] = []
    for host in _hosts(exchange):
        try:
            out = http_get_json(host + path, params, timeout=timeout, retries=2)
            _working_host[exchange] = host
            return out
        except ExchangeError as exc:
            errors.append(exc)
    priority = ["region_blocked", "rate_limited", "http_error", "bad_response", "timeout", "network"]
    errors.sort(key=lambda e: priority.index(e.kind) if e.kind in priority else 99)
    first = errors[0]
    raise ExchangeError("; ".join(str(e) for e in errors), first.kind, first.status, first.host)


def _valid_candle(c: dict) -> bool:
    o, h, l, cl, v = c["open"], c["high"], c["low"], c["close"], c["volume"]
    return (
        min(o, h, l, cl) > 0
        and h >= max(o, cl) - 1e-12
        and l <= min(o, cl) + 1e-12
        and v >= 0
    )


# --------------------------------------------------------------------- klines
def parse_binance_klines(rows: list, tf_ms: int, now: int) -> list[dict]:
    out = []
    for r in rows:
        out.append({
            "open_ts": int(r[0]), "open": float(r[1]), "high": float(r[2]), "low": float(r[3]),
            "close": float(r[4]), "volume": float(r[5]), "quote_volume": float(r[7]),
            "trades": int(r[8]), "taker_buy_volume": float(r[9]),
            "closed": int(int(r[6]) < now),
        })
    return out


def parse_bybit_klines(payload: dict, tf_ms: int, now: int) -> list[dict]:
    if payload.get("retCode") != 0:
        raise ExchangeError(f"bybit kline error: {payload.get('retMsg')}")
    out = []
    for r in payload["result"]["list"]:
        ts = int(r[0])
        out.append({
            "open_ts": ts, "open": float(r[1]), "high": float(r[2]), "low": float(r[3]),
            "close": float(r[4]), "volume": float(r[5]), "quote_volume": float(r[6]),
            "trades": None, "taker_buy_volume": None,
            "closed": int(ts + tf_ms <= now),
        })
    return out


def parse_okx_candles(payload: dict, tf_ms: int, now: int) -> list[dict]:
    if str(payload.get("code")) != "0":
        raise ExchangeError(f"okx candle error: {payload.get('msg')}")
    out = []
    for r in payload["data"]:
        ts = int(r[0])
        confirm = r[8] if len(r) > 8 else ("1" if ts + tf_ms <= now else "0")
        out.append({
            "open_ts": ts, "open": float(r[1]), "high": float(r[2]), "low": float(r[3]),
            "close": float(r[4]), "volume": float(r[5]),
            "quote_volume": float(r[7]) if len(r) > 7 and r[7] not in ("", None) else None,
            "trades": None, "taker_buy_volume": None,
            "closed": int(str(confirm) == "1"),
        })
    return out


def fetch_klines(exchange: str, symbol: str, timeframe: str, limit: int = 1000, end_ms: int | None = None) -> list[dict]:
    """One page of candles ending at end_ms (inclusive), sorted ascending, validated."""
    tf_ms = TIMEFRAME_MS[timeframe]
    sym = exchange_symbol(exchange, symbol)
    now = int(time.time() * 1000)
    if exchange == "binance":
        rows = get_with_fallback("binance", "/api/v3/klines", {
            "symbol": sym, "interval": interval(exchange, timeframe), "limit": min(limit, 1000), "endTime": end_ms})
        out = parse_binance_klines(rows, tf_ms, now)
    elif exchange == "bybit":
        payload = get_with_fallback("bybit", "/v5/market/kline", {
            "category": "spot", "symbol": sym, "interval": interval(exchange, timeframe),
            "limit": min(limit, 1000), "end": end_ms})
        out = parse_bybit_klines(payload, tf_ms, now)
    elif exchange == "okx":
        # /candles serves recent data (max 300); /history-candles serves older data (max 100).
        params = {"instId": sym, "bar": interval(exchange, timeframe),
                  "after": (end_ms + 1) if end_ms is not None else None}
        payload = get_with_fallback("okx", "/api/v5/market/candles", {**params, "limit": min(limit, 300)})
        out = parse_okx_candles(payload, tf_ms, now)
        if not out and end_ms is not None:
            payload = get_with_fallback("okx", "/api/v5/market/history-candles", {**params, "limit": min(limit, 100)})
            out = parse_okx_candles(payload, tf_ms, now)
    else:
        raise ValueError(exchange)
    clean = [c for c in out if _valid_candle(c)]
    if len(clean) != len(out):
        log.warning("%s %s %s: dropped %d invalid candles", exchange, symbol, timeframe, len(out) - len(clean))
    clean.sort(key=lambda c: c["open_ts"])
    return clean


def fetch_history(exchange: str, symbol: str, timeframe: str, bars: int, pause: float = 0.2) -> list[dict]:
    """Paginate backwards until `bars` candles are collected (or history runs out)."""
    collected: dict[int, dict] = {}
    end_ms: int | None = None
    stalls = 0
    while len(collected) < bars:
        page = fetch_klines(exchange, symbol, timeframe, 1000, end_ms)
        if not page:
            break
        before = len(collected)
        for c in page:
            collected[c["open_ts"]] = c
        oldest = page[0]["open_ts"]
        if len(collected) == before:
            stalls += 1
            if stalls >= 2:
                break
        end_ms = oldest - 1
        time.sleep(pause)
    rows = sorted(collected.values(), key=lambda c: c["open_ts"])
    return rows[-bars:]


# ------------------------------------------------------------------ orderbook
def _book(bids: list, asks: list, ts_ms: int | None) -> dict:
    b = [[float(x[0]), float(x[1])] for x in bids if float(x[1]) > 0]
    a = [[float(x[0]), float(x[1])] for x in asks if float(x[1]) > 0]
    b.sort(key=lambda x: -x[0])
    a.sort(key=lambda x: x[0])
    return {"bids": b, "asks": a, "ts_ms": int(ts_ms) if ts_ms else int(time.time() * 1000)}


def fetch_orderbook(exchange: str, symbol: str, depth: int = 50) -> dict:
    sym = exchange_symbol(exchange, symbol)
    if exchange == "binance":
        d = get_with_fallback("binance", "/api/v3/depth", {"symbol": sym, "limit": min(depth, 100)})
        return _book(d.get("bids", []), d.get("asks", []), None)
    if exchange == "bybit":
        d = get_with_fallback("bybit", "/v5/market/orderbook", {"category": "spot", "symbol": sym, "limit": min(depth, 200)})
        if d.get("retCode") != 0:
            raise ExchangeError(d.get("retMsg"))
        r = d["result"]
        return _book(r.get("b", []), r.get("a", []), r.get("ts"))
    if exchange == "okx":
        d = get_with_fallback("okx", "/api/v5/market/books", {"instId": sym, "sz": min(depth, 400)})
        if str(d.get("code")) != "0":
            raise ExchangeError(d.get("msg"))
        r = d["data"][0]
        return _book(r.get("bids", []), r.get("asks", []), r.get("ts"))
    raise ValueError(exchange)


# --------------------------------------------------------------------- ticker
def parse_ticker(exchange: str, payload) -> dict:
    if exchange == "binance":
        d = payload
        return {"last": float(d["lastPrice"]), "change_24h_pct": float(d["priceChangePercent"]),
                "volume_24h": float(d["volume"]), "quote_volume_24h": float(d["quoteVolume"]),
                "high_24h": float(d["highPrice"]), "low_24h": float(d["lowPrice"]),
                "ts_ms": int(d.get("closeTime") or time.time() * 1000)}
    if exchange == "bybit":
        if payload.get("retCode") != 0:
            raise ExchangeError(payload.get("retMsg"))
        d = payload["result"]["list"][0]
        return {"last": float(d["lastPrice"]), "change_24h_pct": float(d["price24hPcnt"]) * 100,
                "volume_24h": float(d["volume24h"]), "quote_volume_24h": float(d["turnover24h"]),
                "high_24h": float(d["highPrice24h"]), "low_24h": float(d["lowPrice24h"]),
                "ts_ms": int(payload.get("time") or time.time() * 1000)}
    if exchange == "okx":
        if str(payload.get("code")) != "0":
            raise ExchangeError(payload.get("msg"))
        d = payload["data"][0]
        last, open24 = float(d["last"]), float(d["open24h"])
        return {"last": last, "change_24h_pct": (last / open24 - 1) * 100 if open24 else None,
                "volume_24h": float(d["vol24h"]), "quote_volume_24h": float(d["volCcy24h"]),
                "high_24h": float(d["high24h"]), "low_24h": float(d["low24h"]), "ts_ms": int(d["ts"])}
    raise ValueError(exchange)


def fetch_ticker(exchange: str, symbol: str) -> dict:
    sym = exchange_symbol(exchange, symbol)
    if exchange == "binance":
        return parse_ticker(exchange, get_with_fallback("binance", "/api/v3/ticker/24hr", {"symbol": sym}))
    if exchange == "bybit":
        return parse_ticker(exchange, get_with_fallback("bybit", "/v5/market/tickers", {"category": "spot", "symbol": sym}))
    if exchange == "okx":
        return parse_ticker(exchange, get_with_fallback("okx", "/api/v5/market/ticker", {"instId": sym}))
    raise ValueError(exchange)
