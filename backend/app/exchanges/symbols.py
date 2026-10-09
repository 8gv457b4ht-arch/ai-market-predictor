"""Symbol / interval normalization for the supported public exchange APIs."""
from __future__ import annotations

EXCHANGES = ("binance", "bybit", "okx")

BYBIT_INTERVALS = {"1m": "1", "5m": "5", "15m": "15", "30m": "30", "1h": "60", "4h": "240", "1d": "D"}
# OKX: hour/day bars are upper-case; "1Dutc" aligns daily bars to 00:00 UTC like Binance/Bybit
# (plain "1D" is aligned to Hong Kong time).
OKX_BARS = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1H", "4h": "4H", "1d": "1Dutc"}
BINANCE_INTERVALS = {k: k for k in ("1m", "5m", "15m", "30m", "1h", "4h", "1d")}


def exchange_symbol(exchange: str, symbol: str) -> str:
    """'BTC/USDT' -> 'BTCUSDT' (Binance/Bybit) or 'BTC-USDT' (OKX)."""
    s = symbol.upper().strip()
    if "/" not in s:
        raise ValueError(f"Symbol must look like BASE/QUOTE, got {symbol!r}")
    return s.replace("/", "-") if exchange == "okx" else s.replace("/", "")


def interval(exchange: str, timeframe: str) -> str:
    table = {"binance": BINANCE_INTERVALS, "bybit": BYBIT_INTERVALS, "okx": OKX_BARS}[exchange]
    try:
        return table[timeframe]
    except KeyError:
        raise ValueError(f"Timeframe {timeframe} not supported for {exchange}") from None


def canonical_symbol(exchange: str, raw: str) -> str:
    """Exchange-native symbol back to canonical BASE/QUOTE (common quotes only)."""
    raw = raw.upper()
    if exchange == "okx" or "-" in raw:
        return raw.replace("-", "/")
    for quote in ("USDT", "USDC", "FDUSD", "BUSD", "USD", "EUR", "BTC", "ETH"):
        if raw.endswith(quote) and len(raw) > len(quote):
            return f"{raw[:-len(quote)]}/{quote}"
    return raw
