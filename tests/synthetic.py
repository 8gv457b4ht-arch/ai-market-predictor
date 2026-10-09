"""TEST-ONLY synthetic market data. Never used by the application itself.

`random_walk_candles` has no predictable structure: a model that "beats" the
naive baseline on it out-of-sample is leaking future information.
`planted_signal_candles` contains a real, learnable lagged dependency so we
can verify the pipeline is able to learn when there is something to learn.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.config import TIMEFRAME_MS


def _ohlc_from_close(close: np.ndarray, rng, start_ts: int, tf: str) -> pd.DataFrame:
    n = len(close)
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, 0.0015, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    vol = rng.lognormal(3, 0.4, n)
    tbv = vol * np.clip(rng.normal(0.5, 0.08, n), 0.05, 0.95)
    ts = start_ts + np.arange(n) * TIMEFRAME_MS[tf]
    return pd.DataFrame({"open_ts": ts, "open": open_, "high": high, "low": low, "close": close, "volume": vol,
                         "quote_volume": vol * close, "taker_buy_volume": tbv, "trades": 100, "closed": 1})


def random_walk_candles(n: int = 3000, tf: str = "15m", seed: int = 1, start_ts: int = 1_700_000_000_000):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.004, n)
    close = 30000 * np.exp(np.cumsum(r))
    return _ohlc_from_close(close, rng, start_ts, tf)


def planted_signal_candles(n: int = 3000, tf: str = "15m", seed: int = 2, start_ts: int = 1_700_000_000_000):
    """Next returns follow the sign of a slowly-varying hidden trend visible in recent returns."""
    rng = np.random.default_rng(seed)
    trend = np.zeros(n)
    for i in range(1, n):
        trend[i] = 0.97 * trend[i - 1] + rng.normal(0, 0.0012)
    r = trend + rng.normal(0, 0.002, n)
    close = 30000 * np.exp(np.cumsum(r))
    return _ohlc_from_close(close, rng, start_ts, tf)


def resample(base: pd.DataFrame, base_tf: str, tf: str) -> pd.DataFrame:
    k = TIMEFRAME_MS[tf] // TIMEFRAME_MS[base_tf]
    g = (base.open_ts - base.open_ts.iloc[0]) // TIMEFRAME_MS[tf]
    agg = base.groupby(g).agg(open_ts=("open_ts", "first"), open=("open", "first"), high=("high", "max"),
                              low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
                              quote_volume=("quote_volume", "sum"), taker_buy_volume=("taker_buy_volume", "sum"),
                              trades=("trades", "sum"), n=("open", "size"))
    agg = agg[agg.n == k].drop(columns="n")
    agg["closed"] = 1
    return agg.reset_index(drop=True)


def fill_db(db, symbol: str, exchange: str, base: pd.DataFrame, base_tf: str, tfs: list[str]) -> None:
    from backend.app.market.candles import upsert_candles
    for tf in tfs:
        df = base if tf == base_tf else resample(base, base_tf, tf)
        upsert_candles(db, exchange, symbol, tf, df.to_dict(orient="records"))
