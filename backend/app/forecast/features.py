"""Inputs for forward-looking forecasts. Every function takes `asof_ms` and uses only data that existed
before that instant (bars whose close time <= asof). The same functions build the training rows, so
training and live forecasts see identically constructed features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TIMEFRAME_MS
from ..db import Database
from ..market.candles import load_candles
from ..ml.features import BASE_FEATURES, compute_indicators

GRID_FEATURES = BASE_FEATURES + ["hour_sin", "hour_cos"]
SECOND_FEATURES = ["r1", "r5", "r15", "r30", "r60", "r300", "rv60", "rv300", "tps60", "vol_z60",
                   "flow30", "flow120", "since_trade", "hour_sin", "hour_cos"]
LOOKBACK_SEC = 600


def _hour(ts_ms: pd.Series | np.ndarray):
    h = (np.asarray(ts_ms, dtype="int64") // 1000 % 86400) / 3600.0
    return np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)


# ------------------------------------------------------------------ candle grid (1m, 15m, 1h)
def grid_frame(candles: pd.DataFrame, grid: str) -> pd.DataFrame:
    """One row per closed bar: features known at the bar close (decision_ts) and the close price."""
    if candles.empty:
        return pd.DataFrame()
    ind = compute_indicators(candles)
    f = pd.DataFrame({"open_ts": ind["open_ts"].astype("int64"), "close": ind["close"].astype(float)})
    f["decision_ts"] = f["open_ts"] + TIMEFRAME_MS[grid]
    for c in BASE_FEATURES:
        f[c] = ind[c].to_numpy()
    f["hour_sin"], f["hour_cos"] = _hour(f["decision_ts"])
    return f


def grid_live_row(db: Database, exchange: str, symbol: str, grid: str, asof_ms: int, bars: int = 400):
    """Features of the newest bar that had closed at asof_ms (never a bar that closes later)."""
    g = TIMEFRAME_MS[grid]
    c = load_candles(db, exchange, symbol, grid, bars, closed_only=True, end_ts=asof_ms - g)
    c = c[c.open_ts + g <= asof_ms]
    f = grid_frame(c, grid)
    if f.empty:
        return None, None
    row = f.iloc[-1]
    return row, int(row["decision_ts"])


def grid_training(db: Database, exchange: str, symbol: str, grid: str, steps: int) -> pd.DataFrame:
    c = load_candles(db, exchange, symbol, grid, None, closed_only=True)
    f = grid_frame(c, grid)
    if f.empty:
        return f
    g = TIMEFRAME_MS[grid]
    fut = f["close"].shift(-steps)
    ok = (f["open_ts"].shift(-steps) - f["open_ts"]) == steps * g  # exactly `steps` bars later in time
    f["fwd_bps"] = np.where(ok, (fut / f["close"] - 1) * 1e4, np.nan)
    return f.iloc[60:].reset_index(drop=True)  # indicator warm-up


# ------------------------------------------------------------------ 1-second grid (live trade stream)
def load_seconds(db: Database, exchange: str, symbol: str, start_sec: int | None = None, end_sec: int | None = None) -> pd.DataFrame:
    sql = "SELECT ts_sec, close, buy_volume, sell_volume, trades FROM price_seconds WHERE exchange=? AND symbol=?"
    p: list = [exchange, symbol]
    if start_sec is not None:
        sql += " AND ts_sec >= ?"
        p.append(start_sec)
    if end_sec is not None:
        sql += " AND ts_sec <= ?"
        p.append(end_sec)
    rows = db.query(sql + " ORDER BY ts_sec", p)
    return pd.DataFrame(rows, columns=["ts_sec", "close", "buy_volume", "sell_volume", "trades"])


def seconds_frame(sec: pd.DataFrame) -> pd.DataFrame:
    """Dense 1-second series with features. A second counts as 'live' when a trade happened within the
    previous 10 s; rows are usable only if the whole look-back window was live."""
    if sec.empty:
        return pd.DataFrame()
    idx = np.arange(int(sec.ts_sec.min()), int(sec.ts_sec.max()) + 1)
    d = sec.set_index("ts_sec").reindex(idx)
    had = d["trades"].notna()
    d["close"] = d["close"].ffill()
    for c in ("buy_volume", "sell_volume", "trades"):
        d[c] = d[c].fillna(0.0)
    vol = d["buy_volume"] + d["sell_volume"]
    last_trade_sec = pd.Series(np.where(had, idx, np.nan), index=idx).ffill()
    since = idx - last_trade_sec.to_numpy()
    lc = np.log(d["close"])
    f = pd.DataFrame({"ts_sec": idx, "close": d["close"].to_numpy()})
    for k in (1, 5, 15, 30, 60, 300):
        f[f"r{k}"] = (lc - lc.shift(k)).to_numpy()
    r1 = lc.diff()
    f["rv60"] = r1.rolling(60).std().to_numpy()
    f["rv300"] = r1.rolling(300).std().to_numpy()
    f["tps60"] = d["trades"].rolling(60).sum().to_numpy() / 60
    v60 = vol.rolling(60).sum()
    f["vol_z60"] = ((v60 - v60.rolling(600, min_periods=120).mean()) / v60.rolling(600, min_periods=120).std()).to_numpy()
    flow = d["buy_volume"] - d["sell_volume"]
    f["flow30"] = (flow.rolling(30).sum() / vol.rolling(30).sum().replace(0, np.nan)).to_numpy()
    f["flow120"] = (flow.rolling(120).sum() / vol.rolling(120).sum().replace(0, np.nan)).to_numpy()
    f["since_trade"] = since
    f["hour_sin"], f["hour_cos"] = _hour(idx * 1000)
    live = pd.Series(since <= 10, index=idx)
    f["live_window"] = live.rolling(300).min().fillna(0).astype(bool).to_numpy()
    return f.replace([np.inf, -np.inf], np.nan)


def seconds_live_row(db: Database, exchange: str, symbol: str, asof_ms: int):
    last_complete = asof_ms // 1000 - 1
    sec = load_seconds(db, exchange, symbol, last_complete - LOOKBACK_SEC - 5, last_complete)
    f = seconds_frame(sec)
    if f.empty or int(f.ts_sec.iloc[-1]) < last_complete - 60:
        return None, None
    row = f.iloc[-1]
    if not bool(row["live_window"]):
        return None, None
    return row, (int(row["ts_sec"]) - int(row["since_trade"])) * 1000 + 999  # newest trade second used


def seconds_training(db: Database, exchange: str, symbol: str, horizon_sec: int, every: int = 5) -> pd.DataFrame:
    sec = load_seconds(db, exchange, symbol)
    f = seconds_frame(sec)
    if f.empty:
        return f
    fut = f["close"].shift(-horizon_sec)
    live_ahead = pd.Series(f["since_trade"].to_numpy() <= 10).shift(-horizon_sec).fillna(False).to_numpy()
    f["fwd_bps"] = np.where(f["live_window"] & live_ahead, (fut / f["close"] - 1) * 1e4, np.nan)
    f["decision_ts"] = (f["ts_sec"] + 1) * 1000
    return f[(f.ts_sec % every == 0) & f["live_window"]].reset_index(drop=True)
