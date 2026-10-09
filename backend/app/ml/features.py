"""Causal feature engineering (no look-ahead).

Every indicator at row t uses only candles with open_ts <= t. Higher-timeframe
features are joined on the time the higher bar *closes*, so a 4h bar is not
visible to a 15m row until that 4h bar is complete.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TIMEFRAME_MS

FEATURE_VERSION = "fv2-mtf-1"
CLASSES = ["DOWN", "FLAT", "UP"]  # label encoding 0, 1, 2

BASE_FEATURES = [
    "ret_1", "ret_3", "ret_6", "ret_12", "roc_10",
    "ema_gap_9_21", "ema_gap_21_55", "close_vs_ema200", "ema21_slope",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "atr_pct", "adx_14", "di_diff",
    "bb_width", "bb_pos", "rv_20", "rv_ratio",
    "vol_z", "trend_strength", "range_pct", "body_pct",
    "buy_ratio", "flow_imbalance_20",
]
CONTEXT_FEATURES = [
    "ret_1", "ret_3", "rsi_14", "macd_hist", "adx_14", "di_diff", "atr_pct",
    "bb_pos", "trend_strength", "ema_gap_9_21", "vol_z", "flow_imbalance_20",
]
REGIMES = ["trending", "ranging", "high_volatility", "low_liquidity", "abnormal"]


def _wilder(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def compute_indicators(candles: pd.DataFrame) -> pd.DataFrame:
    """Indicators for one timeframe. Input: ascending candles with OHLCV columns."""
    x = candles.copy().reset_index(drop=True)
    c, h, l, o, v = x["close"], x["high"], x["low"], x["open"], x["volume"]
    logc = np.log(c)
    x["ret_1"] = logc.diff(1)
    x["ret_3"] = logc.diff(3)
    x["ret_6"] = logc.diff(6)
    x["ret_12"] = logc.diff(12)
    x["roc_10"] = c.pct_change(10)

    ema = {n: c.ewm(span=n, adjust=False, min_periods=n).mean() for n in (9, 12, 21, 26, 55, 200)}
    x["ema_9"], x["ema_21"], x["ema_55"], x["ema_200"] = ema[9], ema[21], ema[55], ema[200]
    x["ema_gap_9_21"] = (ema[9] - ema[21]) / c
    x["ema_gap_21_55"] = (ema[21] - ema[55]) / c
    x["close_vs_ema200"] = c / ema[200] - 1
    x["ema21_slope"] = ema[21].pct_change(5)

    delta = c.diff()
    gain = _wilder(delta.clip(lower=0), 14)
    loss = _wilder(-delta.clip(upper=0), 14)
    rs = gain / loss.replace(0, np.nan)
    x["rsi_14"] = np.where(loss == 0, 100.0, 100 - 100 / (1 + rs))
    x.loc[gain.isna(), "rsi_14"] = np.nan

    macd = ema[12] - ema[26]
    sig = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    x["macd"] = macd / c
    x["macd_signal"] = sig / c
    x["macd_hist"] = (macd - sig) / c

    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = _wilder(tr, 14)
    x["atr_14"] = atr
    x["atr_pct"] = atr / c

    up_move, down_move = h.diff(), -l.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_di = 100 * _wilder(pd.Series(plus_dm, index=x.index), 14) / atr
    minus_di = 100 * _wilder(pd.Series(minus_dm, index=x.index), 14) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    x["di_plus"], x["di_minus"] = plus_di, minus_di
    x["di_diff"] = (plus_di - minus_di) / 100
    x["adx_14"] = _wilder(dx, 14)

    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    upper, lower = mid + 2 * sd, mid - 2 * sd
    x["bb_mid"], x["bb_upper"], x["bb_lower"] = mid, upper, lower
    x["bb_width"] = (upper - lower) / mid
    x["bb_pos"] = (c - lower) / (upper - lower).replace(0, np.nan)

    x["rv_20"] = x["ret_1"].rolling(20).std()
    rv100 = x["ret_1"].rolling(100, min_periods=50).std()
    x["rv_100"] = rv100
    x["rv_ratio"] = x["rv_20"] / rv100

    logv = np.log1p(v)
    x["vol_z"] = (logv - logv.rolling(50, min_periods=20).mean()) / logv.rolling(50, min_periods=20).std()
    x["trend_strength"] = (ema[9] - ema[55]) / atr
    x["range_pct"] = (h - l) / c
    x["body_pct"] = (c - o) / c

    if "taker_buy_volume" in x and x["taker_buy_volume"].notna().any():
        tbv = x["taker_buy_volume"]
        x["buy_ratio"] = tbv / v.replace(0, np.nan)
        flow = 2 * tbv - v  # buy volume - sell volume (taker side)
        x["flow_delta"] = flow
        x["cvd"] = flow.cumsum()
        x["flow_imbalance_20"] = flow.rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    else:
        x["buy_ratio"] = np.nan
        x["flow_delta"] = np.nan
        x["cvd"] = np.nan
        x["flow_imbalance_20"] = np.nan

    # regime inputs (rolling percentiles: rank of the current value within the trailing window)
    x["atr_pct_rank"] = x["atr_pct"].rolling(200, min_periods=50).rank(pct=True)
    x["ret_z"] = x["ret_1"] / rv100
    return x.replace([np.inf, -np.inf], np.nan)


def classify_regime_row(adx, atr_rank, vol_z, ret_z, spread_bps=None, max_spread_bps=None, quality_ok=True) -> str:
    """Priority: abnormal > high_volatility > low_liquidity > trending > ranging."""
    def nan(v):
        return v is None or (isinstance(v, float) and np.isnan(v))
    if not quality_ok or (not nan(ret_z) and abs(ret_z) > 6):
        return "abnormal"
    if not nan(atr_rank) and atr_rank >= 0.9:
        return "high_volatility"
    if (not nan(vol_z) and vol_z <= -1.5) or (
            spread_bps is not None and max_spread_bps is not None and spread_bps > max_spread_bps):
        return "low_liquidity"
    if not nan(adx) and adx >= 25:
        return "trending"
    return "ranging"


def regime_series(ind: pd.DataFrame) -> pd.Series:
    return pd.Series([
        classify_regime_row(a, r, vz, rz)
        for a, r, vz, rz in zip(ind["adx_14"], ind["atr_pct_rank"], ind["vol_z"], ind["ret_z"])
    ], index=ind.index)


def build_feature_frame(candles_by_tf: dict[str, pd.DataFrame], base_tf: str) -> pd.DataFrame:
    """Base-TF rows with base features + features of every higher timeframe available."""
    base = candles_by_tf[base_tf]
    if base.empty:
        return pd.DataFrame()
    bms = TIMEFRAME_MS[base_tf]
    ind = compute_indicators(base)
    frame = pd.DataFrame({
        "open_ts": ind["open_ts"].astype("int64"),
        "decision_ts": ind["open_ts"].astype("int64") + bms,  # the bar is closed at this instant
        "open": ind["open"], "close": ind["close"], "high": ind["high"], "low": ind["low"],
        "volume": ind["volume"], "atr_pct": ind["atr_pct"],
    })
    for f in BASE_FEATURES:
        frame[f"{base_tf}_{f}"] = ind[f].to_numpy()
    reg = regime_series(ind)
    frame["regime"] = reg.to_numpy()
    for r in REGIMES:
        frame[f"regime_{r}"] = (reg == r).astype(float).to_numpy()

    for tf, df in candles_by_tf.items():
        if tf == base_tf or TIMEFRAME_MS[tf] <= bms or df is None or df.empty:
            continue
        hi = compute_indicators(df)
        right = pd.DataFrame({"avail_ts": hi["open_ts"].astype("int64") + TIMEFRAME_MS[tf]})
        for f in CONTEXT_FEATURES:
            right[f"{tf}_{f}"] = hi[f].to_numpy()
        right = right.sort_values("avail_ts")
        frame = pd.merge_asof(frame.sort_values("decision_ts"), right, left_on="decision_ts",
                              right_on="avail_ts", direction="backward", allow_exact_matches=True)
        frame = frame.drop(columns=["avail_ts"])
    return frame.reset_index(drop=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    skip = {"open_ts", "decision_ts", "open", "close", "high", "low", "volume", "atr_pct", "regime",
            "label", "fwd_return", "label_threshold"}
    cols = [c for c in frame.columns if c not in skip]
    # Drop columns that carry no information at all (e.g. taker flow on exchanges that do not provide it).
    return [c for c in cols if frame[c].notna().any()]


def label_threshold(atr_pct: pd.Series | float, horizon: int, atr_mult: float, cost_bps: float):
    """Minimum move that counts as UP/DOWN: max(round-trip cost, atr_mult * ATR% * sqrt(h))."""
    return np.maximum(cost_bps / 1e4, atr_mult * np.asarray(atr_pct, dtype=float) * np.sqrt(horizon))


def add_labels(frame: pd.DataFrame, horizon: int, atr_mult: float, cost_bps: float) -> pd.DataFrame:
    """fwd_return(t) = close[t+h]/close[t]-1. Rows whose future is unknown get label NaN."""
    f = frame.copy()
    f["fwd_return"] = f["close"].shift(-horizon) / f["close"] - 1
    thr = label_threshold(f["atr_pct"].fillna(f["atr_pct"].median()), horizon, atr_mult, cost_bps)
    f["label_threshold"] = thr
    lab = np.where(f["fwd_return"] > thr, 2, np.where(f["fwd_return"] < -thr, 0, 1)).astype(float)
    lab[f["fwd_return"].isna().to_numpy()] = np.nan
    f["label"] = lab
    return f


def direction_of(ret: float, threshold: float) -> str:
    if ret > threshold:
        return "UP"
    if ret < -threshold:
        return "DOWN"
    return "FLAT"
