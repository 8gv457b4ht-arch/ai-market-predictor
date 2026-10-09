"""Build model-ready datasets from candles stored in the database."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TIMEFRAME_MS, Settings
from ..db import Database
from ..market.candles import load_candles
from .features import add_labels, build_feature_frame, feature_columns

WARMUP_ROWS = 60


def context_timeframes(settings: Settings, base_tf: str) -> list[str]:
    return [tf for tf in settings.timeframes if TIMEFRAME_MS[tf] >= TIMEFRAME_MS[base_tf]]


def load_mtf_candles(db: Database, settings: Settings, symbol: str, base_tf: str,
                     base_limit: int, context_limit: int, exchange: str | None = None,
                     end_ts: int | None = None) -> dict[str, pd.DataFrame]:
    ex = exchange or settings.primary_exchange
    out = {}
    for tf in context_timeframes(settings, base_tf):
        out[tf] = load_candles(db, ex, symbol, tf, base_limit if tf == base_tf else context_limit,
                               closed_only=True, end_ts=end_ts)
    return out


def build_training_frame(db: Database, settings: Settings, symbol: str, base_tf: str) -> tuple[pd.DataFrame, list[str]]:
    candles = load_mtf_candles(db, settings, symbol, base_tf, 1_000_000, 1_000_000)
    frame = build_feature_frame(candles, base_tf)
    if frame.empty:
        return frame, []
    frame = add_labels(frame, settings.horizon_bars, settings.label_atr_mult, settings.round_trip_cost_bps)
    frame = frame.iloc[WARMUP_ROWS:].reset_index(drop=True)
    feats = feature_columns(frame)
    # keep rows where the large majority of features exist (higher-TF history may start later)
    ok = frame[feats].notna().mean(axis=1) >= 0.8
    frame = frame[ok].reset_index(drop=True)
    return frame, feats


def latest_feature_row(db: Database, settings: Settings, symbol: str, base_tf: str,
                       features: list[str]) -> tuple[pd.Series | None, pd.DataFrame]:
    candles = load_mtf_candles(db, settings, symbol, base_tf, 700, 400)
    frame = build_feature_frame(candles, base_tf)
    if frame.empty:
        return None, frame
    for f in features:
        if f not in frame:
            frame[f] = np.nan
    return frame.iloc[-1], frame
