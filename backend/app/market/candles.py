"""Candle persistence and loading helpers."""
from __future__ import annotations

import pandas as pd

from ..config import TIMEFRAME_MS
from ..db import Database, now_ms

COLUMNS = ["open_ts", "open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume", "trades", "closed"]


def upsert_candles(db: Database, exchange: str, symbol: str, timeframe: str, rows: list[dict]) -> int:
    if not rows:
        return 0
    t = now_ms()
    data = [(exchange, symbol, timeframe, int(r["open_ts"]), r["open"], r["high"], r["low"], r["close"], r["volume"],
             r.get("quote_volume"), r.get("taker_buy_volume"), r.get("trades"), int(r["closed"]), t) for r in rows]
    with db.transaction():
        db.executemany(
            "INSERT INTO candles(exchange,symbol,timeframe,open_ts,open,high,low,close,volume,quote_volume,"
            "taker_buy_volume,trades,closed,updated_ms) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(exchange,symbol,timeframe,open_ts) DO UPDATE SET open=excluded.open,high=excluded.high,"
            "low=excluded.low,close=excluded.close,volume=excluded.volume,quote_volume=excluded.quote_volume,"
            "taker_buy_volume=excluded.taker_buy_volume,trades=excluded.trades,closed=excluded.closed,"
            "updated_ms=excluded.updated_ms", data)
    return len(data)


def load_candles(db: Database, exchange: str, symbol: str, timeframe: str, limit: int | None = None,
                 closed_only: bool = True, end_ts: int | None = None) -> pd.DataFrame:
    sql = (f"SELECT {','.join(COLUMNS)} FROM candles WHERE exchange=? AND symbol=? AND timeframe=?"
           + (" AND closed=1" if closed_only else "") + (" AND open_ts<=?" if end_ts is not None else "")
           + " ORDER BY open_ts DESC" + (f" LIMIT {int(limit)}" if limit else ""))
    params: list = [exchange, symbol, timeframe] + ([end_ts] if end_ts is not None else [])
    rows = db.query(sql, params)
    df = pd.DataFrame(rows, columns=COLUMNS)
    if df.empty:
        return df
    df = df.sort_values("open_ts").reset_index(drop=True)
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df["open_ts"] = df["open_ts"].astype("int64")
    return df


def missing_bars(df: pd.DataFrame, timeframe: str) -> int:
    """Number of missing bars inside the time range covered by df."""
    if len(df) < 2:
        return 0
    span = (int(df.open_ts.iloc[-1]) - int(df.open_ts.iloc[0])) // TIMEFRAME_MS[timeframe] + 1
    return max(0, int(span - len(df)))
