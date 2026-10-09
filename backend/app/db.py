"""Database access layer.

SQLite (WAL mode) is the default for a single VPS. All SQL is written in the
common subset of SQLite and PostgreSQL (`?` placeholders are translated,
`INSERT .. ON CONFLICT .. DO UPDATE`, `RETURNING`), so switching to PostgreSQL
is a matter of setting DATABASE_URL=postgresql://... and installing
`psycopg[binary]` (see requirements-postgres.txt).

Every service process owns its own Database object; concurrent access between
processes is handled by SQLite WAL + busy_timeout (or by PostgreSQL itself).
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

log = logging.getLogger(__name__)

SCHEMA_VERSION = 4

# Columns added after the first release: (table, column, type). Applied to existing databases.
MIGRATIONS = [("predictions", "gate_json", "TEXT")]

# Append-only ledgers (SQLite): what was predicted can never be changed afterwards; the outcome columns
# can be written exactly once. Checked by tests (no retroactive changes).
SQLITE_TRIGGERS = [
    """CREATE TRIGGER IF NOT EXISTS forecasts_append_only BEFORE UPDATE ON forecasts
       WHEN OLD.resolved_ms IS NOT NULL
         OR NEW.forecast_id IS NOT OLD.forecast_id OR NEW.created_ms IS NOT OLD.created_ms
         OR NEW.data_ts_ms IS NOT OLD.data_ts_ms OR NEW.symbol IS NOT OLD.symbol OR NEW.exchange IS NOT OLD.exchange
         OR NEW.horizon_sec IS NOT OLD.horizon_sec OR NEW.target_ts IS NOT OLD.target_ts
         OR NEW.ref_price IS NOT OLD.ref_price OR NEW.p_up IS NOT OLD.p_up OR NEW.p_down IS NOT OLD.p_down
         OR NEW.p_flat IS NOT OLD.p_flat OR NEW.q10_bps IS NOT OLD.q10_bps OR NEW.q50_bps IS NOT OLD.q50_bps
         OR NEW.q90_bps IS NOT OLD.q90_bps OR NEW.decision IS NOT OLD.decision OR NEW.reasons IS NOT OLD.reasons
         OR NEW.model_version IS NOT OLD.model_version OR NEW.features_json IS NOT OLD.features_json
         OR NEW.input_hash IS NOT OLD.input_hash OR NEW.base_p_up IS NOT OLD.base_p_up
       BEGIN SELECT RAISE(ABORT, 'forecasts are append-only: the prediction cannot be changed'); END""",
    """CREATE TRIGGER IF NOT EXISTS predictions_append_only BEFORE UPDATE ON predictions
       WHEN OLD.resolved_ms IS NOT NULL
         OR NEW.created_ms IS NOT OLD.created_ms OR NEW.candle_ts IS NOT OLD.candle_ts OR NEW.price IS NOT OLD.price
         OR NEW.prediction IS NOT OLD.prediction OR NEW.p_up IS NOT OLD.p_up OR NEW.p_down IS NOT OLD.p_down
         OR NEW.p_flat IS NOT OLD.p_flat OR NEW.model_version IS NOT OLD.model_version
         OR NEW.gate_reasons IS NOT OLD.gate_reasons OR NEW.features_json IS NOT OLD.features_json
       BEGIN SELECT RAISE(ABORT, 'predictions are append-only: the prediction cannot be changed'); END""",
]

# {pk} -> autoincrement primary key; types chosen to be valid in both engines.
SCHEMA = [
    """CREATE TABLE IF NOT EXISTS candles(
        exchange TEXT NOT NULL, symbol TEXT NOT NULL, timeframe TEXT NOT NULL,
        open_ts BIGINT NOT NULL,
        open DOUBLE PRECISION NOT NULL, high DOUBLE PRECISION NOT NULL,
        low DOUBLE PRECISION NOT NULL, close DOUBLE PRECISION NOT NULL,
        volume DOUBLE PRECISION NOT NULL, quote_volume DOUBLE PRECISION,
        taker_buy_volume DOUBLE PRECISION, trades BIGINT,
        closed INTEGER NOT NULL, updated_ms BIGINT NOT NULL,
        PRIMARY KEY(exchange, symbol, timeframe, open_ts))""",
    """CREATE TABLE IF NOT EXISTS flow_bars(
        exchange TEXT NOT NULL, symbol TEXT NOT NULL, open_ts BIGINT NOT NULL,
        buy_volume DOUBLE PRECISION NOT NULL, sell_volume DOUBLE PRECISION NOT NULL,
        delta DOUBLE PRECISION NOT NULL, cvd DOUBLE PRECISION NOT NULL,
        trades BIGINT NOT NULL, vwap DOUBLE PRECISION, high DOUBLE PRECISION, low DOUBLE PRECISION,
        last_price DOUBLE PRECISION, updated_ms BIGINT NOT NULL,
        PRIMARY KEY(exchange, symbol, open_ts))""",
    """CREATE TABLE IF NOT EXISTS trades_raw(
        id {pk}, exchange TEXT NOT NULL, symbol TEXT NOT NULL, trade_id TEXT,
        ts_ms BIGINT NOT NULL, price DOUBLE PRECISION NOT NULL, qty DOUBLE PRECISION NOT NULL,
        side TEXT NOT NULL, recv_ms BIGINT NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS ix_trades_raw_ts ON trades_raw(ts_ms)",
    """CREATE TABLE IF NOT EXISTS book_state(
        exchange TEXT NOT NULL, symbol TEXT NOT NULL, ts_ms BIGINT NOT NULL, recv_ms BIGINT NOT NULL,
        best_bid DOUBLE PRECISION, best_ask DOUBLE PRECISION, mid DOUBLE PRECISION,
        spread_bps DOUBLE PRECISION, bid_depth DOUBLE PRECISION, ask_depth DOUBLE PRECISION,
        imbalance DOUBLE PRECISION, levels_json TEXT,
        PRIMARY KEY(exchange, symbol))""",
    """CREATE TABLE IF NOT EXISTS book_samples(
        exchange TEXT NOT NULL, symbol TEXT NOT NULL, ts_ms BIGINT NOT NULL,
        mid DOUBLE PRECISION, spread_bps DOUBLE PRECISION, imbalance DOUBLE PRECISION,
        bid_depth DOUBLE PRECISION, ask_depth DOUBLE PRECISION,
        PRIMARY KEY(exchange, symbol, ts_ms))""",
    """CREATE TABLE IF NOT EXISTS stream_status(
        exchange TEXT NOT NULL, symbol TEXT NOT NULL, state TEXT NOT NULL,
        connected_since_ms BIGINT, last_msg_ms BIGINT, last_trade_ms BIGINT, last_book_ms BIGINT,
        last_exchange_ts_ms BIGINT, reconnects BIGINT NOT NULL DEFAULT 0, gaps BIGINT NOT NULL DEFAULT 0,
        duplicates BIGINT NOT NULL DEFAULT 0, invalid BIGINT NOT NULL DEFAULT 0,
        stale_events BIGINT NOT NULL DEFAULT 0, clock_skew_ms BIGINT, last_error TEXT,
        updated_ms BIGINT NOT NULL,
        PRIMARY KEY(exchange, symbol))""",
    """CREATE TABLE IF NOT EXISTS market_ticker(
        exchange TEXT NOT NULL, symbol TEXT NOT NULL, last DOUBLE PRECISION,
        change_24h_pct DOUBLE PRECISION, volume_24h DOUBLE PRECISION, quote_volume_24h DOUBLE PRECISION,
        high_24h DOUBLE PRECISION, low_24h DOUBLE PRECISION, ts_ms BIGINT NOT NULL,
        PRIMARY KEY(exchange, symbol))""",
    """CREATE TABLE IF NOT EXISTS news_events(
        id TEXT PRIMARY KEY, published_ms BIGINT NOT NULL, fetched_ms BIGINT NOT NULL,
        source TEXT, title TEXT NOT NULL, url TEXT, summary TEXT,
        category TEXT, event_type TEXT, asset TEXT, direction DOUBLE PRECISION,
        relevance DOUBLE PRECISION, novelty DOUBLE PRECISION, confidence DOUBLE PRECISION,
        horizon_min BIGINT, affected_assets TEXT, analyzer TEXT, raw_json TEXT)""",
    "CREATE INDEX IF NOT EXISTS ix_news_published ON news_events(published_ms)",
    """CREATE TABLE IF NOT EXISTS predictions(
        prediction_id {pk}, created_ms BIGINT NOT NULL, candle_ts BIGINT NOT NULL,
        target_ts BIGINT NOT NULL, symbol TEXT NOT NULL, exchange TEXT NOT NULL,
        timeframe TEXT NOT NULL, horizon_bars INTEGER NOT NULL, price DOUBLE PRECISION NOT NULL,
        prediction TEXT NOT NULL, model_direction TEXT NOT NULL,
        p_up DOUBLE PRECISION NOT NULL, p_down DOUBLE PRECISION NOT NULL, p_flat DOUBLE PRECISION NOT NULL,
        confidence DOUBLE PRECISION NOT NULL, label_threshold DOUBLE PRECISION NOT NULL,
        gate_reasons TEXT, gate_json TEXT, regime TEXT, quality_score DOUBLE PRECISION,
        news_impact DOUBLE PRECISION, news_relevance DOUBLE PRECISION,
        features_version TEXT NOT NULL, model_version TEXT NOT NULL,
        features_json TEXT, snapshot_json TEXT,
        resolved_ms BIGINT, actual_price DOUBLE PRECISION, actual_return DOUBLE PRECISION,
        actual_direction TEXT, error DOUBLE PRECISION, result TEXT, error_class TEXT,
        UNIQUE(symbol, exchange, timeframe, horizon_bars, candle_ts))""",
    "CREATE INDEX IF NOT EXISTS ix_predictions_pending ON predictions(resolved_ms, target_ts)",
    """CREATE TABLE IF NOT EXISTS model_registry(
        version TEXT PRIMARY KEY, model_key TEXT NOT NULL, status TEXT NOT NULL,
        created_ms BIGINT NOT NULL, promoted_ms BIGINT,
        train_start_ts BIGINT, train_end_ts BIGINT, n_train BIGINT, n_validation BIGINT,
        feature_version TEXT NOT NULL, features_json TEXT, params_json TEXT,
        metrics_json TEXT, comparison_json TEXT, artifact_path TEXT,
        parent_version TEXT, reason TEXT)""",
    "CREATE INDEX IF NOT EXISTS ix_registry_key ON model_registry(model_key, status)",
    """CREATE TABLE IF NOT EXISTS learning_events(
        id {pk}, ts_ms BIGINT NOT NULL, model_key TEXT, event_type TEXT NOT NULL,
        payload_json TEXT)""",
    # every REST/WebSocket check of every exchange: error rates over time (audit finding A2)
    """CREATE TABLE IF NOT EXISTS source_checks(
        id {pk}, ts_ms BIGINT NOT NULL, exchange TEXT NOT NULL, channel TEXT NOT NULL,
        ok INTEGER NOT NULL, kind TEXT, latency_ms BIGINT, host TEXT, detail TEXT, skew_ms BIGINT)""",
    "CREATE INDEX IF NOT EXISTS ix_source_checks_ts ON source_checks(ts_ms)",
    # candle closes for which no prediction was made (scheduler delay, outage): shown, never hidden
    """CREATE TABLE IF NOT EXISTS prediction_gaps(
        symbol TEXT NOT NULL, exchange TEXT NOT NULL, timeframe TEXT NOT NULL, candle_ts BIGINT NOT NULL,
        detected_ms BIGINT NOT NULL, reason TEXT,
        PRIMARY KEY(symbol, exchange, timeframe, candle_ts))""",
    # 1-second price/flow bars from the live trade stream (seconds horizons, exact outcome prices)
    """CREATE TABLE IF NOT EXISTS price_seconds(
        exchange TEXT NOT NULL, symbol TEXT NOT NULL, ts_sec BIGINT NOT NULL,
        open DOUBLE PRECISION NOT NULL, high DOUBLE PRECISION NOT NULL, low DOUBLE PRECISION NOT NULL,
        close DOUBLE PRECISION NOT NULL, buy_volume DOUBLE PRECISION NOT NULL, sell_volume DOUBLE PRECISION NOT NULL,
        trades BIGINT NOT NULL, last_trade_ms BIGINT NOT NULL,
        PRIMARY KEY(exchange, symbol, ts_sec))""",
    # forward-looking forecasts for many horizons; append-only (see TRIGGERS)
    """CREATE TABLE IF NOT EXISTS forecasts(
        forecast_id TEXT PRIMARY KEY, created_ms BIGINT NOT NULL, data_ts_ms BIGINT NOT NULL,
        symbol TEXT NOT NULL, exchange TEXT NOT NULL, horizon_sec INTEGER NOT NULL, target_ts BIGINT NOT NULL,
        ref_price DOUBLE PRECISION NOT NULL, ref_source TEXT NOT NULL,
        p_up DOUBLE PRECISION, p_down DOUBLE PRECISION, p_flat DOUBLE PRECISION, cost_bps DOUBLE PRECISION NOT NULL,
        q10_bps DOUBLE PRECISION, q50_bps DOUBLE PRECISION, q90_bps DOUBLE PRECISION, uncertainty DOUBLE PRECISION,
        decision TEXT NOT NULL, reasons TEXT NOT NULL, model_version TEXT, model_status TEXT,
        base_p_up DOUBLE PRECISION, base_p_down DOUBLE PRECISION, base_p_flat DOUBLE PRECISION,
        base_q50_bps DOUBLE PRECISION, features_json TEXT, input_hash TEXT NOT NULL, detail_json TEXT,
        resolved_ms BIGINT, resolution_source TEXT, resolution_lag_ms BIGINT, actual_price DOUBLE PRECISION,
        actual_bps DOUBLE PRECISION, actual_class TEXT, brier DOUBLE PRECISION, brier_base DOUBLE PRECISION,
        logloss DOUBLE PRECISION, logloss_base DOUBLE PRECISION, abs_err_bps DOUBLE PRECISION,
        abs_err_rw_bps DOUBLE PRECISION, in_range INTEGER, net_bps DOUBLE PRECISION)""",
    "CREATE INDEX IF NOT EXISTS ix_forecasts_key ON forecasts(symbol, horizon_sec, created_ms)",
    "CREATE INDEX IF NOT EXISTS ix_forecasts_pending ON forecasts(resolved_ms, target_ts)",
    """CREATE TABLE IF NOT EXISTS system_state(
        key TEXT PRIMARY KEY, value_json TEXT, updated_ms BIGINT NOT NULL)""",
]


def now_ms() -> int:
    return int(time.time() * 1000)


class Database:
    def __init__(self, url: str):
        self.url = url
        self._lock = threading.RLock()
        if url.startswith("sqlite:///"):
            self.dialect = "sqlite"
            path = url[len("sqlite:///"):]
            if path != ":memory:":
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.path = path
            self.conn = sqlite3.connect(path, check_same_thread=False, timeout=30, isolation_level=None)
            self.conn.row_factory = sqlite3.Row
            if path != ":memory:":
                self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.execute("PRAGMA busy_timeout=30000")
        elif url.startswith(("postgres://", "postgresql://")):
            self.dialect = "postgres"
            self.path = None
            try:
                import psycopg  # type: ignore
                from psycopg.rows import dict_row  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("PostgreSQL requires `pip install psycopg[binary]` (requirements-postgres.txt)") from exc
            self.conn = psycopg.connect(url, autocommit=True, row_factory=dict_row)
        else:
            raise ValueError(f"Unsupported DATABASE_URL: {url}")
        self._in_tx = False

    # ---------------------------------------------------------------- schema
    def init_schema(self) -> None:
        pk = "INTEGER PRIMARY KEY AUTOINCREMENT" if self.dialect == "sqlite" else "BIGSERIAL PRIMARY KEY"
        with self.transaction():
            for stmt in SCHEMA:
                self.execute(stmt.replace("{pk}", pk))
            for table, col, typ in MIGRATIONS:
                if col not in self.columns(table):
                    self.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
            if self.dialect == "sqlite":
                for stmt in SQLITE_TRIGGERS:
                    self.execute(stmt)
            self.set_state("schema_version", SCHEMA_VERSION)

    def columns(self, table: str) -> set[str]:
        if self.dialect == "sqlite":
            return {r["name"] for r in self.query(f"PRAGMA table_info({table})")}
        return {r["column_name"] for r in self.query(  # pragma: no cover
            "SELECT column_name FROM information_schema.columns WHERE table_name=?", (table,))}

    # ----------------------------------------------------------------- core
    def _sql(self, sql: str) -> str:
        return sql.replace("?", "%s") if self.dialect == "postgres" else sql

    def execute(self, sql: str, params: Iterable[Any] = ()):
        with self._lock:
            return self.conn.execute(self._sql(sql), tuple(params))

    def executemany(self, sql: str, rows: list[tuple]) -> None:
        if not rows:
            return
        with self._lock:
            if self.dialect == "sqlite":
                self.conn.executemany(sql, rows)
            else:
                with self.conn.cursor() as cur:
                    cur.executemany(self._sql(sql), rows)

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[dict]:
        with self._lock:
            cur = self.conn.execute(self._sql(sql), tuple(params))
            return [dict(r) for r in cur.fetchall()]

    def query_one(self, sql: str, params: Iterable[Any] = ()) -> dict | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def scalar(self, sql: str, params: Iterable[Any] = ()):
        row = self.query_one(sql, params)
        return None if row is None else next(iter(row.values()))

    @contextmanager
    def transaction(self) -> Iterator["Database"]:
        with self._lock:
            if self._in_tx:  # nested: join outer transaction
                yield self
                return
            self._in_tx = True
            try:
                if self.dialect == "sqlite":
                    self.conn.execute("BEGIN IMMEDIATE")
                    try:
                        yield self
                        self.conn.execute("COMMIT")
                    except BaseException:
                        self.conn.execute("ROLLBACK")
                        raise
                else:  # pragma: no cover
                    with self.conn.transaction():
                        yield self
            finally:
                self._in_tx = False

    def ping(self) -> bool:
        try:
            return self.scalar("SELECT 1") == 1
        except Exception:
            return False

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    # ------------------------------------------------------- key/value state
    def set_state(self, key: str, value: Any) -> None:
        self.execute(
            "INSERT INTO system_state(key,value_json,updated_ms) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_ms=excluded.updated_ms",
            (key, json.dumps(value, default=str), now_ms()),
        )

    def get_state(self, key: str, default: Any = None) -> Any:
        row = self.query_one("SELECT value_json FROM system_state WHERE key=?", (key,))
        if row is None:
            return default
        try:
            return json.loads(row["value_json"])
        except (TypeError, ValueError):
            return default

    def get_state_row(self, key: str) -> dict | None:
        return self.query_one("SELECT key,value_json,updated_ms FROM system_state WHERE key=?", (key,))

    def heartbeat(self, service: str, info: dict | None = None) -> None:
        self.set_state(f"heartbeat:{service}", {"ts_ms": now_ms(), **(info or {})})

    def log_event(self, event_type: str, payload: dict, model_key: str | None = None) -> None:
        self.execute(
            "INSERT INTO learning_events(ts_ms,model_key,event_type,payload_json) VALUES(?,?,?,?)",
            (now_ms(), model_key, event_type, json.dumps(payload, default=str)),
        )

    # ----------------------------------------------------------- retention
    def apply_retention(self, settings) -> dict:
        t = now_ms()
        h, d = 3_600_000, 86_400_000
        jobs = {
            "trades_raw": ("DELETE FROM trades_raw WHERE ts_ms < ?", t - int(settings.raw_trades_retention_hours * h)),
            "book_samples": ("DELETE FROM book_samples WHERE ts_ms < ?", t - int(settings.book_samples_retention_days * d)),
            "flow_bars": ("DELETE FROM flow_bars WHERE open_ts < ?", t - int(settings.flow_bars_retention_days * d)),
            "candles_1m": ("DELETE FROM candles WHERE timeframe='1m' AND open_ts < ?", t - int(settings.candles_1m_retention_days * d)),
            "news_events": ("DELETE FROM news_events WHERE published_ms < ?", t - int(settings.news_retention_days * d)),
            "learning_events": ("DELETE FROM learning_events WHERE ts_ms < ?", t - 180 * d),
            "price_seconds": ("DELETE FROM price_seconds WHERE ts_sec < ?",
                              (t - int(getattr(settings, "second_bars_retention_hours", 48) * h)) // 1000),
            # seconds/minutes forecasts are many: raw rows for 14 days, their statistics stay in the exports
            "forecasts_short": ("DELETE FROM forecasts WHERE horizon_sec <= 300 AND created_ms < ?",
                                t - int(getattr(settings, "short_forecast_retention_days", 14) * d)),
            "source_checks": ("DELETE FROM source_checks WHERE ts_ms < ?",
                              t - int(getattr(settings, "source_checks_retention_days", 30) * d)),
        }
        out = {}
        for name, (sql, cutoff) in jobs.items():
            cur = self.execute(sql, (cutoff,))
            out[name] = getattr(cur, "rowcount", None)
        if self.dialect == "sqlite":
            self.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self.execute("PRAGMA optimize")
        return out


_db_cache: dict[str, Database] = {}


def get_db(url: str | None = None) -> Database:
    """Process-wide Database (one connection per process/URL)."""
    from .config import get_settings

    url = url or get_settings().database_url
    db = _db_cache.get(url)
    if db is None:
        db = Database(url)
        db.init_schema()
        _db_cache[url] = db
    return db
