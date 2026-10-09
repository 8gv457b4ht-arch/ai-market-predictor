"""Central configuration. Every value comes from the environment (.env) so that
no secret or deployment detail is hard-coded. Read once per process."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

TIMEFRAME_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}
ALL_TIMEFRAMES = list(TIMEFRAME_MS)


def _list(name: str, default: str) -> list[str]:
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # --- storage -----------------------------------------------------------
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", f"sqlite:///{ROOT / 'data' / 'market.sqlite3'}"))
    model_dir: Path = field(default_factory=lambda: Path(os.getenv("MODEL_DIR", str(ROOT / "models"))))
    backup_dir: Path = field(default_factory=lambda: Path(os.getenv("BACKUP_DIR", str(ROOT / "backups"))))

    # --- markets -----------------------------------------------------------
    exchanges: list[str] = field(default_factory=lambda: [x.lower() for x in _list("EXCHANGES", "binance,bybit,okx")])
    primary_exchange: str = field(default_factory=lambda: os.getenv("PRIMARY_EXCHANGE", "binance").lower())
    symbols: list[str] = field(default_factory=lambda: [x.upper() for x in _list("SYMBOLS", "BTC/USDT,ETH/USDT")])
    timeframes: list[str] = field(default_factory=lambda: _list("TIMEFRAMES", ",".join(ALL_TIMEFRAMES)))
    predict_timeframes: list[str] = field(default_factory=lambda: _list("PREDICT_TIMEFRAMES", "15m,1h"))
    horizon_bars: int = field(default_factory=lambda: _int("HORIZON_BARS", 4))
    history_bars: int = field(default_factory=lambda: _int("HISTORY_BARS", 5000))
    context_history_bars: int = field(default_factory=lambda: _int("CONTEXT_HISTORY_BARS", 1500))
    okx_ws_url: str = field(default_factory=lambda: os.getenv("OKX_WS_URL", "wss://ws.okx.com/ws/v5/public"))
    binance_ws_url: str = field(default_factory=lambda: os.getenv("BINANCE_WS_URL", "wss://stream.binance.com:9443/stream"))
    bybit_ws_url: str = field(default_factory=lambda: os.getenv("BYBIT_WS_URL", "wss://stream.bybit.com/v5/public/spot"))
    binance_rest_url: str = field(default_factory=lambda: os.getenv("BINANCE_REST_URL", "https://api.binance.com"))
    bybit_rest_url: str = field(default_factory=lambda: os.getenv("BYBIT_REST_URL", "https://api.bybit.com"))
    okx_rest_url: str = field(default_factory=lambda: os.getenv("OKX_REST_URL", "https://www.okx.com"))

    # --- collector ---------------------------------------------------------
    stale_after_sec: float = field(default_factory=lambda: _float("STALE_AFTER_SEC", 30))
    reconnect_max_sec: float = field(default_factory=lambda: _float("RECONNECT_MAX_SEC", 60))
    flush_interval_sec: float = field(default_factory=lambda: _float("FLUSH_INTERVAL_SEC", 1.0))
    candle_sync_sec: float = field(default_factory=lambda: _float("CANDLE_SYNC_SEC", 30))
    store_raw_trades: bool = field(default_factory=lambda: _bool("STORE_RAW_TRADES", True))

    # --- retention (hours / days) -----------------------------------------
    raw_trades_retention_hours: float = field(default_factory=lambda: _float("RAW_TRADES_RETENTION_HOURS", 6))
    book_samples_retention_days: float = field(default_factory=lambda: _float("BOOK_SAMPLES_RETENTION_DAYS", 7))
    flow_bars_retention_days: float = field(default_factory=lambda: _float("FLOW_BARS_RETENTION_DAYS", 30))
    candles_1m_retention_days: float = field(default_factory=lambda: _float("CANDLES_1M_RETENTION_DAYS", 30))
    news_retention_days: float = field(default_factory=lambda: _float("NEWS_RETENTION_DAYS", 60))

    # --- prediction & gating ----------------------------------------------
    confidence_threshold: float = field(default_factory=lambda: _float("CONFIDENCE_THRESHOLD", 0.55))
    min_edge: float = field(default_factory=lambda: _float("MIN_EDGE", 0.10))
    label_atr_mult: float = field(default_factory=lambda: _float("LABEL_ATR_MULT", 0.30))
    min_quality_score: float = field(default_factory=lambda: _float("MIN_QUALITY_SCORE", 0.70))
    block_low_liquidity: bool = field(default_factory=lambda: _bool("BLOCK_LOW_LIQUIDITY", True))
    max_spread_bps: float = field(default_factory=lambda: _float("MAX_SPREAD_BPS", 15))
    max_exchange_divergence_bps: float = field(default_factory=lambda: _float("MAX_EXCHANGE_DIVERGENCE_BPS", 40))
    predictor_poll_sec: float = field(default_factory=lambda: _float("PREDICTOR_POLL_SEC", 10))

    # --- costs (used in labels, backtest) ----------------------------------
    fee_bps: float = field(default_factory=lambda: _float("FEE_BPS", 10))           # per side, taker
    slippage_bps: float = field(default_factory=lambda: _float("SLIPPAGE_BPS", 2))  # per side
    spread_bps: float = field(default_factory=lambda: _float("SPREAD_BPS", 1))      # full spread paid per round trip
    latency_ms: float = field(default_factory=lambda: _float("LATENCY_MS", 500))

    # --- learning ----------------------------------------------------------
    learning_interval_sec: float = field(default_factory=lambda: _float("LEARNING_INTERVAL_SEC", 900))
    min_new_labels: int = field(default_factory=lambda: _int("LEARNING_MIN_NEW_LABELS", 300))
    holdout_fraction: float = field(default_factory=lambda: _float("LEARNING_HOLDOUT_FRACTION", 0.5))
    min_holdout: int = field(default_factory=lambda: _int("LEARNING_MIN_HOLDOUT", 150))
    min_logloss_gain: float = field(default_factory=lambda: _float("LEARNING_MIN_LOGLOSS_GAIN", 0.002))
    bootstrap_confidence: float = field(default_factory=lambda: _float("LEARNING_BOOTSTRAP_CONFIDENCE", 0.90))
    walk_forward_folds: int = field(default_factory=lambda: _int("WALK_FORWARD_FOLDS", 5))
    min_train_rows: int = field(default_factory=lambda: _int("MIN_TRAIN_ROWS", 600))
    # a model must beat the naive base-rate forecast out of sample before its signals are used
    require_baseline_edge: bool = field(default_factory=lambda: _bool("REQUIRE_BASELINE_EDGE", True))
    baseline_p_better: float = field(default_factory=lambda: _float("BASELINE_P_BETTER", 0.95))
    # ... and its own out-of-sample signals must have earned a positive average after costs (enough of them)
    require_cost_edge: bool = field(default_factory=lambda: _bool("REQUIRE_COST_EDGE", True))
    min_oos_signals: int = field(default_factory=lambda: _int("MIN_OOS_SIGNALS", 30))

    # --- news --------------------------------------------------------------
    news_feeds: list[str] = field(default_factory=lambda: _list("NEWS_FEEDS", ",".join(DEFAULT_FEEDS)))
    news_interval_sec: float = field(default_factory=lambda: _float("NEWS_INTERVAL_SEC", 300))
    news_half_life_min: float = field(default_factory=lambda: _float("NEWS_HALF_LIFE_MIN", 180))
    news_llm_provider: str = field(default_factory=lambda: os.getenv("NEWS_LLM_PROVIDER", "auto").lower())
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "").strip())
    anthropic_model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5"))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", "").strip())
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "").strip())
    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    news_llm_max_items: int = field(default_factory=lambda: _int("NEWS_LLM_MAX_ITEMS", 20))

    # --- API / security ----------------------------------------------------
    api_key: str = field(default_factory=lambda: os.getenv("API_KEY", "").strip())
    cors_origins: list[str] = field(default_factory=lambda: _list("CORS_ORIGINS", ""))
    rate_limit_per_min: int = field(default_factory=lambda: _int("RATE_LIMIT_PER_MIN", 240))
    trust_proxy_headers: bool = field(default_factory=lambda: _bool("TRUST_PROXY_HEADERS", False))

    # --- notifications -----------------------------------------------------
    notify_events: list[str] = field(default_factory=lambda: _list("NOTIFY_EVENTS", "signals,outages,promotions"))
    ntfy_server: str = field(default_factory=lambda: os.getenv("NTFY_SERVER", "https://ntfy.sh").rstrip("/"))
    ntfy_topic: str = field(default_factory=lambda: os.getenv("NTFY_TOPIC", "").strip())
    telegram_bot_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", "").strip())
    telegram_chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", "").strip())
    dashboard_url: str = field(default_factory=lambda: os.getenv("DASHBOARD_URL", "").strip())
    notify_enabled: bool = field(default_factory=lambda: _bool("NOTIFY_ENABLED", True))
    notify_lang: str = field(default_factory=lambda: os.getenv("NOTIFY_LANG", "ru").strip().lower())
    notify_min_confidence: float = field(default_factory=lambda: _float("NOTIFY_MIN_CONFIDENCE", 0.0))
    notify_timeframes: list[str] = field(default_factory=lambda: _list("NOTIFY_TIMEFRAMES", ""))
    notify_symbols: list[str] = field(default_factory=lambda: [x.upper() for x in _list("NOTIFY_SYMBOLS", "")])
    notify_max_age_sec: float = field(default_factory=lambda: _float("NOTIFY_MAX_AGE_SEC", 600))
    notify_min_quality: float = field(default_factory=lambda: _float("NOTIFY_MIN_QUALITY", 0.0))
    control_dir: Path = field(default_factory=lambda: Path(os.getenv("CONTROL_DIR", str(ROOT / "control"))))

    # --- cloud (scheduled) mode --------------------------------------------
    ws_sample_sec: float = field(default_factory=lambda: _float("WS_SAMPLE_SEC", 45))
    public_dir: Path = field(default_factory=lambda: Path(os.getenv("PUBLIC_DIR", str(ROOT / "public"))))
    # scheduled: periodic job (GitHub Actions); continuous: always-on process (VPS/Docker)
    backend_mode: str = field(default_factory=lambda: os.getenv("BACKEND_MODE", "scheduled").strip().lower())
    # scheduled mode: keep streams open and wait for the next candle close (at most this long) to predict on time
    wait_close_max_sec: float = field(default_factory=lambda: _float("WAIT_CLOSE_MAX_SEC", 0))
    close_settle_sec: float = field(default_factory=lambda: _float("CLOSE_SETTLE_SEC", 6))
    source_checks_retention_days: float = field(default_factory=lambda: _float("SOURCE_CHECKS_RETENTION_DAYS", 30))

    # --- backup ------------------------------------------------------------
    backup_interval_sec: float = field(default_factory=lambda: _float("BACKUP_INTERVAL_SEC", 21600))
    backup_keep: int = field(default_factory=lambda: _int("BACKUP_KEEP", 14))

    def validate(self) -> None:
        bad = [t for t in self.timeframes + self.predict_timeframes if t not in TIMEFRAME_MS]
        if bad:
            raise ValueError(f"Unsupported timeframe(s): {bad}. Allowed: {ALL_TIMEFRAMES}")
        for t in self.predict_timeframes:
            if t not in self.timeframes:
                raise ValueError(f"PREDICT_TIMEFRAMES entry {t} must also be in TIMEFRAMES")
        unknown = [e for e in self.exchanges if e not in {"binance", "bybit", "okx"}]
        if unknown:
            raise ValueError(f"Unsupported exchange(s): {unknown}")
        if self.primary_exchange != "auto" and self.primary_exchange not in self.exchanges:
            raise ValueError("PRIMARY_EXCHANGE must be one of EXCHANGES or 'auto'")
        if not 0 < self.confidence_threshold < 1:
            raise ValueError("CONFIDENCE_THRESHOLD must be in (0,1)")

    @property
    def round_trip_cost_bps(self) -> float:
        return 2 * self.fee_bps + 2 * self.slippage_bps + self.spread_bps


DEFAULT_FEEDS = [
    # Crypto
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
    # Central banks / regulators (official press-release feeds)
    "https://www.federalreserve.gov/feeds/press_all.xml",
    "https://www.ecb.europa.eu/rss/press.html",
    "https://www.sec.gov/news/pressreleases.rss",
    # Macro / markets
    "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
]

_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.validate()
    return _settings


def reset_settings() -> None:
    """Testing hook: re-read the environment."""
    global _settings
    _settings = None
