"""Forecast horizons. The time a forecast looks ahead (horizon) is separate from how often it is
recomputed (refresh). Each horizon is learned on a grid whose bars really exist in the data:
seconds horizons on 1-second bars from the live trade stream, minutes on 1-minute candles,
hours on 15-minute or 1-hour candles. Nothing is resampled to a finer resolution than its source."""
from __future__ import annotations

from dataclasses import dataclass

# evaluation rules; changing compute priorities never changes these.
# fp2: direction hit rate counts only forecasts whose median has a sign (fp1 counted a zero median as a miss)
EVAL_PROTOCOL = "fp2"


@dataclass(frozen=True)
class Horizon:
    seconds: int
    label: str
    refresh_sec: int   # how often a new forecast is issued
    grid: str          # bar series the model learns on: "1s", "1m", "15m", "1h"

    @property
    def grid_sec(self) -> int:
        return {"1s": 1, "1m": 60, "15m": 900, "1h": 3600}[self.grid]

    @property
    def steps(self) -> int:
        """Horizon length in grid bars (also the label overlap -> purge gap and bootstrap block)."""
        return max(1, self.seconds // self.grid_sec)

    @property
    def max_data_age_sec(self) -> int:
        """Newest input must be at most this old, otherwise DATA STALE."""
        if self.grid == "1s":
            return 3
        return self.grid_sec + 120  # the last closed bar plus fetch delay

    @property
    def resolve_tolerance_ms(self) -> int:
        """Largest accepted gap between the target time and the price used to check the forecast."""
        return 2_000 if self.seconds < 900 else 60_000


HORIZONS = [
    Horizon(5, "5s", 15, "1s"),
    Horizon(15, "15s", 15, "1s"),
    Horizon(30, "30s", 30, "1s"),
    Horizon(60, "1m", 20, "1m"),
    Horizon(180, "3m", 30, "1m"),
    Horizon(300, "5m", 50, "1m"),
    Horizon(900, "15m", 150, "1m"),
    Horizon(1800, "30m", 300, "1m"),
    Horizon(3600, "1h", 600, "15m"),
    Horizon(7200, "2h", 1200, "15m"),
    Horizon(14400, "4h", 2400, "1h"),
    Horizon(28800, "8h", 3600, "1h"),
    Horizon(43200, "12h", 3600, "1h"),
    Horizon(86400, "1d", 3600, "1h"),
]
BY_SECONDS = {h.seconds: h for h in HORIZONS}


def model_key(symbol: str, horizon: Horizon) -> str:
    return f"{symbol}|f{horizon.seconds}"
