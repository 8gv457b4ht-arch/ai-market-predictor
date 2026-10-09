"""Local order book maintained from WebSocket snapshots and deltas."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LocalOrderBook:
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    initialized: bool = False
    last_update_id: int | None = None
    last_seq: int | None = None
    ts_ms: int | None = None
    gaps: int = 0
    out_of_order: int = 0

    def apply_snapshot(self, bids, asks, update_id=None, seq=None, ts_ms=None) -> None:
        self.bids = {float(p): float(q) for p, q in bids if float(q) > 0}
        self.asks = {float(p): float(q) for p, q in asks if float(q) > 0}
        self.initialized = True
        self.last_update_id = update_id
        self.last_seq = seq
        self.ts_ms = ts_ms

    def apply_delta(self, bids, asks, update_id=None, seq=None, ts_ms=None) -> str:
        """Returns 'ok', 'gap' (book must be re-synced) or 'stale' (ignored)."""
        if not self.initialized:
            self.gaps += 1
            return "gap"
        if update_id is not None and self.last_update_id is not None and update_id <= self.last_update_id:
            self.out_of_order += 1
            return "stale"
        for side, levels in ((self.bids, bids), (self.asks, asks)):
            for p, q in levels:
                p, q = float(p), float(q)
                if q <= 0:
                    side.pop(p, None)
                else:
                    side[p] = q
        if update_id is not None:
            self.last_update_id = update_id
        if seq is not None:
            self.last_seq = seq
        self.ts_ms = ts_ms
        if self.crossed():
            # A crossed book means we missed something: force a re-sync.
            self.gaps += 1
            self.initialized = False
            return "gap"
        return "ok"

    def crossed(self) -> bool:
        if not self.bids or not self.asks:
            return False
        return max(self.bids) >= min(self.asks)

    def top(self, depth: int = 20) -> tuple[list[list[float]], list[list[float]]]:
        b = sorted(self.bids.items(), key=lambda x: -x[0])[:depth]
        a = sorted(self.asks.items(), key=lambda x: x[0])[:depth]
        return [list(x) for x in b], [list(x) for x in a]

    def metrics(self, depth: int = 20) -> dict:
        return book_metrics(*self.top(depth))


def book_metrics(bids: list, asks: list, imbalance_levels: int = 10) -> dict:
    if not bids or not asks:
        return {"best_bid": None, "best_ask": None, "mid": None, "spread_bps": None,
                "bid_depth": 0.0, "ask_depth": 0.0, "imbalance": None}
    bb, ba = float(bids[0][0]), float(asks[0][0])
    mid = (bb + ba) / 2
    bd = sum(float(q) for _, q in bids[:imbalance_levels])
    ad = sum(float(q) for _, q in asks[:imbalance_levels])
    tot = bd + ad
    return {
        "best_bid": bb, "best_ask": ba, "mid": mid,
        "spread_bps": (ba - bb) / mid * 1e4 if mid > 0 else None,
        "bid_depth": bd, "ask_depth": ad,
        "imbalance": (bd - ad) / tot if tot > 0 else None,
    }
