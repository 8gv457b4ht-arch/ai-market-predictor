"""Research notifications via ntfy.sh and/or Telegram. Never trading instructions.

Events: signals (UP/DOWN predictions), all_predictions, outages (exchange data source changes
state), promotions (a challenger model replaced production). Each event is sent once.
"""
from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request

from .config import Settings
from .db import Database

log = logging.getLogger("notify")
DISCLAIMER = "Research probability, not a trading recommendation."


def _post(url: str, data: bytes, headers: dict) -> None:
    req = urllib.request.Request(url, data=data, method="POST", headers=headers)
    with urllib.request.urlopen(req, timeout=15) as r:
        r.read()


class Notifier:
    def __init__(self, db: Database, settings: Settings, post=None):
        self.db, self.s, self._post_fn = db, settings, post
        self.sent: list[dict] = []
        self.errors: list[str] = []

    @property
    def enabled(self) -> bool:
        return bool(self.s.ntfy_topic or (self.s.telegram_bot_token and self.s.telegram_chat_id))

    def _post(self, url: str, data: bytes, headers: dict) -> None:
        (self._post_fn or _post)(url, data, headers)  # module-level lookup at call time (testable)

    def send(self, title: str, body: str, priority: str = "default", tags: str = "chart_with_upwards_trend") -> None:
        delivered = False
        if self.s.dashboard_url:
            body = f"{body}\n{self.s.dashboard_url}"
        if self.s.ntfy_topic:
            try:
                self._post(f"{self.s.ntfy_server}/{urllib.parse.quote(self.s.ntfy_topic)}", body.encode("utf-8"),
                           {"Title": title.encode("ascii", "replace").decode(), "Priority": priority, "Tags": tags})
                delivered = True
            except Exception as exc:  # noqa: BLE001 - a broken channel must not stop the cycle
                self.errors.append(f"ntfy: {exc}")
        if self.s.telegram_bot_token and self.s.telegram_chat_id:
            try:
                self._post(f"https://api.telegram.org/bot{self.s.telegram_bot_token}/sendMessage",
                           json.dumps({"chat_id": self.s.telegram_chat_id, "text": f"{title}\n{body}",
                                       "disable_web_page_preview": True}).encode(),
                           {"Content-Type": "application/json"})
                delivered = True
            except Exception as exc:  # noqa: BLE001
                self.errors.append(f"telegram: {type(exc).__name__}")  # never log the token-bearing URL
        if delivered:
            self.sent.append({"title": title})

    # ------------------------------------------------------------------ events
    def predictions(self) -> None:
        ev = set(self.s.notify_events)
        if not ({"signals", "all_predictions"} & ev):
            return
        last = int(self.db.get_state("notify_last_prediction_id", 0) or 0)
        rows = self.db.query("SELECT * FROM predictions WHERE prediction_id > ? ORDER BY prediction_id", (last,))
        for r in rows:
            if r["prediction"] in ("UP", "DOWN") or "all_predictions" in ev:
                g = json.loads(r.get("gate_json") or "{}")
                title = f"{r['symbol']} {r['timeframe']}: {r['prediction']}"
                body = (f"Horizon {r['horizon_bars']} x {r['timeframe']}, price {r['price']:,.2f} ({r['exchange']})\n"
                        f"UP {r['p_up']:.0%} / DOWN {r['p_down']:.0%} / FLAT {r['p_flat']:.0%}, confidence {r['confidence']:.0%}\n"
                        f"Data quality {r['quality_score']:.0%}, regime {r['regime']}, model {r['model_version']}\n"
                        + (f"No trade: {', '.join(json.loads(r['gate_reasons'] or '[]'))}\n" if r["prediction"] == "NO TRADE" else "")
                        + (f"Decision delay {g.get('decision_delay_sec')} s\n" if g else "") + DISCLAIMER)
                self.send(title, body, "high" if r["prediction"] != "NO TRADE" else "low")
        if rows:
            self.db.set_state("notify_last_prediction_id", rows[-1]["prediction_id"])

    def outages(self, probes: dict) -> None:
        if "outages" not in self.s.notify_events:
            return
        prev = self.db.get_state("notify_source_state", {}) or {}
        cur = {}
        for ex, p in probes.items():
            ok = bool(p.get("rest", {}).get("ok")) and bool(p.get("ws", {}).get("verified_live"))
            cur[ex] = "ok" if ok else (p.get("rest", {}).get("kind") or p.get("ws", {}).get("kind") or "down")
            if ex in prev and prev[ex] != cur[ex]:
                if cur[ex] == "ok":
                    self.send(f"{ex}: data source back", f"{ex} REST and WebSocket verified again.", "default", "white_check_mark")
                else:
                    self.send(f"{ex}: DATA SOURCE UNAVAILABLE", f"{ex} problem: {cur[ex]}. "
                              f"REST: {p.get('rest', {}).get('error')}; WS: {p.get('ws', {}).get('error')}", "high", "warning")
        self.db.set_state("notify_source_state", cur)

    def promotions(self) -> None:
        if "promotions" not in self.s.notify_events:
            return
        last = int(self.db.get_state("notify_last_event_id", 0) or 0)
        rows = self.db.query("SELECT * FROM learning_events WHERE id > ? ORDER BY id", (last,))
        for r in rows:
            if r["event_type"] in ("challenger_promoted", "baseline_trained"):
                p = json.loads(r["payload_json"] or "{}")
                self.send(f"Model update {r['model_key']}", f"{r['event_type'].replace('_', ' ')}: {p.get('version')}. "
                          f"Out-of-sample metrics are on the dashboard.", "default", "brain")
        if rows:
            self.db.set_state("notify_last_event_id", rows[-1]["id"])
