"""Research notifications via ntfy.sh and/or Telegram. Never trading instructions.

Events: signals (UP/DOWN predictions), all_predictions, outages (exchange data source changes
state), promotions (a challenger model replaced production). Each event is sent once.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request

from .config import Settings
from .db import Database

log = logging.getLogger("notify")
DISCLAIMER = "Research probability, not a trading recommendation."


def _bar_sec(tf: str) -> float:
    from .config import TIMEFRAME_MS
    return TIMEFRAME_MS.get(tf, 0) / 1000


# Notification texts (backend side of the translation dictionary; codes and tickers stay untranslated)
TEXT = {
    "ru": {"time": "Свеча закрыта {when}, горизонт {h} × {tf}", "price": "Цена {price} ({ex})",
           "probs": "Рост {up} / Падение {down} / Боковик {flat}, уверенность {conf}",
           "why_signal": "Все проверки пройдены. Главные факторы: {factors}",
           "why_no_trade": "Нет сделки, причины: {reasons}",
           "quality": "Качество данных {q}, режим {regime}, задержка решения {delay} с",
           "disclaimer": "Исследовательская вероятность, не торговая рекомендация.",
           "src_back": "{ex}: источник данных снова работает", "src_back_body": "{ex}: REST и WebSocket снова проверены.",
           "src_down": "{ex}: ИСТОЧНИК ДАННЫХ НЕДОСТУПЕН", "src_down_body": "{ex}: {kind}. REST: {rest}; WS: {ws}",
           "model": "Обновление модели {key}", "model_body": "{event}: {version}. Метрики вне выборки — на сайте."},
    "uk": {"time": "Свічка закрита {when}, горизонт {h} × {tf}", "price": "Ціна {price} ({ex})",
           "probs": "Зростання {up} / Падіння {down} / Боковик {flat}, впевненість {conf}",
           "why_signal": "Усі перевірки пройдено. Головні чинники: {factors}",
           "why_no_trade": "Без угоди, причини: {reasons}",
           "quality": "Якість даних {q}, режим {regime}, затримка рішення {delay} с",
           "disclaimer": "Дослідницька ймовірність, не торгова рекомендація.",
           "src_back": "{ex}: джерело даних знову працює", "src_back_body": "{ex}: REST і WebSocket знову перевірено.",
           "src_down": "{ex}: ДЖЕРЕЛО ДАНИХ НЕДОСТУПНЕ", "src_down_body": "{ex}: {kind}. REST: {rest}; WS: {ws}",
           "model": "Оновлення моделі {key}", "model_body": "{event}: {version}. Метрики поза вибіркою — на сайті."},
    "en": {"time": "Candle closed {when}, horizon {h} x {tf}", "price": "Price {price} ({ex})",
           "probs": "UP {up} / DOWN {down} / FLAT {flat}, confidence {conf}",
           "why_signal": "All checks passed. Main factors: {factors}",
           "why_no_trade": "No trade, reasons: {reasons}",
           "quality": "Data quality {q}, regime {regime}, decision delay {delay} s",
           "disclaimer": DISCLAIMER,
           "src_back": "{ex}: data source back", "src_back_body": "{ex} REST and WebSocket verified again.",
           "src_down": "{ex}: DATA SOURCE UNAVAILABLE", "src_down_body": "{ex} problem: {kind}. REST: {rest}; WS: {ws}",
           "model": "Model update {key}", "model_body": "{event}: {version}. Out-of-sample metrics are on the dashboard."},
}


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
                # JSON publishing: UTF-8 title and body (HTTP headers would mangle Cyrillic text)
                self._post(self.s.ntfy_server, json.dumps({
                    "topic": self.s.ntfy_topic, "title": title, "message": body,
                    "priority": {"low": 2, "default": 3, "high": 4}.get(priority, 3),
                    "tags": [t for t in tags.split(",") if t]}, ensure_ascii=False).encode("utf-8"),
                    {"Content-Type": "application/json; charset=utf-8"})
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
    def t(self, text_id: str, **kw) -> str:
        lang = self.s.notify_lang if self.s.notify_lang in TEXT else "ru"
        return TEXT[lang].get(text_id, TEXT["en"][text_id]).format(**kw)

    def _passes(self, r: dict, now: int) -> str | None:
        """None if the prediction may be sent, otherwise why not."""
        from .config import TIMEFRAME_MS
        signal = r["prediction"] in ("UP", "DOWN")
        if self.s.notify_timeframes and r["timeframe"] not in self.s.notify_timeframes:
            return "timeframe"
        if self.s.notify_symbols and r["symbol"] not in self.s.notify_symbols:
            return "symbol"
        if signal and r["confidence"] < self.s.notify_min_confidence:
            return "confidence"
        if (r["quality_score"] or 0) < self.s.notify_min_quality:
            return "quality"
        age = (now - (int(r["candle_ts"]) + TIMEFRAME_MS.get(r["timeframe"], 0))) / 1000
        if age > self.s.notify_max_age_sec:
            return "stale"  # e.g. a re-run after a failed save: the signal is no longer current
        return None

    def predictions(self, now: int | None = None) -> dict:
        from .db import now_ms
        now = now or now_ms()
        ev = set(self.s.notify_events)
        if not self.s.notify_enabled or not ({"signals", "all_predictions"} & ev):
            return {"skipped": "disabled"}
        last = int(self.db.get_state("notify_last_prediction_id", 0) or 0)
        sent_keys = list(self.db.get_state("notify_sent_keys", []) or [])
        rows = self.db.query("SELECT * FROM predictions WHERE prediction_id > ? ORDER BY prediction_id", (last,))
        skipped: dict = {}
        for r in rows:
            if not (r["prediction"] in ("UP", "DOWN") or "all_predictions" in ev):
                continue
            k = f"{r['symbol']}|{r['timeframe']}|{r['horizon_bars']}|{r['candle_ts']}"
            if k in sent_keys:  # the same candle is never announced twice (staleness also stops replays after a restore)
                skipped["duplicate"] = skipped.get("duplicate", 0) + 1
                continue
            why = self._passes(r, now)
            if why:
                skipped[why] = skipped.get(why, 0) + 1
                continue
            g = json.loads(r.get("gate_json") or "{}")
            reasons = json.loads(r["gate_reasons"] or "[]")
            factors = ", ".join(f["feature"] for f in (g.get("factors") or [])[:3])
            when = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime((int(r["candle_ts"])) / 1000 + _bar_sec(r["timeframe"])))
            title = f"{r['symbol']} {r['timeframe']}: {r['prediction']}"
            why_line = (self.t("why_signal", factors=factors or "-") if r["prediction"] != "NO TRADE"
                        else self.t("why_no_trade", reasons=", ".join(reasons) or "-"))
            body = "\n".join([
                self.t("time", when=when, h=r["horizon_bars"], tf=r["timeframe"]),
                self.t("price", price=f"{r['price']:,.2f}", ex=r["exchange"]),
                self.t("probs", up=f"{r['p_up']:.0%}", down=f"{r['p_down']:.0%}", flat=f"{r['p_flat']:.0%}",
                       conf=f"{r['confidence']:.0%}"),
                why_line,
                self.t("quality", q=f"{(r['quality_score'] or 0):.0%}", regime=r["regime"],
                       delay=g.get("decision_delay_sec", "-")),
                self.t("disclaimer")])
            before = len(self.sent)
            self.send(title, body, "high" if r["prediction"] != "NO TRADE" else "low")
            if len(self.sent) > before:
                sent_keys.append(k)
        if rows:
            self.db.set_state("notify_last_prediction_id", rows[-1]["prediction_id"])
        self.db.set_state("notify_sent_keys", sent_keys[-500:])
        return {"skipped": skipped}

    def outages(self, probes: dict) -> None:
        if not self.s.notify_enabled or "outages" not in self.s.notify_events:
            return
        prev = self.db.get_state("notify_source_state", {}) or {}
        cur = {}
        for ex, p in probes.items():
            ok = bool(p.get("rest", {}).get("ok")) and bool(p.get("ws", {}).get("verified_live"))
            cur[ex] = "ok" if ok else (p.get("rest", {}).get("kind") or p.get("ws", {}).get("kind") or "down")
            if ex in prev and prev[ex] != cur[ex]:
                if cur[ex] == "ok":
                    self.send(self.t("src_back", ex=ex), self.t("src_back_body", ex=ex), "default", "white_check_mark")
                else:
                    self.send(self.t("src_down", ex=ex), self.t("src_down_body", ex=ex, kind=cur[ex],
                              rest=p.get("rest", {}).get("error"), ws=p.get("ws", {}).get("error")), "high", "warning")
        self.db.set_state("notify_source_state", cur)

    def promotions(self) -> None:
        if not self.s.notify_enabled or "promotions" not in self.s.notify_events:
            return
        last = int(self.db.get_state("notify_last_event_id", 0) or 0)
        rows = self.db.query("SELECT * FROM learning_events WHERE id > ? ORDER BY id", (last,))
        for r in rows:
            if r["event_type"] in ("challenger_promoted", "baseline_trained", "model_rollback"):
                p = json.loads(r["payload_json"] or "{}")
                ver = p.get("version") or (p.get("to") or {}).get("version")
                self.send(self.t("model", key=r["model_key"]), self.t("model_body", event=r["event_type"], version=ver),
                          "default", "brain")
        if rows:
            self.db.set_state("notify_last_event_id", rows[-1]["id"])
