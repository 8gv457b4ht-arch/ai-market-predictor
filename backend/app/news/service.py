"""News worker logic + freshness-weighted news features for the market snapshot."""
from __future__ import annotations

import json
import logging
import math

from ..config import Settings
from ..db import Database, now_ms
from .analyzer import MACRO, LLMAnalyzer, RuleAnalyzer
from .feeds import fetch_feed, item_id

log = logging.getLogger("news")


def store_event(db: Database, eid: str, item: dict, ev: dict) -> None:
    db.execute(
        "INSERT INTO news_events(id,published_ms,fetched_ms,source,title,url,summary,category,event_type,asset,"
        "direction,relevance,novelty,confidence,horizon_min,affected_assets,analyzer,raw_json) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO NOTHING",
        (eid, item["published_ms"], now_ms(), item.get("source"), item["title"], item.get("url"), ev["summary"],
         ev["category"], ev["event_type"], ev["asset"], ev["direction"], ev["relevance"], ev["novelty"],
         ev["confidence"], ev["expected_horizon_min"], json.dumps(ev["affected_assets"]), ev["analyzer"],
         json.dumps({"event": ev["event"], "published_estimated": item.get("published_estimated", False)})))


def run_news_once(db: Database, settings: Settings, fetch=fetch_feed) -> dict:
    t0 = now_ms()
    feed_status, fresh = {}, []
    for url in settings.news_feeds:
        try:
            items = fetch(url)
            feed_status[url] = {"ok": True, "items": len(items)}
            fresh.extend(items)
        except Exception as exc:  # noqa: BLE001 - one broken feed must not stop the rest
            feed_status[url] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:200]}
            log.warning("feed %s failed: %s", url, exc)
    # drop already-stored and very old items
    cutoff = t0 - int(settings.news_retention_days * 86_400_000)
    seen, new_items = set(), []
    for it in sorted(fresh, key=lambda x: x["published_ms"]):
        eid = item_id(it)
        if eid in seen or it["published_ms"] < cutoff:
            continue
        seen.add(eid)
        if db.query_one("SELECT 1 AS x FROM news_events WHERE id=?", (eid,)) is None:
            new_items.append((eid, it))
    recent = [r["title"] for r in db.query(
        "SELECT title FROM news_events WHERE published_ms >= ? ORDER BY published_ms DESC LIMIT 400",
        (t0 - 48 * 3_600_000,))]

    llm, rules = LLMAnalyzer(settings), RuleAnalyzer()
    analyzed, llm_error = 0, None
    pending = list(new_items)
    if llm.enabled and pending:
        batch = pending[: settings.news_llm_max_items]
        try:
            events = llm.analyze_batch([it for _, it in batch], recent)
            for (eid, it), ev in zip(batch, events):
                store_event(db, eid, it, ev)
                recent.append(it["title"])
            analyzed += len(batch)
            pending = pending[len(batch):]
        except Exception as exc:  # noqa: BLE001 - fall back to rules, keep the error visible
            llm_error = f"{type(exc).__name__}: {exc}"[:300]
            log.warning("LLM analysis failed, falling back to rules: %s", exc)
    for eid, it in pending:
        store_event(db, eid, it, rules.analyze(it, recent))
        recent.append(it["title"])
        analyzed += 1
    ok_feeds = sum(1 for v in feed_status.values() if v["ok"])
    status = {"ts_ms": now_ms(), "feeds_ok": ok_feeds, "feeds_total": len(settings.news_feeds),
              "new_events": analyzed, "analyzer": llm.name if llm.enabled else rules.name,
              "llm_enabled": llm.enabled, "llm_error": llm_error, "feeds": feed_status,
              "duration_ms": now_ms() - t0}
    db.set_state("news_status", status)
    return status


def event_weight(ev: dict, symbol: str, now: int, half_life_min: float) -> tuple[float, float]:
    """(weight, asset relevance) of one stored event for `symbol` at time `now`."""
    base = symbol.split("/")[0].upper()
    affected = ev["affected_assets"] if isinstance(ev["affected_assets"], list) else json.loads(ev["affected_assets"] or "[]")
    if base in affected or (ev.get("asset") or "").upper() == base:
        rel = float(ev["relevance"])
    elif "CRYPTO_MARKET" in affected or ev.get("category") in MACRO or ev.get("category") in ("crypto", "etf", "regulation"):
        rel = float(ev["relevance"]) * 0.6
    else:
        return 0.0, 0.0
    age_min = max(0.0, (now - int(ev["published_ms"])) / 60_000)
    decay = 0.5 ** (age_min / half_life_min)
    if age_min > 2 * float(ev.get("horizon_min") or 360):
        decay *= 0.25  # beyond the expected impact horizon
    return rel * float(ev["confidence"]) * float(ev["novelty"] or 0.5) ** 0.5 * decay, rel * decay


def news_features(db: Database, symbol: str, now: int, half_life_min: float, lookback_h: float = 48) -> dict:
    rows = db.query("SELECT * FROM news_events WHERE published_ms >= ? AND published_ms <= ? ORDER BY published_ms DESC",
                    (now - int(lookback_h * 3_600_000), now))
    total, rel_max, count, top = 0.0, 0.0, 0, []
    for ev in rows:
        w, rel = event_weight(ev, symbol, now, half_life_min)
        if w <= 0:
            continue
        total += w * float(ev["direction"])
        rel_max = max(rel_max, rel)
        if w > 0.02:
            count += 1
            top.append({"title": ev["title"], "weight": round(w, 4), "direction": ev["direction"],
                        "category": ev["category"], "published_ms": ev["published_ms"]})
    top.sort(key=lambda x: -x["weight"])
    return {"news_impact": round(math.tanh(total), 4), "news_relevance": round(rel_max, 4),
            "news_event_count": count, "news_top": top[:5]}
