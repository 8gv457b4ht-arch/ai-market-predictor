"""Data Quality Engine, News AI pipeline (RSS/Atom, rules, LLM adapter, fusion decay) and backups."""
import gzip
import json
import sqlite3

from backend.app.market.candles import load_candles, upsert_candles
from backend.app.market.quality import QualityTracker, evaluate
from backend.app.news import analyzer as an
from backend.app.news.feeds import item_id, parse_feed
from backend.app.news.service import news_features, run_news_once
from backend.app.workers.backup import restore_sqlite, run_backup_once
from tests.helpers import fresh_settings
from tests.synthetic import random_walk_candles

T0 = 1_700_000_000_000


def _candles(db, n=300, drop=()):
    df = random_walk_candles(n, "15m", start_ts=T0)
    df = df[~df.index.isin(drop)]
    upsert_candles(db, "binance", "BTC/USDT", "15m", df.to_dict(orient="records"))
    return load_candles(db, "binance", "BTC/USDT", "15m")


def test_quality_fresh_vs_stale_and_missing(monkeypatch, tmp_path):
    s, db = fresh_settings(monkeypatch, tmp_path)
    c = _candles(db, 300, drop=range(100, 120))
    now = int(c.open_ts.iloc[-1]) + 900_000 + 10_000
    r = evaluate(db, s, "BTC/USDT", "15m", c, now=now)
    codes = {i["code"]: i["severity"] for i in r.issues}
    assert codes["missing_bars"] == "critical" and not r.ok
    s2, db2 = fresh_settings(monkeypatch, tmp_path / "b")
    c2 = _candles(db2, 300)
    now2 = int(c2.open_ts.iloc[-1]) + 900_000 + 10_000
    db2.execute("INSERT INTO stream_status(exchange,symbol,state,last_msg_ms,updated_ms) VALUES('binance','BTC/USDT','connected',?,?)", (now2, now2))
    ok = evaluate(db2, s2, "BTC/USDT", "15m", c2, now=now2)
    assert ok.ok and ok.score >= 0.9, ok.to_dict()
    stale = evaluate(db2, s2, "BTC/USDT", "15m", c2, now=now2 + 3_600_000)
    assert not stale.ok and any(i["code"] == "stale_candles" for i in stale.issues)


def test_quality_exchange_disagreement_spread_and_new_gaps(monkeypatch, tmp_path):
    s, db = fresh_settings(monkeypatch, tmp_path)
    c = _candles(db)
    now = int(c.open_ts.iloc[-1]) + 900_000 + 1000
    for ex, mid, spread in (("binance", 100.0, 1.0), ("bybit", 100.02, 1.0), ("okx", 101.5, 1.0)):
        db.execute("INSERT INTO book_state(exchange,symbol,ts_ms,recv_ms,mid,spread_bps) VALUES(?,?,?,?,?,?)",
                   (ex, "BTC/USDT", now, now, mid, spread))
    db.execute("INSERT INTO stream_status(exchange,symbol,state,last_msg_ms,gaps,updated_ms) VALUES('binance','BTC/USDT','connected',?,0,?)", (now, now))
    tr = QualityTracker()
    r = evaluate(db, s, "BTC/USDT", "15m", c, tr, now=now)
    dis = [i for i in r.issues if i["code"] == "exchange_disagreement"]
    assert dis and dis[0]["severity"] == "warning"  # okx is the outlier, not the primary exchange
    db.execute("UPDATE stream_status SET gaps=3")
    db.execute("UPDATE book_state SET spread_bps=60 WHERE exchange='binance'")
    r2 = evaluate(db, s, "BTC/USDT", "15m", c, tr, now=now)
    codes = {i["code"] for i in r2.issues}
    assert {"ws_gaps", "wide_spread"} <= codes


RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>x</title>
<item><title>Bitcoin ETF sees record inflows as price surges</title><link>https://n.example/a</link>
<description>&lt;p&gt;Spot bitcoin ETFs drew inflows.&lt;/p&gt;</description><pubDate>Tue, 06 Oct 2026 10:00:00 GMT</pubDate></item>
<item><title>Major crypto exchange hacked, $200M drained</title><link>https://n.example/b</link>
<pubDate>Tue, 06 Oct 2026 11:00:00 +0000</pubDate></item></channel></rss>"""
ATOM = b"""<?xml version="1.0" encoding="utf-8"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><title>Fed holds rates, signals rate cut later this year</title><link rel="alternate" href="https://f.example/1"/>
<updated>2026-10-06T18:00:00Z</updated><summary>The FOMC kept the policy rate unchanged.</summary></entry></feed>"""


def test_feed_parsing_rss_and_atom():
    rss = parse_feed(RSS, "n.example")
    assert len(rss) == 2 and rss[0]["summary"] == "Spot bitcoin ETFs drew inflows."
    assert rss[1]["published_ms"] - rss[0]["published_ms"] == 3_600_000
    atom = parse_feed(ATOM, "f.example")
    assert atom[0]["url"] == "https://f.example/1" and atom[0]["published_ms"]
    assert item_id(rss[0]) != item_id(rss[1])


def test_rule_analyzer_structured_events():
    ra = an.RuleAnalyzer()
    items = parse_feed(RSS, "n") + parse_feed(ATOM, "f")
    etf = ra.analyze(items[0], [])
    hack = ra.analyze(items[1], [])
    fed = ra.analyze(items[2], [])
    assert etf["category"] == "etf" and etf["direction"] > 0 and "BTC" in etf["affected_assets"]
    assert hack["category"] == "hacks" and hack["direction"] < 0 and hack["expected_horizon_min"] == 60
    assert fed["category"] in ("central_banks", "interest_rates") and "CRYPTO_MARKET" in fed["affected_assets"]
    for e in (etf, hack, fed):
        for k in ("event", "category", "asset", "direction", "relevance", "novelty", "confidence",
                  "expected_horizon_min", "affected_assets", "summary", "analyzer"):
            assert k in e
        assert e["confidence"] <= 0.6  # rule-based output is never presented as high confidence
    dup = ra.analyze(items[0], [items[0]["title"]])
    assert dup["novelty"] == 0.0


def test_llm_analyzer_parses_and_clamps(monkeypatch, tmp_path):
    s, db = fresh_settings(monkeypatch, tmp_path, ANTHROPIC_API_KEY="test-key")
    llm = an.LLMAnalyzer(s)
    assert llm.enabled and llm.provider == "anthropic"
    reply = "```json\n" + json.dumps({"events": [{"index": 0, "event": "ETF inflows", "category": "etf",
                                                   "asset": "btc", "direction": 3, "relevance": 0.9, "novelty": 0.8,
                                                   "confidence": 0.7, "expected_horizon_min": 1440,
                                                   "affected_assets": ["btc", "eth"], "summary": "Inflows."}]}) + "\n```"
    monkeypatch.setattr(llm, "_complete", lambda user: reply)
    ev = llm.analyze_batch(parse_feed(RSS, "n")[:1], [])[0]
    assert ev["direction"] == 1.0 and ev["asset"] == "BTC" and ev["affected_assets"] == ["BTC", "ETH"]
    assert ev["analyzer"].startswith("llm:anthropic:")


def test_news_worker_falls_back_to_rules_and_dedupes(monkeypatch, tmp_path):
    s, db = fresh_settings(monkeypatch, tmp_path, ANTHROPIC_API_KEY="test-key",
                           NEWS_FEEDS="https://a.example/rss,https://b.example/atom,https://down.example/x")

    def fetch(url):
        if "down" in url:
            raise OSError("unreachable")
        return parse_feed(RSS if "rss" in url else ATOM, url)

    def boom(self, user):
        raise RuntimeError("LLM unavailable")
    monkeypatch.setattr(an.LLMAnalyzer, "_complete", boom)
    st = run_news_once(db, s, fetch=fetch)
    assert st["new_events"] == 3 and st["feeds_ok"] == 2 and "LLM unavailable" in st["llm_error"]
    assert {r["analyzer"] for r in db.query("SELECT analyzer FROM news_events")} == {"rules-v1"}
    assert run_news_once(db, s, fetch=fetch)["new_events"] == 0  # already stored


def test_news_features_freshness_decay(monkeypatch, tmp_path):
    s, db = fresh_settings(monkeypatch, tmp_path)
    now = T0
    for eid, age_min in (("fresh", 10), ("old", 600)):
        db.execute("INSERT INTO news_events(id,published_ms,fetched_ms,title,category,direction,relevance,novelty,"
                   "confidence,horizon_min,affected_assets,analyzer) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                   (eid, now - age_min * 60_000, now, eid, "etf", 1.0, 0.9, 1.0, 0.8, 1440, '["BTC"]', "t"))
    f = news_features(db, "BTC/USDT", now, half_life_min=180)
    top = {t["title"]: t["weight"] for t in f["news_top"]}
    assert top["fresh"] > 4 * top["old"]  # older news weighs much less
    assert 0 < f["news_impact"] < 1 and f["news_relevance"] > 0.8
    assert news_features(db, "ETH/USDT", now, 180)["news_event_count"] == 2  # ETF news -> market-wide at 60% weight
    assert news_features(db, "BTC/USDT", now - 3_600_000 * 24 * 3, 180)["news_event_count"] == 0  # no future leakage


def test_backup_rotation_and_restore(monkeypatch, tmp_path):
    s, db = fresh_settings(monkeypatch, tmp_path, BACKUP_KEEP=2)
    db.set_state("marker", {"v": 1})
    for _ in range(3):
        res = run_backup_once(db, s)
    files = sorted(s.backup_dir.glob("market_*.sqlite3.gz"))
    assert 1 <= len(files) <= 2 and res["database"].endswith(".gz")
    assert db.get_state("last_backup")["database"] == res["database"]
    with gzip.open(files[-1]) as f:
        assert f.read(16).startswith(b"SQLite format 3")
    target = tmp_path / "restored.sqlite3"
    restore_sqlite(files[-1], str(target))
    con = sqlite3.connect(target)
    assert json.loads(con.execute("SELECT value_json FROM system_state WHERE key='marker'").fetchone()[0]) == {"v": 1}
