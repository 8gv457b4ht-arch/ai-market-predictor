"""Scheduled-backend cycle end to end with SYNTHETIC exchanges (test only): one exchange region-blocked,
the others answering REST + WebSocket. Checks primary selection, history download, verified-live
streams, model training, prediction with gate details, export files, notifications and persistence."""
import asyncio
import json
import time

from backend.app.config import TIMEFRAME_MS
from backend.app.exchanges import rest
from tests.helpers import fresh_settings
from tests.synthetic import planted_signal_candles, resample


def _series():
    now = int(time.time() * 1000)
    tf = TIMEFRAME_MS["15m"]
    start = now - now % 86_400_000 - 30 * 86_400_000  # day-aligned so resampling is exact
    n = (now - start) // tf + 1                         # last bar is the one still forming
    base = planted_signal_candles(int(n), "15m", seed=3, start_ts=start)
    out = {"15m": base}
    for t in ("1h", "4h", "1d"):
        out[t] = resample(base, "15m", t)
    for t, df in out.items():  # the bar containing "now" is not closed yet
        df["closed"] = (df.open_ts + TIMEFRAME_MS[t] <= now).astype(int)
    return {t: df.to_dict(orient="records") for t, df in out.items()}


SER = _series()


def fake_klines(exchange, symbol, tf, limit=1000, end_ms=None):
    if exchange == "binance":
        raise rest.ExchangeError("HTTP 451 from api.binance.com: restricted location", "region_blocked", 451, "api.binance.com")
    rows = [dict(r) for r in SER[tf] if end_ms is None or r["open_ts"] <= end_ms]
    return rows[-limit:]


def fake_history(exchange, symbol, tf, bars, pause=0.2):
    return fake_klines(exchange, symbol, tf, bars)


def fake_ticker(exchange, symbol):
    if exchange == "binance":
        raise rest.ExchangeError("HTTP 451", "region_blocked", 451, "api.binance.com")
    return {"last": SER["15m"][-1]["close"], "change_24h_pct": 1.0, "volume_24h": 1.0, "quote_volume_24h": 1.0,
            "high_24h": 1.0, "low_24h": 1.0, "ts_ms": int(time.time() * 1000)}


class FakeWS:
    def __init__(self, exchange):
        self.ex, self.n = exchange, 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def send(self, m):
        pass

    async def recv(self):
        await asyncio.sleep(0.05)
        self.n += 1
        t = int(time.time() * 1000)
        px = SER["15m"][-1]["close"]
        if self.ex == "okx":
            if self.n % 2:
                return json.dumps({"arg": {"channel": "trades", "instId": "BTC-USDT"}, "data": [
                    {"instId": "BTC-USDT", "tradeId": str(self.n), "px": f"{px:.2f}", "sz": "0.1", "side": "buy", "ts": str(t)}]})
            return json.dumps({"arg": {"channel": "books5", "instId": "BTC-USDT"}, "data": [
                {"bids": [[f"{px - 1:.2f}", "2", "0", "1"]], "asks": [[f"{px + 1:.2f}", "1", "0", "1"]], "ts": str(t), "seqId": self.n}]})
        if self.n % 2:
            return json.dumps({"topic": "publicTrade.BTCUSDT", "ts": t, "data": [
                {"T": t, "s": "BTCUSDT", "S": "Sell", "v": "0.2", "p": f"{px:.2f}", "i": str(self.n)}]})
        return json.dumps({"topic": "orderbook.50.BTCUSDT", "type": "snapshot", "ts": t, "data": {
            "s": "BTCUSDT", "b": [[f"{px - 1.2:.2f}", "1"]], "a": [[f"{px + 1.2:.2f}", "1"]], "u": self.n, "seq": self.n}})


def fake_connect(url, **kw):
    if "binance" in url:
        raise OSError("HTTP 451: restricted location")
    return FakeWS("okx" if "okx" in url else "bybit")


def test_full_cycle_with_one_blocked_exchange(monkeypatch, tmp_path):
    s, db = fresh_settings(monkeypatch, tmp_path, PRIMARY_EXCHANGE="auto", SYMBOLS="BTC/USDT", PREDICT_TIMEFRAMES="15m",
                           TIMEFRAMES="15m,1h,4h,1d", HISTORY_BARS=2400, CONTEXT_HISTORY_BARS=800, WS_SAMPLE_SEC=3,
                           WALK_FORWARD_FOLDS=3, MIN_TRAIN_ROWS=600, PUBLIC_DIR=tmp_path / "public",
                           NTFY_TOPIC="test-topic", NOTIFY_EVENTS="signals,all_predictions,outages,promotions",
                           NOTIFY_MAX_AGE_SEC=3600)  # the synthetic last close can be up to 15 min old
    monkeypatch.setattr(rest, "fetch_klines", fake_klines)
    monkeypatch.setattr(rest, "fetch_history", fake_history)
    monkeypatch.setattr(rest, "fetch_ticker", fake_ticker)
    sent = []
    from backend.app import notify
    monkeypatch.setattr(notify, "_post", lambda url, data, headers: sent.append((url, data.decode(), headers)))
    from backend.app.cloud.cycle import run_cycle

    r = run_cycle(db, s, connect=fake_connect, news=False)
    assert r["primary_exchange"] == "okx", r.get("errors")
    assert r["rest"]["binance"]["kind"] == "region_blocked" and r["rest"]["okx"]["ok"]
    ws = r["ws"]["streams"]
    assert ws["okx"]["verified_live"] and ws["bybit"]["verified_live"] and not ws["binance"]["verified_live"]
    assert ws["binance"]["kind"] == "region_blocked"
    assert r["learning"]["BTC/USDT|15m|h4"]["status"] == "baseline_trained"
    assert not r["errors"], r["errors"]

    # second run: model exists -> a prediction on the latest closed candle with full gate details
    r2 = run_cycle(db, s, connect=fake_connect, news=False)
    pred = r2["predictions"]["BTC/USDT|15m"]
    assert pred["status"] in ("predicted", "already_predicted"), pred
    row = db.query_one("SELECT * FROM predictions")
    assert row is not None and row["exchange"] == "okx"
    gate = json.loads(row["gate_json"])
    for k in ("confidence", "threshold", "edge", "min_edge", "p_flat", "quality_score", "quality_issues", "decision_delay_sec"):
        assert k in gate, k
    snap = json.loads(row["snapshot_json"])
    assert snap["bid"] and snap["ask"] and snap["spread_bps"] > 0  # from the live okx book sample

    st = json.loads((tmp_path / "public" / "state.json").read_text())
    assert st["primary_exchange"] == "okx" and st["probes"]["probes"]["binance"]["rest"]["kind"] == "region_blocked"
    m = st["models"]["BTC/USDT|15m|h4"]["production"]["metrics"]
    assert "gate_diagnostics" in m and "label_diagnostics" in m and m["gate_diagnostics"]["first_failing_condition"]
    assert st["latest_predictions"]["BTC/USDT|15m|h4"]["gate"]["threshold"] == s.confidence_threshold
    led = json.loads((tmp_path / "public" / "ledger.json").read_text())
    assert led["predictions"] and "features" not in led["predictions"][0]
    assert (tmp_path / "public" / "candles_BTC-USDT_15m.json").exists()
    assert "API_KEY" not in json.dumps(st) and "test-topic" not in json.dumps(st)  # no secrets in public files

    msgs = [json.loads(d) for _, d, _ in sent]
    titles = [m["title"] for m in msgs]
    assert any("Обновление модели" in t for t in titles) and any("BTC/USDT 15m" in t for t in titles)  # Russian by default
    assert all(u == "https://ntfy.sh" for u, _, _ in sent) and all(m["topic"] == "test-topic" for m in msgs)
    body = next(m["message"] for m in msgs if "BTC/USDT 15m" in m["title"])
    assert "Рост" in body and "Качество данных" in body and "не торговая рекомендация" in body
    n_before = len(sent)
    run_cycle(db, s, connect=fake_connect, news=False, learn=False)
    assert not any("BTC/USDT 15m" in json.loads(d)["title"] for _, d, _ in sent[n_before:])  # each prediction notified once


def test_schema_migration_adds_columns(tmp_path):
    import sqlite3
    from backend.app.db import Database
    path = tmp_path / "old.sqlite3"
    from backend.app.db import SCHEMA
    v1 = [x for x in SCHEMA if "CREATE TABLE IF NOT EXISTS predictions" in x][0]
    con = sqlite3.connect(path)
    con.execute(v1.replace("gate_json TEXT, ", "").replace("{pk}", "INTEGER PRIMARY KEY AUTOINCREMENT"))
    con.commit()
    con.close()
    db = Database(f"sqlite:///{path}")
    db.init_schema()
    assert "gate_json" in db.columns("predictions")


def test_cycle_waits_for_candle_close_and_records_health(monkeypatch, tmp_path):
    """Scheduled mode with WAIT_CLOSE_MAX_SEC: streams stay open until the (simulated) close, then the newest
    bars are fetched and predictions made; source checks, health and gaps are exported (SYNTHETIC)."""
    s, db = fresh_settings(monkeypatch, tmp_path, PRIMARY_EXCHANGE="auto", SYMBOLS="BTC/USDT", PREDICT_TIMEFRAMES="15m",
                           TIMEFRAMES="15m,1h,4h,1d", HISTORY_BARS=2400, CONTEXT_HISTORY_BARS=800, WS_SAMPLE_SEC=2,
                           WALK_FORWARD_FOLDS=3, MIN_TRAIN_ROWS=600, PUBLIC_DIR=tmp_path / "public")
    monkeypatch.setattr(rest, "fetch_klines", fake_klines)
    monkeypatch.setattr(rest, "fetch_history", fake_history)
    monkeypatch.setattr(rest, "fetch_ticker", fake_ticker)
    from backend.app.cloud import cycle
    cycle.run_cycle(db, s, connect=fake_connect, news=False)  # trains
    db.execute("DELETE FROM predictions")
    monkeypatch.setattr(cycle, "next_close_wait", lambda settings, now=None: (3.0, 1))
    r = cycle.run_cycle(db, s, connect=fake_connect, news=False, learn=False)
    assert not r["errors"], r["errors"]
    assert r["wait_for_close"]["wait_sec"] == 3.0 and r["timings_sec"]["ws_close"] >= 2.5
    assert r["predictions_at_close"]["BTC/USDT|15m"]["status"] in ("predicted", "already_predicted")
    assert db.scalar("SELECT COUNT(*) FROM source_checks WHERE channel='ws'") == 9  # 3 exchanges x 3 stream samples
    st = json.loads((tmp_path / "public" / "state.json").read_text())
    assert st["export_version"] == 2 and st["health"]["database"]["ok"] and st["health"]["last_prediction_ms"]
    assert st["source_stats"]["binance"]["rest"]["24h"]["error_rate"] == 1.0
    assert st["source_stats"]["okx"]["ws"]["24h"]["error_rate"] == 0.0
    m = st["models"]["BTC/USDT|15m|h4"]
    assert m["baseline_test"] and "passed" in m["baseline_test"] and "gaps" in m
    assert st["health"]["notifications"]["channels"] == {"ntfy": False, "telegram": False}
