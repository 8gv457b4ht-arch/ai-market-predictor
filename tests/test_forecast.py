"""Multi-horizon forecasts (SYNTHETIC data, test only): append-only ledger, no look-ahead, exact outcome
prices, reproducibility, per-horizon training/validation and the compute scheduler."""
import json
import time

import numpy as np
import pytest

from backend.app.config import TIMEFRAME_MS
from tests.helpers import fresh_settings
from tests.synthetic import fill_db, planted_signal_candles


def _db_with_candles(monkeypatch, tmp_path, n=3000, **kw):
    s, db = fresh_settings(monkeypatch, tmp_path, **kw)
    base = planted_signal_candles(n, "15m", seed=5)
    fill_db(db, "BTC/USDT", "binance", base, "15m", ["15m", "1h", "4h", "1d"])
    return s, db, base


def _seconds(db, start_sec, n, price=100.0, step=0.01, exchange="binance", symbol="BTC/USDT"):
    rows = []
    p = price
    rng = np.random.default_rng(1)
    for i in range(n):
        p *= 1 + rng.normal(0, 2e-4)
        ts = start_sec + i
        rows.append((exchange, symbol, ts, p, p, p, p, 1.0, 0.5, 3, ts * 1000 + 500))
    db.executemany("INSERT INTO price_seconds(exchange,symbol,ts_sec,open,high,low,close,buy_volume,sell_volume,trades,last_trade_ms) "
                   "VALUES(?,?,?,?,?,?,?,?,?,?,?)", rows)
    return rows


def test_ledgers_are_append_only(monkeypatch, tmp_path):
    s, db = fresh_settings(monkeypatch, tmp_path)
    db.execute("INSERT INTO forecasts(forecast_id,created_ms,data_ts_ms,symbol,exchange,horizon_sec,target_ts,ref_price,ref_source,"
               "p_up,p_down,p_flat,cost_bps,decision,reasons,input_hash) VALUES('f1',1000,900,'BTC/USDT','binance',60,61000,100,'trade',"
               "0.2,0.2,0.6,25,'NO TRADE','[]','h')")
    with pytest.raises(Exception, match="append-only"):
        db.execute("UPDATE forecasts SET p_up=0.9 WHERE forecast_id='f1'")
    with pytest.raises(Exception, match="append-only"):
        db.execute("UPDATE forecasts SET decision='LONG' WHERE forecast_id='f1'")
    db.execute("UPDATE forecasts SET resolved_ms=70000, actual_bps=3.0, resolution_source='trade' WHERE forecast_id='f1'")
    with pytest.raises(Exception, match="append-only"):  # the outcome is written exactly once
        db.execute("UPDATE forecasts SET actual_bps=99 WHERE forecast_id='f1'")
    db.execute("INSERT INTO predictions(created_ms,candle_ts,target_ts,symbol,exchange,timeframe,horizon_bars,price,prediction,"
               "model_direction,p_up,p_down,p_flat,confidence,label_threshold,gate_reasons,features_version,model_version) "
               "VALUES(1,0,1,'BTC/USDT','binance','15m',4,100,'NO TRADE','FLAT',0.2,0.2,0.6,0.2,0.0025,'[]','fv','v')")
    with pytest.raises(Exception, match="append-only"):
        db.execute("UPDATE predictions SET prediction='UP'")


def test_second_bars_from_trades_and_late_trades(monkeypatch, tmp_path):
    s, db = fresh_settings(monkeypatch, tmp_path)
    from backend.app.market.aggregator import Aggregator
    agg = Aggregator(db, store_raw_trades=False, second_bar_exchanges={"binance"})
    t0 = (int(time.time()) - 20) * 1000
    for i, (dt, px, side) in enumerate([(100, 10.0, "buy"), (400, 11.0, "sell"), (900, 10.5, "buy"), (1200, 12.0, "buy")]):
        agg.on_event({"kind": "trade", "exchange": "binance", "symbol": "BTC/USDT", "price": px, "qty": 1.0, "side": side,
                      "trade_id": i + 1, "ts_ms": t0 + dt}, recv_ms=t0 + dt + 50)
    agg.on_event({"kind": "trade", "exchange": "okx", "symbol": "BTC/USDT", "price": 1.0, "qty": 1.0, "side": "buy",
                  "trade_id": "x", "ts_ms": t0 + 100}, recv_ms=t0 + 150)  # not the model exchange: not stored
    agg.flush(now=t0 + 3000)
    rows = db.query("SELECT * FROM price_seconds ORDER BY ts_sec")
    assert [(r["exchange"], r["open"], r["high"], r["low"], r["close"], r["trades"]) for r in rows] == [
        ("binance", 10.0, 11.0, 10.0, 10.5, 3), ("binance", 12.0, 12.0, 12.0, 12.0, 1)]
    assert rows[0]["buy_volume"] == 2.0 and rows[0]["sell_volume"] == 1.0 and rows[0]["last_trade_ms"] == t0 + 900
    # a trade that arrives 10 s late must not overwrite the already written second
    agg.on_event({"kind": "trade", "exchange": "binance", "symbol": "BTC/USDT", "price": 99.0, "qty": 1.0, "side": "buy",
                  "trade_id": 9, "ts_ms": t0 + 200}, recv_ms=t0 + 10_200)
    agg.flush(now=t0 + 11_000)
    assert db.query_one("SELECT close FROM price_seconds WHERE ts_sec=?", (t0 // 1000,))["close"] == 10.5


def test_grid_training_validation_and_forecast_without_lookahead(monkeypatch, tmp_path):
    s, db, base = _db_with_candles(monkeypatch, tmp_path)
    from backend.app.forecast.engine import Forecaster
    from backend.app.forecast.horizons import BY_SECONDS
    from backend.app.forecast.train import train_one
    h = BY_SECONDS[3600]  # 1 hour, learned on 15-minute candles
    st = train_one(db, s, "BTC/USDT", h)
    assert st["status"] in ("validated", "not_better", "collecting") and st["holdout"]["n"] > 300
    ho = st["holdout"]
    for k in ("brier", "brier_base", "gain", "gain_ci95", "p_better", "direction_hit", "mae_bps", "mae_rw_bps",
              "coverage_10_90", "signals", "n_independent", "protocol"):
        assert k in ho, k
    # forecast "now" = right after a bar in the middle of the data, then add data from the future and repeat
    asof = int(base.open_ts.iloc[2000]) + 900_000 + 5_000
    db.execute("INSERT INTO price_seconds(exchange,symbol,ts_sec,open,high,low,close,buy_volume,sell_volume,trades,last_trade_ms) "
               "VALUES('binance','BTC/USDT',?,1,1,1,?,1,1,1,?)", (asof // 1000 - 1, float(base.close.iloc[2000]), asof - 1500))
    f1 = Forecaster(db, s)
    fid = f1.issue("BTC/USDT", h, asof)
    r1 = db.query_one("SELECT * FROM forecasts WHERE forecast_id=?", (fid,))
    assert r1["data_ts_ms"] <= r1["created_ms"] and r1["target_ts"] == asof + 3600_000
    assert r1["q10_bps"] <= r1["q50_bps"] <= r1["q90_bps"] and abs(r1["p_up"] + r1["p_down"] + r1["p_flat"] - 1) < 1e-9
    # wild future candles must not change anything about a forecast made at `asof`
    db.execute("UPDATE candles SET close=close*3, high=high*3 WHERE timeframe='15m' AND open_ts > ?", (asof - 900_000,))
    r2 = Forecaster(db, s).compute("BTC/USDT", h, asof)
    assert r2["features_json"] == r1["features_json"] and r2["p_up"] == r1["p_up"] and r2["q50_bps"] == r1["q50_bps"]
    assert r2["input_hash"] == r1["input_hash"]
    # reproducible from the stored snapshot
    rep = f1.reproduce(fid)
    assert rep["same"] and rep["input_hash_ok"]


def test_resolution_uses_only_prices_at_or_before_the_target(monkeypatch, tmp_path):
    s, db = fresh_settings(monkeypatch, tmp_path)
    from backend.app.forecast.engine import Forecaster
    from backend.app.forecast.horizons import BY_SECONDS
    f = Forecaster(db, s)
    t = 1_791_500_000  # seconds
    db.executemany("INSERT INTO price_seconds(exchange,symbol,ts_sec,open,high,low,close,buy_volume,sell_volume,trades,last_trade_ms) "
                   "VALUES('binance','BTC/USDT',?,?,?,?,?,1,1,1,?)",
                   [(t, 100, 100, 100, 100, t * 1000 + 100), (t + 5, 101, 101, 101, 101, t * 1000 + 5_300),
                    (t + 6, 105, 105, 105, 105, t * 1000 + 6_100)])
    h = BY_SECONDS[5]
    target = t * 1000 + 5_200  # the 101 trade happened 100 ms AFTER the target -> must not be used
    price, src, lag = f.price_at("binance", "BTC/USDT", target, h)
    assert (price, src, lag) == (100.0, "trade", 5_100)  # the last trade at or before the target
    price, src, lag = f.price_at("binance", "BTC/USDT", t * 1000 + 5_900, h)
    assert (price, src, lag) == (101.0, "trade", 600)


def test_issue_resolve_stats_cycle_on_seconds(monkeypatch, tmp_path):
    s, db = fresh_settings(monkeypatch, tmp_path)
    from backend.app.forecast.engine import Forecaster
    from backend.app.forecast.horizons import BY_SECONDS
    from backend.app.forecast.train import train_one
    h = BY_SECONDS[15]
    now_sec = int(time.time())
    _seconds(db, now_sec - 2 * 3600, 2 * 3600)  # two hours of live 1-second data
    st = train_one(db, s, "BTC/USDT", h)
    assert st["status"] in ("collecting", "not_better", "validated"), st
    assert st.get("moves_beyond_costs_rare") is True  # 15 s moves never exceed 25 bps of costs here
    f = Forecaster(db, s)
    asof = (now_sec - 600) * 1000 + 200
    fid = f.issue("BTC/USDT", h, asof)
    assert fid, f.missing
    r = db.query_one("SELECT * FROM forecasts WHERE forecast_id=?", (fid,))
    assert r["ref_source"] == "trade"
    assert {"no_edge_after_costs", "model_not_validated", "model_not_better_than_baseline"} & set(json.loads(r["reasons"]))
    assert r["decision"] == "NO TRADE"
    res = f.resolve(now_sec * 1000)
    assert res["resolved"] == 1
    r = db.query_one("SELECT * FROM forecasts WHERE forecast_id=?", (fid,))
    tgt = db.query_one("SELECT close FROM price_seconds WHERE ts_sec=? ", ((asof + 15_000) // 1000 - 1,))
    assert r["resolution_source"] == "trade" and abs(r["actual_price"] - tgt["close"]) < 1e-12
    stats = f.stats(now_sec * 1000)["BTC/USDT|f15"]
    assert stats["resolved"] == 1 and stats["live"]["n"] == 1 and not stats["live"]["sufficient"]


def test_missing_outcome_is_recorded_not_guessed(monkeypatch, tmp_path):
    s, db = fresh_settings(monkeypatch, tmp_path)
    from backend.app.forecast.engine import Forecaster
    db.execute("INSERT INTO forecasts(forecast_id,created_ms,data_ts_ms,symbol,exchange,horizon_sec,target_ts,ref_price,ref_source,"
               "p_up,p_down,p_flat,cost_bps,q10_bps,q50_bps,q90_bps,decision,reasons,input_hash,base_p_up,base_p_down,base_p_flat) "
               "VALUES('m1',0,0,'BTC/USDT','binance',60,60000,100,'trade',0.1,0.1,0.8,25,-5,0,5,'NO TRADE','[]','h',0.1,0.1,0.8)")
    f = Forecaster(db, s)
    assert f.resolve(30 * 60_000) == {"resolved": 0, "missing": 0}  # still waiting
    assert f.resolve(3 * 3600_000) == {"resolved": 0, "missing": 1}
    assert db.query_one("SELECT resolution_source FROM forecasts")["resolution_source"] == "missing"


def test_scheduler_trains_most_promising_first_within_budget(monkeypatch, tmp_path):
    s, db = fresh_settings(monkeypatch, tmp_path)
    from backend.app.forecast import train as T
    from backend.app.forecast.horizons import BY_SECONDS, model_key
    db.set_state(T.state_key(model_key("BTC/USDT", BY_SECONDS[3600])), {"status": "validated", "holdout": {"gain": 0.02}, "last_train_ms": 1})
    db.set_state(T.state_key(model_key("BTC/USDT", BY_SECONDS[60])), {"status": "collecting", "last_train_ms": 1})
    db.set_state(T.state_key(model_key("BTC/USDT", BY_SECONDS[7200])), {"status": "degraded", "last_train_ms": int(time.time() * 1000)})
    order = []
    monkeypatch.setattr(T, "train_one", lambda db_, s_, sym, h: order.append(h.label) or {"status": "x", "train_sec": 0})
    out = T.run_training(db, s, budget_sec=1000)
    assert order[0] == "2h"            # degraded: re-validated first, even though it was trained just now
    assert order.index("1h") < order.index("1m")
    # an empty budget postpones everything but the first job, criteria untouched
    order.clear()
    out = T.run_training(db, s, budget_sec=-1)
    assert len(order) == 0 and out["postponed"]


def test_horizon_without_model_is_retried_soon(monkeypatch, tmp_path):
    """A horizon that could not be issued (model trained later in the same run) must not wait a full refresh."""
    s, db = fresh_settings(monkeypatch, tmp_path)
    from backend.app.forecast.engine import Forecaster
    f = Forecaster(db, s)
    calls = []
    monkeypatch.setattr(f, "issue", lambda sym, h, now: calls.append((h.label, now)) or None)
    f.tick(1_000_000)
    n = len(calls)
    f.tick(1_000_000 + 31_000)
    assert len(calls) == 2 * n  # every horizon tried again after 30 s, including 1 d (refresh 1 h)
