"""Self-learning (baseline, waiting, challenger vs production on unseen data), model
registry, prediction ledger, gating (quality / news veto) and outcome resolution."""
import json
from pathlib import Path

import numpy as np

from backend.app.learning import engine, registry
from backend.app.learning.predictor import Predictor
from backend.app.ml.features import FEATURE_VERSION
from tests.helpers import fresh_settings
from tests.synthetic import planted_signal_candles, resample

TFS = ["15m", "1h", "4h", "1d"]
FAST = dict(WALK_FORWARD_FOLDS=3, LEARNING_MIN_NEW_LABELS=200, LEARNING_MIN_HOLDOUT=100, MIN_TRAIN_ROWS=600)


def _load_until(db, base, cutoff_ts):
    """Insert all candles that are closed at `cutoff_ts` (base and resampled higher TFs)."""
    from backend.app.config import TIMEFRAME_MS
    from backend.app.market.candles import upsert_candles
    for tf in TFS:
        df = base if tf == "15m" else resample(base, "15m", tf)
        df = df[df.open_ts + TIMEFRAME_MS[tf] <= cutoff_ts]
        upsert_candles(db, "binance", "BTC/USDT", tf, df.to_dict(orient="records"))


def test_compare_models_promotes_only_genuine_improvement():
    from backend.app.config import Settings
    s = Settings()
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 400)
    good = np.full((400, 3), 0.15)
    good[np.arange(400), y] = 0.7
    flat = np.full((400, 3), 1 / 3)
    c = engine.compare_models(y, flat, good, s, 4)
    assert c["promote"] and c["bootstrap_p_better"] > 0.99
    c2 = engine.compare_models(y, good, good, s, 4)  # identical model is NOT an improvement
    assert not c2["promote"] and not c2["checks"]["logloss_gain_ok"]
    c3 = engine.compare_models(y, good, flat, s, 4)
    assert not c3["promote"]


def test_block_bootstrap_probability():
    assert engine.block_bootstrap_prob(np.full(100, 0.1), 4) == 1.0
    assert engine.block_bootstrap_prob(np.full(100, -0.1), 4) == 0.0
    rng = np.random.default_rng(1)
    p = engine.block_bootstrap_prob(rng.normal(0, 1, 500), 4)
    assert 0.05 < p < 0.95


def test_learning_cycle_baseline_wait_and_challenger(monkeypatch, tmp_path):
    s, db = fresh_settings(monkeypatch, tmp_path, **FAST)
    base = planted_signal_candles(2600, "15m")
    cut1 = int(base.open_ts.iloc[1999]) + 900_000
    _load_until(db, base, cut1)
    key = registry.model_key("BTC/USDT", "15m", 4)

    r1 = engine.challenger_cycle(db, s, "BTC/USDT", "15m")
    assert r1["status"] == "baseline_trained"
    prod = registry.get_production(db, key)
    m = prod["metrics"]
    for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "brier", "log_loss", "ece",
              "confusion_matrix", "by_regime", "folds", "baseline_prior", "backtest", "reliability"):
        assert k in m, k
    assert prod["feature_version"] == FEATURE_VERSION and Path(prod["artifact_path"]).exists()
    assert prod["train_end_ts"] < cut1

    r2 = engine.challenger_cycle(db, s, "BTC/USDT", "15m")
    assert r2["status"] == "waiting" and r2["new_labels"] == 0  # no retraining without new data

    cut2 = int(base.open_ts.iloc[2599]) + 900_000
    _load_until(db, base, cut2)
    r3 = engine.challenger_cycle(db, s, "BTC/USDT", "15m")
    assert r3["status"] in ("promoted", "rejected"), r3
    versions = registry.list_versions(db, key)
    cand = [v for v in versions if v["version"] == r3["version"]][0]
    cmp_ = cand["comparison"]
    assert cmp_["n_holdout"] >= 100 and set(cmp_["checks"]) >= {"logloss_gain_ok", "bootstrap_ok", "brier_ok", "accuracy_ok", "both_halves_ok", "no_regime_worse"}
    # the holdout starts after production's training data + purge gap -> unseen by both models
    assert cand["params"]["holdout_start"] > prod["train_end_ts"] + 4 * 900_000
    assert cand["train_end_ts"] < cand["params"]["holdout_start"]
    prods = [v for v in versions if v["status"] == "production"]
    assert len(prods) == 1
    if r3["status"] == "promoted":
        assert prods[0]["version"] == r3["version"] and cmp_["promote"]
    else:
        assert prods[0]["version"] == prod["version"] and cand["status"] == "rejected"

    r4 = engine.challenger_cycle(db, s, "BTC/USDT", "15m")
    assert r4["status"] == "waiting"  # an attempt is not repeated on the same data
    assert db.query("SELECT event_type FROM learning_events")


class StubModel:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.tile(self.p, (len(X), 1))


PASSED_BASELINE = {"baseline_test": {"n": 500, "gain": 0.05, "gain_ci95": [0.02, 0.08], "p_better": 0.99,
                                     "log_loss_model": 0.95, "log_loss_naive": 1.0, "passed": True},
                   "backtest": {"trades": 40, "avg_net_bps": 3.0, "win_rate": 0.55}}


def _stub_production(db, s, p, metrics=None):
    from backend.app.ml.dataset import build_training_frame
    frame, feats = build_training_frame(db, s, "BTC/USDT", "15m")
    key = registry.model_key("BTC/USDT", "15m", 4)
    version = registry.new_version(key)
    path = registry.save_artifact(s.model_dir, version, {"model": StubModel(np.array(p)), "features": feats,
                                                        "feature_version": FEATURE_VERSION, "version": version})
    registry.register(db, version=version, key=key, status="production", train_start_ts=0,
                      train_end_ts=int(frame.open_ts.iloc[-1]), n_train=len(frame), n_validation=0,
                      feature_version=FEATURE_VERSION, features=feats, params={},
                      metrics=PASSED_BASELINE if metrics is None else metrics, comparison=None,
                      artifact_path=path, parent_version=None, reason="test stub")
    return version


def _book(db, now, spread_ok=True):
    db.execute("INSERT INTO book_state(exchange,symbol,ts_ms,recv_ms,best_bid,best_ask,mid,spread_bps,bid_depth,"
               "ask_depth,imbalance,levels_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(exchange,symbol) DO UPDATE "
               "SET recv_ms=excluded.recv_ms, spread_bps=excluded.spread_bps",
               ("binance", "BTC/USDT", now, now, 99.99, 100.01, 100, 2.0 if spread_ok else 80.0, 5, 4, 0.11, "{}"))
    db.execute("INSERT INTO stream_status(exchange,symbol,state,last_msg_ms,updated_ms) VALUES(?,?,?,?,?) "
               "ON CONFLICT(exchange,symbol) DO UPDATE SET last_msg_ms=excluded.last_msg_ms, state=excluded.state",
               ("binance", "BTC/USDT", "connected", now, now))


def test_predictor_ledger_snapshot_gating_and_resolution(monkeypatch, tmp_path):
    s, db = fresh_settings(monkeypatch, tmp_path, **FAST)
    base = planted_signal_candles(1500, "15m")
    cut = int(base.open_ts.iloc[1399]) + 900_000
    _load_until(db, base, cut)
    v = _stub_production(db, s, [0.1, 0.1, 0.8])
    pr = Predictor(db, s)
    now = cut + 5_000
    _book(db, now)

    out = pr.predict("BTC/USDT", "15m", now=now)
    assert out["status"] == "predicted" and out["prediction"] == "UP", out
    assert out["model_version"] == v
    assert pr.predict("BTC/USDT", "15m", now=now)["status"] == "already_predicted"
    row = db.query_one("SELECT * FROM predictions")
    snap = json.loads(row["snapshot_json"])
    for k in ("timestamp", "exchange", "symbol", "timeframe", "price", "bid", "ask", "spread_bps", "volume",
              "cvd_live", "orderbook_imbalance", "news_impact", "news_relevance", "regime", "features", "freshness"):
        assert k in snap, k
    assert snap["freshness"]["weights"]["order_book"] > 0.5 and snap["features"]
    assert row["features_version"] == FEATURE_VERSION and row["target_ts"] == row["candle_ts"] + 4 * 900_000

    # outcome resolution once the horizon has passed
    cut2 = int(base.open_ts.iloc[1409]) + 900_000
    _load_until(db, base, cut2)
    assert pr.resolve("BTC/USDT", "15m", now=cut2 + 1000) == 1
    row = db.query_one("SELECT * FROM predictions")
    assert row["actual_direction"] in ("UP", "DOWN", "FLAT") and row["result"] in ("correct", "wrong")
    assert row["error_class"] in ("correct", "wrong_direction", "false_move")
    target = base[base.open_ts == row["target_ts"]].close.iloc[0]
    assert abs(row["actual_price"] - target) < 1e-9

    # strongly negative, fresh, relevant news vetoes an UP signal (news never creates one)
    db.execute("DELETE FROM predictions")
    db.execute("INSERT INTO news_events(id,published_ms,fetched_ms,title,category,direction,relevance,novelty,confidence,"
               "horizon_min,affected_assets,analyzer) VALUES('n1',?,?,'Major exchange hacked','hacks',-1,0.95,1,0.95,60,"
               "'[\"BTC\"]','test')", (cut2 - 60_000, cut2))
    _book(db, cut2 + 1000)
    out = pr.predict("BTC/USDT", "15m", now=cut2 + 1000)
    assert out["prediction"] == "NO TRADE" and "news_conflict" in out["reasons"], out

    # stale market data -> data quality failure -> NO TRADE
    db.execute("DELETE FROM predictions")
    db.execute("DELETE FROM news_events")
    late = cut2 + 3 * 3_600_000
    _book(db, late)
    out = pr.predict("BTC/USDT", "15m", now=late)
    assert out["prediction"] == "NO TRADE" and "data_quality" in out["reasons"] and "late_decision" in out["reasons"]

    # weak model -> NO TRADE (confidence gate)
    db.execute("DELETE FROM predictions")
    _stub_production(db, s, [0.34, 0.33, 0.33])
    registry.promote(db, registry.model_key("BTC/USDT", "15m", 4),
                     db.query_one("SELECT version FROM model_registry ORDER BY created_ms DESC LIMIT 1")["version"], "t")
    _book(db, cut2 + 1000)
    out = pr.predict("BTC/USDT", "15m", now=cut2 + 1000)
    assert out["prediction"] == "NO TRADE" and "confidence_below_threshold" in out["reasons"]
    assert out["gate"]["confidence"] < out["gate"]["threshold"]


def test_no_model_means_no_prediction(monkeypatch, tmp_path):
    s, db = fresh_settings(monkeypatch, tmp_path)
    assert Predictor(db, s).predict("BTC/USDT", "15m")["status"] == "model_not_ready"
    assert db.scalar("SELECT COUNT(*) FROM predictions") == 0


def test_procedure_upgrade_is_a_head_to_head_out_of_sample_comparison(monkeypatch, tmp_path):
    s, db = fresh_settings(monkeypatch, tmp_path, **FAST)
    base = planted_signal_candles(2200, "15m")
    _load_until(db, base, int(base.open_ts.iloc[-1]) + 900_000)
    key = registry.model_key("BTC/USDT", "15m", 4)
    assert engine.challenger_cycle(db, s, "BTC/USDT", "15m")["status"] == "baseline_trained"
    prod = registry.get_production(db, key)
    params = dict(prod["params"], procedure="old-procedure")  # pretend production came from an older procedure
    db.execute("UPDATE model_registry SET params_json=? WHERE version=?", (json.dumps(params), prod["version"]))
    r = engine.challenger_cycle(db, s, "BTC/USDT", "15m")
    assert r["status"] in ("promoted", "rejected") and r["comparison"]["n_holdout"] >= 100
    assert engine.challenger_cycle(db, s, "BTC/USDT", "15m")["status"] == "waiting"  # checked once, not repeated
    assert len([v for v in registry.list_versions(db, key) if v["status"] == "production"]) == 1
