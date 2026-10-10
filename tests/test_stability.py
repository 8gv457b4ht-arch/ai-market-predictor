"""Fault tolerance and recovery (SYNTHETIC data, test only): source-check history and error rates,
candle-gap repair, damaged-database restore, model rollback (automatic and by operator command),
model-vs-baseline gate, waiting for the candle close, notification conditions/staleness/dedupe,
missed-candle recording and time-consistent labels."""
import gzip
import json
import time

import joblib
import numpy as np
import pandas as pd

from backend.app.config import TIMEFRAME_MS
from backend.app.exchanges import rest
from tests.helpers import fresh_settings
from tests.synthetic import fill_db, planted_signal_candles, random_walk_candles


class ConstModel:
    """Picklable stand-in for a trained model (fixed probabilities)."""
    def __init__(self, p):
        self.p = np.asarray(p, dtype=float)

    def predict_proba(self, X):
        return np.tile(self.p, (len(X), 1))


def _register(db, s, version, status, metrics=None, artifact=True):
    from backend.app.learning import registry
    path = None
    if artifact:
        path = registry.save_artifact(s.model_dir, version, {"model": ConstModel([0.3, 0.4, 0.3]), "features": ["a"]})
    registry.register(db, version=version, key="BTC/USDT|15m|h4", status=status, train_start_ts=0, train_end_ts=1,
                      n_train=10, n_validation=0, feature_version="x", features=["a"], params={},
                      metrics=metrics or {"log_loss": 1.0}, comparison=None, artifact_path=path, parent_version=None,
                      reason="test")
    return path


def test_source_checks_history_and_error_rates(monkeypatch, tmp_path):
    s, db = fresh_settings(monkeypatch, tmp_path)
    from backend.app.cloud.cycle import record_source_checks, source_stats
    for i in range(4):
        record_source_checks(
            db, {"binance": {"ok": False, "kind": "region_blocked", "error": "HTTP 451"},
                 "okx": {"ok": True, "candles": {"BTC/USDT 15m": {"ok": True, "latency_ms": 120}}, "host": "https://www.okx.com"}},
            {"okx": {"verified_live": i != 0, "kind": None if i else "timeout", "symbols": {"BTC/USDT": {"clock_skew_ms": -80}}}})
    st = source_stats(db)
    assert st["binance"]["rest"]["1h"] == {"checks": 4, "ok": 0, "error_rate": 1.0}
    assert st["binance"]["rest"]["top_error_24h"] == "region_blocked"
    assert st["okx"]["rest"]["24h"]["error_rate"] == 0.0 and st["okx"]["ws"]["1h"]["error_rate"] == 0.25
    row = db.query_one("SELECT * FROM source_checks WHERE exchange='okx' AND channel='ws' AND ok=1")
    assert row["skew_ms"] == 80 and st["okx"]["ws"]["last_ok_ms"]


def test_missing_candles_are_detected_and_refetched(monkeypatch, tmp_path):
    s, db = fresh_settings(monkeypatch, tmp_path)
    from backend.app.cloud.cycle import sync_candles
    from backend.app.cloud.health import candle_gaps
    from backend.app.market.candles import upsert_candles
    bar = TIMEFRAME_MS["15m"]
    now = int(time.time() * 1000) // bar * bar
    rows = [{"open_ts": now - (600 - i) * bar, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1.0,
             "quote_volume": 100.0, "taker_buy_volume": 0.5, "trades": 3, "closed": 1} for i in range(600)]
    upsert_candles(db, "okx", "BTC/USDT", "15m", [r for i, r in enumerate(rows) if not 200 <= i < 230])  # outage hole
    assert candle_gaps(db, "okx", "BTC/USDT", "15m", 600) == [(rows[200]["open_ts"], 30)]
    monkeypatch.setattr(rest, "fetch_klines", lambda ex, sym, tf, limit=1000, end_ms=None: rows[-limit:])
    monkeypatch.setattr(rest, "fetch_history", lambda ex, sym, tf, bars, pause=0.2: rows[-bars:])
    r = sync_candles(db, s, "okx", "BTC/USDT", "15m", 600)
    assert r["gaps_found"] == 30 and r["gaps_remaining"] == 0
    assert db.scalar("SELECT COUNT(*) FROM candles WHERE exchange='okx'") == 600
    # duplicates are impossible: the primary key upserts the same bar
    upsert_candles(db, "okx", "BTC/USDT", "15m", rows[-10:])
    assert db.scalar("SELECT COUNT(*) FROM candles WHERE exchange='okx'") == 600


def test_damaged_database_is_restored_from_newest_good_backup(monkeypatch, tmp_path):
    s, db = fresh_settings(monkeypatch, tmp_path)
    from backend.app.cloud.health import ensure_database, sqlite_check
    from backend.app.db import Database
    from backend.app.workers.backup import run_backup_once
    db.set_state("marker", "before-backup")
    db.execute("INSERT INTO source_checks(ts_ms,exchange,channel,ok) VALUES(1,'okx','rest',1)")
    res = run_backup_once(db, s)
    assert res["database"].endswith(".sqlite3.gz")
    # a newer but broken backup must be skipped
    (s.backup_dir / "market_29990101T000000Z.sqlite3.gz").write_bytes(gzip.compress(b"not a database" * 100))
    db.conn.close()
    path = tmp_path / "test.sqlite3"
    raw = bytearray(path.read_bytes())
    raw[:100] = b"\x00" * 100  # destroy the header
    path.write_bytes(bytes(raw))
    for suffix in ("-wal", "-shm"):
        (tmp_path / f"test.sqlite3{suffix}").unlink(missing_ok=True)
    assert sqlite_check(path) != "ok"
    out = ensure_database(path, s.backup_dir)
    assert out["status"] == "damaged" and out["restored"] and out["backup"] == res["database"], out
    assert [t["result"] != "ok" for t in out["tried"]][0]  # the newest (broken) one was tried and rejected
    assert sqlite_check(path, full=True) == "ok"
    assert list(tmp_path.glob("test.sqlite3.damaged-*"))  # the damaged file is kept for inspection
    db2 = Database(f"sqlite:///{path}")
    assert db2.get_state("marker") == "before-backup"


def test_damaged_database_without_backup_starts_empty(tmp_path):
    from backend.app.cloud.health import ensure_database
    p = tmp_path / "x.sqlite3"
    p.write_bytes(b"garbage" * 1000)
    out = ensure_database(p, tmp_path / "no-backups")
    assert out["status"] == "damaged_no_backup" and not p.exists()


def test_broken_production_model_rolls_back_automatically(monkeypatch, tmp_path):
    s, db = fresh_settings(monkeypatch, tmp_path)
    from backend.app.cloud.health import verify_models
    from backend.app.learning import registry
    _register(db, s, "v-old", "archived", {"log_loss": 1.01})
    time.sleep(0.01)
    path = _register(db, s, "v-new", "production", {"log_loss": 0.99})
    open(path, "wb").write(b"truncated")  # e.g. an interrupted copy
    out = verify_models(db, s)["BTC/USDT|15m|h4"]
    assert out["status"] == "broken" and out["rollback"]["status"] == "rolled_back"
    assert registry.get_production(db, "BTC/USDT|15m|h4")["version"] == "v-old"
    assert db.query_one("SELECT status FROM model_registry WHERE version='v-new'")["status"] == "rolled_back"
    ev = db.query_one("SELECT payload_json FROM learning_events WHERE event_type='model_rollback'")
    p = json.loads(ev["payload_json"])
    assert p["from"]["version"] == "v-new" and p["to"]["version"] == "v-old" and p["to"]["log_loss"] == 1.01


def test_operator_rollback_command_runs_exactly_once(monkeypatch, tmp_path):
    s, db = fresh_settings(monkeypatch, tmp_path)
    from backend.app import control
    from backend.app.learning import registry
    _register(db, s, "v1", "archived")
    time.sleep(0.01)
    _register(db, s, "v2", "production")
    s.control_dir.mkdir(parents=True)
    (s.control_dir / "commands.json").write_text(json.dumps({"commands": [
        {"id": "rb-1", "action": "rollback", "model_key": "BTC/USDT|15m|h4"}, {"id": "", "action": "rollback"}]}))
    r = control.run_commands(db, s)
    assert [x["id"] for x in r] == ["rb-1"] and r[0]["result"]["status"] == "rolled_back"
    assert registry.get_production(db, "BTC/USDT|15m|h4")["version"] == "v1"
    assert control.run_commands(db, s) == []  # remembered, also outside the database:
    assert "rb-1" in json.loads((tmp_path / "control_done.json").read_text())["done"]
    db.execute("DELETE FROM system_state WHERE key='control_done'")
    assert control.run_commands(db, s) == []


def test_restore_command_is_handled_before_open_and_only_once(monkeypatch, tmp_path):
    s, db = fresh_settings(monkeypatch, tmp_path)
    from backend.app import control
    s.control_dir.mkdir(parents=True)
    (s.control_dir / "commands.json").write_text(json.dumps({"commands": [{"id": "r1", "action": "restore_backup"}]}))
    assert control.pending_restore(s) == "r1"
    control.mark_done(s, "r1")
    assert control.pending_restore(s) is None


def test_notify_file_overrides_environment(monkeypatch, tmp_path):
    s, db = fresh_settings(monkeypatch, tmp_path)
    from backend.app.control import apply_notify_file
    s.control_dir.mkdir(parents=True)
    (s.control_dir / "notify.json").write_text(json.dumps({"enabled": False, "min_confidence": 0.62, "timeframes": ["1h"],
                                                           "language": "uk", "max_age_sec": 300}))
    out = apply_notify_file(s)
    assert out["source"] == "control/notify.json"
    assert (s.notify_enabled, s.notify_min_confidence, s.notify_timeframes, s.notify_lang, s.notify_max_age_sec) == \
        (False, 0.62, ["1h"], "uk", 300.0)


def test_baseline_test_separates_skill_from_base_rates():
    from backend.app.ml.evaluation import baseline_test
    rng = np.random.default_rng(0)
    n = 1200
    y = rng.choice(3, size=n, p=[0.2, 0.6, 0.2])
    prior = np.array([0.2, 0.6, 0.2])
    base = pd.DataFrame({"label": y})
    base[["p_naive_down", "p_naive_flat", "p_naive_up"]] = prior
    same = base.copy()
    same[["p_down", "p_flat", "p_up"]] = prior + rng.normal(0, 0.02, (n, 3)).clip(-0.1, 0.1)
    same[["p_down", "p_flat", "p_up"]] = same[["p_down", "p_flat", "p_up"]].div(same[["p_down", "p_flat", "p_up"]].sum(1), axis=0)
    r = baseline_test(same, 4)
    assert not r["passed"] and r["gain"] <= 0.01 and r["gain_ci95"][0] < 0
    good = base.copy()
    onehot = np.eye(3)[y]
    good[["p_down", "p_flat", "p_up"]] = 0.5 * onehot + 0.5 * prior
    r = baseline_test(good, 4)
    assert r["passed"] and r["gain"] > 0.1 and r["gain_ci95"][0] > 0 and r["p_better"] > 0.99


def test_signal_blocked_when_model_not_better_than_baseline(monkeypatch, tmp_path):
    from tests.test_learning_predictor import _book, _load_until, _stub_production, FAST
    from backend.app.learning.predictor import Predictor
    s, db = fresh_settings(monkeypatch, tmp_path, **FAST)
    base = planted_signal_candles(1500, "15m")
    cut = int(base.open_ts.iloc[1399]) + 900_000
    _load_until(db, base, cut)
    failed = {"baseline_test": {"n": 900, "gain": 0.001, "gain_ci95": [-0.004, 0.006], "p_better": 0.61,
                                "log_loss_model": 0.949, "log_loss_naive": 0.950, "passed": False},
              "backtest": {"trades": 40, "avg_net_bps": 3.0}}
    _stub_production(db, s, [0.1, 0.1, 0.8], metrics=failed)
    _book(db, cut + 5_000)
    out = Predictor(db, s).predict("BTC/USDT", "15m", now=cut + 5_000)
    assert out["prediction"] == "NO TRADE" and out["reasons"] == ["model_not_better_than_baseline"], out
    g = out["gate"]
    assert g["baseline"]["p_better"] == 0.61 and not g["baseline"]["passed"]
    assert g["costs"]["round_trip_bps"] >= s.round_trip_cost_bps and g["check_ts"] == out["candle_ts"] + 5 * 900_000


def test_prediction_details_factors_and_missed_candles(monkeypatch, tmp_path):
    from tests.test_learning_predictor import _book, _load_until, FAST, PASSED_BASELINE
    from backend.app.learning import registry
    from backend.app.learning.predictor import Predictor
    from backend.app.ml.dataset import build_training_frame
    from backend.app.ml.model import EnsembleModel
    s, db = fresh_settings(monkeypatch, tmp_path, **FAST)
    base = planted_signal_candles(1500, "15m")
    cut = int(base.open_ts.iloc[1399]) + 900_000
    _load_until(db, base, cut)
    frame, feats = build_training_frame(db, s, "BTC/USDT", "15m")
    lab = frame.dropna(subset=["label"])
    m = EnsembleModel().fit(lab[feats].to_numpy(float), lab["label"].astype(int).to_numpy(), feats)
    key = registry.model_key("BTC/USDT", "15m", 4)
    path = registry.save_artifact(s.model_dir, "v-real", {"model": m, "features": feats})
    registry.register(db, version="v-real", key=key, status="production", train_start_ts=0,
                      train_end_ts=int(lab.open_ts.iloc[-1]), n_train=len(lab), n_validation=0, feature_version="fv",
                      features=feats, params={}, metrics=PASSED_BASELINE, comparison=None, artifact_path=path,
                      parent_version=None, reason="test")
    _book(db, cut + 5_000)
    pr = Predictor(db, s)
    out = pr.predict("BTC/USDT", "15m", now=cut + 5_000)
    f = out["gate"]["factors"]
    assert 1 <= len(f) <= 5 and all(x["feature"] in feats for x in f)
    assert abs(f[0]["effect"]) >= abs(f[-1]["effect"])
    # three candles later, with no run in between: two closes were missed and are recorded
    cut2 = cut + 3 * 900_000
    _load_until(db, base, cut2)
    _book(db, cut2 + 5_000)
    out2 = pr.predict("BTC/USDT", "15m", now=cut2 + 5_000)
    assert out2["missed_before"] == 2
    assert db.scalar("SELECT COUNT(*) FROM prediction_gaps") == 2


def test_next_close_wait():
    from backend.app.cloud.cycle import next_close_wait
    from backend.app.config import Settings
    s = Settings()
    s.predict_timeframes, s.wait_close_max_sec, s.close_settle_sec = ["15m", "1h"], 960, 6
    close = 1_791_545_400_000  # a 15m boundary
    w, c = next_close_wait(s, close - 100_000)
    assert c == close and abs(w - 106) < 1e-6
    w, c = next_close_wait(s, close + 10_000)  # just after a close: next one is ~15 min away, still <= 16 min
    assert c == close + 900_000 and w <= 960
    s.wait_close_max_sec = 300
    assert next_close_wait(s, close + 10_000) == (0.0, None)
    s.wait_close_max_sec = 0
    assert next_close_wait(s, close - 1000) == (0.0, None)


def _pred_row(db, candle_ts, prediction="UP", conf=0.7, tf="15m", quality=0.95):
    db.execute("INSERT INTO predictions(created_ms,candle_ts,target_ts,symbol,exchange,timeframe,horizon_bars,price,prediction,"
               "model_direction,p_up,p_down,p_flat,confidence,label_threshold,gate_reasons,gate_json,regime,quality_score,"
               "features_version,model_version) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
               (candle_ts, candle_ts, candle_ts + 4 * 900_000, "BTC/USDT", "okx", tf, 4, 100.0, prediction, "UP",
                conf, 0.1, 1 - conf - 0.1, conf, 0.0025, "[]",
                json.dumps({"decision_delay_sec": 5, "factors": [{"feature": "15m_rsi_14"}]}), "trending", quality, "fv", "v1"))


def test_notifications_conditions_staleness_and_no_duplicates(monkeypatch, tmp_path):
    s, db = fresh_settings(monkeypatch, tmp_path, NTFY_TOPIC="t", NOTIFY_MIN_CONFIDENCE="0.65", NOTIFY_MAX_AGE_SEC="600",
                           NOTIFY_EVENTS="signals,outages",
                           NOTIFY_LANG="en")
    from backend.app.notify import Notifier
    sent = []
    post = lambda url, data, headers: sent.append(json.loads(data))  # noqa: E731
    bar = 900_000
    now = int(time.time() * 1000)
    fresh = now - bar - 60_000  # this candle closed one minute ago (independent of the wall clock)
    _pred_row(db, fresh, conf=0.7)
    _pred_row(db, fresh - bar, conf=0.6)                 # below the minimum confidence
    _pred_row(db, fresh - 8 * bar, conf=0.9)             # 2 h old: stale, must not be sent
    _pred_row(db, fresh - 2 * bar, prediction="NO TRADE")  # not a signal ("signals" event only)
    n = Notifier(db, s, post=post)
    out = n.predictions(now=now)
    assert len(sent) == 1 and out["skipped"] == {"confidence": 1, "stale": 1}
    m = sent[0]
    assert m["title"] == "BTC/USDT 15m: UP" and m["topic"] == "t"
    for part in ("UP 70%", "confidence 70%", "Main factors: 15m_rsi_14", "Data quality 95%", "not a trading recommendation"):
        assert part in m["message"], (part, m["message"])
    # a re-run (or a restored id counter) never repeats the same candle
    db.set_state("notify_last_prediction_id", 0)
    out = Notifier(db, s, post=post).predictions(now=now)
    assert len(sent) == 1 and out["skipped"]["duplicate"] == 1
    # disabled -> nothing at all
    s.notify_enabled = False
    assert Notifier(db, s, post=post).predictions(now=now) == {"skipped": "disabled"}


def test_labels_require_exact_horizon_in_time():
    from backend.app.ml.features import add_labels
    bar = 900_000
    ts = [i * bar for i in range(10) if i != 5]  # bar 5 missing
    f = pd.DataFrame({"open_ts": ts, "close": np.linspace(100, 110, len(ts)), "atr_pct": [np.nan] + [0.001] * (len(ts) - 1)})
    lab = add_labels(f, horizon=2, atr_mult=0.3, cost_bps=25)
    # rows whose +2 rows would jump over the hole have no label instead of a stretched horizon
    assert lab.loc[lab.open_ts == 3 * bar, "label"].isna().all() and lab.loc[lab.open_ts == 4 * bar, "label"].isna().all()
    assert lab.loc[lab.open_ts == 0, "label"].notna().all() and lab.label_threshold.iloc[0] == 0.0025  # cost floor


def test_challenger_needs_gain_in_both_halves(monkeypatch, tmp_path):
    s, _ = fresh_settings(monkeypatch, tmp_path)
    from backend.app.learning.engine import compare_models
    rng = np.random.default_rng(1)
    n = 800
    y = rng.choice(3, size=n)
    p_prod = np.full((n, 3), 1 / 3)
    p_chal = p_prod.copy()
    p_chal[: n // 2] = 0.6 * np.eye(3)[y[: n // 2]] + 0.4 / 3  # great in the first half only
    p_chal[n // 2:] = 0.9 * (1 - np.eye(3)[y[n // 2:]]) / 2 + 0.1 / 3  # worse later
    c = compare_models(y, p_prod, p_chal, s, 4, np.array(["trending"] * n))
    assert not c["checks"]["both_halves_ok"] and not c["promote"]
    assert c["halves_gain"][0] > 0 > c["halves_gain"][1] and c["by_regime"]["trending"]["n"] == n


def test_signal_blocked_without_after_cost_edge(monkeypatch, tmp_path):
    """Better probabilities than the base rates are not enough: the model's own out-of-sample signals
    must have earned money after costs (here: 4 signals, -20 bps) -> NO TRADE / NO EDGE AFTER COSTS."""
    from tests.test_learning_predictor import _book, _load_until, _stub_production, FAST, PASSED_BASELINE
    from backend.app.learning.predictor import Predictor
    s, db = fresh_settings(monkeypatch, tmp_path, **FAST)
    base = planted_signal_candles(1500, "15m")
    cut = int(base.open_ts.iloc[1399]) + 900_000
    _load_until(db, base, cut)
    _stub_production(db, s, [0.1, 0.1, 0.8], metrics={**PASSED_BASELINE, "backtest": {"trades": 4, "avg_net_bps": -20.7}})
    _book(db, cut + 5_000)
    out = Predictor(db, s).predict("BTC/USDT", "15m", now=cut + 5_000)
    assert out["prediction"] == "NO TRADE" and out["reasons"] == ["no_edge_after_costs"], out
    assert out["model_version"] and out["gate"]["costs"]["oos_signals"] == 4


def test_continuous_publisher_builds_one_orphan_commit(monkeypatch, tmp_path):
    """Continuous mode (SYNTHETIC GitHub API): stream status -> source checks + probes, export with
    BACKEND_MODE=continuous, one orphan commit with public JSON (+ state snapshot), branch force-updated."""
    s, db = fresh_settings(monkeypatch, tmp_path, PUBLIC_DIR=tmp_path / "public", PRIMARY_EXCHANGE="okx")
    from backend.app.exchanges import rest as rest_mod
    from backend.app.workers import publisher as P
    now = int(time.time() * 1000)
    for ex, age in (("okx", 2_000), ("bybit", 3_000), ("binance", 600_000)):
        db.execute("INSERT INTO stream_status(exchange,symbol,state,last_msg_ms,clock_skew_ms,updated_ms) VALUES(?,?,?,?,?,?)",
                   (ex, "BTC/USDT", "connected", now - age, 50, now))
    monkeypatch.setattr(rest_mod, "fetch_ticker", lambda ex, sym: (_ for _ in ()).throw(
        rest_mod.ExchangeError("HTTP 451", "region_blocked")) if ex == "binance" else {"last": 1.0})
    calls = []

    def fake(method, path, body):
        calls.append((method, path, body))
        if path.endswith("/git/blobs"):
            return {"sha": f"b{len(calls)}"}
        if path.endswith("/git/trees"):
            return {"sha": "t1"}
        if path.endswith("/git/commits"):
            return {"sha": "c1"}
        return {}
    pub = P.GitHubPublisher("owner/repo", "TOKEN", request=fake)
    out = P.run_publish_once(db, s, pub, with_state=True, rest_check=True)
    assert out["commit"] == "c1"
    commit = next(b for m, p, b in calls if p.endswith("/git/commits"))
    assert commit["parents"] == []  # orphan: the branch never accumulates history
    tree = next(b for m, p, b in calls if p.endswith("/git/trees"))["tree"]
    paths = {x["path"] for x in tree}
    assert "public/state.json" in paths and "public/ledger.json" in paths and "state/market.sqlite3.gz" in paths
    ref = [c for c in calls if c[0] == "PATCH"][0]
    assert ref[1].endswith("/git/refs/heads/data") and ref[2]["force"] is True
    st = json.loads((tmp_path / "public" / "state.json").read_text())
    assert st["mode"] == "continuous" and st["health"]["backend_mode"] == "continuous"
    pr = st["probes"]["probes"]
    assert pr["okx"]["ws"]["verified_live"] and not pr["binance"]["ws"]["verified_live"] and not pr["binance"]["rest"]["ok"]
    rates = st["source_stats"]
    assert rates["binance"]["ws"]["1h"]["error_rate"] == 1.0 and rates["okx"]["ws"]["1h"]["error_rate"] == 0.0
    assert "TOKEN" not in json.dumps(st)
    # later rounds reuse the uploaded state blobs instead of re-sending the database
    calls.clear()
    P.run_publish_once(db, s, pub, with_state=False, rest_check=False)
    tree = next(b for m, p, b in calls if p.endswith("/git/trees"))["tree"]
    assert "state/market.sqlite3.gz" in {x["path"] for x in tree}
    assert sum(1 for m, p, b in calls if p.endswith("/git/blobs")) == len([x for x in tree if x["path"].startswith("public/")])


def test_candle_sync_refreshes_after_backfill_and_right_after_a_close(monkeypatch, tmp_path):
    """Regression (found by the 3-hour real-data soak): the continuous collector reset its sync timestamp on every
    pass, so after the initial backfill no candle was ever refreshed and the predictor made 0 predictions."""
    import asyncio
    from backend.app.market import collector
    s, db = fresh_settings(monkeypatch, tmp_path, EXCHANGES="okx", PRIMARY_EXCHANGE="okx",
                           TIMEFRAMES="1m,15m", CANDLE_SYNC_SEC=1, FORECAST_ENABLED=0)
    sync = collector.CandleSync(db, s)
    # periodic: due again once the interval passed, not before
    t0 = 1_791_574_210_000  # 10 s after a 15m (and 1m) close
    sync.last_sync[("okx", "BTC/USDT", "15m")] = t0
    assert not sync.due(("okx", "BTC/USDT", "15m"), "15m", t0 + 30_000)
    assert sync.due(("okx", "BTC/USDT", "15m"), "15m", t0 + 60_000)
    # close-aware: a bar close since the last sync makes it due ~2 s after the close, even inside the period
    sync.last_sync[("okx", "BTC/USDT", "1m")] = t0 + 49_000
    sync.s.candle_sync_sec = 3600
    assert not sync.due(("okx", "BTC/USDT", "1m"), "1m", t0 + 50_500)   # close at +50 s, settling
    assert sync.due(("okx", "BTC/USDT", "1m"), "1m", t0 + 52_100)
    sync.s.candle_sync_sec = 1

    calls = []
    monkeypatch.setattr(rest, "fetch_klines", lambda ex, sym, tf, limit=1000, end_ms=None: calls.append((tf, limit)) or [])
    monkeypatch.setattr(rest, "fetch_history", lambda ex, sym, tf, bars, pause=0.2: calls.append((tf, "history")) or [])
    monkeypatch.setattr(rest, "fetch_ticker", lambda ex, sym: (_ for _ in ()).throw(rest.ExchangeError("x")))

    async def run():
        stop = asyncio.Event()
        task = asyncio.create_task(sync.run_exchange("okx", stop))
        await asyncio.sleep(4.5)
        stop.set()
        await task
    asyncio.run(run())
    refreshes = [c for c in calls if c[1] == 5]
    assert len([c for c in calls if c[1] in (1000, "history")]) == 2       # one backfill per timeframe
    assert len(refreshes) >= 4, calls                                       # then refreshed again and again
