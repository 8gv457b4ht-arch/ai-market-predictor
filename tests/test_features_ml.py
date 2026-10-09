"""Indicators, causality (no look-ahead), multi-timeframe join, labels, ensemble,
gating, walk-forward honesty and the cost-aware backtest."""
import numpy as np
import pandas as pd

from backend.app.ml.evaluation import backtest, gate, gate_array, walk_forward
from backend.app.ml.features import (BASE_FEATURES, add_labels, build_feature_frame, classify_regime_row,
                                     compute_indicators, feature_columns)
from backend.app.ml.model import EnsembleModel
from tests.synthetic import planted_signal_candles, random_walk_candles, resample


def test_indicator_ranges():
    ind = compute_indicators(random_walk_candles(600))
    rsi = ind["rsi_14"].dropna()
    assert rsi.between(0, 100).all()
    assert ind["adx_14"].dropna().between(0, 100).all()
    assert (ind["atr_pct"].dropna() > 0).all()
    assert ind["bb_pos"].notna().sum() > 500
    assert ind["cvd"].notna().all()  # from taker-buy volume
    for f in BASE_FEATURES:
        assert f in ind, f


def test_features_are_causal_no_lookahead():
    df = random_walk_candles(800)
    a = compute_indicators(df)
    shocked = df.copy()
    shocked.loc[600:, ["open", "high", "low", "close"]] *= 1.5  # change only the future
    shocked.loc[600:, "volume"] *= 10
    b = compute_indicators(shocked)
    cols = BASE_FEATURES + ["atr_pct_rank", "ret_z"]
    pd.testing.assert_frame_equal(a.loc[:599, cols], b.loc[:599, cols])


def test_multi_timeframe_join_uses_only_closed_higher_bars():
    base = random_walk_candles(2000, "15m")
    h1 = resample(base, "15m", "1h")
    frame = build_feature_frame({"15m": base, "1h": h1}, "15m")
    assert "1h_rsi_14" in frame and "15m_macd_hist" in frame
    hi = compute_indicators(h1)
    # pick a row: decision at the close of a 15m bar; the matching 1h value must come from a 1h bar
    # that closed at or before that instant
    for i in (500, 777, 1500):
        dts = frame.decision_ts.iloc[i]
        avail = hi.open_ts + 3_600_000
        j = int(np.where(avail <= dts)[0][-1])
        v = frame["1h_rsi_14"].iloc[i]
        assert np.isclose(v, hi["rsi_14"].iloc[j], equal_nan=True)
        assert hi.open_ts.iloc[j] + 3_600_000 <= dts
    # a 15m row in the middle of an hour must NOT see the still-open hour
    mid = frame[((frame.decision_ts - base.open_ts.iloc[0]) % 3_600_000) == 1_800_000].iloc[100]
    j_open = int(np.where(hi.open_ts <= mid.open_ts)[0][-1])  # the hour containing this bar (still open)
    assert not np.isclose(mid["1h_rsi_14"], hi["rsi_14"].iloc[j_open])


def test_labels_and_threshold():
    f = pd.DataFrame({"close": [100, 101, 99, 100, 103], "atr_pct": [0.001] * 5})
    lab = add_labels(f, horizon=1, atr_mult=0.0, cost_bps=50)  # threshold = 0.5%
    assert lab["label"].tolist()[:4] == [2, 0, 2, 2]
    assert np.isnan(lab["label"].iloc[-1])  # future unknown -> no label


def test_regime_priority():
    assert classify_regime_row(40, 0.5, 0, 7) == "abnormal"
    assert classify_regime_row(40, 0.95, 0, 0) == "high_volatility"
    assert classify_regime_row(40, 0.5, -2, 0) == "low_liquidity"
    assert classify_regime_row(30, 0.5, 0, 0) == "trending"
    assert classify_regime_row(10, 0.5, 0, 0) == "ranging"
    assert classify_regime_row(10, 0.5, 0, 0, spread_bps=50, max_spread_bps=15) == "low_liquidity"
    assert classify_regime_row(10, 0.5, 0, 0, quality_ok=False) == "abnormal"


def test_gate_confidence_threshold():
    assert gate(0.2, 0.1, 0.7, 0.55, 0.1)[0] == "UP"
    assert gate(0.3, 0.3, 0.4, 0.55, 0.1)[0] == "NO TRADE"          # weak confidence
    assert gate(0.56, 0.0, 0.44, 0.55, 0.15)[0] == "NO TRADE"       # edge too small
    assert gate(0.6, 0.65, 0.0, 0.55, 0.1)[0] == "NO TRADE"         # FLAT more likely than direction
    p = np.array([[0.7, 0.1, 0.2], [0.3, 0.4, 0.3], [0.1, 0.2, 0.7]])
    assert gate_array(p, 0.55, 0.1).tolist() == [0, -1, 2]


def _frame(gen, n):
    base = gen(n, "15m")
    candles = {"15m": base, "1h": resample(base, "15m", "1h"), "4h": resample(base, "15m", "4h")}
    f = add_labels(build_feature_frame(candles, "15m"), 4, 0.3, 26).iloc[60:].reset_index(drop=True)
    feats = feature_columns(f)
    return f.dropna(subset=["label"]).reset_index(drop=True), feats


def test_ensemble_outputs_valid_probabilities():
    f, feats = _frame(random_walk_candles, 900)
    m = EnsembleModel().fit(f[feats].to_numpy(float), f["label"].astype(int).to_numpy(), feats)
    p = m.predict_proba(f[feats].tail(50).to_numpy(float))
    assert p.shape == (50, 3)
    assert np.allclose(p.sum(axis=1), 1) and (p > 0).all()
    assert set(m.models) == {"logreg", "random_forest", "hist_gb"}
    assert 0.3 <= m.temperature <= 5


def test_walk_forward_does_not_find_signal_in_random_walk():
    """Leakage canary: on pure noise the OOS log loss must not beat the naive prior meaningfully."""
    f, feats = _frame(random_walk_candles, 1700)
    wf = walk_forward(f, feats, horizon=4, folds=3, min_train=600)
    m = wf["metrics"]
    assert m["log_loss"] > m["baseline_prior"]["log_loss"] - 0.02, (m["log_loss"], m["baseline_prior"])
    assert m["accuracy"] < 0.45
    assert wf["oos"].open_ts.is_monotonic_increasing
    assert all(fr["train_rows"] < fr["test_rows"] + fr["train_rows"] for fr in m["folds"])
    # purge: the last training row is at least `horizon` rows before the first test row
    first_test = m["folds"][0]["test_start"]
    assert (f.open_ts < first_test).sum() - m["folds"][0]["train_rows"] >= 4


def test_walk_forward_learns_a_real_signal():
    f, feats = _frame(planted_signal_candles, 1700)
    m = walk_forward(f, feats, horizon=4, folds=3, min_train=600)["metrics"]
    assert m["log_loss"] < m["baseline_prior"]["log_loss"] - 0.1
    assert m["signals"]["signals"] > 0 and m["signals"]["hit_rate"] > 0.6
    assert "by_regime" in m and m["confusion_matrix"]["labels"] == ["DOWN", "FLAT", "UP"]


def _oos(n=400, always=2):
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.002, n)))
    p = np.tile([0.1, 0.1, 0.8] if always == 2 else [0.8, 0.1, 0.1], (n, 1))
    return pd.DataFrame({"open_ts": np.arange(n) * 900_000, "open": np.r_[close[0], close[:-1]], "close": close,
                         "p_down": p[:, 0], "p_flat": p[:, 1], "p_up": p[:, 2]})


def test_backtest_costs_and_latency():
    oos = _oos()
    free = backtest(oos, "15m", 4, 0.55, 0.1, 0, 0, 0, 0)
    costly = backtest(oos, "15m", 4, 0.55, 0.1, 10, 5, 2, 0)
    assert free["trades"] == costly["trades"] > 50
    assert costly["avg_net_bps"] < free["avg_net_bps"]
    assert abs((free["avg_net_bps"] - costly["avg_net_bps"]) - 32) < 1.0  # 2*10 fee + 2*5 slip + 2 spread
    slow = backtest(oos, "15m", 4, 0.55, 0.1, 0, 0, 0, latency_ms=2 * 900_000)
    assert slow["costs"]["latency_bars"] == 2
    assert slow["avg_gross_bps"] < free["avg_gross_bps"]  # later entry misses part of the move
    none = backtest(_oos().assign(p_up=0.34, p_down=0.33, p_flat=0.33), "15m", 4, 0.55, 0.1, 0, 0, 0, 0)
    assert none["trades"] == 0


def test_gate_diagnostics_decompose_every_condition():
    from backend.app.ml.evaluation import gate_diagnostics
    y = np.array([2, 0, 1, 2])
    p = np.array([[0.1, 0.2, 0.7],    # passes
                  [0.4, 0.3, 0.3],    # confidence below 0.55
                  [0.5, 0.0, 0.5],    # edge 0 < 0.1 (conf 0.5 < 0.55 fails first)
                  [0.0, 0.6, 0.58]])  # conf passes, edge passes, FLAT more likely
    d = gate_diagnostics(y, p, np.array([0.01, -0.01, 0.0, 0.002]), 0.55, 0.1)
    assert d["passed_all"] == 1
    assert d["first_failing_condition"] == {"confidence_below_threshold": 2, "edge_below_min_edge": 0,
                                            "flat_more_likely": 1, "passed": 1}
    assert d["per_class_calibration"]["UP"]["actual_frequency"] == 0.5
    sweep = {r["threshold"]: r for r in d["threshold_sweep_informational"]}
    assert sweep[0.40]["signals"] >= sweep[0.70]["signals"]
    assert d["confidence_quantiles"]["max"] == 0.7


def test_probabilities_match_base_rates_on_imbalanced_labels():
    """Regression for the real-data finding: balanced class weights inflated P(UP)/P(DOWN)."""
    from backend.app.ml.evaluation import walk_forward
    f, feats = _frame(random_walk_candles, 1700)
    f = f.copy()
    rng = np.random.default_rng(0)
    f["label"] = rng.choice([0, 1, 2], size=len(f), p=[0.15, 0.7, 0.15])  # FLAT-heavy, no information
    m = walk_forward(f, feats, horizon=4, folds=3, min_train=600)["metrics"]
    cal = m["gate_diagnostics"]["per_class_calibration"]
    for k in ("DOWN", "FLAT", "UP"):
        assert abs(cal[k]["mean_predicted"] - cal[k]["actual_frequency"]) < 0.06, (k, cal[k])
    assert m["log_loss"] < m["baseline_prior"]["log_loss"] + 0.03  # not worse than the naive prior
    assert m["gate_diagnostics"]["passed_all"] <= 0.02 * m["n"]   # no information -> (almost) no signals
