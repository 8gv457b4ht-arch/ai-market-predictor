"""Metrics, confidence gating, purged walk-forward validation and a cost-aware backtest."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)

from ..config import TIMEFRAME_MS
from .features import CLASSES
from .model import EnsembleModel

DOWN, FLAT, UP = 0, 1, 2


# ------------------------------------------------------------------ gating
def gate(p_down: float, p_flat: float, p_up: float, threshold: float, min_edge: float) -> tuple[str, str, float]:
    """Return (signal, raw model direction, confidence).

    A directional signal needs P(dir) >= threshold, P(dir) - P(opposite) >= min_edge
    and P(dir) >= P(FLAT); otherwise NO TRADE.
    """
    probs = {"DOWN": p_down, "FLAT": p_flat, "UP": p_up}
    model_dir = max(probs, key=probs.get)
    d = "UP" if p_up >= p_down else "DOWN"
    conf = max(p_up, p_down)
    edge = abs(p_up - p_down)
    if conf >= threshold and edge >= min_edge and conf >= p_flat:
        return d, model_dir, conf
    return "NO TRADE", model_dir, conf


def gate_array(proba: np.ndarray, threshold: float, min_edge: float) -> np.ndarray:
    """Vectorised gate: returns 2=UP, 0=DOWN, -1=NO TRADE."""
    pd_, pf, pu = proba[:, DOWN], proba[:, FLAT], proba[:, UP]
    d = np.where(pu >= pd_, UP, DOWN)
    conf = np.maximum(pu, pd_)
    ok = (conf >= threshold) & (np.abs(pu - pd_) >= min_edge) & (conf >= pf)
    return np.where(ok, d, -1)


# ----------------------------------------------------------------- metrics
def expected_calibration_error(y: np.ndarray, proba: np.ndarray, bins: int = 10) -> tuple[float, list]:
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece, table = 0.0, []
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= 0) & (conf <= edges[1])
        if m.any():
            gap = abs(correct[m].mean() - conf[m].mean())
            ece += m.mean() * gap
            table.append({"bin": f"{edges[i]:.1f}-{edges[i+1]:.1f}", "n": int(m.sum()),
                          "confidence": float(conf[m].mean()), "accuracy": float(correct[m].mean())})
    return float(ece), table


def classification_metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    pred = proba.argmax(axis=1)
    onehot = np.eye(3)[y]
    ece, reliability = expected_calibration_error(y, proba)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision_macro": float(precision_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0)),
        "recall_macro": float(recall_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0)),
        "f1_macro": float(f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0)),
        "precision": dict(zip(CLASSES, map(float, precision_score(y, pred, average=None, labels=[0, 1, 2], zero_division=0)))),
        "recall": dict(zip(CLASSES, map(float, recall_score(y, pred, average=None, labels=[0, 1, 2], zero_division=0)))),
        "brier": float(np.mean(np.sum((proba - onehot) ** 2, axis=1))),
        "log_loss": float(log_loss(y, proba, labels=[0, 1, 2])),
        "ece": ece,
        "reliability": reliability,
        "confusion_matrix": {"labels": CLASSES, "matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()},
        "label_distribution": dict(zip(CLASSES, map(int, np.bincount(y, minlength=3)))),
    }


def baseline_metrics(y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """Naive benchmark: always predict the training class frequencies."""
    prior = np.bincount(np.asarray(y_train, dtype=int), minlength=3) / len(y_train)
    prior = np.clip(prior, 1e-6, 1)
    prior = prior / prior.sum()
    proba = np.tile(prior, (len(y_test), 1))
    return {"log_loss": float(log_loss(y_test, proba, labels=[0, 1, 2])),
            "brier": float(np.mean(np.sum((proba - np.eye(3)[np.asarray(y_test, dtype=int)]) ** 2, axis=1))),
            "accuracy": float(np.mean(np.asarray(y_test) == prior.argmax()))}


def signal_metrics(y: np.ndarray, proba: np.ndarray, threshold: float, min_edge: float) -> dict:
    sig = gate_array(proba, threshold, min_edge)
    active = sig >= 0
    n = int(active.sum())
    hits = int((sig[active] == np.asarray(y)[active]).sum()) if n else 0
    wrong_dir = int(((sig == UP) & (y == DOWN)).sum() + ((sig == DOWN) & (y == UP)).sum())
    return {"threshold": threshold, "min_edge": min_edge, "signals": n,
            "coverage": float(active.mean()) if len(y) else 0.0,
            "hit_rate": hits / n if n else None, "wrong_direction": wrong_dir,
            "flat_after_signal": n - hits - wrong_dir}


def gate_diagnostics(y: np.ndarray, proba: np.ndarray, fwd_return: np.ndarray | None, threshold: float,
                     min_edge: float) -> dict:
    """Why do (or don't) predictions pass the gate? Decomposes every condition on out-of-sample data.

    The threshold sweep is INFORMATIONAL ONLY: it shows what other thresholds would have done on the
    same out-of-sample predictions; the configured threshold is never changed automatically.
    """
    y = np.asarray(y, dtype=int)
    pd_, pf, pu = proba[:, DOWN], proba[:, FLAT], proba[:, UP]
    conf = np.maximum(pu, pd_)
    edge = np.abs(pu - pd_)
    c_conf, c_edge, c_flat = conf >= threshold, edge >= min_edge, conf >= pf
    passed = c_conf & c_edge & c_flat
    first_fail = np.where(~c_conf, "confidence_below_threshold",
                          np.where(~c_edge, "edge_below_min_edge", np.where(~c_flat, "flat_more_likely", "passed")))
    blocked = {k: int((first_fail == k).sum()) for k in ("confidence_below_threshold", "edge_below_min_edge",
                                                           "flat_more_likely", "passed")}
    q = lambda a, p: float(np.quantile(a, p)) if len(a) else None  # noqa: E731
    per_class = {CLASSES[k]: {"mean_predicted": float(proba[:, k].mean()), "actual_frequency": float((y == k).mean())}
                 for k in range(3)}
    sweep = []
    direction = np.where(pu >= pd_, UP, DOWN)
    for thr in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        m = (conf >= thr) & c_edge & c_flat
        n = int(m.sum())
        row = {"threshold": thr, "signals": n, "coverage": float(m.mean()) if len(m) else 0.0,
               "hit_rate": float((direction[m] == y[m]).mean()) if n else None}
        if fwd_return is not None and n:
            sign = np.where(direction[m] == UP, 1.0, -1.0)
            row["avg_gross_bps"] = float(np.nanmean(sign * np.asarray(fwd_return)[m]) * 1e4)
        sweep.append(row)
    return {
        "n": int(len(y)), "threshold": threshold, "min_edge": min_edge,
        "confidence_quantiles": {"p50": q(conf, 0.5), "p90": q(conf, 0.9), "p99": q(conf, 0.99), "max": q(conf, 1.0)},
        "p_flat_mean": float(pf.mean()) if len(pf) else None,
        "argmax_flat_share": float((proba.argmax(1) == FLAT).mean()) if len(y) else None,
        "pass_confidence": int(c_conf.sum()), "pass_edge": int(c_edge.sum()), "pass_flat": int(c_flat.sum()),
        "passed_all": int(passed.sum()), "first_failing_condition": blocked,
        "per_class_calibration": per_class, "threshold_sweep_informational": sweep,
    }


def full_metrics(y, proba, regimes=None, threshold=0.55, min_edge=0.1) -> dict:
    y = np.asarray(y, dtype=int)
    m = classification_metrics(y, proba)
    m["signals"] = signal_metrics(y, proba, threshold, min_edge)
    if regimes is not None:
        regimes = np.asarray(regimes)
        by = {}
        for r in sorted(set(regimes)):
            mask = regimes == r
            if mask.sum() >= 10:
                pr = proba[mask]
                by[r] = {"n": int(mask.sum()),
                         "accuracy": float(np.mean(pr.argmax(axis=1) == y[mask])),
                         "log_loss": float(log_loss(y[mask], pr, labels=[0, 1, 2])),
                         "signals": signal_metrics(y[mask], pr, threshold, min_edge)}
        m["by_regime"] = by
    return m


# ------------------------------------------------------------ walk-forward
def walk_forward(frame: pd.DataFrame, features: list[str], horizon: int, folds: int = 5,
                 min_train: int = 600, threshold: float = 0.55, min_edge: float = 0.1,
                 model_factory=EnsembleModel) -> dict:
    """Expanding-window, purged walk-forward evaluation.

    The last part of the labelled data is split into `folds` consecutive test
    blocks. For a test block starting at row s, training uses rows < s - horizon
    (purge: their labels would otherwise peek into the test block).
    """
    d = frame.dropna(subset=["label"]).reset_index(drop=True)
    n = len(d)
    if n < min_train + folds * 50:
        raise ValueError(f"not enough labelled rows for walk-forward: {n}")
    test_total = min(int(n * 0.4), n - min_train)
    block = test_total // folds
    start0 = n - block * folds
    oos_parts, fold_rows = [], []
    for k in range(folds):
        s, e = start0 + k * block, start0 + (k + 1) * block
        tr = d.iloc[: max(0, s - horizon)]
        te = d.iloc[s:e]
        ytr = tr["label"].astype(int).to_numpy()
        if len(np.unique(ytr)) < 2:
            continue
        model = model_factory().fit(tr[features].to_numpy(float), ytr, features)
        p = model.predict_proba(te[features].to_numpy(float))
        part = te[["open_ts", "open", "close", "fwd_return", "label", "label_threshold", "regime"]].copy()
        part[["p_down", "p_flat", "p_up"]] = p
        # naive benchmark for exactly these rows: class frequencies of this fold's training window
        prior = np.clip(np.bincount(ytr, minlength=3) / len(ytr), 1e-6, 1)
        part[["p_naive_down", "p_naive_flat", "p_naive_up"]] = prior / prior.sum()
        part["fold"] = k
        oos_parts.append(part)
        fm = classification_metrics(te["label"].astype(int).to_numpy(), p)
        fold_rows.append({"fold": k, "train_rows": len(tr), "test_rows": len(te),
                          "test_start": int(te.open_ts.iloc[0]), "test_end": int(te.open_ts.iloc[-1]),
                          "accuracy": fm["accuracy"], "log_loss": fm["log_loss"], "brier": fm["brier"],
                          "baseline": baseline_metrics(ytr, te["label"].astype(int).to_numpy())})
    if not oos_parts:
        raise ValueError("walk-forward produced no valid folds")
    oos = pd.concat(oos_parts, ignore_index=True)
    proba = oos[["p_down", "p_flat", "p_up"]].to_numpy()
    y = oos["label"].astype(int).to_numpy()
    metrics = full_metrics(y, proba, oos["regime"].to_numpy(), threshold, min_edge)
    first_train = d.iloc[: max(0, start0 - horizon)]["label"].astype(int).to_numpy()
    metrics["baseline_prior"] = baseline_metrics(first_train, y)
    metrics["gate_diagnostics"] = gate_diagnostics(y, proba, oos["fwd_return"].to_numpy(float), threshold, min_edge)
    thr = oos["label_threshold"].to_numpy(float)
    fr = np.abs(oos["fwd_return"].to_numpy(float))
    metrics["label_diagnostics"] = {
        "median_label_threshold_bps": float(np.nanmedian(thr) * 1e4),
        "median_abs_forward_return_bps": float(np.nanmedian(fr) * 1e4),
        "share_moves_above_threshold": float(np.nanmean(fr > thr)),
    }
    metrics["baseline_test"] = baseline_test(oos, horizon)
    metrics["folds"] = fold_rows
    metrics["method"] = f"expanding walk-forward, {len(fold_rows)} folds, purge={horizon} bars, out-of-sample only"
    return {"metrics": metrics, "oos": oos}


# ---------------------------------------------------------------- backtest
def backtest(oos: pd.DataFrame, timeframe: str, horizon: int, threshold: float, min_edge: float,
             fee_bps: float, slippage_bps: float, spread_bps: float, latency_ms: float) -> dict:
    """Non-overlapping, cost-aware simulation of gated OOS signals.

    Signal at the close of bar t. Entry at the open of bar t+1+L where
    L = floor(latency / bar) extra bars of latency; entry/exit prices are worsened
    by half the spread + slippage each; fees charged on both sides. Exit at the
    close of bar t+h (the label horizon). Read-only research: no orders anywhere.
    """
    d = oos.sort_values("open_ts").reset_index(drop=True)
    if d.empty:
        return {"trades": 0}
    proba = d[["p_down", "p_flat", "p_up"]].to_numpy()
    sig = gate_array(proba, threshold, min_edge)
    lat_bars = int(latency_ms // TIMEFRAME_MS[timeframe])
    per_side = (spread_bps / 2 + slippage_bps) / 1e4
    fee = fee_bps / 1e4
    opens, closes = d["open"].to_numpy(float), d["close"].to_numpy(float)
    ts = d["open_ts"].to_numpy()
    bar_ms = TIMEFRAME_MS[timeframe]
    trades, i, n = [], 0, len(d)
    while i < n:
        if sig[i] < 0:
            i += 1
            continue
        e_idx, x_idx = i + 1 + lat_bars, i + horizon
        # rows must be contiguous in time (fold boundaries / gaps break the trade)
        if x_idx >= n or e_idx > x_idx or ts[x_idx] - ts[i] != horizon * bar_ms:
            i += 1
            continue
        side = 1 if sig[i] == UP else -1
        entry = opens[e_idx] * (1 + side * per_side)
        exit_ = closes[x_idx] * (1 - side * per_side)
        gross = side * (closes[x_idx] / opens[e_idx] - 1)
        net = side * (exit_ / entry - 1) - 2 * fee
        trades.append({"ts": int(ts[i]), "side": "LONG" if side > 0 else "SHORT", "gross": gross, "net": net})
        i = x_idx + 1
    if not trades:
        return {"trades": 0, "note": "no signal passed the confidence gate"}
    t = pd.DataFrame(trades)
    eq = (1 + t["net"]).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    wins, losses = t.loc[t.net > 0, "net"].sum(), -t.loc[t.net <= 0, "net"].sum()
    span_bars = max(1, (int(ts[-1]) - int(ts[0])) // bar_ms)
    bh = closes[-1] / opens[0] - 1
    return {
        "trades": int(len(t)), "long": int((t.side == "LONG").sum()), "short": int((t.side == "SHORT").sum()),
        "win_rate": float((t.net > 0).mean()),
        "avg_gross_bps": float(t.gross.mean() * 1e4), "avg_net_bps": float(t.net.mean() * 1e4),
        "total_net_return": float(eq.iloc[-1] - 1), "total_gross_return": float((1 + t.gross).prod() - 1),
        "max_drawdown": float(dd), "profit_factor": float(wins / losses) if losses > 0 else None,
        "exposure": float(len(t) * horizon / span_bars), "buy_and_hold_return": float(bh),
        "sharpe_per_trade": float(t.net.mean() / t.net.std() * math.sqrt(len(t))) if len(t) > 2 and t.net.std() > 0 else None,
        "costs": {"fee_bps_per_side": fee_bps, "slippage_bps_per_side": slippage_bps, "spread_bps": spread_bps,
                  "latency_ms": latency_ms, "latency_bars": lat_bars,
                  "round_trip_cost_bps": 2 * fee_bps + 2 * slippage_bps + spread_bps},
        "period": {"start": int(ts[0]), "end": int(ts[-1])},
        "disclaimer": "Historical out-of-sample simulation. Not a guarantee of future results.",
    }


# ------------------------------------------------------- model vs naive baseline
NAIVE_COLS = ["p_naive_down", "p_naive_flat", "p_naive_up"]


def _block_bootstrap(diff: np.ndarray, block: int, n_boot: int = 2000, seed: int = 11) -> np.ndarray:
    n = len(diff)
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    return diff[idx].mean(axis=1)


def baseline_test(oos: pd.DataFrame, horizon: int, p_required: float = 0.95) -> dict | None:
    """Is the model better than always predicting the training base rates, on the same unseen rows?

    Per-row log-loss difference (naive - model), moving-block bootstrap (block = horizon, because
    overlapping labels are autocorrelated): mean gain, 95% interval, P(gain > 0)."""
    if oos is None or len(oos) < 50 or not set(NAIVE_COLS) <= set(oos.columns):
        return None
    y = oos["label"].astype(int).to_numpy()
    pm = np.clip(oos[["p_down", "p_flat", "p_up"]].to_numpy(float), 1e-9, 1)
    pn = np.clip(oos[NAIVE_COLS].to_numpy(float), 1e-9, 1)
    ll_m = -np.log(pm[np.arange(len(y)), y])
    ll_n = -np.log(pn[np.arange(len(y)), y])
    diff = ll_n - ll_m
    boots = _block_bootstrap(diff, horizon)
    onehot = np.eye(3)[y]
    p_better = float((boots > 0).mean())
    gain = float(diff.mean())
    return {"n": int(len(y)), "log_loss_model": float(ll_m.mean()), "log_loss_naive": float(ll_n.mean()),
            "gain": gain, "gain_ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "relative_gain": float(gain / ll_n.mean()) if ll_n.mean() > 0 else None,
            "p_better": p_better, "p_required": p_required,
            "brier_model": float(np.mean(np.sum((pm - onehot) ** 2, axis=1))),
            "brier_naive": float(np.mean(np.sum((pn - onehot) ** 2, axis=1))),
            "passed": bool(gain > 0 and p_better >= p_required)}
