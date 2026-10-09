"""Reproducible model research on REAL stored candles (read-only on the database).

Protocol (decided before looking at results, to avoid picking a winner on one lucky period):
  * per market (symbol x timeframe) the labelled rows are split chronologically:
      development = first 70 %, final holdout = last 30 % (minus a purge gap of `horizon` bars)
  * candidates are compared ONLY on development, with a purged expanding walk-forward (4 folds)
  * one configuration is chosen for ALL markets: best mean log-loss gain vs the naive base rates on development
  * the chosen configuration, the current production procedure and the naive baseline are then
    evaluated ONCE on the final holdout; results are reported with block-bootstrap 95 % intervals
  * metrics: log loss, Brier, ECE, accuracy, macro precision/recall, signal coverage/hit rate at the
    production gate, average result per signal after costs; per market regime as well

    python scripts/research.py --db state/market.sqlite3 --exchange binance --out docs/research_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

GROUPS = {
    "trend": ("ema_gap", "close_vs_ema200", "ema21_slope", "adx_14", "di_diff", "trend_strength", "macd"),
    "momentum": ("ret_", "roc_10", "rsi_14", "bb_pos"),
    "volatility": ("atr_pct", "bb_width", "rv_20", "rv_ratio", "range_pct", "body_pct"),
    "volume_flow": ("vol_z", "buy_ratio", "flow_imbalance"),
    "regime": ("regime_",),
}


def group_of(f: str) -> str:
    n = f.split("_", 1)[1] if f.split("_", 1)[0] in ("15m", "1h", "4h", "1d") else f
    for g, pats in GROUPS.items():
        if any(n.startswith(p) or p in n for p in pats):
            return g
    return "other"


def components(name: str):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    imp = ("impute", SimpleImputer(strategy="median", keep_empty_features=True))
    if name == "logreg":
        return Pipeline([imp, ("scale", StandardScaler()), ("clf", LogisticRegression(C=0.5, max_iter=2000))])
    if name == "logreg_strong_l2":
        return Pipeline([imp, ("scale", StandardScaler()), ("clf", LogisticRegression(C=0.02, max_iter=2000))])
    if name == "random_forest":
        return Pipeline([imp, ("clf", RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=25,
                                                             max_features="sqrt", random_state=42, n_jobs=-1))])
    if name == "hist_gb":
        return Pipeline([imp, ("clf", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_depth=4,
                                                                      min_samples_leaf=40, l2_regularization=1.0,
                                                                      early_stopping=False, random_state=42))])
    raise ValueError(name)


class Single:
    """One sklearn model + the same temperature/bias calibration as production (fitted on the last 20 %)."""
    def __init__(self, name):
        self.name = name

    def fit(self, X, y, feats=None):
        from backend.app.ml.model import EnsembleModel
        m = EnsembleModel(weights={self.name: 1.0})
        import backend.app.ml.model as mm
        orig = mm._components
        mm._components = lambda seed=42, balanced=False: {self.name: components(self.name)}
        try:
            m.fit(X, y, feats)
        finally:
            mm._components = orig
        self.m = m
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)


def candidates():
    from backend.app.ml.model import EnsembleModel
    out = {"ensemble_current": lambda: EnsembleModel()}
    for n in ("logreg", "logreg_strong_l2", "random_forest", "hist_gb"):
        out[n] = (lambda n=n: Single(n))
    return out


def metrics(y, p, prior, fwd, thr_lab, regimes, cost_bps, threshold=0.55, min_edge=0.1, horizon=4):
    from backend.app.ml.evaluation import classification_metrics, gate_array
    p = np.clip(p, 1e-9, 1)
    ll = -np.log(p[np.arange(len(y)), y])
    lln = -np.log(np.clip(prior, 1e-9, 1)[np.arange(len(y)), y])
    m = classification_metrics(y, p)
    sig = gate_array(p, threshold, min_edge)
    act = sig >= 0
    side = np.where(sig == 2, 1, np.where(sig == 0, -1, 0))
    net = side[act] * fwd[act] * 1e4 - cost_bps
    from backend.app.ml.evaluation import _block_bootstrap
    boots = _block_bootstrap(lln - ll, horizon)
    return {"n": int(len(y)), "log_loss": float(ll.mean()), "log_loss_naive": float(lln.mean()), "gain": float((lln - ll).mean()),
            "gain_ci95": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))], "p_better": float((boots > 0).mean()),
            "brier": m["brier"], "ece": m.get("ece"), "accuracy": m["accuracy"],
            "precision_macro": m.get("precision_macro"), "recall_macro": m.get("recall_macro"),
            "signals": int(act.sum()), "coverage": float(act.mean()), "hit_rate": float((sig[act] == y[act]).mean()) if act.any() else None,
            "avg_net_bps_per_signal": float(net.mean()) if act.any() else None,
            "by_regime": {r: {"n": int((regimes == r).sum()), "gain": float((lln - ll)[regimes == r].mean())}
                          for r in sorted(set(regimes)) if (regimes == r).sum() >= 30}}


def walk(model_fn, d, feats, horizon, folds, start, end):
    """Purged expanding walk-forward over rows [start, end): returns OOS proba and naive prior per row."""
    block = (end - start) // folds
    P, N, idx = [], [], []
    for k in range(folds):
        s, e = start + k * block, start + (k + 1) * block if k < folds - 1 else end
        tr = d.iloc[: max(0, s - horizon)]
        te = d.iloc[s:e]
        ytr = tr.label.astype(int).to_numpy()
        model = model_fn().fit(tr[feats].to_numpy(float), ytr, feats)
        P.append(model.predict_proba(te[feats].to_numpy(float)))
        prior = np.bincount(ytr, minlength=3) / len(ytr)
        N.append(np.tile(prior, (len(te), 1)))
        idx.extend(range(s, e))
    return np.vstack(P), np.vstack(N), np.array(idx)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--exchange", default="binance")
    ap.add_argument("--out", default="docs/research_results.json")
    ap.add_argument("--folds", type=int, default=4)
    a = ap.parse_args()
    os.environ["DATABASE_URL"] = f"sqlite:///{Path(a.db).resolve()}"
    os.environ["PRIMARY_EXCHANGE"] = a.exchange
    from backend.app.config import get_settings
    from backend.app.db import get_db
    from backend.app.ml.dataset import build_training_frame
    s = get_settings()
    db = get_db()
    h, cost = s.horizon_bars, s.round_trip_cost_bps
    markets = [(sym, tf) for sym in s.symbols for tf in s.predict_timeframes]
    data = {}
    for sym, tf in markets:
        f, feats = build_training_frame(db, s, sym, tf)
        d = f.dropna(subset=["label"]).reset_index(drop=True)
        data[(sym, tf)] = (d, feats)
    report: dict = {"generated_utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime()), "exchange": a.exchange,
                    "horizon_bars": h, "round_trip_cost_bps": cost, "protocol": __doc__.strip().split("\n\n")[1], "markets": {}}

    # ---------------- stage 1: data / labels
    for (sym, tf), (d, feats) in data.items():
        dist = d.label.value_counts(normalize=True).sort_index()
        report["markets"][f"{sym} {tf}"] = {
            "rows": len(d), "from": int(d.open_ts.iloc[0]), "to": int(d.open_ts.iloc[-1]),
            "label_share": {"DOWN": float(dist.get(0.0, 0)), "FLAT": float(dist.get(1.0, 0)), "UP": float(dist.get(2.0, 0))},
            "median_threshold_bps": float(d.label_threshold.median() * 1e4),
            "threshold_is_cost_floor_share": float((d.label_threshold <= cost / 1e4 + 1e-12).mean()),
            "median_abs_move_bps": float(d.fwd_return.abs().median() * 1e4),
            "duplicate_timestamps": int(d.open_ts.duplicated().sum()),
            "n_features": len(feats), "feature_groups": {g: sum(group_of(x) == g for x in feats) for g in list(GROUPS) + ["other"]},
        }

    # ---------------- stage 2: development comparison (walk-forward inside the first 70 %)
    cands = candidates()
    variants = {name: (fn, None) for name, fn in cands.items()}
    # feature-group ablations and "no higher timeframes" with the cheapest strong model to keep run time sane
    for g in GROUPS:
        variants[f"logreg_strong_l2-without_{g}"] = (cands["logreg_strong_l2"], ("drop_group", g))
    variants["logreg_strong_l2-base_tf_only"] = (cands["logreg_strong_l2"], ("base_only", None))
    variants["ensemble_current-base_tf_only"] = (cands["ensemble_current"], ("base_only", None))
    dev = {}
    for vname, (fn, mod) in variants.items():
        dev[vname] = {}
        for (sym, tf), (d, feats) in data.items():
            fs = feats
            if mod and mod[0] == "drop_group":
                fs = [x for x in feats if group_of(x) != mod[1]]
            elif mod and mod[0] == "base_only":
                fs = [x for x in feats if x.startswith(tf + "_") or x.startswith("regime_")]
            n_dev = int(len(d) * 0.7)
            start = int(n_dev * 0.4)
            t0 = time.time()
            P, N, idx = walk(fn, d, fs, h, a.folds, start, n_dev)
            sub = d.iloc[idx]
            dev[vname][f"{sym} {tf}"] = metrics(sub.label.astype(int).to_numpy(), P, N, sub.fwd_return.to_numpy(float),
                                                sub.label_threshold.to_numpy(float), sub.regime.to_numpy().astype(str), cost)
            print(f"dev {vname:40s} {sym} {tf}: gain {dev[vname][f'{sym} {tf}']['gain']:+.4f} "
                  f"p {dev[vname][f'{sym} {tf}']['p_better']:.2f} ({time.time() - t0:.0f}s)", flush=True)
    report["development"] = dev
    mean_gain = {v: float(np.mean([m["gain"] for m in r.values()])) for v, r in dev.items()}
    report["development_mean_gain"] = mean_gain
    chosen = max(mean_gain, key=mean_gain.get)
    report["chosen_on_development"] = chosen

    # ---------------- stage 3: final holdout, evaluated once
    final = {}
    for vname in sorted({chosen, "ensemble_current"}):
        fn, mod = variants[vname]
        final[vname] = {}
        for (sym, tf), (d, feats) in data.items():
            fs = feats
            if mod and mod[0] == "drop_group":
                fs = [x for x in feats if group_of(x) != mod[1]]
            elif mod and mod[0] == "base_only":
                fs = [x for x in feats if x.startswith(tf + "_") or x.startswith("regime_")]
            n_dev = int(len(d) * 0.7)
            tr = d.iloc[: n_dev]
            te = d.iloc[n_dev + h:]
            ytr = tr.label.astype(int).to_numpy()
            model = fn().fit(tr[fs].to_numpy(float), ytr, fs)
            P = model.predict_proba(te[fs].to_numpy(float))
            N = np.tile(np.bincount(ytr, minlength=3) / len(ytr), (len(te), 1))
            final[vname][f"{sym} {tf}"] = metrics(te.label.astype(int).to_numpy(), P, N, te.fwd_return.to_numpy(float),
                                                  te.label_threshold.to_numpy(float), te.regime.to_numpy().astype(str), cost)
            final[vname][f"{sym} {tf}"]["period"] = [int(te.open_ts.iloc[0]), int(te.open_ts.iloc[-1])]
            print(f"HOLDOUT {vname:30s} {sym} {tf}: {final[vname][f'{sym} {tf}']['gain']:+.4f} "
                  f"CI {final[vname][f'{sym} {tf}']['gain_ci95']}", flush=True)
    report["final_holdout"] = final
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1, default=float))
    print("chosen:", chosen, "->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
