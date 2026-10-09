"""Guarded self-learning (champion / challenger).

1. No production model yet (or feature-set changed) -> train a baseline,
   evaluated with purged walk-forward (out-of-sample) and registered as production.
2. Otherwise wait until enough NEW labelled bars exist that the production model
   never saw (single errors never trigger retraining).
3. Split the new region chronologically: challenger trains on everything before
   the holdout (minus a purge gap); the holdout is unseen by BOTH models.
4. Promote only if the challenger is better on the holdout:
   - mean log-loss gain >= LEARNING_MIN_LOGLOSS_GAIN,
   - block-bootstrap P(gain > 0) >= LEARNING_BOOTSTRAP_CONFIDENCE,
   - Brier not worse, accuracy not materially worse.
   Every candidate, accepted or rejected, is stored in the registry.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import TIMEFRAME_MS, Settings
from ..db import Database, now_ms
from ..ml.dataset import build_training_frame
from ..ml.evaluation import backtest, baseline_test, full_metrics, walk_forward
from ..ml.features import FEATURE_VERSION
from ..ml.model import procedure_features, PROCEDURE_VERSION, EnsembleModel
from . import registry

log = logging.getLogger("learning")


def _per_sample_logloss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1))


def block_bootstrap_prob(diff: np.ndarray, block: int, n_boot: int = 2000, seed: int = 7) -> float:
    """P(mean(diff) > 0) under a moving-block bootstrap (respects autocorrelation of overlapping labels)."""
    n = len(diff)
    if n == 0:
        return 0.0
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    means = diff[idx].mean(axis=1)
    return float((means > 0).mean())


def compare_models(y: np.ndarray, p_prod: np.ndarray, p_chal: np.ndarray, settings: Settings, horizon: int,
                   regimes: np.ndarray | None = None) -> dict:
    ll_p, ll_c = _per_sample_logloss(y, p_prod), _per_sample_logloss(y, p_chal)
    onehot = np.eye(3)[y]
    brier_p = float(np.mean(np.sum((p_prod - onehot) ** 2, axis=1)))
    brier_c = float(np.mean(np.sum((p_chal - onehot) ** 2, axis=1)))
    acc_p = float(np.mean(p_prod.argmax(1) == y))
    acc_c = float(np.mean(p_chal.argmax(1) == y))
    gain = float(np.mean(ll_p - ll_c))
    prob = block_bootstrap_prob(ll_p - ll_c, block=horizon)
    checks = {
        "logloss_gain_ok": gain >= settings.min_logloss_gain,
        "bootstrap_ok": prob >= settings.bootstrap_confidence,
        "brier_ok": brier_c <= brier_p + 1e-4,
        "accuracy_ok": acc_c >= acc_p - 0.01,
    }
    # stability: the gain must hold in the earlier AND the later half of the holdout (not one lucky stretch)
    half = len(y) // 2
    halves = [float(np.mean((ll_p - ll_c)[:half])) if half else 0.0, float(np.mean((ll_p - ll_c)[half:])) if half else 0.0]
    checks["both_halves_ok"] = all(g > 0 for g in halves)
    by_regime = {}
    if regimes is not None:
        for r in sorted(set(map(str, regimes))):
            m = np.asarray(regimes).astype(str) == r
            if m.sum() >= 1:
                by_regime[r] = {"n": int(m.sum()), "logloss_production": float(ll_p[m].mean()),
                                "logloss_challenger": float(ll_c[m].mean()), "gain": float((ll_p[m] - ll_c[m]).mean())}
        # no regime with enough rows may get clearly worse
        checks["no_regime_worse"] = all(v["gain"] > -0.05 for v in by_regime.values() if v["n"] >= 50)
    return {"n_holdout": int(len(y)), "logloss_production": float(ll_p.mean()), "logloss_challenger": float(ll_c.mean()),
            "logloss_gain": gain, "bootstrap_p_better": prob, "brier_production": brier_p,
            "brier_challenger": brier_c, "accuracy_production": acc_p, "accuracy_challenger": acc_c,
            "halves_gain": halves, "by_regime": by_regime,
            "checks": checks, "promote": all(checks.values())}


def _fit(train: pd.DataFrame, features: list[str]) -> EnsembleModel:
    return EnsembleModel().fit(train[features].to_numpy(float), train["label"].astype(int).to_numpy(), features)


def _payload(model, features, key, version, symbol, tf, settings) -> dict:
    return {"model": model, "features": features, "feature_version": FEATURE_VERSION, "version": version,
            "model_key": key, "symbol": symbol, "timeframe": tf, "horizon": settings.horizon_bars,
            "label_atr_mult": settings.label_atr_mult, "cost_bps": settings.round_trip_cost_bps}


def bootstrap_model(db: Database, settings: Settings, symbol: str, tf: str, frame: pd.DataFrame,
                    features: list[str], reason: str, parent: str | None, wf: dict | None = None,
                    comparison: dict | None = None) -> dict:
    key = registry.model_key(symbol, tf, settings.horizon_bars)
    labelled = frame.dropna(subset=["label"]).reset_index(drop=True)
    if wf is None:
        wf = walk_forward(labelled, features, settings.horizon_bars, settings.walk_forward_folds,
                          settings.min_train_rows, settings.confidence_threshold, settings.min_edge)
    metrics = wf["metrics"]
    metrics["backtest"] = backtest(wf["oos"], tf, settings.horizon_bars, settings.confidence_threshold,
                                   settings.min_edge, settings.fee_bps, settings.slippage_bps,
                                   settings.spread_bps, settings.latency_ms)
    model = _fit(labelled, features)
    version = registry.new_version(key)
    path = registry.save_artifact(settings.model_dir, version, _payload(model, features, key, version, symbol, tf, settings))
    oos_path = str(settings.model_dir / f"{version}.oos.csv")
    wf["oos"].to_csv(oos_path, index=False)
    if parent:
        db.execute("UPDATE model_registry SET status='archived' WHERE version=?", (parent,))
    registry.register(
        db, version=version, key=key, status="production",
        train_start_ts=int(labelled.open_ts.iloc[0]), train_end_ts=int(labelled.open_ts.iloc[-1]),
        n_train=len(labelled), n_validation=int(metrics["n"]), feature_version=FEATURE_VERSION,
        features=features, params={**model.describe(), "oos_path": oos_path, "horizon": settings.horizon_bars,
                                    "procedure": PROCEDURE_VERSION},
        metrics=metrics, comparison=comparison, artifact_path=path, parent_version=parent, reason=reason)
    db.log_event("baseline_trained", {"version": version, "reason": reason, "oos_log_loss": metrics["log_loss"],
                                      "oos_accuracy": metrics["accuracy"], "rows": len(labelled)}, key)
    log.info("%s baseline %s trained (OOS logloss %.4f, acc %.3f)", key, version, metrics["log_loss"], metrics["accuracy"])
    return {"status": "baseline_trained", "version": version, "metrics": {k: metrics[k] for k in ("accuracy", "log_loss", "brier")}}


def procedure_upgrade(db: Database, settings: Settings, symbol: str, tf: str, labelled: pd.DataFrame,
                      features: list[str], prod: dict) -> dict:
    """A new training procedure must beat production on the SAME out-of-sample predictions.

    Both sides are walk-forward out-of-sample predictions (each model trained only on data before
    the predicted bar); they are compared on the timestamps they share.
    """
    key = registry.model_key(symbol, tf, settings.horizon_bars)
    db.set_state(f"procedure_checked:{key}:{PROCEDURE_VERSION}", now_ms())
    prod_oos_path = registry.resolve_path((prod.get("params") or {}).get("oos_path"), settings.model_dir)
    if prod_oos_path is None or not prod_oos_path.exists():
        db.log_event("procedure_upgrade_skipped", {"reason": "production has no stored out-of-sample predictions"}, key)
        return {"status": "waiting", "reason": "no production OOS to compare"}
    wf = walk_forward(labelled, features, settings.horizon_bars, settings.walk_forward_folds, settings.min_train_rows,
                      settings.confidence_threshold, settings.min_edge)
    new_oos = wf["oos"]
    old_oos = pd.read_csv(prod_oos_path)[["open_ts", "p_down", "p_flat", "p_up"]]
    both = new_oos.merge(old_oos, on="open_ts", suffixes=("", "_prod"))
    if len(both) < settings.min_holdout:
        db.log_event("procedure_upgrade_skipped", {"reason": f"only {len(both)} shared out-of-sample rows"}, key)
        return {"status": "waiting", "reason": "not enough shared out-of-sample rows"}
    y = both["label"].astype(int).to_numpy()
    p_new = both[["p_down", "p_flat", "p_up"]].to_numpy()
    p_old = both[["p_down_prod", "p_flat_prod", "p_up_prod"]].to_numpy()
    cmp_ = compare_models(y, p_old, p_new, settings, settings.horizon_bars, both["regime"].to_numpy())
    cmp_["kind"] = "procedure_upgrade"
    if cmp_["promote"]:
        out = bootstrap_model(db, settings, symbol, tf, labelled, features,
                              f"procedure upgrade to {PROCEDURE_VERSION}: out-of-sample log-loss gain "
                              f"{cmp_['logloss_gain']:.4f}, P(better) {cmp_['bootstrap_p_better']:.2f}",
                              prod["version"], wf=wf, comparison=cmp_)
        db.log_event("challenger_promoted", {"version": out["version"], "replaced": prod["version"], **_short(cmp_)}, key)
        out["status"] = "promoted"
        out["comparison"] = _short(cmp_)
    else:
        version = registry.new_version(key)
        registry.register(db, version=version, key=key, status="rejected", train_start_ts=int(labelled.open_ts.iloc[0]),
                          train_end_ts=int(labelled.open_ts.iloc[-1]), n_train=len(labelled), n_validation=len(both),
                          feature_version=FEATURE_VERSION, features=features, params={"procedure": PROCEDURE_VERSION},
                          metrics=wf["metrics"], comparison=cmp_, artifact_path=None, parent_version=prod["version"],
                          reason="procedure upgrade rejected: " + ", ".join(k for k, v in cmp_["checks"].items() if not v))
        db.log_event("challenger_rejected", {"version": version, "production": prod["version"], **_short(cmp_)}, key)
        out = {"status": "rejected", "version": version, "comparison": _short(cmp_)}
    db.set_state(f"learning_status:{key}", {**out, "ts_ms": now_ms()})
    return out


def challenger_cycle(db: Database, settings: Settings, symbol: str, tf: str, force: bool = False) -> dict:
    key = registry.model_key(symbol, tf, settings.horizon_bars)
    frame, features = build_training_frame(db, settings, symbol, tf)
    features = procedure_features(features, tf)
    labelled = frame.dropna(subset=["label"]).reset_index(drop=True) if not frame.empty else frame
    if labelled.empty or len(labelled) < settings.min_train_rows + 250:
        db.log_event("waiting", {"reason": "not_enough_history", "labelled_rows": int(len(labelled))}, key)
        return {"status": "waiting", "reason": "not_enough_history", "labelled_rows": int(len(labelled))}

    prod = registry.get_production(db, key)
    if prod is None:
        return bootstrap_model(db, settings, symbol, tf, frame, features, "initial baseline (no production model)", None)
    if prod["feature_version"] != FEATURE_VERSION:
        return bootstrap_model(db, settings, symbol, tf, frame, features,
                               f"feature version changed {prod['feature_version']} -> {FEATURE_VERSION}", prod["version"])

    if (prod.get("params") or {}).get("procedure") != PROCEDURE_VERSION and \
            not db.get_state(f"procedure_checked:{key}:{PROCEDURE_VERSION}"):
        return procedure_upgrade(db, settings, symbol, tf, labelled, features, prod)

    bar = TIMEFRAME_MS[tf]
    h = settings.horizon_bars
    embargo_start = int(prod["train_end_ts"]) + h * bar  # production's labels saw prices up to here
    new = labelled[labelled.open_ts > embargo_start]
    last_attempt = db.get_state(f"learning_last_attempt:{key}", 0) or 0
    since_attempt = int((labelled.open_ts > last_attempt).sum())
    need = settings.min_new_labels
    if len(new) < need or (since_attempt < need // 2 and not force):
        out = {"status": "waiting", "new_labels": int(len(new)), "required": need,
               "new_since_last_attempt": since_attempt}
        db.set_state(f"learning_status:{key}", {**out, "ts_ms": now_ms()})
        return out

    n_hold = max(settings.min_holdout, int(len(new) * settings.holdout_fraction))
    holdout = new.iloc[-n_hold:]
    if last_attempt:
        # each decision uses rows no earlier decision has seen: repeated testing on the same
        # holdout would eventually promote a lucky challenger
        unseen = new[new.open_ts > last_attempt]
        if len(unseen) < settings.min_holdout:
            out = {"status": "waiting", "reason": "holdout_not_fresh", "unseen_labels": int(len(unseen)),
                   "required": settings.min_holdout}
            db.set_state(f"learning_status:{key}", {**out, "ts_ms": now_ms()})
            return out
        holdout = unseen
    hold_start = int(holdout.open_ts.iloc[0])
    train = labelled[labelled.open_ts < hold_start - h * bar]  # purge gap before the holdout
    y = holdout["label"].astype(int).to_numpy()
    if train["label"].nunique() < 2 or len(np.unique(y)) < 2:
        return {"status": "waiting", "reason": "class_diversity"}

    prod_art = registry.load_artifact(prod["artifact_path"], settings.model_dir)
    prod_feats = prod_art["features"]
    for f in prod_feats:  # a feature can disappear only if the data source changed
        if f not in holdout:
            holdout = holdout.assign(**{f: np.nan})
    p_prod = prod_art["model"].predict_proba(holdout[prod_feats].to_numpy(float))
    challenger = _fit(train, features)
    p_chal = challenger.predict_proba(holdout[features].to_numpy(float))
    cmp_ = compare_models(y, p_prod, p_chal, settings, h, holdout["regime"].to_numpy())
    chal_metrics = full_metrics(y, p_chal, holdout["regime"].to_numpy(), settings.confidence_threshold, settings.min_edge)
    prod_metrics = full_metrics(y, p_prod, holdout["regime"].to_numpy(), settings.confidence_threshold, settings.min_edge)
    oos = holdout[["open_ts", "open", "close", "fwd_return", "label", "label_threshold", "regime"]].copy()
    oos[["p_down", "p_flat", "p_up"]] = p_chal
    chal_metrics["backtest"] = backtest(oos, tf, h, settings.confidence_threshold, settings.min_edge,
                                        settings.fee_bps, settings.slippage_bps, settings.spread_bps, settings.latency_ms)
    chal_metrics["method"] = "chronological holdout unseen by production and challenger"
    prior = np.clip(np.bincount(train["label"].astype(int).to_numpy(), minlength=3) / len(train), 1e-6, 1)
    oos[["p_naive_down", "p_naive_flat", "p_naive_up"]] = prior / prior.sum()
    chal_metrics["baseline_test"] = baseline_test(oos, h, settings.baseline_p_better)
    cmp_["production_metrics"] = {k: prod_metrics[k] for k in ("accuracy", "log_loss", "brier", "ece", "signals", "by_regime")}

    version = registry.new_version(key)
    path = registry.save_artifact(settings.model_dir, version,
                                  _payload(challenger, features, key, version, symbol, tf, settings))
    status = "candidate"
    registry.register(
        db, version=version, key=key, status=status, train_start_ts=int(train.open_ts.iloc[0]),
        train_end_ts=int(train.open_ts.iloc[-1]), n_train=len(train), n_validation=len(holdout),
        feature_version=FEATURE_VERSION, features=features,
        params={**challenger.describe(), "holdout_start": hold_start, "horizon": h, "procedure": PROCEDURE_VERSION},
        metrics=chal_metrics, comparison=cmp_, artifact_path=path, parent_version=prod["version"],
        reason="challenger evaluation")
    db.set_state(f"learning_last_attempt:{key}", int(labelled.open_ts.iloc[-1]))
    if cmp_["promote"]:
        registry.promote(db, key, version,
                         f"promoted: logloss gain {cmp_['logloss_gain']:.4f}, P(better)={cmp_['bootstrap_p_better']:.2f}")
        db.log_event("challenger_promoted", {"version": version, "replaced": prod["version"], **_short(cmp_)}, key)
        out = {"status": "promoted", "version": version, "comparison": _short(cmp_)}
    else:
        db.execute("UPDATE model_registry SET status='rejected', reason=?, artifact_path=NULL WHERE version=?",
                   ("rejected: " + ", ".join(k for k, v in cmp_["checks"].items() if not v), version))
        Path(path).unlink(missing_ok=True)  # metrics stay in the registry; the unused model file is dropped
        db.log_event("challenger_rejected", {"version": version, "production": prod["version"], **_short(cmp_)}, key)
        out = {"status": "rejected", "version": version, "comparison": _short(cmp_)}
    db.set_state(f"learning_status:{key}", {**out, "ts_ms": now_ms()})
    return out


def _short(c: dict) -> dict:
    return {k: c.get(k) for k in ("n_holdout", "logloss_production", "logloss_challenger", "logloss_gain",
                                  "bootstrap_p_better", "accuracy_production", "accuracy_challenger", "checks",
                                  "halves_gain", "by_regime")}


def ledger_review(db: Database, key_symbol: str, tf: str, horizon: int) -> dict:
    """Error taxonomy of resolved live predictions, per model version and regime."""
    rows = db.query(
        "SELECT model_version, prediction, regime, result, error_class, error FROM predictions "
        "WHERE symbol=? AND timeframe=? AND horizon_bars=? AND resolved_ms IS NOT NULL", (key_symbol, tf, horizon))
    if not rows:
        return {"resolved": 0}
    df = pd.DataFrame(rows)
    sig = df[df.prediction != "NO TRADE"]
    out = {
        "resolved": int(len(df)),
        "signals": int(len(sig)),
        "signal_hit_rate": float((sig.result == "correct").mean()) if len(sig) else None,
        "mean_prob_error": float(df.error.mean()),
        "error_classes": {k: int(v) for k, v in df.error_class.value_counts().items()},
        "by_regime": {r: {"n": int(len(g)), "signals": int((g.prediction != "NO TRADE").sum()),
                          "hit_rate": float((g[g.prediction != "NO TRADE"].result == "correct").mean())
                          if (g.prediction != "NO TRADE").any() else None,
                          "mean_prob_error": float(g.error.mean())}
                      for r, g in df.groupby("regime")},
        "by_model": {m: {"n": int(len(g)), "mean_prob_error": float(g.error.mean())} for m, g in df.groupby("model_version")},
    }
    return out


def run_learning_once(db: Database, settings: Settings, force: bool = False) -> dict:
    results = {}
    for symbol in settings.symbols:
        for tf in settings.predict_timeframes:
            key = registry.model_key(symbol, tf, settings.horizon_bars)
            try:
                results[key] = challenger_cycle(db, settings, symbol, tf, force=force)
            except Exception as exc:  # noqa: BLE001 - one market must not stop the others
                log.exception("learning cycle failed for %s", key)
                results[key] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
                db.log_event("error", {"error": results[key]["error"]}, key)
            try:
                review = ledger_review(db, symbol, tf, settings.horizon_bars)
                db.set_state(f"ledger_review:{key}", review)
            except Exception:  # noqa: BLE001
                log.exception("ledger review failed for %s", key)
    db.set_state("learning_last_run", {"ts_ms": now_ms(), "results": json.loads(json.dumps(results, default=str))})
    return results
