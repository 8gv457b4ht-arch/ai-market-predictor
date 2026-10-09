"""Three-model ensemble (Logistic Regression + Random Forest + HistGradientBoosting)
producing calibrated P(DOWN), P(FLAT), P(UP).

Calibration: the ensemble is first fitted on the earliest 80% of the training
window, a single temperature is fitted on the chronologically later 20%
(never on random folds), then the components are refitted on 100% of the
window. The walk-forward evaluation measures this whole procedure out of
sample, so the reported calibration is honest.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

N_CLASSES = 3
# Training/calibration procedure. A change here is evaluated head-to-head against production
# before it can replace it (see learning.engine.procedure_upgrade).
PROCEDURE_VERSION = "ens3-base-tf-features"


def procedure_features(features: list[str], base_tf: str) -> list[str]:
    """Inputs used by the current procedure. Research on real data (docs/RESEARCH_2026-10-09.md): the 4h/1d
    context features made out-of-sample log loss worse (few independent values in the available history),
    so the model uses the base timeframe and the regime flags only. Higher timeframes still define regimes."""
    keep = [f for f in features if f.startswith(base_tf + "_") or f.startswith("regime_")]
    return keep or list(features)
EPS = 1e-6


def _components(seed: int = 42, balanced: bool = False) -> dict:
    cw = "balanced" if balanced else None
    return {
        "logreg": Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=0.5, max_iter=2000, class_weight=cw)),
        ]),
        "random_forest": Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=25,
                                           max_features="sqrt", class_weight="balanced_subsample" if balanced else None,
                                           random_state=seed, n_jobs=-1)),
        ]),
        # Imputer first: HGB's binning fails on columns that are entirely NaN inside a training window.
        "hist_gb": Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("clf", HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=200, max_depth=4, min_samples_leaf=40, l2_regularization=1.0,
                class_weight=cw, early_stopping=False, random_state=seed)),
        ]),
    }


def _full_proba(model, X, n_classes=N_CLASSES) -> np.ndarray:
    """predict_proba aligned to classes 0..n-1 even if a class was absent in training."""
    p = model.predict_proba(X)
    out = np.zeros((len(X), n_classes))
    for j, cls in enumerate(model.classes_):
        out[:, int(cls)] = p[:, j]
    return out


def _softmax_temp(p: np.ndarray, t: float, bias: np.ndarray | None = None) -> np.ndarray:
    z = np.log(np.clip(p, EPS, 1)) / t
    if bias is not None:
        z = z + bias
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class EnsembleModel:
    def __init__(self, weights: dict | None = None, seed: int = 42, balanced: bool = False, calibration: str = "temperature_bias"):
        # balanced=True was the first release's behaviour. On real BTC/ETH data it pushed P(UP)/P(DOWN)
        # far above their true frequencies (out-of-sample log loss worse than the naive prior).
        self.balanced = balanced
        self.calibration = calibration
        self.bias = np.zeros(N_CLASSES)
        self.weights = weights or {"logreg": 1.0, "random_forest": 1.0, "hist_gb": 1.0}
        self.seed = seed
        self.models: dict = {}
        self.temperature = 1.0
        self.feature_names: list[str] = []
        self.class_prior = np.full(N_CLASSES, 1 / N_CLASSES)

    def _fit_components(self, X, y) -> dict:
        models = _components(self.seed, self.balanced)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for m in models.values():
                m.fit(X, y)
        return models

    def _raw_proba(self, models, X) -> np.ndarray:
        total = sum(self.weights.values())
        p = sum(self.weights[k] * _full_proba(m, X) for k, m in models.items()) / total
        p = np.clip(p, EPS, 1)
        return p / p.sum(axis=1, keepdims=True)

    def fit(self, X, y, feature_names: list[str] | None = None, calibrate: bool = True) -> "EnsembleModel":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if len(np.unique(y)) < 2:
            raise ValueError("training labels contain a single class")
        self.feature_names = list(feature_names or [])
        self.class_prior = np.bincount(y, minlength=N_CLASSES) / len(y)
        self.temperature, self.bias = 1.0, np.zeros(N_CLASSES)
        n_cal = int(len(y) * 0.2)
        if calibrate and n_cal >= 100 and len(np.unique(y[:-n_cal])) >= 2:
            early = self._fit_components(X[:-n_cal], y[:-n_cal])
            p_cal = self._raw_proba(early, X[-n_cal:])
            yc = y[-n_cal:]
            if self.calibration == "temperature_bias":
                # temperature + per-class bias (vector scaling): also corrects a shift in base rates
                def nll(theta):
                    q = _softmax_temp(p_cal, np.exp(theta[0]), np.array([theta[1], theta[2], 0.0]))
                    return -np.mean(np.log(q[np.arange(len(yc)), yc] + EPS)) + 1e-3 * (theta[1] ** 2 + theta[2] ** 2)
                res = minimize(nll, np.zeros(3), method="L-BFGS-B",
                               bounds=[(np.log(0.3), np.log(5.0)), (-3, 3), (-3, 3)])
                if res.success or res.fun < nll(np.zeros(3)):
                    self.temperature = float(np.exp(res.x[0]))
                    self.bias = np.array([res.x[1], res.x[2], 0.0])
            else:
                def nll_t(t):
                    q = _softmax_temp(p_cal, t)
                    return -np.mean(np.log(q[np.arange(len(yc)), yc] + EPS))
                res = minimize_scalar(nll_t, bounds=(0.3, 5.0), method="bounded")
                self.temperature = float(res.x) if res.success else 1.0
        self.models = self._fit_components(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return _softmax_temp(self._raw_proba(self.models, X), self.temperature, getattr(self, "bias", None))

    def component_proba(self, X) -> dict[str, np.ndarray]:
        X = np.asarray(X, dtype=float)
        return {k: _full_proba(m, X) for k, m in self.models.items()}

    def factors(self, x, cls: int, top: int = 5) -> list[dict]:
        """Which inputs moved P(cls) most: each feature in turn is replaced by its training median
        (the imputer statistic) and the change of P(cls) is measured. Local, model-agnostic, approximate."""
        x = np.asarray(x, dtype=float).reshape(1, -1)
        try:
            med = np.asarray(self.models["logreg"].named_steps["impute"].statistics_, dtype=float)
        except (KeyError, AttributeError):
            return []
        if len(med) != x.shape[1]:
            return []
        base = float(self.predict_proba(x)[0, cls])
        idx = [i for i in range(x.shape[1]) if np.isfinite(x[0, i]) and abs(x[0, i] - med[i]) > 1e-12]
        if not idx:
            return []
        Xp = np.repeat(x, len(idx), axis=0)
        for r, i in enumerate(idx):
            Xp[r, i] = med[i]
        p = self.predict_proba(Xp)[:, cls]
        out = [{"feature": self.feature_names[i] if i < len(self.feature_names) else str(i),
                "value": float(x[0, i]), "median": float(med[i]), "effect": float(base - p[r])}
               for r, i in enumerate(idx)]
        out.sort(key=lambda d: -abs(d["effect"]))
        return out[:top]

    def describe(self) -> dict:
        return {"components": list(self.models), "weights": self.weights,
                "temperature": self.temperature, "bias": [float(b) for b in getattr(self, "bias", np.zeros(3))],
                "balanced_class_weights": getattr(self, "balanced", True), "n_features": len(self.feature_names),
                "class_prior": {"DOWN": self.class_prior[0], "FLAT": self.class_prior[1], "UP": self.class_prior[2]}}
