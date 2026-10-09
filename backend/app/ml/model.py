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
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

N_CLASSES = 3
EPS = 1e-6


def _components(seed: int = 42) -> dict:
    return {
        "logreg": Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced")),
        ]),
        "random_forest": Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=25,
                                           max_features="sqrt", class_weight="balanced_subsample",
                                           random_state=seed, n_jobs=-1)),
        ]),
        # Imputer first: HGB's binning fails on columns that are entirely NaN inside a training window.
        "hist_gb": Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("clf", HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=200, max_depth=4, min_samples_leaf=40, l2_regularization=1.0,
                class_weight="balanced", early_stopping=False, random_state=seed)),
        ]),
    }


def _full_proba(model, X, n_classes=N_CLASSES) -> np.ndarray:
    """predict_proba aligned to classes 0..n-1 even if a class was absent in training."""
    p = model.predict_proba(X)
    out = np.zeros((len(X), n_classes))
    for j, cls in enumerate(model.classes_):
        out[:, int(cls)] = p[:, j]
    return out


def _softmax_temp(p: np.ndarray, t: float) -> np.ndarray:
    z = np.log(np.clip(p, EPS, 1)) / t
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class EnsembleModel:
    def __init__(self, weights: dict | None = None, seed: int = 42):
        self.weights = weights or {"logreg": 1.0, "random_forest": 1.0, "hist_gb": 1.0}
        self.seed = seed
        self.models: dict = {}
        self.temperature = 1.0
        self.feature_names: list[str] = []
        self.class_prior = np.full(N_CLASSES, 1 / N_CLASSES)

    def _fit_components(self, X, y) -> dict:
        models = _components(self.seed)
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
        self.temperature = 1.0
        n_cal = int(len(y) * 0.2)
        if calibrate and n_cal >= 100 and len(np.unique(y[:-n_cal])) >= 2:
            early = self._fit_components(X[:-n_cal], y[:-n_cal])
            p_cal = self._raw_proba(early, X[-n_cal:])
            yc = y[-n_cal:]

            def nll(t):
                q = _softmax_temp(p_cal, t)
                return -np.mean(np.log(q[np.arange(len(yc)), yc] + EPS))
            res = minimize_scalar(nll, bounds=(0.3, 5.0), method="bounded")
            self.temperature = float(res.x) if res.success else 1.0
        self.models = self._fit_components(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return _softmax_temp(self._raw_proba(self.models, X), self.temperature)

    def component_proba(self, X) -> dict[str, np.ndarray]:
        X = np.asarray(X, dtype=float)
        return {k: _full_proba(m, X) for k, m in self.models.items()}

    def describe(self) -> dict:
        return {"components": list(self.models), "weights": self.weights,
                "temperature": self.temperature, "n_features": len(self.feature_names),
                "class_prior": {"DOWN": self.class_prior[0], "FLAT": self.class_prior[1], "UP": self.class_prior[2]}}
