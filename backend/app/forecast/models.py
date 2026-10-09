"""Forecast model for one symbol x horizon:
  * calibrated 3-class probabilities: return above +cost (UP), below -cost (DOWN), in between (FLAT)
  * return quantiles q10 / q50 / q90 in bps (expected range and median move)
and the naive baseline it must beat (training-window class frequencies and return quantiles).
Deterministic: fixed seeds, no shuffling, so a stored input reproduces the same forecast."""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EPS = 1e-6
QUANTILES = (0.1, 0.5, 0.9)
MODEL_FAMILY = "fc1-lr-hgb-cal+hgb-quantiles"


def labels(fwd_bps: np.ndarray, cost_bps: float) -> np.ndarray:
    return np.where(fwd_bps > cost_bps, 2, np.where(fwd_bps < -cost_bps, 0, 1)).astype(int)


class Baseline:
    """What you would say knowing only the past distribution: class frequencies and return quantiles."""
    def __init__(self, fwd_bps: np.ndarray, cost_bps: float):
        y = labels(fwd_bps, cost_bps)
        p = np.bincount(y, minlength=3) / max(1, len(y))
        p = np.clip(p, 1e-4, 1)
        self.prior = p / p.sum()
        self.q = {q: float(np.quantile(fwd_bps, q)) for q in QUANTILES} if len(fwd_bps) else {q: 0.0 for q in QUANTILES}

    def proba(self, n: int) -> np.ndarray:
        return np.tile(self.prior, (n, 1))


def _softmax_temp(p, t, bias):
    z = np.log(np.clip(p, EPS, 1)) / t + bias
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class ForecastModel:
    def __init__(self, seed: int = 7):
        self.seed = seed
        self.features: list[str] = []
        self.classes_present: list[int] = []
        self.temperature, self.bias = 1.0, np.zeros(3)
        self.classifiers: list = []
        self.quantile_models: dict = {}
        self.prior = np.full(3, 1 / 3)
        self.degenerate = False

    def _clf(self):
        imp = ("impute", SimpleImputer(strategy="median", keep_empty_features=True))
        return [Pipeline([imp, ("scale", StandardScaler()), ("clf", LogisticRegression(C=0.1, max_iter=2000))]),
                Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                          ("clf", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=150, max_depth=3,
                                                                 min_samples_leaf=60, l2_regularization=1.0,
                                                                 early_stopping=False, random_state=self.seed))])]

    def _raw(self, models, X):
        out = np.zeros((len(X), 3))
        for m in models:
            p = m.predict_proba(X)
            for j, c in enumerate(m.classes_):
                out[:, int(c)] += p[:, j]
        out /= len(models)
        out = np.clip(out, EPS, 1)
        return out / out.sum(axis=1, keepdims=True)

    def fit(self, X, fwd_bps, cost_bps: float, features: list[str]) -> "ForecastModel":
        X = np.asarray(X, dtype=float)
        y = labels(fwd_bps, cost_bps)
        self.features = list(features)
        self.prior = np.clip(np.bincount(y, minlength=3) / len(y), 1e-4, 1)
        self.prior /= self.prior.sum()
        counts = np.bincount(y, minlength=3)
        self.classes_present = [int(c) for c in range(3) if counts[c] > 0]
        # moves beyond the costs are too rare to learn a classifier: probabilities stay at the base rates
        self.degenerate = (counts[0] < 30 or counts[2] < 30)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if not self.degenerate:
                n_cal = int(len(y) * 0.2)
                early = [m.fit(X[:-n_cal], y[:-n_cal]) for m in self._clf()]
                pc, yc = self._raw(early, X[-n_cal:]), y[-n_cal:]

                def nll(th):
                    q = _softmax_temp(pc, np.exp(th[0]), np.array([th[1], th[2], 0.0]))
                    return -np.mean(np.log(q[np.arange(len(yc)), yc] + EPS)) + 1e-3 * (th[1] ** 2 + th[2] ** 2)
                r = minimize(nll, np.zeros(3), method="L-BFGS-B", bounds=[(np.log(0.3), np.log(5)), (-3, 3), (-3, 3)])
                if r.fun < nll(np.zeros(3)):
                    self.temperature, self.bias = float(np.exp(r.x[0])), np.array([r.x[1], r.x[2], 0.0])
                self.classifiers = [m.fit(X, y) for m in self._clf()]
            for q in QUANTILES:
                m = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                              ("reg", HistGradientBoostingRegressor(loss="quantile", quantile=q, learning_rate=0.05,
                                                                    max_iter=150, max_depth=3, min_samples_leaf=60,
                                                                    early_stopping=False, random_state=self.seed))])
                self.quantile_models[q] = m.fit(X, fwd_bps)
        return self

    def proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.degenerate or not self.classifiers:
            return np.tile(self.prior, (len(X), 1))
        return _softmax_temp(self._raw(self.classifiers, X), self.temperature, self.bias)

    def quantiles(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        qs = np.column_stack([self.quantile_models[q].predict(X) for q in QUANTILES])
        return np.sort(qs, axis=1)  # no crossing quantiles

    def describe(self) -> dict:
        return {"family": MODEL_FAMILY, "seed": self.seed, "n_features": len(self.features), "temperature": self.temperature,
                "bias": [float(b) for b in self.bias], "degenerate_classifier": self.degenerate,
                "prior": {"DOWN": float(self.prior[0]), "FLAT": float(self.prior[1]), "UP": float(self.prior[2])}}
