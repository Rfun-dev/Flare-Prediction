"""Tangga model — dari yang paling sederhana ke yang lebih rumit.

Semua kelas memakai antarmuka yang sama (`fit`, `predict_proba`) sehingga
modul evaluasi bisa memperlakukannya secara identik. Naik tangga hanya sah
kalau tingkat sebelumnya sudah dikalahkan pada set test yang sama.

  Tingkat 0a  Climatology  — selalu menebak basis rate. Lantai untuk BSS.
  Tingkat 0b  Persistence  — "AR yang baru saja flare akan flare lagi".
                             Lantai NYATA yang harus dikalahkan; sering
                             mengejutkan kuat (TSS ~ 0.4-0.5 untuk kelas M).
  Tingkat 1   Ambang satu fitur — mis. TOTUSJH > c. Dapat dibaca manusia.
  Tingkat 2   Regresi logistik — model ML sejati paling sederhana.
  Tingkat 3   Random Forest / Gradient Boosting — non-linear, interaksi fitur.
  Tingkat 4   CNN di atas piksel magnetogram — ada di `cnn.py`, bukan di sini,
              karena butuh PyTorch (opsional) dan cache citra tersendiri.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ==============================================================================
# Prapemrosesan
# ==============================================================================

class SignedLog(BaseEstimator, TransformerMixin):
    """log10(1 + |x|) dengan tanda dipertahankan.

    Parameter SHARP membentang belasan orde besaran (USFLUX ~ 1e22 Mx,
    MEANGAM ~ 40 derajat). Tanpa kompresi log, regresi logistik akan didominasi
    satu-dua kolom bernilai raksasa dan gagal konvergen dengan mulus.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xa = np.asarray(X, dtype=float)
        return np.sign(Xa) * np.log10(1.0 + np.abs(Xa))


def _num(X) -> np.ndarray:
    return np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


# ==============================================================================
# Tingkat 0 — baseline tanpa pembelajaran
# ==============================================================================

class ClimatologyBaseline:
    """Selalu mengembalikan basis rate set latih. TSS-nya nol menurut definisi."""

    name = "0a. Climatology"

    def fit(self, X, y):
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.rate_)
        return np.column_stack([1 - p, p])


class PersistenceBaseline:
    """Prediksi = apakah AR ini sudah flare >= ambang pada window observasi.

    Tidak ada parameter yang dipelajari. Ini benchmark wajib di literatur:
    model ML yang tidak mengalahkannya tidak layak dipublikasikan.
    """

    name = "0b. Persistence"

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        # X di sini adalah vektor biner y_persistence (sudah dihitung saat labeling)
        p = _num(X).ravel().astype(float)
        return np.column_stack([1 - p, p])


# ==============================================================================
# Tingkat 1 — ambang satu fitur
# ==============================================================================

class SingleFeatureThreshold:
    """Pilih SATU fitur SHARP dan satu ambang yang memaksimalkan TSS di set latih.

    Ini adalah pendekatan Bloomfield et al. (2012) / Barnes & Leka: sangat
    dapat ditafsirkan, dan menjadi tolok ukur "apakah ML kompleks memang perlu".
    """

    name = "1. Ambang satu fitur"

    def __init__(self, n_grid: int = 200):
        self.n_grid = n_grid

    def fit(self, X: pd.DataFrame, y):
        from .evaluate import deterministic_metrics

        y = np.asarray(y).astype(int)
        best = (-2.0, None, None, 1)
        for col in X.columns:
            v = _num(X[col].values)
            qs = np.unique(np.quantile(v, np.linspace(0.01, 0.99, self.n_grid)))
            for sign in (1, -1):
                for thr in qs:
                    pred = ((v * sign) >= (thr * sign)).astype(int)
                    tss = deterministic_metrics(y, pred)["TSS"]
                    if tss > best[0]:
                        best = (tss, col, float(thr), sign)
        self.train_tss_, self.feature_, self.threshold_, self.sign_ = best
        # Skala nilai fitur menjadi pseudo-probabilitas agar ROC/PR tetap bisa dihitung
        v = _num(X[self.feature_].values) * self.sign_
        self._lo, self._hi = float(np.min(v)), float(np.max(v))
        return self

    def predict_proba(self, X: pd.DataFrame):
        v = _num(X[self.feature_].values) * self.sign_
        span = max(self._hi - self._lo, 1e-12)
        p = np.clip((v - self._lo) / span, 0.0, 1.0)
        return np.column_stack([1 - p, p])

    def describe(self) -> str:
        op = ">=" if self.sign_ == 1 else "<="
        return (f"prakiraan positif bila {self.feature_} {op} {self.threshold_:.4g} "
                f"(TSS latih {self.train_tss_:.3f})")


# ==============================================================================
# Tingkat 2 & 3 — model scikit-learn
# ==============================================================================

def make_logistic(random_state: int = 42) -> Pipeline:
    """Model ML paling sederhana: linear, terkalibrasi wajar, koefisiennya terbaca.

    `class_weight="balanced"` penting: tanpa itu, dengan basis rate ~2%,
    solusi optimal loss adalah memprediksi nol untuk semuanya.
    """
    return Pipeline([
        ("log", SignedLog()),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0, max_iter=5000, class_weight="balanced",
            solver="lbfgs", random_state=random_state)),
    ])


class ProbabilityCalibrator:
    """Kalibrasi isotonik di atas model apa pun yang punya `predict_proba`.

    `class_weight="balanced"` (dan resampling apa pun) sengaja mendistorsi
    probabilitas keluaran agar ambang keputusan bergeser — konsekuensinya
    Brier, BSS, dan diagram reliabilitas menjadi tidak bermakna, dan BSS bisa
    jauh negatif meski TSS bagus. Pemetaan monoton ini memulihkan makna
    probabilistiknya TANPA mengubah urutan skor, jadi TSS, ROC-AUC, dan PR-AUC
    sama persis seperti sebelumnya.

    HARUS di-fit pada data yang TIDAK dipakai melatih model dasar, dan pada
    potongan yang berbeda dari yang dipakai memilih ambang keputusan.
    """

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        from sklearn.isotonic import IsotonicRegression

        p = self.model.predict_proba(X)[:, 1]
        self.iso_ = IsotonicRegression(
            out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, np.asarray(y, dtype=float))
        return self

    def predict_proba(self, X):
        p = np.clip(self.iso_.predict(self.model.predict_proba(X)[:, 1]), 0.0, 1.0)
        return np.column_stack([1 - p, p])


def make_random_forest(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400, max_depth=12, min_samples_leaf=20,
        max_features="sqrt", class_weight="balanced_subsample",
        n_jobs=-1, random_state=random_state)


def make_gradient_boosting(random_state: int = 42) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0,
        class_weight="balanced", early_stopping=True,
        validation_fraction=0.15, random_state=random_state)


MODEL_FACTORY = {
    "2. Regresi logistik": make_logistic,
    "3a. Random Forest": make_random_forest,
    "3b. Gradient Boosting": make_gradient_boosting,
}


def unwrap(model):
    """Kupas pembungkus kalibrasi untuk mendapatkan estimator dasarnya."""
    return model.model if isinstance(model, ProbabilityCalibrator) else model


def extract_importance(model, feature_names: list[str]) -> tuple[np.ndarray, str]:
    """Ambil koefisien/kepentingan fitur apa pun jenis modelnya."""
    model = unwrap(model)
    est = model[-1] if isinstance(model, Pipeline) else model
    if hasattr(est, "coef_"):
        return est.coef_.ravel(), "Koefisien (satuan standar)"
    if hasattr(est, "feature_importances_"):
        return est.feature_importances_, "Kepentingan fitur (impurity)"
    return np.zeros(len(feature_names)), "tidak tersedia"


def permutation_importance_tss(model, X, y, feature_names, threshold=0.5,
                               n_repeats: int = 5, random_state: int = 42):
    """Kepentingan permutasi diukur dengan TSS.

    Lebih jujur daripada importance berbasis impurity, yang bias terhadap
    fitur berkardinalitas tinggi dan tidak peduli pada metrik yang kita pakai.
    """
    from .evaluate import deterministic_metrics

    rng = np.random.default_rng(random_state)
    X = pd.DataFrame(X, columns=feature_names).reset_index(drop=True)
    y = np.asarray(y).astype(int)

    base = deterministic_metrics(y, (model.predict_proba(X)[:, 1] >= threshold).astype(int))["TSS"]
    drops = np.zeros(len(feature_names))
    for j, col in enumerate(feature_names):
        vals = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].values)
            p = (model.predict_proba(Xp)[:, 1] >= threshold).astype(int)
            vals.append(base - deterministic_metrics(y, p)["TSS"])
        drops[j] = float(np.mean(vals))
    return drops
