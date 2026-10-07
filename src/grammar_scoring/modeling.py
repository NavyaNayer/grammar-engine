"""Estimators, cross-validation and the blended scorer.

Design constraints that drive every choice here:

* **769 training samples, ~150 features.** That ratio punishes flexible models, so the
  zoo is dominated by regularised linear models and small ensembles, and every estimate
  comes from repeated cross-validation rather than a single split.
* **The target is an ordinal Likert scale used as a continuous score.** We regress
  directly on it (the leaderboard metrics are RMSE and Pearson, both continuous) but
  stratify the folds on the rounded label so no fold is missing a score band.
* **Blend weights must be non-negative.** Unconstrained stacking on so few samples
  produces large cancelling weights that do not survive out of sample; non-negative
  least squares keeps the blend interpretable and stable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.model_selection import KFold, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import SVR

from grammar_scoring.config import SCORE_MAX, SCORE_MIN, ModelConfig
from grammar_scoring.evaluate import compute_metrics

LOGGER = logging.getLogger(__name__)


def _preprocessor(scale: bool) -> list[tuple[str, object]]:
    """Shared preprocessing: median imputation, constant-feature removal, scaling.

    Scaling uses a quantile transform rather than standardisation because several
    features (error counts, pause durations) are heavily right-skewed with outliers
    that would otherwise dominate the distance-based and linear models.
    """
    steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("variance", VarianceThreshold(threshold=0.0)),
    ]
    if scale:
        steps.append(
            (
                "scale",
                QuantileTransformer(
                    output_distribution="normal", n_quantiles=200, subsample=100_000
                ),
            )
        )
    return steps


def build_estimator(name: str, random_state: int = 42) -> Pipeline:
    """Construct one named estimator from the zoo."""
    name = name.lower()
    if name == "ridge":
        model = RidgeCV(alphas=np.logspace(-2, 4, 40))
        scale = True
    elif name == "elasticnet":
        model = ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.99],
            alphas=60,
            cv=5,
            random_state=random_state,
            max_iter=20_000,
        )
        scale = True
    elif name == "svr_rbf":
        model = SVR(kernel="rbf", C=3.0, epsilon=0.15, gamma="scale")
        scale = True
    elif name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=3,
            max_features=0.3,
            random_state=random_state,
            n_jobs=-1,
        )
        scale = False
    elif name == "gradient_boosting":
        model = HistGradientBoostingRegressor(
            max_depth=3,
            learning_rate=0.05,
            max_iter=400,
            min_samples_leaf=15,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=random_state,
        )
        scale = False
    elif name == "knn":
        model = KNeighborsRegressor(n_neighbors=15, weights="distance", metric="euclidean")
        scale = True
    else:
        raise ValueError(f"Unknown estimator: {name}")

    return Pipeline([*_preprocessor(scale), ("model", model)])


def make_cv(y: np.ndarray, config: ModelConfig):
    """Repeated K-fold, stratified on the rounded score when possible."""
    if not config.stratify:
        return RepeatedKFold(
            n_splits=config.n_splits, n_repeats=config.n_repeats, random_state=config.random_state
        )

    bins = np.clip(np.round(np.asarray(y, dtype=float)), SCORE_MIN, SCORE_MAX).astype(int)
    _, counts = np.unique(bins, return_counts=True)
    if counts.min() < config.n_splits:
        LOGGER.warning(
            "Rarest score band has %d samples (< %d folds); falling back to unstratified CV.",
            counts.min(),
            config.n_splits,
        )
        return RepeatedKFold(
            n_splits=config.n_splits, n_repeats=config.n_repeats, random_state=config.random_state
        )
    return RepeatedStratifiedKFold(
        n_splits=config.n_splits, n_repeats=config.n_repeats, random_state=config.random_state
    )


def stratification_labels(y: np.ndarray) -> np.ndarray:
    return np.clip(np.round(np.asarray(y, dtype=float)), SCORE_MIN, SCORE_MAX).astype(int)


def clip_scores(values: np.ndarray) -> np.ndarray:
    """Clip predictions to the rubric's defined range."""
    return np.clip(values, SCORE_MIN, SCORE_MAX)


@dataclass
class CVResult:
    """Cross-validated predictions and metrics for one estimator."""

    name: str
    oof_predictions: np.ndarray  # averaged across repeats
    fold_metrics: pd.DataFrame
    metrics: dict[str, float]

    @property
    def rmse(self) -> float:
        return self.metrics["rmse"]

    @property
    def pearson(self) -> float:
        return self.metrics["pearson"]


def cross_validate_estimator(
    X: pd.DataFrame,
    y: np.ndarray,
    name: str,
    config: ModelConfig,
    verbose: bool = True,
) -> CVResult:
    """Repeated CV for one estimator, returning repeat-averaged OOF predictions."""
    X_values = X.to_numpy(dtype=float)
    y = np.asarray(y, dtype=float)
    cv = make_cv(y, config)
    groups = stratification_labels(y)

    # Each sample is held out once per repeat; accumulate and average.
    oof_sum = np.zeros(len(y))
    oof_count = np.zeros(len(y))
    fold_rows: list[dict] = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_values, groups)):
        estimator = build_estimator(name, config.random_state + fold)
        estimator.fit(X_values[train_idx], y[train_idx])
        predictions = estimator.predict(X_values[test_idx])
        if config.clip_predictions:
            predictions = clip_scores(predictions)

        oof_sum[test_idx] += predictions
        oof_count[test_idx] += 1
        fold_metrics = compute_metrics(y[test_idx], predictions)
        fold_rows.append({"model": name, "fold": fold, **fold_metrics})

    oof = np.divide(oof_sum, np.maximum(oof_count, 1))
    result = CVResult(
        name=name,
        oof_predictions=oof,
        fold_metrics=pd.DataFrame(fold_rows),
        metrics=compute_metrics(y, oof),
    )
    if verbose:
        LOGGER.info(
            "%-18s OOF RMSE=%.4f  Pearson=%.4f",
            name,
            result.metrics["rmse"],
            result.metrics["pearson"],
        )
    return result


def nnls_weights(predictions: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative blend weights, normalised to sum to one."""
    from scipy.optimize import nnls

    weights, _ = nnls(predictions, np.asarray(y, dtype=float))
    total = weights.sum()
    if total <= 1e-8:
        return np.full(predictions.shape[1], 1.0 / predictions.shape[1])
    return weights / total


@dataclass
class ZooResult:
    """Everything produced by a full model-zoo run."""

    results: dict[str, CVResult] = field(default_factory=dict)
    blend_weights: dict[str, float] = field(default_factory=dict)
    blend_oof: np.ndarray | None = None
    blend_metrics: dict[str, float] = field(default_factory=dict)
    honest_blend_metrics: dict[str, float] = field(default_factory=dict)
    y_true: np.ndarray | None = None

    def leaderboard(self) -> pd.DataFrame:
        """Per-model metrics, best RMSE first, with the blend appended."""
        rows = [{"model": name, **result.metrics} for name, result in self.results.items()]
        frame = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
        if self.blend_metrics:
            blend = pd.DataFrame([{"model": "blend (stacked)", **self.blend_metrics}])
            frame = pd.concat([frame, blend], ignore_index=True)
        return frame

    @property
    def best_single(self) -> str:
        return min(self.results, key=lambda n: self.results[n].rmse)

    def oof_frame(self) -> pd.DataFrame:
        data = {name: result.oof_predictions for name, result in self.results.items()}
        if self.blend_oof is not None:
            data["blend"] = self.blend_oof
        frame = pd.DataFrame(data)
        if self.y_true is not None:
            frame.insert(0, "actual", self.y_true)
        return frame


def run_model_zoo(
    X: pd.DataFrame,
    y: np.ndarray,
    config: ModelConfig | None = None,
    verbose: bool = True,
) -> ZooResult:
    """Cross-validate every candidate estimator and learn a blend over them."""
    config = config or ModelConfig()
    y = np.asarray(y, dtype=float)

    results = {
        name: cross_validate_estimator(X, y, name, config, verbose=verbose)
        for name in config.candidates
    }
    zoo = ZooResult(results=results, y_true=y)
    if not config.use_stacking or len(results) < 2:
        return zoo

    names = list(results)
    oof_matrix = np.column_stack([results[n].oof_predictions for n in names])
    weights = nnls_weights(oof_matrix, y)
    blend = clip_scores(oof_matrix @ weights) if config.clip_predictions else oof_matrix @ weights

    zoo.blend_weights = {n: float(w) for n, w in zip(names, weights, strict=True)}
    zoo.blend_oof = blend
    zoo.blend_metrics = compute_metrics(y, blend)
    zoo.honest_blend_metrics = _nested_blend_metrics(oof_matrix, y, config)

    if verbose:
        active = {n: round(w, 3) for n, w in zoo.blend_weights.items() if w > 1e-3}
        LOGGER.info("Blend weights: %s", active)
        LOGGER.info(
            "Blend OOF RMSE=%.4f  Pearson=%.4f (nested check RMSE=%.4f)",
            zoo.blend_metrics["rmse"],
            zoo.blend_metrics["pearson"],
            zoo.honest_blend_metrics.get("rmse", float("nan")),
        )
    return zoo


def _nested_blend_metrics(
    oof_matrix: np.ndarray, y: np.ndarray, config: ModelConfig
) -> dict[str, float]:
    """Estimate how well the blend generalises.

    Fitting blend weights on the same OOF predictions we then score is mildly optimistic.
    Here the weights are re-fit inside an outer K-fold and applied to the held-out part,
    which removes that optimism and tells us whether the blend is real or fitted noise.
    """
    outer = KFold(n_splits=config.n_splits, shuffle=True, random_state=config.random_state)
    predictions = np.zeros(len(y))
    for train_idx, test_idx in outer.split(oof_matrix):
        weights = nnls_weights(oof_matrix[train_idx], y[train_idx])
        predictions[test_idx] = oof_matrix[test_idx] @ weights
    if config.clip_predictions:
        predictions = clip_scores(predictions)
    return compute_metrics(y, predictions)


class GrammarScorer(BaseEstimator, RegressorMixin):
    """The fitted end-to-end scorer: a non-negative blend of the zoo's estimators.

    Fitting refits every base estimator on the full training set; the blend weights come
    from cross-validation, so they are never fit on in-sample predictions.
    """

    def __init__(self, config: ModelConfig | None = None, weights: dict[str, float] | None = None):
        self.config = config or ModelConfig()
        self.weights = weights
        self.estimators_: dict[str, Pipeline] = {}
        self.weights_: dict[str, float] = {}
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> GrammarScorer:
        y = np.asarray(y, dtype=float)
        self.feature_names_ = list(X.columns)

        weights = self.weights
        if weights is None:
            zoo = run_model_zoo(X, y, self.config, verbose=self.config is not None)
            weights = zoo.blend_weights or {zoo.best_single: 1.0}

        active = {n: w for n, w in weights.items() if w > 1e-4}
        total = sum(active.values()) or 1.0
        self.weights_ = {n: w / total for n, w in active.items()}

        X_values = X.to_numpy(dtype=float)
        for name in self.weights_:
            estimator = build_estimator(name, self.config.random_state)
            estimator.fit(X_values, y)
            self.estimators_[name] = estimator
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.estimators_:
            raise RuntimeError("GrammarScorer must be fitted before calling predict().")
        X = X.reindex(columns=self.feature_names_)
        X_values = X.to_numpy(dtype=float)
        prediction = sum(
            weight * self.estimators_[name].predict(X_values)
            for name, weight in self.weights_.items()
        )
        return clip_scores(prediction) if self.config.clip_predictions else prediction

    def predict_components(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-base-model predictions, useful for diagnosing a disagreeing ensemble."""
        X = X.reindex(columns=self.feature_names_)
        X_values = X.to_numpy(dtype=float)
        return pd.DataFrame(
            {name: est.predict(X_values) for name, est in self.estimators_.items()},
            index=X.index,
        )
