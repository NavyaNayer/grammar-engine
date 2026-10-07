"""Metrics and visualisations.

The leaderboard uses Pearson correlation and RMSE, so those lead. The extra metrics are
there because they answer different questions: RMSE says how far off we are, Pearson
says whether we rank candidates correctly, and quadratic-weighted kappa says whether we
land in the right rubric band -- which is what a hiring decision actually consumes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from grammar_scoring.config import SCORE_MAX, SCORE_MIN

#: Consistent colours across every figure in the notebook.
PALETTE = {
    "primary": "#2563eb",
    "secondary": "#f97316",
    "accent": "#10b981",
    "muted": "#94a3b8",
    "danger": "#dc2626",
}


# ----------------------------------------------------------------------- metrics
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson r, returning 0.0 when either input is constant (r undefined)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.std() < 1e-12 or y_pred.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if np.std(y_pred) < 1e-12:
        return 0.0
    return float(spearmanr(y_true, y_pred).statistic)


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """QWK after rounding both sides to the nearest rubric level."""
    from sklearn.metrics import cohen_kappa_score

    true_bins = np.clip(np.round(y_true), SCORE_MIN, SCORE_MAX).astype(int)
    pred_bins = np.clip(np.round(y_pred), SCORE_MIN, SCORE_MAX).astype(int)
    if len(np.unique(true_bins)) < 2 or len(np.unique(pred_bins)) < 2:
        return 0.0
    labels = list(range(int(SCORE_MIN), int(SCORE_MAX) + 1))
    return float(cohen_kappa_score(true_bins, pred_bins, weights="quadratic", labels=labels))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """All reported metrics for one set of predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_true - y_pred
    denominator = np.sum((y_true - y_true.mean()) ** 2)

    return {
        "rmse": rmse(y_true, y_pred),
        "pearson": pearson(y_true, y_pred),
        "mae": float(np.mean(np.abs(residual))),
        "spearman": spearman(y_true, y_pred),
        "r2": float(1.0 - np.sum(residual**2) / denominator) if denominator > 0 else 0.0,
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
        # Share of candidates placed within half a rubric level of the human score.
        "within_0.5": float(np.mean(np.abs(residual) <= 0.5)),
        "within_1.0": float(np.mean(np.abs(residual) <= 1.0)),
    }


def metrics_table(metrics: dict[str, float], title: str = "Metrics") -> pd.DataFrame:
    return pd.DataFrame({title: metrics}).T


def baseline_metrics(y_true: np.ndarray) -> dict[str, float]:
    """Metrics for the mean-prediction baseline: the bar any model must clear."""
    y_true = np.asarray(y_true, dtype=float)
    return compute_metrics(y_true, np.full_like(y_true, y_true.mean()))


# ---------------------------------------------------------------------- plotting
def _setup(ax=None, figsize=(8, 5)):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def apply_plot_style() -> None:
    """Consistent, readable defaults for every figure."""
    import matplotlib as mpl
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 140,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": "#cbd5e1",
            "grid.color": "#e2e8f0",
            "legend.frameon": False,
            "figure.facecolor": "white",
        }
    )


def plot_label_distribution(y: np.ndarray, ax=None, title: str = "Grammar score distribution"):
    """Histogram of the training labels with the mean marked."""
    import matplotlib.pyplot as plt

    ax = _setup(ax)
    y = np.asarray(y, dtype=float)
    bins = np.arange(SCORE_MIN - 0.25, SCORE_MAX + 0.75, 0.5)
    ax.hist(y, bins=bins, color=PALETTE["primary"], edgecolor="white", alpha=0.85)
    ax.axvline(
        y.mean(), color=PALETTE["danger"], linestyle="--", label=f"mean = {y.mean():.2f}"
    )
    ax.set_xlabel("MOS Likert grammar score")
    ax.set_ylabel("Number of samples")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return ax


def plot_duration_distribution(durations: np.ndarray, ax=None):
    ax = _setup(ax)
    durations = np.asarray(durations, dtype=float)
    ax.hist(durations, bins=30, color=PALETTE["accent"], edgecolor="white", alpha=0.85)
    ax.axvline(
        float(np.median(durations)),
        color=PALETTE["danger"],
        linestyle="--",
        label=f"median = {np.median(durations):.1f}s",
    )
    ax.set_xlabel("Clip duration (s)")
    ax.set_ylabel("Number of samples")
    ax.set_title("Audio duration distribution")
    ax.legend()
    return ax


def plot_waveform_with_segments(audio, title: str | None = None, figsize=(12, 6)):
    """Waveform with the detected voiced runs shaded, above its mel spectrogram.

    This is the diagnostic that shows the pause segmentation is behaving: the shaded
    bands should track the speech, and the gaps between them are what the pause
    features measure.
    """
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    times = np.arange(len(audio.waveform)) / audio.sample_rate

    axes[0].plot(times, audio.waveform, linewidth=0.4, color=PALETTE["primary"])
    for start, end in audio.voiced_intervals:
        axes[0].axvspan(start, end, color=PALETTE["accent"], alpha=0.18)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(title or f"{audio.path.name} - waveform with detected speech regions")

    mel = librosa.feature.melspectrogram(
        y=audio.waveform, sr=audio.sample_rate, n_mels=96, fmax=8000
    )
    librosa.display.specshow(
        librosa.power_to_db(mel, ref=np.max),
        sr=audio.sample_rate,
        x_axis="time",
        y_axis="mel",
        ax=axes[1],
        cmap="magma",
    )
    axes[1].set_title("Mel spectrogram")
    plt.tight_layout()
    return fig


def target_correlations(X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    """Pearson correlation of every feature with the target, strongest first."""
    y = pd.Series(np.asarray(y, dtype=float), index=X.index)
    numeric = X.select_dtypes(include=[np.number])
    correlations = numeric.apply(lambda column: column.corr(y))
    return correlations.dropna().sort_values(key=np.abs, ascending=False)


def plot_top_correlations(X: pd.DataFrame, y: np.ndarray, top_n: int = 20, ax=None):
    """Horizontal bar chart of the features most correlated with the grammar score."""
    ax = _setup(ax, figsize=(8, max(4, top_n * 0.32)))
    correlations = target_correlations(X, y).head(top_n)[::-1]
    colours = [
        PALETTE["primary"] if value > 0 else PALETTE["secondary"] for value in correlations
    ]
    ax.barh(correlations.index, correlations.to_numpy(), color=colours)
    ax.axvline(0, color="#475569", linewidth=0.8)
    ax.set_xlabel("Pearson correlation with grammar score")
    ax.set_title(f"Top {top_n} features by correlation with the target")
    return ax


def plot_correlation_heatmap(X: pd.DataFrame, features: list[str], ax=None):
    """Correlation structure among the strongest features (collinearity check)."""
    import seaborn as sns

    ax = _setup(ax, figsize=(9, 7.5))
    matrix = X[features].corr()
    sns.heatmap(
        matrix,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.4,
        cbar_kws={"shrink": 0.7, "label": "Pearson r"},
        ax=ax,
    )
    ax.set_title("Correlation between top features")
    return ax


def plot_predictions_vs_actual(y_true, y_pred, ax=None, title: str = "Out-of-fold predictions"):
    """Scatter of predicted against actual, with the identity and best-fit lines."""
    ax = _setup(ax, figsize=(6.5, 6))
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Jitter the (discrete) true scores horizontally so overlapping points stay visible.
    jitter = np.random.default_rng(0).normal(0, 0.045, size=len(y_true))
    ax.scatter(
        y_true + jitter, y_pred, alpha=0.45, s=26, color=PALETTE["primary"], edgecolor="none"
    )

    limits = [SCORE_MIN - 0.3, SCORE_MAX + 0.3]
    ax.plot(limits, limits, "--", color="#475569", linewidth=1.2, label="perfect prediction")
    slope, intercept = np.polyfit(y_true, y_pred, 1)
    grid = np.linspace(*limits, 50)
    ax.plot(
        grid, slope * grid + intercept, color=PALETTE["secondary"], linewidth=1.8, label="best fit"
    )

    metrics = compute_metrics(y_true, y_pred)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Actual grammar score")
    ax.set_ylabel("Predicted grammar score")
    ax.set_title(f"{title}\nRMSE = {metrics['rmse']:.3f}   Pearson r = {metrics['pearson']:.3f}")
    ax.legend(loc="upper left")
    return ax


def plot_residuals(y_true, y_pred, figsize=(12, 4.5)):
    """Residuals against the true score, plus the residual distribution."""
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_pred - y_true

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    jitter = np.random.default_rng(1).normal(0, 0.045, size=len(y_true))
    axes[0].scatter(
        y_true + jitter, residual, alpha=0.45, s=24, color=PALETTE["primary"], edgecolor="none"
    )
    axes[0].axhline(0, color=PALETTE["danger"], linestyle="--")
    axes[0].set_xlabel("Actual grammar score")
    axes[0].set_ylabel("Residual (predicted - actual)")
    axes[0].set_title("Residuals across the score range")

    axes[1].hist(residual, bins=30, color=PALETTE["accent"], edgecolor="white", alpha=0.85)
    axes[1].axvline(0, color=PALETTE["danger"], linestyle="--")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].set_title(
        f"Residual distribution (mean = {residual.mean():.3f}, sd = {residual.std():.3f})"
    )
    plt.tight_layout()
    return fig


def plot_model_comparison(leaderboard: pd.DataFrame, figsize=(11, 4.5)):
    """Side-by-side RMSE (lower better) and Pearson (higher better) per model."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    frame = leaderboard.copy()
    colours = [
        PALETTE["accent"] if "blend" in str(name) else PALETTE["primary"]
        for name in frame["model"]
    ]

    axes[0].barh(frame["model"], frame["rmse"], color=colours)
    axes[0].set_xlabel("RMSE (lower is better)")
    axes[0].set_title("Cross-validated RMSE")
    axes[0].invert_yaxis()
    for i, value in enumerate(frame["rmse"]):
        axes[0].text(value, i, f" {value:.3f}", va="center", fontsize=9)

    axes[1].barh(frame["model"], frame["pearson"], color=colours)
    axes[1].set_xlabel("Pearson r (higher is better)")
    axes[1].set_title("Cross-validated Pearson correlation")
    axes[1].invert_yaxis()
    for i, value in enumerate(frame["pearson"]):
        axes[1].text(value, i, f" {value:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    return fig


def plot_score_band_confusion(y_true, y_pred, ax=None):
    """Confusion matrix after rounding both sides to rubric levels."""
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    ax = _setup(ax, figsize=(6, 5))
    labels = list(range(int(SCORE_MIN), int(SCORE_MAX) + 1))
    true_bins = np.clip(np.round(y_true), SCORE_MIN, SCORE_MAX).astype(int)
    pred_bins = np.clip(np.round(y_pred), SCORE_MIN, SCORE_MAX).astype(int)
    matrix = confusion_matrix(true_bins, pred_bins, labels=labels)

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Samples"},
        ax=ax,
    )
    ax.set_xlabel("Predicted band (rounded)")
    ax.set_ylabel("Actual band (rounded)")
    ax.set_title(f"Rubric-band agreement (QWK = {quadratic_weighted_kappa(y_true, y_pred):.3f})")
    return ax


def error_by_band(y_true, y_pred) -> pd.DataFrame:
    """Per-rubric-band error breakdown, to expose where the model is weakest."""
    frame = pd.DataFrame({"actual": np.asarray(y_true, float), "pred": np.asarray(y_pred, float)})
    frame["band"] = np.clip(np.round(frame["actual"]), SCORE_MIN, SCORE_MAX).astype(int)
    grouped = frame.groupby("band").apply(
        lambda g: pd.Series(
            {
                "n": len(g),
                "mean_actual": g["actual"].mean(),
                "mean_predicted": g["pred"].mean(),
                "bias": (g["pred"] - g["actual"]).mean(),
                "rmse": rmse(g["actual"], g["pred"]),
                "mae": np.abs(g["pred"] - g["actual"]).mean(),
            }
        ),
        include_groups=False,
    )
    return grouped


def plot_error_by_band(y_true, y_pred, ax=None):
    """Mean predicted score per band against the ideal, showing regression to the mean."""
    ax = _setup(ax, figsize=(7.5, 5))
    table = error_by_band(y_true, y_pred).reset_index()

    ax.plot(
        table["band"],
        table["mean_predicted"],
        "o-",
        color=PALETTE["primary"],
        linewidth=2,
        label="mean prediction",
    )
    ax.plot(
        table["band"],
        table["mean_actual"],
        "s--",
        color=PALETTE["danger"],
        linewidth=1.5,
        label="mean actual",
    )
    for _, row in table.iterrows():
        ax.annotate(
            f"n={int(row['n'])}",
            (row["band"], row["mean_predicted"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
            color="#475569",
        )
    ax.set_xlabel("Actual rubric band")
    ax.set_ylabel("Grammar score")
    ax.set_title("Prediction shrinkage towards the mean, by band")
    ax.set_xticks(table["band"])
    ax.legend()
    return ax


def permutation_importance_frame(
    estimator, X: pd.DataFrame, y: np.ndarray, n_repeats: int = 8, random_state: int = 42
) -> pd.DataFrame:
    """Permutation importance in RMSE units (how much RMSE worsens when shuffled)."""
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import make_scorer

    scorer = make_scorer(rmse, greater_is_better=False)
    result = permutation_importance(
        estimator, X, y, scoring=scorer, n_repeats=n_repeats, random_state=random_state, n_jobs=1
    )
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance": result.importances_mean,
                "std": result.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def plot_permutation_importance(frame: pd.DataFrame, top_n: int = 20, ax=None):
    ax = _setup(ax, figsize=(8, max(4, top_n * 0.32)))
    top = frame.head(top_n)[::-1]
    ax.barh(
        top["feature"],
        top["importance"],
        xerr=top["std"],
        color=PALETTE["primary"],
        error_kw={"ecolor": PALETTE["muted"], "elinewidth": 1},
    )
    ax.set_xlabel("Increase in RMSE when the feature is shuffled")
    ax.set_title(f"Top {top_n} features by permutation importance")
    return ax


def plot_block_contribution(block_scores: pd.DataFrame, ax=None):
    """Ablation chart: CV RMSE achieved by each feature block on its own."""
    ax = _setup(ax, figsize=(8, 4.5))
    frame = block_scores.sort_values("rmse")
    ax.barh(frame["block"], frame["rmse"], color=PALETTE["secondary"])
    ax.set_xlabel("Cross-validated RMSE using only this feature block")
    ax.set_title("How much each feature block contributes on its own")
    ax.invert_yaxis()
    for i, value in enumerate(frame["rmse"]):
        ax.text(value, i, f" {value:.3f}", va="center", fontsize=9)
    return ax


def plot_learning_curve(estimator, X: pd.DataFrame, y: np.ndarray, cv=5, ax=None):
    """Training-set-size curve: shows whether more data would still help."""
    from sklearn.model_selection import learning_curve

    ax = _setup(ax, figsize=(7.5, 5))
    sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        train_sizes=np.linspace(0.2, 1.0, 6),
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
        random_state=42,
        shuffle=True,
    )
    train_rmse = -train_scores.mean(axis=1)
    test_rmse = -test_scores.mean(axis=1)

    ax.plot(sizes, train_rmse, "o-", color=PALETTE["primary"], label="training RMSE")
    ax.plot(sizes, test_rmse, "s-", color=PALETTE["secondary"], label="validation RMSE")
    ax.fill_between(
        sizes,
        -test_scores.mean(axis=1) - test_scores.std(axis=1),
        -test_scores.mean(axis=1) + test_scores.std(axis=1),
        alpha=0.15,
        color=PALETTE["secondary"],
    )
    ax.set_xlabel("Training samples")
    ax.set_ylabel("RMSE")
    ax.set_title("Learning curve")
    ax.legend()
    return ax
