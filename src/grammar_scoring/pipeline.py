"""End-to-end orchestration: audio in, submission file out."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from grammar_scoring.config import PipelineConfig
from grammar_scoring.data import Dataset, resolve_dataset
from grammar_scoring.evaluate import baseline_metrics, compute_metrics
from grammar_scoring.features.assemble import FeatureBundle, FeaturePipeline, align_features
from grammar_scoring.features.embeddings import reduce_wav2vec2_embeddings
from grammar_scoring.features.wavlm import reduce_wavlm_embeddings
from grammar_scoring.modeling import GrammarScorer, ZooResult, run_model_zoo

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineOutput:
    """Everything a run produces, so the notebook can inspect any stage."""

    config: PipelineConfig
    dataset: Dataset
    train_bundle: FeatureBundle
    test_bundle: FeatureBundle
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_test: pd.DataFrame
    zoo: ZooResult
    scorer: GrammarScorer
    submission: pd.DataFrame

    @property
    def oof_predictions(self) -> np.ndarray:
        if self.zoo.blend_oof is not None:
            return self.zoo.blend_oof
        return self.zoo.results[self.zoo.best_single].oof_predictions

    def metrics(self) -> dict[str, float]:
        return compute_metrics(self.y_train, self.oof_predictions)

    def summary(self) -> str:
        metrics = self.metrics()
        baseline = baseline_metrics(self.y_train)
        return (
            f"Training samples : {len(self.X_train)}\n"
            f"Test samples     : {len(self.X_test)}\n"
            f"Features         : {self.X_train.shape[1]}\n"
            f"Baseline RMSE    : {baseline['rmse']:.4f}  (predicting the mean)\n"
            f"Model OOF RMSE   : {metrics['rmse']:.4f}\n"
            f"Model OOF Pearson: {metrics['pearson']:.4f}\n"
            f"Improvement      : {100 * (1 - metrics['rmse'] / baseline['rmse']):.1f}% lower RMSE"
        )


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # These libraries are chatty at INFO and drown out the pipeline's own progress.
    for noisy in ("transformers", "matplotlib", "numba", "urllib3", "filelock", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def build_submission(
    filenames: list[str], predictions: np.ndarray, template: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Assemble the submission frame, matching the sample template's column names.

    The grader reads the template's exact header, so when a ``sample_submission.csv`` is
    present we copy its column names. Its row *list* is used too, but only when every row
    has a real matching prediction. Kaggle mirrors of this kind of competition have
    shipped a ``sample_submission.csv`` left over from a different train/test split than
    the audio actually packaged with it; reindexing to such a template with no real
    prediction for some rows has no correct value to put there, so rather than fabricate
    one (e.g. the mean of the other predictions), the real predicted test set is used
    instead, with the template's column names kept for format compliance.
    """
    frame = pd.DataFrame({"filename": filenames, "label": np.asarray(predictions, dtype=float)})
    if template is None or template.empty:
        return frame

    columns = list(template.columns)
    name_col = columns[0]
    label_col = columns[1] if len(columns) > 1 else "label"

    lookup = dict(zip(frame["filename"], frame["label"], strict=True))
    template_names = template[name_col].astype(str)
    missing = [name for name in template_names if name not in lookup]

    if missing:
        LOGGER.warning(
            "%d/%d sample_submission.csv rows have no matching prediction (e.g. %s) -- "
            "submitting the %d real predictions instead of the template's row list, "
            "using the template's column names.",
            len(missing),
            len(template_names),
            missing[:3],
            len(frame),
        )
        return frame.rename(columns={"filename": name_col, "label": label_col})

    ordered = template.copy()
    ordered[label_col] = [lookup[str(name)] for name in template_names]
    return ordered[[name_col, label_col]]


def run_pipeline(
    config: PipelineConfig | None = None,
    allow_synthetic_fallback: bool = True,
    synthetic_kwargs: dict | None = None,
    use_feature_cache: bool = True,
) -> PipelineOutput:
    """Run every stage: load -> featurise -> cross-validate -> fit -> predict -> save."""
    config = config or PipelineConfig()
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    config.cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = resolve_dataset(
        data_dir=config.data_dir,
        allow_synthetic_fallback=allow_synthetic_fallback,
        **(synthetic_kwargs or {}),
    )
    LOGGER.info("Dataset ready:\n%s", dataset.describe())

    features = FeaturePipeline(config)
    try:
        train_bundle = features.transform(
            dataset.train.audio_paths,
            dataset.train.filenames,
            split="train",
            use_cache=use_feature_cache,
        )
        test_bundle = features.transform(
            dataset.test.audio_paths,
            dataset.test.filenames,
            split="test",
            use_cache=use_feature_cache,
        )
    finally:
        features.close()

    X_train, X_test = align_features(train_bundle.features, test_bundle.features)
    X_train, X_test = reduce_wav2vec2_embeddings(X_train, X_test, config.features)
    X_train, X_test = reduce_wavlm_embeddings(X_train, X_test, config.features)
    y_train = dataset.train.frame.set_index("filename").loc[X_train.index, "label"].to_numpy(float)

    LOGGER.info("Cross-validating %d candidate models ...", len(config.model.candidates))
    zoo = run_model_zoo(X_train, y_train, config.model, verbose=config.verbose)

    LOGGER.info("Fitting the final scorer on the full training set ...")
    weights = zoo.blend_weights or {zoo.best_single: 1.0}
    scorer = GrammarScorer(config.model, weights=weights).fit(X_train, y_train)

    predictions = scorer.predict(X_test)
    submission = build_submission(list(X_test.index), predictions, dataset.sample_submission)

    output = PipelineOutput(
        config=config,
        dataset=dataset,
        train_bundle=train_bundle,
        test_bundle=test_bundle,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        zoo=zoo,
        scorer=scorer,
        submission=submission,
    )
    save_artifacts(output)
    return output


def save_artifacts(output: PipelineOutput) -> dict[str, Path]:
    """Persist the submission, metrics, leaderboard, OOF predictions and transcripts."""
    directory = output.config.artifacts_dir
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    paths["submission"] = directory / "submission.csv"
    output.submission.to_csv(paths["submission"], index=False)

    paths["leaderboard"] = directory / "model_leaderboard.csv"
    output.zoo.leaderboard().to_csv(paths["leaderboard"], index=False)

    paths["oof"] = directory / "oof_predictions.csv"
    oof = output.zoo.oof_frame()
    oof.insert(0, "filename", list(output.X_train.index))
    oof.to_csv(paths["oof"], index=False)

    paths["transcripts"] = directory / "train_transcripts.csv"
    output.train_bundle.transcripts.to_csv(paths["transcripts"])

    paths["features"] = directory / "train_features.csv"
    output.X_train.to_csv(paths["features"])

    metrics = {
        "train_oof": output.metrics(),
        "baseline_mean_predictor": baseline_metrics(output.y_train),
        "per_model": {name: r.metrics for name, r in output.zoo.results.items()},
        "blend_weights": output.zoo.blend_weights,
        "blend_nested_check": output.zoo.honest_blend_metrics,
        "n_train": int(len(output.X_train)),
        "n_test": int(len(output.X_test)),
        "n_features": int(output.X_train.shape[1]),
        "dataset_is_synthetic": bool(output.dataset.is_synthetic),
    }
    paths["metrics"] = directory / "metrics.json"
    paths["metrics"].write_text(json.dumps(metrics, indent=2))

    paths["config"] = output.config.save(directory / "config.json")
    LOGGER.info("Artifacts written to %s", directory.resolve())
    return paths
