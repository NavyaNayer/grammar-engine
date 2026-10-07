"""Command line interface.

    grammar-scoring generate --n-train 60 --n-test 20   # build a local demo corpus
    grammar-scoring run --data-dir data/raw             # full pipeline + submission
    grammar-scoring score path/to/clip.wav              # score a single audio file
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from grammar_scoring.config import ASRConfig, ModelConfig, PipelineConfig
from grammar_scoring.pipeline import configure_logging, run_pipeline

LOGGER = logging.getLogger("grammar_scoring")


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", default="data/raw", help="Root of the competition data")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Where outputs are written")
    parser.add_argument("--cache-dir", default=".cache", help="ASR and feature cache location")
    parser.add_argument("--asr-model", default=ASRConfig.model_name, help="Whisper checkpoint")
    parser.add_argument(
        "--no-language-tool", action="store_true", help="Skip rule-based grammar checks"
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce logging to warnings")


def _build_config(args: argparse.Namespace) -> PipelineConfig:
    config = PipelineConfig(
        data_dir=Path(args.data_dir),
        artifacts_dir=Path(args.artifacts_dir),
        cache_dir=Path(args.cache_dir),
        verbose=not args.quiet,
    )
    config.asr.model_name = args.asr_model
    if getattr(args, "no_language_tool", False):
        config.features.use_language_tool = False
    if getattr(args, "folds", None):
        config.model = ModelConfig(n_splits=args.folds, n_repeats=args.repeats)
    return config


def command_generate(args: argparse.Namespace) -> int:
    from grammar_scoring.synthetic import build_synthetic_dataset, espeak_available

    if not espeak_available():
        LOGGER.warning("espeak-ng is not installed; generated audio will not contain speech.")
    dataset = build_synthetic_dataset(
        output_dir=args.output,
        n_train=args.n_train,
        n_test=args.n_test,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(dataset.describe())
    return 0


def command_run(args: argparse.Namespace) -> int:
    config = _build_config(args)
    output = run_pipeline(
        config,
        allow_synthetic_fallback=not args.no_synthetic_fallback,
        synthetic_kwargs={"n_train": args.synthetic_train, "n_test": args.synthetic_test},
        use_feature_cache=not args.no_cache,
    )

    print("\n" + output.summary())
    print("\nCross-validated model comparison:")
    print(output.zoo.leaderboard().to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSubmission written to {(config.artifacts_dir / 'submission.csv').resolve()}")
    return 0


def command_score(args: argparse.Namespace) -> int:
    """Score individual clips with a scorer trained on the configured dataset."""
    import pandas as pd

    config = _build_config(args)
    output = run_pipeline(config, use_feature_cache=not args.no_cache)

    from grammar_scoring.features.assemble import FeaturePipeline

    features = FeaturePipeline(config)
    try:
        rows, transcripts = [], []
        for path in args.audio:
            feature_row, transcript = features.process_one(path)
            rows.append({"filename": Path(path).name, **feature_row})
            transcripts.append(transcript.text)
    finally:
        features.close()

    frame = pd.DataFrame(rows).set_index("filename").reindex(columns=output.X_train.columns)
    predictions = output.scorer.predict(frame)

    for path, score, text in zip(args.audio, predictions, transcripts, strict=True):
        print(json.dumps({"file": str(path), "grammar_score": round(float(score), 3),
                          "transcript": text}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grammar-scoring",
        description="Grammar Scoring Engine for spoken audio (SHL Hiring Assessment 2026).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate the local synthetic corpus")
    generate.add_argument("--output", default="data/synthetic")
    generate.add_argument("--n-train", type=int, default=769)
    generate.add_argument("--n-test", type=int, default=216)
    generate.add_argument("--seed", type=int, default=20260501)
    generate.add_argument("--overwrite", action="store_true")
    generate.set_defaults(func=command_generate)

    run = subparsers.add_parser("run", help="Run the full pipeline and write a submission")
    _add_common_arguments(run)
    run.add_argument("--folds", type=int, default=5)
    run.add_argument("--repeats", type=int, default=2)
    run.add_argument("--no-cache", action="store_true", help="Recompute features from scratch")
    run.add_argument("--no-synthetic-fallback", action="store_true",
                     help="Fail instead of generating a synthetic corpus when data is missing")
    run.add_argument("--synthetic-train", type=int, default=769)
    run.add_argument("--synthetic-test", type=int, default=216)
    run.set_defaults(func=command_run)

    score = subparsers.add_parser("score", help="Score one or more audio files")
    _add_common_arguments(score)
    score.add_argument("audio", nargs="+", help="Audio files to score")
    score.add_argument("--no-cache", action="store_true")
    score.set_defaults(func=command_score)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(logging.WARNING if getattr(args, "quiet", False) else logging.INFO)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
