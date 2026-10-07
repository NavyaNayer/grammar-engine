"""Grammar Scoring Engine for spoken audio -- SHL Hiring Assessment 2026.

Given a 45-60 s recording of spontaneous English speech, predict the MOS Likert grammar
score in [0, 5] that a human rater would assign.

The pipeline is::

    audio -> preprocessing -> Whisper ASR -> feature extraction -> regression ensemble

Quick start::

    from grammar_scoring import PipelineConfig, run_pipeline

    output = run_pipeline(PipelineConfig(data_dir="data/raw"))
    print(output.summary())
"""

from grammar_scoring.config import (
    SAMPLE_RATE,
    SCORE_MAX,
    SCORE_MIN,
    ASRConfig,
    AudioConfig,
    FeatureConfig,
    ModelConfig,
    PipelineConfig,
)
from grammar_scoring.data import Dataset, load_dataset, resolve_dataset
from grammar_scoring.evaluate import compute_metrics
from grammar_scoring.modeling import GrammarScorer, run_model_zoo
from grammar_scoring.pipeline import PipelineOutput, configure_logging, run_pipeline

__version__ = "1.0.0"

__all__ = [
    "SAMPLE_RATE",
    "SCORE_MAX",
    "SCORE_MIN",
    "ASRConfig",
    "AudioConfig",
    "Dataset",
    "FeatureConfig",
    "GrammarScorer",
    "ModelConfig",
    "PipelineConfig",
    "PipelineOutput",
    "compute_metrics",
    "configure_logging",
    "load_dataset",
    "resolve_dataset",
    "run_model_zoo",
    "run_pipeline",
    "__version__",
]
