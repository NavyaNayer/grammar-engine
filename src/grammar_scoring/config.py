"""Configuration objects for the grammar scoring pipeline.

Every stage of the pipeline reads its knobs from :class:`PipelineConfig` so that an
experiment can be reproduced from a single serialisable object.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

# The competition brief defines the output range as continuous [0, 5] (the rubric table
# only names bands 1-5, but 0 is a valid score: ~5% of real training labels are exactly
# 0.0, for clips judged to have no discernible grammatical structure at all). Model
# outputs are clipped to this range because scores outside it are not defined.
SCORE_MIN = 0.0
SCORE_MAX = 5.0

# Whisper is trained on 16 kHz mono audio; every downstream feature uses the same rate
# so that frame indices are comparable across extractors.
SAMPLE_RATE = 16_000


@dataclass
class AudioConfig:
    """Controls waveform loading and the silence/pause segmentation."""

    sample_rate: int = SAMPLE_RATE
    #: Peak value used for amplitude normalisation after loading.
    peak_normalize_to: float = 0.95
    #: Frames quieter than (max_db - top_db) are treated as silence.
    top_db: float = 35.0
    #: Analysis frame length / hop for the energy envelope, in samples.
    frame_length: int = 1024
    hop_length: int = 256
    #: Gaps at least this long (seconds) count as a *pause* rather than a stop consonant.
    min_pause_s: float = 0.30
    #: Pauses at least this long are "long" pauses, a strong hesitation marker.
    long_pause_s: float = 1.00
    #: Voiced runs shorter than this are dropped before computing segment statistics.
    min_segment_s: float = 0.10
    #: Trim leading/trailing silence before analysis.
    trim_silence: bool = True


@dataclass
class ASRConfig:
    """Controls Whisper transcription."""

    model_name: str = "openai/whisper-small.en"
    #: Fall back to this (much smaller) checkpoint if the primary one cannot be loaded.
    fallback_model_name: str = "openai/whisper-tiny.en"
    language: str | None = "en"
    #: Optional decoder prompt. A disfluent prompt (fillers, false starts, self-corrections)
    #: stops Whisper tidying speech into fluent text, so disfluency survives transcription.
    #: Off by default; enabling it changes the transcript and therefore the cache key.
    initial_prompt: str | None = None
    #: Whisper always consumes 30 s windows; long-form audio is chunked with overlap.
    chunk_length_s: int = 30
    stride_length_s: int = 5
    batch_size: int = 1
    num_beams: int = 1
    #: Ask the decoder for token scores so we can derive confidence features.
    return_timestamps: bool = True
    device: str = "auto"
    #: Transcripts are expensive; cache them as JSON keyed by file content hash.
    use_cache: bool = True


@dataclass
class FeatureConfig:
    """Switches for each feature family.

    Defaults ship the *interpretable* subset only: rule-based grammar errors, fluency
    and lexical features are all named ratios/counts a human can read and defend
    directly. The acoustic family and the two neural grammar signals below are fully
    implemented and tested but off by default -- each is a black box relative to the
    rubric (which judges sentence structure and syntax, not vocal timbre or a neural
    net's internal score), each adds a large model download, and with 769 training
    samples a smaller, more interpretable feature set also overfits less. Flip the
    booleans back on to use them.
    """

    #: Off by default: MFCCs/spectral shape/pitch are the family least tied to the
    #: rubric and the hardest to justify in an interview. Pause/speech-rate signal is
    #: not lost by disabling this -- that lives in the fluency family below.
    acoustic: bool = False
    fluency: bool = True
    lexical: bool = True
    grammar: bool = True

    # --- grammar sub-features -------------------------------------------------
    #: Rule-based grammatical error detection (requires a Java runtime). Kept on by
    #: default: fully interpretable ("N errors of type X detected").
    use_language_tool: bool = True
    language_tool_lang: str = "en-US"
    #: Pin the engine version: the default ``latest`` re-downloads a ~260 MB nightly
    #: snapshot on every start, whereas a pinned release is cached once and reused.
    language_tool_version: str = "6.6"
    #: Neural grammatical-acceptability classifier (CoLA). Off by default: a black-box
    #: score is hard to defend under questioning, and it's a second large model
    #: download beyond Whisper. Available for a higher-accuracy, lower-explainability
    #: run by setting this to True.
    use_cola: bool = False
    cola_model_name: str = "textattack/roberta-base-CoLA"
    #: Causal-LM surprisal of the transcript. Off by default for the same reason as
    #: CoLA -- it also conflates grammatical awkwardness with topic novelty (see
    #: README).
    use_perplexity: bool = False
    perplexity_model_name: str = "distilgpt2"
    perplexity_max_tokens: int = 512

    #: Frozen wav2vec2 speech embeddings (mean-pooled, PCA-reduced). Off by default:
    #: the least explainable family of all -- individual PCA components of a neural
    #: embedding carry no interpretable meaning -- and a large model download. Never
    #: fine-tuned; used only as a fixed feature extractor, so the overfitting risk stays
    #: far below end-to-end fine-tuning. Available as an accuracy lever once a
    #: competitive score, not explainability, is the binding constraint.
    wav2vec2: bool = False
    wav2vec2_model_name: str = "facebook/wav2vec2-base-960h"
    #: Clips are truncated to this many seconds before embedding, to bound the cost of
    #: a transformer forward pass per clip. Clips run 45-60s (measured max 60.1s), so
    #: 65s covers all of them with margin -- the cost of covering the full clip instead
    #: of a 30s prefix is modest (~25% more time per clip, benchmarked), and a prefix
    #: truncation was blind to a third to half of every clip's audio for no good reason.
    wav2vec2_max_duration_s: float = 65.0
    wav2vec2_pca_components: int = 32

    #: Frozen WavLM speech embeddings (mean-pooled per layer, concatenated across
    #: ``wavlm_layers``, PCA-reduced). Off by default for the same reason as wav2vec2.
    #: WavLM adds a speaker-overlap/denoising pretraining objective on top of the same
    #: base architecture as wav2vec2 -- empirically the single biggest real-world
    #: accuracy lever found while developing this pipeline (see the notebook's results
    #: section). Layers 8-10 and PCA=64 were chosen by cross-validated sweep, not guessed.
    wavlm: bool = False
    wavlm_model_name: str = "microsoft/wavlm-base-plus"
    wavlm_layers: tuple[int, ...] = (8, 9, 10)
    wavlm_max_duration_s: float = 65.0
    wavlm_pca_components: int = 64

    #: Moving-average type-token-ratio window (length-robust lexical diversity).
    mattr_window: int = 50
    #: Torch inference threads for the neural text models.
    torch_num_threads: int = 4


@dataclass
class ModelConfig:
    """Cross-validation and estimator settings."""

    n_splits: int = 5
    n_repeats: int = 2
    random_state: int = 42
    #: Stratify folds on the rounded score so every fold sees the full score range.
    stratify: bool = True
    #: Estimators evaluated in the model zoo (see :mod:`grammar_scoring.modeling`).
    #: Two deliberately: a regularised linear baseline (ridge) and a non-linear
    #: ensemble (random forest), one of each inductive bias. ``build_estimator`` still
    #: supports "elasticnet", "svr_rbf", "gradient_boosting" and "knn" for a larger
    #: zoo if more accuracy is worth more surface area to explain.
    candidates: tuple[str, ...] = (
        "ridge",
        "random_forest",
    )
    #: Blend candidate out-of-fold predictions with non-negative least squares.
    use_stacking: bool = True
    clip_predictions: bool = True


@dataclass
class PipelineConfig:
    """Top-level configuration bundle."""

    data_dir: Path = Path("data/raw")
    artifacts_dir: Path = Path("artifacts")
    cache_dir: Path = Path(".cache")
    audio: AudioConfig = field(default_factory=AudioConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    n_jobs: int = 4
    verbose: bool = True

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.artifacts_dir = Path(self.artifacts_dir)
        self.cache_dir = Path(self.cache_dir)

    def to_dict(self) -> dict:
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path
