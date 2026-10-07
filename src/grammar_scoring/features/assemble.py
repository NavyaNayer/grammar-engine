"""Turns a list of audio files into a feature matrix.

Processing is streamed one clip at a time: a 60 s clip at 16 kHz is ~3.8 MB as float32,
so holding a whole split in memory would cost several gigabytes for no benefit. Each
clip is loaded, transcribed, featurised and then released.

Both expensive stages are cached. ASR results are cached inside
:class:`~grammar_scoring.asr.WhisperTranscriber`; the assembled matrix is cached here,
keyed by the configuration and the exact list of input files, so re-running a notebook
cell is instant.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from grammar_scoring.asr import Transcript, WhisperTranscriber
from grammar_scoring.audio import load_audio
from grammar_scoring.config import PipelineConfig
from grammar_scoring.features.acoustic import extract_acoustic_features
from grammar_scoring.features.embeddings import Wav2Vec2EmbeddingExtractor
from grammar_scoring.features.fluency import extract_fluency_features
from grammar_scoring.features.grammar import GrammarFeatureExtractor
from grammar_scoring.features.lexical import extract_lexical_features
from grammar_scoring.features.wavlm import WavLMEmbeddingExtractor

LOGGER = logging.getLogger(__name__)

#: Ordering of feature blocks in the output matrix, for readable inspection.
_PREFIX_ORDER = ("gr_", "fl_", "sy_", "lx_", "ac_", "wv_", "wl_")


@dataclass
class FeatureBundle:
    """The feature matrix plus the transcripts it was derived from."""

    features: pd.DataFrame  # index = filename
    transcripts: pd.DataFrame  # filename, text, n_words, avg_logprob

    def __len__(self) -> int:
        return len(self.features)

    @property
    def feature_names(self) -> list[str]:
        return list(self.features.columns)


def order_columns(columns: list[str]) -> list[str]:
    """Group feature columns by block, alphabetically within each block."""

    def key(name: str) -> tuple[int, str]:
        for i, prefix in enumerate(_PREFIX_ORDER):
            if name.startswith(prefix):
                return (i, name)
        return (len(_PREFIX_ORDER), name)

    return sorted(columns, key=key)


class FeaturePipeline:
    """Audio files in, feature matrix out."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.transcriber = WhisperTranscriber(
            self.config.asr, cache_dir=self.config.cache_dir / "asr"
        )
        self.grammar = GrammarFeatureExtractor(self.config.features)
        self.embeddings = Wav2Vec2EmbeddingExtractor(
            self.config.features, cache_dir=self.config.cache_dir / "wav2vec2"
        )
        self.wavlm = WavLMEmbeddingExtractor(
            self.config.features, cache_dir=self.config.cache_dir / "wavlm_multilayer"
        )

    # ---------------------------------------------------------------- caching
    def _cache_key(self, paths: list[Path], split: str) -> str:
        digest = hashlib.sha1()
        digest.update(split.encode())
        for path in paths:
            digest.update(str(path).encode())
            try:
                digest.update(str(Path(path).stat().st_size).encode())
            except OSError:
                pass
        digest.update(
            json.dumps(
                {
                    "asr": self.config.asr.__dict__,
                    "audio": self.config.audio.__dict__,
                    "features": self.config.features.__dict__,
                },
                sort_keys=True,
                default=str,
            ).encode()
        )
        return digest.hexdigest()[:16]

    def _cache_paths(self, key: str) -> tuple[Path, Path]:
        base = self.config.cache_dir / "features"
        base.mkdir(parents=True, exist_ok=True)
        return base / f"{key}_features.csv", base / f"{key}_transcripts.csv"

    # -------------------------------------------------------------- extraction
    def process_one(self, path: str | Path) -> tuple[dict[str, float], Transcript]:
        """Full feature dictionary for a single audio file."""
        audio = load_audio(path, self.config.audio)
        transcript = self.transcriber.transcribe(audio)

        features: dict[str, float] = {}
        if self.config.features.acoustic:
            features.update(extract_acoustic_features(audio, self.config.audio))
        if self.config.features.fluency:
            features.update(extract_fluency_features(transcript, audio))
        if self.config.features.lexical:
            features.update(
                extract_lexical_features(transcript.text, self.config.features.mattr_window)
            )
        if self.config.features.grammar:
            features.update(self.grammar.extract(transcript.text))
        if self.config.features.wav2vec2:
            features.update(self.embeddings.extract(audio))
        if self.config.features.wavlm:
            features.update(self.wavlm.extract(audio))
        return features, transcript

    def transform(
        self,
        paths: list[str | Path],
        filenames: list[str] | None = None,
        split: str = "data",
        use_cache: bool = True,
    ) -> FeatureBundle:
        """Featurise a list of audio files."""
        paths = [Path(p) for p in paths]
        filenames = filenames or [p.name for p in paths]
        if len(filenames) != len(paths):
            raise ValueError("filenames and paths must have the same length")

        cache_key = self._cache_key(paths, split)
        feature_cache, transcript_cache = self._cache_paths(cache_key)
        if use_cache and feature_cache.exists() and transcript_cache.exists():
            LOGGER.info("Loading cached %s features from %s", split, feature_cache)
            features = pd.read_csv(feature_cache, index_col=0)
            transcripts = pd.read_csv(transcript_cache, index_col=0, keep_default_na=False)
            return FeatureBundle(features=features, transcripts=transcripts)

        iterator = zip(filenames, paths, strict=True)
        if self.config.verbose:
            from tqdm.auto import tqdm

            iterator = tqdm(
                list(iterator), desc=f"Featurising {split}", unit="clip", total=len(paths)
            )

        rows: list[dict[str, float]] = []
        transcript_rows: list[dict] = []
        for filename, path in iterator:
            try:
                features, transcript = self.process_one(path)
            except Exception as exc:  # noqa: BLE001 - one bad file must not kill a run
                LOGGER.error("Failed to featurise %s: %s", path, exc)
                features, transcript = {}, Transcript(text="")
            rows.append({"filename": filename, **features})
            transcript_rows.append(
                {
                    "filename": filename,
                    "text": transcript.text,
                    "n_words": transcript.n_words,
                    "avg_logprob": transcript.avg_logprob,
                    "duration": transcript.duration,
                }
            )

        features_df = pd.DataFrame(rows).set_index("filename")
        features_df = features_df[order_columns(list(features_df.columns))]
        transcripts_df = pd.DataFrame(transcript_rows).set_index("filename")

        if use_cache:
            features_df.to_csv(feature_cache)
            transcripts_df.to_csv(transcript_cache)
            LOGGER.info("Cached %s features to %s", split, feature_cache)

        return FeatureBundle(features=features_df, transcripts=transcripts_df)

    def close(self) -> None:
        self.grammar.close()


def align_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Give train and test an identical, identically-ordered column set.

    A feature can go missing from one split when, for example, every clip in it produced
    an empty transcript. Reindexing keeps the matrices compatible instead of failing at
    predict time.
    """
    columns = order_columns(sorted(set(train.columns) | set(test.columns)))
    return train.reindex(columns=columns), test.reindex(columns=columns)
