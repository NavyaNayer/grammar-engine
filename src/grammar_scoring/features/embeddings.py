"""Frozen wav2vec2 speech embeddings, an optional feature family (``wv_``).

Every other feature family reasons about the *transcript* or simple acoustic statistics
(MFCCs, pitch, timing). This one is different: it is the mean-pooled hidden state of a
pretrained speech encoder, used **frozen** -- never fine-tuned -- as a fixed feature
vector. That keeps the overfitting risk far below end-to-end fine-tuning (which the
top-level README already argues against for 769 samples) while still adding a learned
acoustic-phonetic representation that nothing else in the pipeline provides.

It is off by default (see ``FeatureConfig.wav2vec2``) for the same reason CoLA and LM
surprisal are: individual PCA components of a neural embedding have no interpretable
meaning, so this is an accuracy lever for when a competitive score, not explainability,
is the binding constraint.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from grammar_scoring.audio import PreprocessedAudio
from grammar_scoring.config import FeatureConfig

LOGGER = logging.getLogger(__name__)


class Wav2Vec2EmbeddingExtractor:
    """Lazily-loaded, frozen wav2vec2 wrapper. One instance can embed a whole dataset."""

    def __init__(self, config: FeatureConfig, cache_dir: str | Path = ".cache/wav2vec2"):
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._processor = None
        self._hidden_size: int | None = None
        self._failed = False

    # ------------------------------------------------------------------ loading
    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if self._failed:
            return False
        try:
            import torch
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

            torch.set_num_threads(max(1, self.config.torch_num_threads))
            LOGGER.info("Loading wav2vec2 model %s ...", self.config.wav2vec2_model_name)
            self._processor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.config.wav2vec2_model_name
            )
            self._model = Wav2Vec2Model.from_pretrained(self.config.wav2vec2_model_name)
            self._model.eval()
            self._hidden_size = self._model.config.hidden_size
            return True
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("wav2vec2 model unavailable (%s); wv_ features disabled.", exc)
            self._failed = True
            return False

    # ------------------------------------------------------------------- caching
    def _cache_key(self, audio: PreprocessedAudio) -> str:
        digest = hashlib.sha1()
        digest.update(np.ascontiguousarray(audio.waveform).tobytes())
        digest.update(
            json.dumps(
                {
                    "model": self.config.wav2vec2_model_name,
                    "max_duration_s": self.config.wav2vec2_max_duration_s,
                },
                sort_keys=True,
            ).encode()
        )
        return digest.hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    # ---------------------------------------------------------------- inference
    def _forward(self, audio: PreprocessedAudio) -> np.ndarray | None:
        import torch

        if not self._ensure_loaded():
            return None
        assert self._model is not None and self._processor is not None

        max_samples = int(self.config.wav2vec2_max_duration_s * audio.sample_rate)
        waveform = audio.waveform[:max_samples]
        if waveform.size < audio.sample_rate * 0.2:
            return np.zeros(self._hidden_size, dtype=float)

        inputs = self._processor(
            waveform, sampling_rate=audio.sample_rate, return_tensors="pt"
        )
        with torch.no_grad():
            hidden_states = self._model(inputs.input_values).last_hidden_state
        # Mean-pool over the time axis: the simplest, standard way to collapse a
        # variable-length sequence of frame embeddings into one fixed-size vector.
        return hidden_states.mean(dim=1).squeeze(0).numpy()

    def extract(self, audio: PreprocessedAudio) -> dict[str, float]:
        """Mean-pooled wav2vec2 embedding for one clip, using the on-disk cache."""
        key = self._cache_key(audio)
        cache_path = self._cache_path(key)
        if cache_path.exists():
            try:
                vector = np.asarray(json.loads(cache_path.read_text()), dtype=float)
            except (json.JSONDecodeError, TypeError) as exc:
                LOGGER.warning("Ignoring corrupt wav2vec2 cache entry %s: %s", cache_path, exc)
                vector = None
        else:
            vector = None

        if vector is None:
            computed = self._forward(audio)
            if computed is None:
                return {}
            vector = computed
            cache_path.write_text(json.dumps(vector.tolist()))

        return {f"wv_raw_{i:03d}": float(v) for i, v in enumerate(vector)}


def reduce_wav2vec2_embeddings(
    X_train: pd.DataFrame, X_test: pd.DataFrame, config: FeatureConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace raw ``wv_raw_*`` columns with PCA components fit on the training split.

    Pure sklearn logic, no neural network involved. A no-op when no ``wv_raw_*`` columns
    are present, so it is always safe to call regardless of whether the family is
    enabled. PCA is fit once on ``X_train`` and applied to both splits -- standard
    practice for unsupervised dimensionality reduction done before cross-validation, not
    per-fold.
    """
    raw_columns = [c for c in X_train.columns if c.startswith("wv_raw_")]
    if not raw_columns:
        return X_train, X_test

    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="median")
    train_raw = imputer.fit_transform(X_train[raw_columns])
    test_raw = imputer.transform(X_test[raw_columns])

    n_components = min(config.wav2vec2_pca_components, train_raw.shape[0], train_raw.shape[1])
    pca = PCA(n_components=n_components, random_state=0)
    train_components = pca.fit_transform(train_raw)
    test_components = pca.transform(test_raw)

    pc_columns = [f"wv_pc{i:02d}" for i in range(n_components)]
    train_pcs = pd.DataFrame(train_components, columns=pc_columns, index=X_train.index)
    test_pcs = pd.DataFrame(test_components, columns=pc_columns, index=X_test.index)

    X_train = pd.concat([X_train.drop(columns=raw_columns), train_pcs], axis=1)
    X_test = pd.concat([X_test.drop(columns=raw_columns), test_pcs], axis=1)
    return X_train, X_test
