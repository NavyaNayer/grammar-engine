"""Frozen WavLM speech embeddings, an optional feature family (``wl_``).

Same rationale as ``features/embeddings.py``'s wav2vec2 family -- a frozen, never
fine-tuned pretrained speech encoder's hidden state, used as a fixed feature vector.
WavLM differs from wav2vec2 in its pretraining objective (it adds a speaker-overlap /
denoising task on top of the same base architecture), which empirically made it the
single biggest real-world accuracy lever found while developing this pipeline -- see
the notebook's results section for the comparison. It is off by default for the same
explainability reason as the other neural feature families: individual PCA components
of a neural embedding have no interpretable meaning.

Unlike the wav2vec2 extractor (which keeps only the final layer's hidden state), this
one caches **every** transformer layer's mean-pooled output per clip -- a single forward
pass already computes every layer internally, so keeping all of them costs almost
nothing extra, and it means changing which layers ``FeatureConfig.wavlm_layers`` asks
for later never requires re-running the (expensive) forward pass.
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


class WavLMEmbeddingExtractor:
    """Lazily-loaded, frozen WavLM wrapper. Caches every layer's pooled output per clip."""

    def __init__(self, config: FeatureConfig, cache_dir: str | Path = ".cache/wavlm_multilayer"):
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._processor = None
        self._hidden_size: int | None = None
        self._num_layers: int | None = None
        self._failed = False

    # ------------------------------------------------------------------ loading
    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if self._failed:
            return False
        try:
            import torch
            from transformers import Wav2Vec2FeatureExtractor, WavLMModel

            torch.set_num_threads(max(1, self.config.torch_num_threads))
            LOGGER.info("Loading WavLM model %s ...", self.config.wavlm_model_name)
            self._processor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.config.wavlm_model_name
            )
            self._model = WavLMModel.from_pretrained(
                self.config.wavlm_model_name, output_hidden_states=True
            )
            self._model.eval()
            self._hidden_size = self._model.config.hidden_size
            self._num_layers = self._model.config.num_hidden_layers + 1  # + embedding layer
            return True
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("WavLM model unavailable (%s); wl_ features disabled.", exc)
            self._failed = True
            return False

    # ------------------------------------------------------------------- caching
    def _cache_key(self, audio: PreprocessedAudio) -> str:
        # NOTE: key shape (unsorted, "model" then "max_duration_s") and .npy storage
        # intentionally match scripts/experiment_wavlm_multilayer.py's cache format, so
        # this reuses the embeddings already computed by that exploration run instead of
        # re-extracting all 985 clips.
        digest = hashlib.sha1()
        digest.update(np.ascontiguousarray(audio.waveform).tobytes())
        digest.update(
            json.dumps(
                {
                    "model": self.config.wavlm_model_name,
                    "max_duration_s": self.config.wavlm_max_duration_s,
                }
            ).encode()
        )
        return digest.hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.npy"

    # ---------------------------------------------------------------- inference
    def _forward_all_layers(self, audio: PreprocessedAudio) -> np.ndarray | None:
        """Returns an (num_layers, hidden_size) array: one mean-pooled vector per layer."""
        import torch

        if not self._ensure_loaded():
            return None
        assert self._model is not None and self._processor is not None

        max_samples = int(self.config.wavlm_max_duration_s * audio.sample_rate)
        waveform = audio.waveform[:max_samples]
        if waveform.size < audio.sample_rate * 0.2:
            return np.zeros((self._num_layers, self._hidden_size), dtype=np.float32)

        inputs = self._processor(waveform, sampling_rate=audio.sample_rate, return_tensors="pt")
        with torch.no_grad():
            output = self._model(inputs.input_values)
        pooled = np.stack(
            [layer.mean(dim=1).squeeze(0).numpy() for layer in output.hidden_states], axis=0
        )
        return pooled.astype(np.float32)

    def extract(self, audio: PreprocessedAudio) -> dict[str, float]:
        """Configured layers' mean-pooled WavLM embeddings for one clip, cached on disk."""
        key = self._cache_key(audio)
        cache_path = self._cache_path(key)
        if cache_path.exists():
            try:
                all_layers = np.load(cache_path)
            except (OSError, ValueError) as exc:
                LOGGER.warning("Ignoring corrupt WavLM cache entry %s: %s", cache_path, exc)
                all_layers = None
        else:
            all_layers = None

        if all_layers is None:
            computed = self._forward_all_layers(audio)
            if computed is None:
                return {}
            all_layers = computed
            np.save(cache_path, all_layers)

        features: dict[str, float] = {}
        for layer in self.config.wavlm_layers:
            if layer >= all_layers.shape[0]:
                continue
            vector = all_layers[layer]
            features.update(
                {f"wl_L{layer:02d}_raw_{i:03d}": float(v) for i, v in enumerate(vector)}
            )
        return features


def reduce_wavlm_embeddings(
    X_train: pd.DataFrame, X_test: pd.DataFrame, config: FeatureConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace raw ``wl_L*_raw_*`` columns with PCA components fit on the training split.

    Pure sklearn logic, no neural network involved. Mirrors
    ``embeddings.reduce_wav2vec2_embeddings`` exactly, just grouping across whichever
    layers were configured rather than a single layer's columns. A no-op when no
    ``wl_`` columns are present.
    """
    raw_columns = [c for c in X_train.columns if c.startswith("wl_L")]
    if not raw_columns:
        return X_train, X_test

    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="median")
    train_raw = imputer.fit_transform(X_train[raw_columns])
    test_raw = imputer.transform(X_test[raw_columns])

    n_components = min(config.wavlm_pca_components, train_raw.shape[0], train_raw.shape[1])
    pca = PCA(n_components=n_components, random_state=0)
    train_components = pca.fit_transform(train_raw)
    test_components = pca.transform(test_raw)

    pc_columns = [f"wl_pc{i:02d}" for i in range(n_components)]
    train_pcs = pd.DataFrame(train_components, columns=pc_columns, index=X_train.index)
    test_pcs = pd.DataFrame(test_components, columns=pc_columns, index=X_test.index)

    X_train = pd.concat([X_train.drop(columns=raw_columns), train_pcs], axis=1)
    X_test = pd.concat([X_test.drop(columns=raw_columns), test_pcs], axis=1)
    return X_train, X_test
