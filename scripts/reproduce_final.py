"""Rebuild the final submission from the raw competition data.

Steps (see ARCHITECTURE.md):
  1. Text features: Whisper-small.en transcripts, LanguageTool, fluency and lexical features.
  2. Audio features: frozen WavLM-base-plus, layers 8-10, mean-pooled over the first 65 s,
     concatenated and reduced to 64 principal components (fitted on training clips only).
  3. Ridge + gradient boosting, combined with non-negative weights from 5-fold CV.
  4. Affine recalibration fitted on the out-of-fold predictions.
  5. Clip to [0, 5].

Usage (from this folder):
    python scripts/reproduce_final.py --data-dir path/to/data/raw --cache-dir path/to/.cache

The first run extracts WavLM features for all 985 clips, which takes about an hour on CPU.
Later runs reuse the cache. The output is compared with submission.csv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

from grammar_scoring.audio import load_audio  # noqa: E402
from grammar_scoring.config import AudioConfig, ModelConfig, PipelineConfig  # noqa: E402
from grammar_scoring.data import load_dataset  # noqa: E402
from grammar_scoring.features.assemble import FeaturePipeline, align_features  # noqa: E402
from grammar_scoring.modeling import GrammarScorer, run_model_zoo  # noqa: E402
from grammar_scoring.pipeline import build_submission  # noqa: E402

WAVLM_MODEL = "microsoft/wavlm-base-plus"
LAYERS = (8, 9, 10)
MAX_DURATION_S = 65.0
PCA_COMPONENTS = 64
SCORE_MIN, SCORE_MAX = 0.0, 5.0


def wavlm_layers(audio, cache_dir: Path, extractor_state: dict) -> np.ndarray:
    """Mean-pooled hidden state of every WavLM-base layer for one clip, cached on disk."""
    digest = hashlib.sha1()
    digest.update(np.ascontiguousarray(audio.waveform).tobytes())
    digest.update(json.dumps({"model": WAVLM_MODEL, "max_duration_s": MAX_DURATION_S}).encode())
    path = cache_dir / f"{digest.hexdigest()}.npy"
    if path.exists():
        return np.load(path)

    if "model" not in extractor_state:
        import torch
        from transformers import Wav2Vec2FeatureExtractor, WavLMModel

        torch.set_num_threads(4)
        extractor_state["processor"] = Wav2Vec2FeatureExtractor.from_pretrained(WAVLM_MODEL)
        extractor_state["model"] = WavLMModel.from_pretrained(WAVLM_MODEL, output_hidden_states=True).eval()
    import torch

    model, processor = extractor_state["model"], extractor_state["processor"]
    waveform = audio.waveform[: int(MAX_DURATION_S * audio.sample_rate)]
    if waveform.size < audio.sample_rate * 0.2:
        pooled = np.zeros((model.config.num_hidden_layers + 1, model.config.hidden_size), np.float32)
    else:
        inputs = processor(waveform, sampling_rate=audio.sample_rate, return_tensors="pt")
        with torch.no_grad():
            out = model(inputs.input_values)
        pooled = np.stack([h.mean(dim=1).squeeze(0).numpy() for h in out.hidden_states]).astype(np.float32)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, pooled)
    return pooled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True, help="folder with train.csv, test.csv, train/, test/")
    parser.add_argument("--cache-dir", type=Path, default=HERE / ".cache")
    parser.add_argument("--out", type=Path, default=HERE / "reproduced_submission.csv")
    args = parser.parse_args()

    dataset = load_dataset(args.data_dir)
    config = PipelineConfig(data_dir=args.data_dir, cache_dir=args.cache_dir)
    config.asr.model_name = "openai/whisper-small.en"
    config.features.wav2vec2 = False
    config.features.wavlm = False

    pipeline = FeaturePipeline(config)
    try:
        train = pipeline.transform(dataset.train.audio_paths, dataset.train.filenames, split="train")
        test = pipeline.transform(dataset.test.audio_paths, dataset.test.filenames, split="test")
    finally:
        pipeline.close()
    X_tr_text, X_te_text = align_features(train.features, test.features)
    tr_idx, te_idx = list(X_tr_text.index), list(X_te_text.index)
    y = dataset.train.frame.set_index("filename").loc[tr_idx, "label"].to_numpy(float)

    # WavLM-base layers 8-10 for every clip.
    cache = args.cache_dir / "wavlm_multilayer"
    state: dict = {}
    audio_config = AudioConfig()

    def clip_block(paths, names):
        return pd.DataFrame(
            [np.concatenate([wavlm_layers(load_audio(p, audio_config), cache, state)[L] for L in LAYERS]) for p in paths],
            index=names,
        )

    tr_audio = clip_block(dataset.train.audio_paths, dataset.train.filenames)
    te_audio = clip_block(dataset.test.audio_paths, dataset.test.filenames)
    tr_audio = tr_audio.loc[tr_idx]
    te_audio = te_audio.loc[te_idx]

    # PCA fitted on training clips only.
    imputer = SimpleImputer(strategy="median")
    tr_imp = imputer.fit_transform(tr_audio.to_numpy(float))
    te_imp = imputer.transform(te_audio.to_numpy(float))
    pca = PCA(n_components=PCA_COMPONENTS, random_state=0).fit(tr_imp)
    X_tr = np.hstack([X_tr_text.to_numpy(float), pca.transform(tr_imp)])
    X_te = np.hstack([X_te_text.to_numpy(float), pca.transform(te_imp)])

    # Ridge + gradient boosting, non-negative blend, out-of-fold predictions.
    zoo_config = ModelConfig(candidates=("ridge", "gradient_boosting"), n_splits=5, n_repeats=2, clip_predictions=False)
    zoo = run_model_zoo(pd.DataFrame(X_tr), y, zoo_config, verbose=False)
    names = list(zoo.blend_weights)
    weights = np.array([zoo.blend_weights[n] for n in names])
    oof = zoo.oof_frame()[names].to_numpy(float)

    # Affine recalibration on the out-of-fold blend.
    raw_oof = oof @ weights
    recal = LinearRegression().fit(raw_oof.reshape(-1, 1), y)
    print(f"blend weights {dict(zip(names, np.round(weights, 3)))}")
    print(f"recalibration slope {recal.coef_[0]:.3f}, intercept {recal.intercept_:.3f}")

    # Refit the blend on all training clips and predict the test clips.
    scorer = GrammarScorer(zoo_config, weights=zoo.blend_weights).fit(pd.DataFrame(X_tr), y)
    raw_test = scorer.predict(pd.DataFrame(X_te))
    final = np.clip(recal.predict(raw_test.reshape(-1, 1)), SCORE_MIN, SCORE_MAX)

    submission = build_submission(te_idx, final, dataset.sample_submission)
    submission.to_csv(args.out, index=False)
    print("wrote", args.out)

    reference = pd.read_csv(HERE / "submission.csv")
    merged = submission.merge(reference, on="filename", suffixes=("_repro", "_final"))
    corr = merged["label_repro"].corr(merged["label_final"])
    diff = (merged["label_repro"] - merged["label_final"]).abs().mean()
    print(f"agreement with submission.csv: correlation {corr:.4f}, mean absolute difference {diff:.3f}")


if __name__ == "__main__":
    main()
