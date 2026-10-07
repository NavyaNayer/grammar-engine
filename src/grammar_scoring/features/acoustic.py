"""Acoustic and prosodic features.

These do not measure grammar directly. They earn their place for two reasons:

* **Delivery correlates with control.** Speakers with weaker grammatical control tend to
  produce more, and longer, hesitation pauses and a flatter, slower prosodic contour,
  because planning competes with articulation.
* **They are ASR-independent.** When the recogniser struggles the transcript degrades,
  but the timing and spectral features stay valid, so they keep the model honest on
  exactly the samples where the text features are least reliable.

All features are prefixed ``ac_``.
"""

from __future__ import annotations

import warnings

import numpy as np

from grammar_scoring.audio import PreprocessedAudio, estimate_syllable_rate
from grammar_scoring.config import AudioConfig
from grammar_scoring.features.text_utils import safe_ratio

F0_MIN_HZ = 65.0
F0_MAX_HZ = 350.0
N_MFCC = 13


def _summary(name: str, values: np.ndarray) -> dict[str, float]:
    """Mean/std/percentile summary of a per-frame contour."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {f"ac_{name}_{stat}": 0.0 for stat in ("mean", "std", "p10", "p90", "range")}
    p10, p90 = np.percentile(values, [10, 90])
    return {
        f"ac_{name}_mean": float(np.mean(values)),
        f"ac_{name}_std": float(np.std(values)),
        f"ac_{name}_p10": float(p10),
        f"ac_{name}_p90": float(p90),
        f"ac_{name}_range": float(p90 - p10),
    }


def _pitch_features(audio: PreprocessedAudio, config: AudioConfig) -> dict[str, float]:
    """Pitch statistics, expressed in semitones relative to the speaker's own median.

    Semitone normalisation removes the (irrelevant) speaker sex/register difference so
    the features describe *intonational movement* rather than absolute voice pitch.
    """
    import librosa

    hop = 512
    frame = 2048
    if audio.waveform.size < frame * 2:
        return {
            **_summary("f0_semitone", np.zeros(0)),
            "ac_f0_median_hz": 0.0,
            "ac_voiced_frame_ratio": 0.0,
        }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f0 = librosa.yin(
            audio.waveform,
            fmin=F0_MIN_HZ,
            fmax=F0_MAX_HZ,
            frame_length=frame,
            hop_length=hop,
            sr=audio.sample_rate,
        )

    times = librosa.frames_to_time(np.arange(len(f0)), sr=audio.sample_rate, hop_length=hop)

    # Keep only frames that the energy segmentation already called speech; YIN always
    # returns a value, including for silence, and those values are meaningless.
    if audio.voiced_intervals.size:
        in_speech = np.zeros(len(f0), dtype=bool)
        for start, end in audio.voiced_intervals:
            in_speech |= (times >= start) & (times <= end)
    else:
        in_speech = np.ones(len(f0), dtype=bool)

    plausible = (f0 > F0_MIN_HZ * 1.05) & (f0 < F0_MAX_HZ * 0.95) & in_speech
    voiced = f0[plausible]
    voiced_ratio = safe_ratio(plausible.sum(), len(f0))

    if voiced.size < 5:
        return {
            **_summary("f0_semitone", np.zeros(0)),
            "ac_f0_median_hz": 0.0,
            "ac_voiced_frame_ratio": voiced_ratio,
        }

    median_hz = float(np.median(voiced))
    semitones = 12.0 * np.log2(voiced / median_hz)
    return {
        **_summary("f0_semitone", semitones),
        "ac_f0_median_hz": median_hz,
        "ac_voiced_frame_ratio": voiced_ratio,
    }


def _timing_features(audio: PreprocessedAudio, config: AudioConfig) -> dict[str, float]:
    """Pause structure and speech-rate features derived purely from the waveform."""
    duration = audio.duration
    pauses = audio.pauses
    long_pauses = pauses[pauses >= config.long_pause_s] if pauses.size else np.zeros(0)
    minutes = max(duration / 60.0, 1e-6)
    syl_rate, articulation_rate = estimate_syllable_rate(audio, config)

    features = {
        "ac_duration_s": duration,
        "ac_original_duration_s": audio.original_duration,
        "ac_speech_duration_s": audio.speech_duration,
        "ac_silence_ratio": 1.0 - safe_ratio(audio.speech_duration, duration, 1.0),
        "ac_n_segments": float(len(audio.voiced_intervals)),
        "ac_segments_per_min": len(audio.voiced_intervals) / minutes,
        "ac_pause_count": float(pauses.size),
        "ac_pauses_per_min": pauses.size / minutes,
        "ac_long_pause_count": float(long_pauses.size),
        "ac_long_pauses_per_min": long_pauses.size / minutes,
        "ac_pause_total_s": float(pauses.sum()) if pauses.size else 0.0,
        "ac_pause_mean_s": float(pauses.mean()) if pauses.size else 0.0,
        "ac_pause_std_s": float(pauses.std()) if pauses.size else 0.0,
        "ac_pause_max_s": float(pauses.max()) if pauses.size else 0.0,
        "ac_syllable_rate": syl_rate,
        "ac_articulation_rate": articulation_rate,
    }

    if audio.voiced_intervals.size:
        lengths = audio.voiced_intervals[:, 1] - audio.voiced_intervals[:, 0]
        features.update(
            {
                "ac_segment_mean_s": float(lengths.mean()),
                "ac_segment_std_s": float(lengths.std()),
                "ac_segment_max_s": float(lengths.max()),
                # Mean length of run between pauses: how much a speaker can utter in one
                # planned breath group. Fluent speakers sustain longer runs.
                "ac_segment_cv": safe_ratio(lengths.std(), lengths.mean()),
            }
        )
    else:
        features.update(
            {
                "ac_segment_mean_s": 0.0,
                "ac_segment_std_s": 0.0,
                "ac_segment_max_s": 0.0,
                "ac_segment_cv": 0.0,
            }
        )
    return features


def _spectral_features(audio: PreprocessedAudio, config: AudioConfig) -> dict[str, float]:
    """Loudness dynamics, spectral shape and MFCC statistics."""
    import librosa

    y = audio.waveform
    sr = audio.sample_rate
    hop = config.hop_length
    frame = config.frame_length

    if y.size < frame * 2:
        empty = {}
        for name in ("rms_db", "zcr", "centroid", "bandwidth", "rolloff", "flatness"):
            empty.update(_summary(name, np.zeros(0)))
        for i in range(N_MFCC):
            empty[f"ac_mfcc{i:02d}_mean"] = 0.0
            empty[f"ac_mfcc{i:02d}_std"] = 0.0
        empty["ac_spectral_tilt"] = 0.0
        return empty

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
        rms_db = librosa.amplitude_to_db(np.maximum(rms, 1e-10), ref=1.0)
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame, hop_length=hop)[0]
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop)[0]
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop, roll_percent=0.9)[0]
        flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop)[0]
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=hop)

    features: dict[str, float] = {}
    features.update(_summary("rms_db", rms_db))
    features.update(_summary("zcr", zcr))
    features.update(_summary("centroid", centroid))
    features.update(_summary("bandwidth", bandwidth))
    features.update(_summary("rolloff", rolloff))
    features.update(_summary("flatness", flatness))

    for i in range(N_MFCC):
        features[f"ac_mfcc{i:02d}_mean"] = float(np.mean(mfcc[i]))
        features[f"ac_mfcc{i:02d}_std"] = float(np.std(mfcc[i]))

    # Long-term average spectrum tilt: slope of log-power against log-frequency.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spectrum = np.mean(np.abs(librosa.stft(y, hop_length=hop)) ** 2, axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    band = (freqs > 80) & (freqs < 6000)
    if band.sum() > 10:
        slope = np.polyfit(np.log10(freqs[band]), np.log10(spectrum[band] + 1e-12), 1)[0]
        features["ac_spectral_tilt"] = float(slope)
    else:
        features["ac_spectral_tilt"] = 0.0

    # Crude SNR proxy: loudness gap between speech frames and the quietest 10%.
    noise_floor = float(np.percentile(rms_db, 10))
    speech_level = float(np.percentile(rms_db, 90))
    features["ac_snr_proxy_db"] = speech_level - noise_floor
    return features


def extract_acoustic_features(
    audio: PreprocessedAudio, config: AudioConfig | None = None
) -> dict[str, float]:
    """Full acoustic feature block for one clip."""
    config = config or AudioConfig()
    features: dict[str, float] = {}
    features.update(_timing_features(audio, config))
    features.update(_spectral_features(audio, config))
    features.update(_pitch_features(audio, config))
    return features
