"""Audio preprocessing and temporal (pause / segment) analysis.

Preprocessing is deliberately conservative: the grammatical signal we are after lives
in *what* was said, so we only do what ASR and the acoustic features both need --
mono-mixing, resampling to 16 kHz, DC removal, peak normalisation and silence trimming.
Aggressive denoising is avoided because it removes the hesitation cues (breaths, filled
pauses) that carry genuine information about the speaker's control of the language.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

from grammar_scoring.config import AudioConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class PreprocessedAudio:
    """A normalised waveform plus its voiced/silent segmentation."""

    path: Path
    waveform: np.ndarray  # float32, mono, `sample_rate` Hz, peak-normalised
    sample_rate: int
    #: (start, end) in seconds for every detected voiced run, in order.
    voiced_intervals: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    #: Duration of the file *before* silence trimming.
    original_duration: float = 0.0

    @property
    def duration(self) -> float:
        return len(self.waveform) / self.sample_rate

    @cached_property
    def speech_duration(self) -> float:
        """Total voiced time in seconds (i.e. duration minus internal pauses)."""
        if self.voiced_intervals.size == 0:
            return self.duration
        return float(np.sum(self.voiced_intervals[:, 1] - self.voiced_intervals[:, 0]))

    @cached_property
    def pauses(self) -> np.ndarray:
        """Durations (seconds) of silent gaps *between* voiced runs.

        Leading and trailing silence is excluded: it reflects recording setup rather
        than the speaker's planning effort.
        """
        if len(self.voiced_intervals) < 2:
            return np.zeros(0)
        return self.voiced_intervals[1:, 0] - self.voiced_intervals[:-1, 1]


def _to_mono(y: np.ndarray) -> np.ndarray:
    return y if y.ndim == 1 else np.mean(y, axis=0)


def load_audio(path: str | Path, config: AudioConfig | None = None) -> PreprocessedAudio:
    """Load ``path`` and apply the standard preprocessing chain."""
    import librosa

    config = config or AudioConfig()
    path = Path(path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(path, sr=config.sample_rate, mono=True)

    y = _to_mono(np.asarray(y, dtype=np.float32))
    original_duration = len(y) / config.sample_rate

    if y.size == 0:
        LOGGER.warning("Empty audio file: %s", path)
        return PreprocessedAudio(path, y, config.sample_rate, original_duration=0.0)

    # Remove any DC offset before energy-based segmentation, otherwise a biased
    # microphone makes silence look voiced.
    y = y - float(np.mean(y))

    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y * (config.peak_normalize_to / peak)

    if config.trim_silence:
        y_trimmed, _ = librosa.effects.trim(
            y,
            top_db=config.top_db,
            frame_length=config.frame_length,
            hop_length=config.hop_length,
        )
        if y_trimmed.size > config.sample_rate * 0.1:
            y = y_trimmed

    intervals = detect_voiced_intervals(y, config)
    return PreprocessedAudio(
        path=path,
        waveform=y.astype(np.float32),
        sample_rate=config.sample_rate,
        voiced_intervals=intervals,
        original_duration=original_duration,
    )


def detect_voiced_intervals(y: np.ndarray, config: AudioConfig) -> np.ndarray:
    """Split the waveform into voiced runs using a relative energy threshold.

    Returns an ``(n, 2)`` array of (start, end) times in seconds. Adjacent runs
    separated by less than ``min_pause_s`` are merged, so that short stop closures and
    plosive gaps are not mistaken for hesitation pauses.
    """
    import librosa

    if y.size == 0:
        return np.zeros((0, 2))

    raw = librosa.effects.split(
        y,
        top_db=config.top_db,
        frame_length=config.frame_length,
        hop_length=config.hop_length,
    )
    if raw.size == 0:
        return np.zeros((0, 2))

    intervals = raw / float(config.sample_rate)

    merged: list[list[float]] = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start - merged[-1][1] < config.min_pause_s:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    out = np.asarray(merged, dtype=float)
    keep = (out[:, 1] - out[:, 0]) >= config.min_segment_s
    return out[keep] if keep.any() else out


def estimate_syllable_rate(audio: PreprocessedAudio, config: AudioConfig) -> tuple[float, float]:
    """Estimate syllables/second from intensity peaks, independent of the ASR output.

    This is a simplified version of the De Jong & Wempe nucleus detector: peaks in the
    smoothed loudness contour that rise at least 2 dB above the neighbouring dip and sit
    within 25 dB of the global peak are counted as syllable nuclei. It gives a speech-rate
    estimate that does not inherit ASR deletion errors, which matters for weak speakers
    whose audio the recogniser transcribes poorly.

    Returns ``(syllables_per_second_overall, syllables_per_second_of_speech)``.
    """
    import librosa
    from scipy.signal import find_peaks

    if audio.waveform.size < config.frame_length:
        return 0.0, 0.0

    rms = librosa.feature.rms(
        y=audio.waveform, frame_length=config.frame_length, hop_length=config.hop_length
    )[0]
    db = librosa.amplitude_to_db(np.maximum(rms, 1e-10), ref=np.max)

    # ~55 ms minimum spacing: faster than the fastest plausible syllable rate (~18/s).
    min_distance = max(1, int(round(0.055 * config.sample_rate / config.hop_length)))
    peaks, _ = find_peaks(db, height=-25.0, prominence=2.0, distance=min_distance)

    n_syllables = float(len(peaks))
    overall = n_syllables / audio.duration if audio.duration > 0 else 0.0
    articulated = n_syllables / audio.speech_duration if audio.speech_duration > 0 else 0.0
    return overall, articulated
