"""Whisper-based transcription with caching and decoder-confidence features.

Grammar can only be measured on words, so the transcript is the hinge of the whole
pipeline. Three details matter here:

1. **Chunking at pauses.** Whisper consumes fixed 30 s windows but the clips are
   45-60 s. Instead of cutting blindly (which clips words in half) or using an
   overlapping sliding window (which duplicates text), we cut inside the longest
   silence in the second half of each window, reusing the pause segmentation from
   :mod:`grammar_scoring.audio`.
2. **Confidence is a feature, not a diagnostic.** The decoder's mean token
   log-probability is a good proxy for how intelligible the speech was, and
   intelligibility correlates with the rubric. We therefore keep the scores.
3. **Caching.** Transcription dominates runtime, so results are cached on disk keyed
   by (audio content hash, model, decoding params).
"""

from __future__ import annotations

import hashlib
import json
import logging
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from grammar_scoring.audio import PreprocessedAudio
from grammar_scoring.config import ASRConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    avg_logprob: float


@dataclass
class Transcript:
    """A transcript plus the decoder statistics used as features."""

    text: str
    segments: list[TranscriptSegment] = field(default_factory=list)
    avg_logprob: float = 0.0
    #: Whisper's repetition detector: text length divided by its zlib-compressed length.
    compression_ratio: float = 1.0
    duration: float = 0.0
    model_name: str = ""

    @property
    def n_words(self) -> int:
        return len(self.text.split())

    def to_dict(self) -> dict:
        return {**asdict(self)}

    @classmethod
    def from_dict(cls, payload: dict) -> Transcript:
        segments = [TranscriptSegment(**s) for s in payload.get("segments", [])]
        return cls(**{**payload, "segments": segments})


def _compression_ratio(text: str) -> float:
    if not text:
        return 1.0
    raw = text.encode("utf-8")
    return len(raw) / max(1, len(zlib.compress(raw)))


def chunk_bounds(audio: PreprocessedAudio, max_chunk_s: float = 30.0) -> list[tuple[float, float]]:
    """Split the clip into <= ``max_chunk_s`` windows, cutting inside silences.

    Within each window we look for the longest pause in its second half and cut at the
    pause midpoint, which keeps words intact. If the window contains no pause (rapid,
    unbroken speech) we fall back to a hard cut.
    """
    duration = audio.duration
    if duration <= max_chunk_s:
        return [(0.0, duration)]

    intervals = audio.voiced_intervals
    bounds: list[tuple[float, float]] = []
    start = 0.0
    while start < duration - 1e-3:
        limit = start + max_chunk_s
        if limit >= duration:
            bounds.append((start, duration))
            break

        cut = limit
        if len(intervals) >= 2:
            gap_starts = intervals[:-1, 1]
            gap_ends = intervals[1:, 0]
            # Only consider gaps that leave a usefully long chunk behind.
            usable = (gap_starts > start + max_chunk_s * 0.5) & (gap_ends < limit)
            if usable.any():
                widths = gap_ends[usable] - gap_starts[usable]
                best = int(np.argmax(widths))
                cut = float((gap_starts[usable][best] + gap_ends[usable][best]) / 2.0)

        bounds.append((start, cut))
        start = cut
    return bounds


class WhisperTranscriber:
    """Lazily-loaded Whisper wrapper. One instance can transcribe a whole dataset."""

    def __init__(self, config: ASRConfig | None = None, cache_dir: str | Path = ".cache/asr"):
        self.config = config or ASRConfig()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._processor = None
        self._model_name: str | None = None

    # ------------------------------------------------------------------ loading
    def _resolve_device(self):
        import torch

        if self.config.device != "auto":
            return torch.device(self.config.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        from transformers.utils import logging as hf_logging

        # Whisper emits a deprecation notice about `forced_decoder_ids` on every single
        # call, which would bury the pipeline's own progress output.
        hf_logging.set_verbosity_error()

        candidates = [self.config.model_name]
        if self.config.fallback_model_name not in candidates:
            candidates.append(self.config.fallback_model_name)

        last_error: Exception | None = None
        for name in candidates:
            try:
                LOGGER.info("Loading ASR model %s ...", name)
                processor = WhisperProcessor.from_pretrained(name)
                model = WhisperForConditionalGeneration.from_pretrained(name)
                model.eval()
                model.to(self._resolve_device())
                self._processor, self._model, self._model_name = processor, model, name
                LOGGER.info("ASR model ready: %s (device=%s)", name, self._resolve_device())
                return
            except Exception as exc:  # noqa: BLE001 - we genuinely want any failure here
                last_error = exc
                LOGGER.warning("Could not load ASR model %s: %s", name, exc)

        raise RuntimeError(f"No Whisper checkpoint could be loaded: {last_error}")

    @property
    def model_name(self) -> str:
        return self._model_name or self.config.model_name

    # ------------------------------------------------------------------- caching
    def _cache_key(self, audio: PreprocessedAudio) -> str:
        digest = hashlib.sha1()
        digest.update(np.ascontiguousarray(audio.waveform).tobytes())
        payload = {
            "model": self.config.model_name,
            "fallback": self.config.fallback_model_name,
            "beams": self.config.num_beams,
            "chunk": self.config.chunk_length_s,
            "lang": self.config.language,
        }
        # Only added when set, so existing cache keys stay valid.
        if self.config.initial_prompt:
            payload["prompt"] = self.config.initial_prompt
        digest.update(json.dumps(payload, sort_keys=True).encode())
        return digest.hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    # -------------------------------------------------------------- transcription
    def transcribe(self, audio: PreprocessedAudio) -> Transcript:
        """Transcribe one preprocessed clip, using the on-disk cache when possible."""
        key = self._cache_key(audio)
        cache_path = self._cache_path(key)
        if self.config.use_cache and cache_path.exists():
            try:
                return Transcript.from_dict(json.loads(cache_path.read_text()))
            except (json.JSONDecodeError, TypeError) as exc:
                LOGGER.warning("Ignoring corrupt ASR cache entry %s: %s", cache_path, exc)

        transcript = self._transcribe_uncached(audio)
        if self.config.use_cache:
            cache_path.write_text(json.dumps(transcript.to_dict()))
        return transcript

    def _transcribe_uncached(self, audio: PreprocessedAudio) -> Transcript:
        import torch

        if audio.waveform.size < audio.sample_rate * 0.2:
            return Transcript(text="", duration=audio.duration, model_name=self.model_name)

        self._ensure_loaded()
        assert self._model is not None and self._processor is not None
        device = self._resolve_device()
        sr = audio.sample_rate

        segments: list[TranscriptSegment] = []
        for start, end in chunk_bounds(audio, float(self.config.chunk_length_s)):
            window = audio.waveform[int(start * sr) : int(end * sr)]
            if window.size < sr * 0.1:
                continue

            inputs = self._processor(
                window, sampling_rate=sr, return_tensors="pt", return_attention_mask=True
            )
            features = inputs.input_features.to(device)
            attention_mask = getattr(inputs, "attention_mask", None)
            gen_kwargs = {
                "max_new_tokens": 440,
                "num_beams": self.config.num_beams,
                "do_sample": False,
                "return_dict_in_generate": True,
                "output_scores": True,
            }
            if attention_mask is not None:
                gen_kwargs["attention_mask"] = attention_mask.to(device)
            if self.config.initial_prompt:
                prompt_ids = self._processor.get_prompt_ids(
                    self.config.initial_prompt, return_tensors="pt"
                ).to(device)
                gen_kwargs["prompt_ids"] = prompt_ids
                # Whisper's decoder holds 448 positions; the prompt uses some of them.
                gen_kwargs["max_new_tokens"] = max(64, 440 - int(prompt_ids.shape[-1]))
            if self.config.language and not self.model_name.endswith(".en"):
                gen_kwargs["language"] = self.config.language
                gen_kwargs["task"] = "transcribe"

            with torch.no_grad():
                output = self._model.generate(features, **gen_kwargs)

            text = self._processor.batch_decode(output.sequences, skip_special_tokens=True)[0]
            # Whisper returns the decoder prompt as the start of the output; it is not speech.
            prompt = self.config.initial_prompt
            if prompt and text.strip().startswith(prompt.strip()):
                text = text.strip()[len(prompt.strip()) :]
            segments.append(
                TranscriptSegment(
                    start=start,
                    end=end,
                    text=text.strip(),
                    avg_logprob=self._mean_logprob(output),
                )
            )

        text = " ".join(s.text for s in segments if s.text).strip()
        weights = np.array([max(s.end - s.start, 1e-6) for s in segments]) if segments else None
        avg_logprob = (
            float(np.average([s.avg_logprob for s in segments], weights=weights))
            if segments
            else 0.0
        )

        return Transcript(
            text=text,
            segments=segments,
            avg_logprob=avg_logprob,
            compression_ratio=_compression_ratio(text),
            duration=audio.duration,
            model_name=self.model_name,
        )

    def _mean_logprob(self, output) -> float:
        """Mean per-token log probability of the generated sequence."""
        import torch

        try:
            scores = self._model.compute_transition_scores(
                output.sequences, output.scores, normalize_logits=True
            )
            valid = scores[torch.isfinite(scores)]
            return float(valid.mean()) if valid.numel() else 0.0
        except Exception as exc:  # noqa: BLE001 - confidence is optional, never fatal
            LOGGER.debug("Could not compute transition scores: %s", exc)
            return 0.0

    def transcribe_many(
        self, audios: list[PreprocessedAudio], progress: bool = True
    ) -> list[Transcript]:
        iterator = audios
        if progress:
            from tqdm.auto import tqdm

            iterator = tqdm(audios, desc="Transcribing", unit="clip")
        return [self.transcribe(a) for a in iterator]
