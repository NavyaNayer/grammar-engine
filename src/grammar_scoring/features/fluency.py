"""Fluency, disfluency and self-repair features.

These sit at the junction of the waveform and the transcript. The rubric explicitly
mentions incomplete sentences (level 2) and self-correction (levels 4-5), so
disfluency is scored evidence, not noise -- but the *direction* differs by type:
filled pauses and stalled restarts point down, while overt self-repair points up.

All features are prefixed ``fl_``.
"""

from __future__ import annotations

from grammar_scoring.asr import Transcript
from grammar_scoring.audio import PreprocessedAudio
from grammar_scoring.features.text_utils import (
    DANGLING_ENDINGS,
    FILLER_PHRASES,
    FILLERS,
    REPAIR_MARKERS,
    count_phrases,
    looks_verbless,
    safe_ratio,
    split_sentences,
    tokenize,
)


def _repetition_counts(tokens: list[str]) -> tuple[int, int]:
    """Count immediate unigram repeats ("the the") and bigram repeats ("I was I was")."""
    unigram = sum(1 for a, b in zip(tokens, tokens[1:], strict=False) if a == b and len(a) > 1)

    bigram = 0
    i = 0
    while i + 3 < len(tokens):
        if tokens[i] == tokens[i + 2] and tokens[i + 1] == tokens[i + 3]:
            bigram += 1
            i += 4
        else:
            i += 1
    return unigram, bigram


def extract_fluency_features(
    transcript: Transcript, audio: PreprocessedAudio | None = None
) -> dict[str, float]:
    """Fluency block for one clip."""
    tokens = tokenize(transcript.text)
    sentences = split_sentences(transcript.text)
    n_words = len(tokens)
    per_100 = lambda count: 100.0 * safe_ratio(count, n_words)  # noqa: E731

    duration = audio.duration if audio is not None else transcript.duration
    speech_duration = audio.speech_duration if audio is not None else duration
    n_segments = len(audio.voiced_intervals) if audio is not None else 0

    n_fillers = sum(1 for t in tokens if t in FILLERS)
    n_filler_phrases = count_phrases(transcript.text, FILLER_PHRASES)
    n_repairs = count_phrases(transcript.text, REPAIR_MARKERS)
    unigram_reps, bigram_reps = _repetition_counts(tokens)

    fragments = sum(1 for s in sentences if looks_verbless(s))
    dangling = 0
    for sentence in sentences:
        sentence_tokens = tokenize(sentence)
        if sentence_tokens and sentence_tokens[-1] in DANGLING_ENDINGS:
            dangling += 1

    sentence_lengths = [len(tokenize(s)) for s in sentences]
    n_sentences = len(sentences)

    features = {
        "fl_n_words": float(n_words),
        "fl_n_sentences": float(n_sentences),
        "fl_words_per_second": safe_ratio(n_words, duration),
        # Articulation rate excludes pause time, separating "thinks slowly" from
        # "speaks slowly" -- only the former indicates weak command of the language.
        "fl_articulation_wps": safe_ratio(n_words, speech_duration),
        "fl_words_per_segment": safe_ratio(n_words, n_segments),
        "fl_pause_time_per_word": safe_ratio(duration - speech_duration, n_words),
        "fl_filler_per_100w": per_100(n_fillers),
        "fl_filler_phrase_per_100w": per_100(n_filler_phrases),
        "fl_filler_total_per_100w": per_100(n_fillers + n_filler_phrases),
        "fl_repeat_unigram_per_100w": per_100(unigram_reps),
        "fl_repeat_bigram_per_100w": per_100(bigram_reps),
        "fl_repair_per_100w": per_100(n_repairs),
        "fl_fragment_ratio": safe_ratio(fragments, n_sentences),
        "fl_dangling_end_ratio": safe_ratio(dangling, n_sentences),
        # --- ASR decoder statistics: proxies for intelligibility -----------------
        "fl_asr_avg_logprob": transcript.avg_logprob,
        "fl_asr_compression_ratio": transcript.compression_ratio,
        "fl_asr_segment_count": float(len(transcript.segments)),
    }

    if transcript.segments:
        logprobs = [s.avg_logprob for s in transcript.segments]
        features["fl_asr_logprob_min"] = float(min(logprobs))
        features["fl_asr_logprob_std"] = float(
            (sum((x - sum(logprobs) / len(logprobs)) ** 2 for x in logprobs) / len(logprobs)) ** 0.5
        )
    else:
        features["fl_asr_logprob_min"] = 0.0
        features["fl_asr_logprob_std"] = 0.0

    if sentence_lengths:
        mean_len = sum(sentence_lengths) / len(sentence_lengths)
        features["fl_sentence_len_mean"] = float(mean_len)
        features["fl_sentence_len_max"] = float(max(sentence_lengths))
        features["fl_sentence_len_min"] = float(min(sentence_lengths))
        features["fl_sentence_len_std"] = float(
            (sum((x - mean_len) ** 2 for x in sentence_lengths) / len(sentence_lengths)) ** 0.5
        )
        features["fl_short_sentence_ratio"] = safe_ratio(
            sum(1 for x in sentence_lengths if x <= 4), len(sentence_lengths)
        )
    else:
        for key in (
            "fl_sentence_len_mean",
            "fl_sentence_len_max",
            "fl_sentence_len_min",
            "fl_sentence_len_std",
            "fl_short_sentence_ratio",
        ):
            features[key] = 0.0

    return features
