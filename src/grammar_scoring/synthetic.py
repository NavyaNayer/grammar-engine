"""A locally generated stand-in for the SHL audio corpus.

The competition audio is behind Kaggle competition access. To keep this repository
runnable and verifiable without it, this module synthesises a corpus with the *same
shape*: spoken monologues of 45-60 s, each with an MOS Likert grammar score in [1, 5],
written out in the exact CSV layout the real dataset uses.

How the labels become real signal rather than a fake target:

1. A latent proficiency level (1-5) is drawn for each speaker.
2. A monologue is composed from well-formed sentences, then degraded by *level-dependent
   error operators* -- dropped articles, subject-verb disagreement, tense flattening,
   preposition substitution, omitted auxiliaries, abandoned fragments, filled pauses and
   restarts. These are the error types the rubric enumerates.
3. The text is spoken by ``espeak-ng``, with speaking rate, pause length and voice
   varied by level, and light room noise added.
4. The published label is the latent level plus rater noise.

The audio is genuine waveform data, so the *entire* pipeline -- loading, pause
segmentation, Whisper ASR, every feature extractor -- runs on it unchanged.

Two honest caveats, both repeated in the notebook: ``espeak-ng`` is a formant
synthesiser, so it is far more intelligible and far less prosodically varied than real
L2 speech, and the error operators are the same ones the feature extractors look for.
Absolute metrics on this corpus are therefore optimistic and are **not** an estimate of
leaderboard performance. Its purpose is to prove the pipeline runs end to end and that
every stage is wired up correctly.
"""

from __future__ import annotations

import logging
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from grammar_scoring.config import SAMPLE_RATE, SCORE_MAX, SCORE_MIN

LOGGER = logging.getLogger(__name__)

# --------------------------------------------------------------------------- text

#: Well-formed source sentences, grouped by the prompt a candidate might answer.
SENTENCE_POOL: dict[str, list[str]] = {
    "work": [
        "I have been working as a support engineer for about three years now",
        "My main responsibility is to handle escalations that the first level team cannot fix",
        "Last quarter I led a small project to automate our weekly reporting",
        "The team I work with is spread across two different time zones",
        "I usually start my day by reviewing the tickets that came in overnight",
        "What I enjoy most about the role is that no two days look the same",
        "We migrated the entire billing system to a new platform without any downtime",
        "I had to learn a completely new framework in order to finish that assignment",
    ],
    "education": [
        "I studied computer science at a university in my home city",
        "During my final year I worked on a project about image classification",
        "One of my professors encouraged me to apply for an internship abroad",
        "The course that influenced me the most was the one on operating systems",
        "I graduated two years ago with a reasonably good grade",
        "Alongside my degree I completed several online certifications in data analysis",
    ],
    "city": [
        "The city where I grew up is quite small compared to where I live now",
        "There is a large park near my apartment where I go running every morning",
        "Public transport here is reliable which makes commuting much easier",
        "I moved to this city four years ago because of a job opportunity",
        "What surprised me most when I arrived was how friendly everyone was",
        "The weather can be difficult in winter but the summers are lovely",
    ],
    "hobby": [
        "In my free time I like to cook food from different countries",
        "I started playing the guitar when I was about fourteen years old",
        "Reading has always been the way I relax after a long day",
        "Every weekend I try to go hiking somewhere outside the city",
        "I recently joined a local football club and we train twice a week",
        "Photography is something I picked up during the pandemic and never stopped",
    ],
    "future": [
        "In the next few years I would like to move into a more technical role",
        "I am hoping to take on more responsibility for architecture decisions",
        "My long term goal is to lead a team of my own",
        "I want to keep learning because this industry changes very quickly",
        "Eventually I would like to mentor junior engineers the way I was mentored",
    ],
}

#: Discourse connectives. Higher levels get the syntactically richer ones.
SIMPLE_CONNECTIVES = ["and", "but", "so", "also", "then"]
COMPLEX_CONNECTIVES = [
    "which meant that",
    "even though",
    "whereas",
    "as a result",
    "in other words",
    "the reason being that",
    "what I found particularly useful was that",
    "having said that",
]

FILLERS = ["um", "uh", "er", "hmm"]
FILLER_PHRASES = ["you know", "I mean", "kind of", "sort of", "how to say"]
REPAIR_PHRASES = ["I mean", "sorry", "or rather", "what I meant was"]

#: Irregular past tenses, so "tense flattening" produces realistic learner errors.
PAST_TO_BASE = {
    "have": "has",
    "been": "be",
    "was": "is",
    "were": "is",
    "had": "have",
    "led": "lead",
    "went": "go",
    "made": "make",
    "took": "take",
    "started": "start",
    "worked": "work",
    "studied": "study",
    "graduated": "graduate",
    "moved": "move",
    "joined": "join",
    "learned": "learn",
    "completed": "complete",
    "migrated": "migrate",
    "picked": "pick",
    "encouraged": "encourage",
    "influenced": "influence",
    "surprised": "surprise",
    "arrived": "arrive",
    "mentored": "mentor",
}

ARTICLES = {"a", "an", "the"}

PREPOSITIONS = ["in", "on", "at", "for", "to", "with", "about", "from", "of"]

#: Error probabilities per latent level: (article drop, agreement, tense, preposition,
#: auxiliary drop, fragment, filler, restart).
_LEVEL_ERROR_RATES = {
    1: dict(article=0.55, agreement=0.45, tense=0.45, prep=0.35, aux=0.40, fragment=0.30,
            filler=0.55, restart=0.30),
    2: dict(article=0.35, agreement=0.30, tense=0.28, prep=0.22, aux=0.22, fragment=0.18,
            filler=0.40, restart=0.20),
    3: dict(article=0.16, agreement=0.14, tense=0.12, prep=0.12, aux=0.08, fragment=0.07,
            filler=0.25, restart=0.10),
    4: dict(article=0.05, agreement=0.04, tense=0.04, prep=0.05, aux=0.02, fragment=0.02,
            filler=0.12, restart=0.05),
    5: dict(article=0.01, agreement=0.01, tense=0.01, prep=0.01, aux=0.00, fragment=0.00,
            filler=0.05, restart=0.02),
}

#: Speaking style per level: words per minute and the pause inserted between sentences.
#: ``n_sentences`` is deliberately generous -- synthesis stops once the clip reaches its
#: target duration, so the surplus simply goes unused rather than being cut mid-word.
_LEVEL_STYLE = {
    1: dict(wpm=112, pause=(0.55, 1.50), n_sentences=(14, 18)),
    2: dict(wpm=124, pause=(0.45, 1.10), n_sentences=(14, 18)),
    3: dict(wpm=138, pause=(0.32, 0.80), n_sentences=(14, 18)),
    4: dict(wpm=150, pause=(0.25, 0.60), n_sentences=(13, 17)),
    5: dict(wpm=160, pause=(0.20, 0.50), n_sentences=(13, 17)),
}

_ESPEAK_VOICES = [
    "en-us+m1", "en-us+m2", "en-us+m3", "en-us+f2", "en-us+f3",
    "en-gb+m4", "en-gb+f4", "en-us+m5", "en-us+f5", "en-gb+m7",
]


def _drop_article(words: list[str], rng: random.Random) -> list[str]:
    indices = [i for i, w in enumerate(words) if w.lower() in ARTICLES]
    if not indices:
        return words
    drop = rng.choice(indices)
    return [w for i, w in enumerate(words) if i != drop]


def _break_agreement(words: list[str], rng: random.Random) -> list[str]:
    swaps = {"is": "are", "are": "is", "was": "were", "were": "was",
             "has": "have", "have": "has", "does": "do", "do": "does"}
    indices = [i for i, w in enumerate(words) if w.lower() in swaps]
    if indices:
        i = rng.choice(indices)
        words = list(words)
        words[i] = swaps[words[i].lower()]
        return words
    # Otherwise attach a spurious third-person -s to a verb-looking word.
    candidates = [i for i, w in enumerate(words) if w.endswith("ing") and len(w) > 5]
    if candidates:
        i = rng.choice(candidates)
        words = list(words)
        words[i] = words[i][:-3] + "s"
    return words


def _flatten_tense(words: list[str], rng: random.Random) -> list[str]:
    indices = [i for i, w in enumerate(words) if w.lower() in PAST_TO_BASE]
    if not indices:
        return words
    i = rng.choice(indices)
    words = list(words)
    words[i] = PAST_TO_BASE[words[i].lower()]
    return words


def _swap_preposition(words: list[str], rng: random.Random) -> list[str]:
    indices = [i for i, w in enumerate(words) if w.lower() in PREPOSITIONS]
    if not indices:
        return words
    i = rng.choice(indices)
    words = list(words)
    replacement = rng.choice([p for p in PREPOSITIONS if p != words[i].lower()])
    words[i] = replacement
    return words


def _drop_auxiliary(words: list[str], rng: random.Random) -> list[str]:
    auxiliaries = {"is", "are", "was", "were", "am", "have", "has", "had", "been", "will", "would"}
    indices = [i for i, w in enumerate(words) if w.lower() in auxiliaries]
    if not indices or len(words) < 6:
        return words
    drop = rng.choice(indices)
    return [w for i, w in enumerate(words) if i != drop]


def _truncate_to_fragment(words: list[str], rng: random.Random) -> list[str]:
    if len(words) < 6:
        return words
    return words[: rng.randint(3, max(3, len(words) // 2))]


def _insert_fillers(words: list[str], rng: random.Random, rate: float) -> list[str]:
    out: list[str] = []
    for word in words:
        if rng.random() < rate / 6.0:
            out.append(rng.choice(FILLERS) if rng.random() < 0.6 else rng.choice(FILLER_PHRASES))
        out.append(word)
    return out


def _insert_restart(words: list[str], rng: random.Random) -> list[str]:
    """Simulate a false start: repeat the opening words, optionally with a repair marker."""
    if len(words) < 5:
        return words
    n = rng.randint(2, 3)
    prefix = words[:n]
    if rng.random() < 0.4:
        return prefix + [rng.choice(REPAIR_PHRASES)] + words
    return prefix + words


def _degrade(sentence: str, level: int, rng: random.Random) -> str:
    """Apply level-appropriate error operators to a well-formed sentence."""
    rates = _LEVEL_ERROR_RATES[level]
    words = sentence.split()

    for name, operator in (
        ("article", _drop_article),
        ("agreement", _break_agreement),
        ("tense", _flatten_tense),
        ("prep", _swap_preposition),
        ("aux", _drop_auxiliary),
    ):
        if rng.random() < rates[name]:
            words = operator(words, rng)

    if rng.random() < rates["fragment"]:
        words = _truncate_to_fragment(words, rng)
    if rng.random() < rates["restart"]:
        words = _insert_restart(words, rng)
    if rates["filler"] > 0:
        words = _insert_fillers(words, rng, rates["filler"])

    return " ".join(words)


def compose_monologue(level: int, rng: random.Random) -> list[str]:
    """Build a level-appropriate monologue as a list of spoken sentences."""
    style = _LEVEL_STYLE[level]
    n_sentences = rng.randint(*style["n_sentences"])

    topics = rng.sample(list(SENTENCE_POOL), k=min(3, len(SENTENCE_POOL)))
    pool = [s for topic in topics for s in SENTENCE_POOL[topic]]
    rng.shuffle(pool)

    sentences: list[str] = []
    for i in range(n_sentences):
        base = pool[i % len(pool)]

        # Higher levels join clauses with richer connectives; lower levels string simple
        # clauses together or do not connect them at all.
        if i > 0:
            if level >= 4 and rng.random() < 0.45:
                base = f"{rng.choice(COMPLEX_CONNECTIVES)} {base[0].lower()}{base[1:]}"
            elif level <= 3 and rng.random() < 0.35:
                base = f"{rng.choice(SIMPLE_CONNECTIVES)} {base[0].lower()}{base[1:]}"

        sentences.append(_degrade(base, level, rng))
    return sentences


# -------------------------------------------------------------------------- audio
def espeak_available() -> bool:
    return shutil.which("espeak-ng") is not None or shutil.which("espeak") is not None


def _espeak_binary() -> str:
    return shutil.which("espeak-ng") or shutil.which("espeak") or "espeak-ng"


def _synthesize_sentence(text: str, voice: str, wpm: int, tmpdir: Path) -> np.ndarray:
    """Render one sentence to a 16 kHz mono waveform via espeak-ng."""
    import soundfile as sf

    out_path = tmpdir / "chunk.wav"
    subprocess.run(
        [_espeak_binary(), "-v", voice, "-s", str(wpm), "-w", str(out_path), text],
        check=True,
        capture_output=True,
        timeout=60,
    )
    wave, sr = sf.read(out_path, dtype="float32")
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa

        wave = librosa.resample(wave, orig_sr=sr, target_sr=SAMPLE_RATE)
    return wave.astype(np.float32)


def _fallback_waveform(sentences: list[str], level: int, rng: random.Random) -> np.ndarray:
    """Formant-free stand-in used only when espeak-ng is missing.

    It reproduces the timing and amplitude envelope of speech (so the acoustic and pause
    features remain meaningful) but carries no intelligible words, so ASR returns little.
    """
    style = _LEVEL_STYLE[level]
    pieces: list[np.ndarray] = []
    for sentence in sentences:
        n_words = max(1, len(sentence.split()))
        duration = n_words / (style["wpm"] / 60.0)
        n_samples = int(duration * SAMPLE_RATE)
        t = np.arange(n_samples) / SAMPLE_RATE
        f0 = rng.uniform(95, 190)
        carrier = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 6))
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * (style["wpm"] / 60.0) * t)
        pieces.append((carrier * envelope * 0.15).astype(np.float32))
        pause = rng.uniform(*style["pause"])
        pieces.append(np.zeros(int(pause * SAMPLE_RATE), dtype=np.float32))
    return np.concatenate(pieces) if pieces else np.zeros(SAMPLE_RATE, dtype=np.float32)


def synthesize_monologue(
    sentences: list[str],
    level: int,
    rng: random.Random,
    target_duration: tuple[float, float] = (45.0, 60.0),
) -> tuple[np.ndarray, list[str]]:
    """Speak a monologue, inserting level-dependent pauses between sentences.

    Sentences are rendered one at a time and appended until the clip reaches its target
    duration, then the last sentence is dropped if that lands closer to the target. A
    clip therefore always ends on a complete sentence rather than mid-word. Returns the
    waveform and the sentences that actually made it into the audio.
    """
    style = _LEVEL_STYLE[level]
    voice = rng.choice(_ESPEAK_VOICES)
    wpm = int(rng.gauss(style["wpm"], 6))
    target_samples = int(rng.uniform(*target_duration) * SAMPLE_RATE)

    spoken: list[str] = []
    pieces: list[np.ndarray] = []
    total = 0

    if espeak_available():
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            for sentence in sentences:
                if total >= target_samples:
                    break
                try:
                    chunk = _synthesize_sentence(sentence, voice, wpm, tmpdir)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                    LOGGER.warning("espeak failed for a sentence (%s); skipping it.", exc)
                    continue
                pause = np.zeros(
                    int(rng.uniform(*style["pause"]) * SAMPLE_RATE), dtype=np.float32
                )
                pieces.extend([chunk, pause])
                total += len(chunk) + len(pause)
                spoken.append(sentence)

        # The loop stops at the first prefix that reaches the target, so it always
        # overshoots. Keep whichever of the last two prefixes is closer to the target.
        if len(spoken) > 1:
            without_last = total - len(pieces[-1]) - len(pieces[-2])
            if abs(without_last - target_samples) < abs(total - target_samples):
                pieces = pieces[:-2]
                spoken.pop()

    if not pieces:
        waveform = _fallback_waveform(sentences, level, rng)
        spoken = list(sentences)
    else:
        waveform = np.concatenate(pieces)

    # Light room noise and a random gain, so the acoustic features see realistic variation
    # instead of a pristine synthetic signal.
    noise = np.random.default_rng(rng.randint(0, 2**31)).normal(0, 0.0025, len(waveform))
    waveform = (waveform + noise) * rng.uniform(0.55, 0.95)
    peak = float(np.max(np.abs(waveform))) or 1.0
    return (waveform / peak * 0.9).astype(np.float32), spoken


# ------------------------------------------------------------------------ corpus
#: Mid-heavy score distribution, as in most spoken-proficiency corpora.
_LEVEL_WEIGHTS = [0.10, 0.18, 0.30, 0.27, 0.15]


@dataclass
class SyntheticSample:
    filename: str
    level: int
    label: float
    text: str
    waveform: np.ndarray


def _draw_label(level: int, rng: random.Random) -> float:
    """Latent level plus rater noise, rounded to the half-point Likert grid."""
    noisy = rng.gauss(level, 0.33)
    return float(np.clip(round(noisy * 2) / 2, SCORE_MIN, SCORE_MAX))


def generate_sample(
    index: int, prefix: str, seed: int, duration: tuple[float, float]
) -> SyntheticSample:
    """Render one sample. Seeded per-sample so generation is deterministic in parallel."""
    rng = random.Random(seed)
    level = rng.choices([1, 2, 3, 4, 5], weights=_LEVEL_WEIGHTS)[0]
    sentences = compose_monologue(level, rng)
    waveform, spoken = synthesize_monologue(sentences, level, rng, duration)
    return SyntheticSample(
        filename=f"{prefix}_{index:04d}.wav",
        level=level,
        label=_draw_label(level, rng),
        text=". ".join(spoken),
        waveform=waveform,
    )


def generate_samples(
    n_samples: int, prefix: str, rng: random.Random, duration: tuple[float, float]
) -> list[SyntheticSample]:
    """Sequential generation, kept for tests and small corpora."""
    return [
        generate_sample(i, prefix, rng.randint(0, 2**31 - 1), duration) for i in range(n_samples)
    ]


def _render_to_disk(
    index: int, prefix: str, seed: int, duration: tuple[float, float], audio_dir: Path
) -> dict:
    """Worker entry point: render one clip, write it, and return only its metadata.

    Waveforms are never returned across the process boundary -- at ~3.5 MB each, a
    competition-sized corpus would be several gigabytes of inter-process traffic.
    """
    import soundfile as sf

    sample = generate_sample(index, prefix, seed, duration)
    sf.write(audio_dir / sample.filename, sample.waveform, SAMPLE_RATE, subtype="PCM_16")
    return {
        "filename": sample.filename,
        "level": sample.level,
        "true_label": sample.label,
        "reference_text": sample.text,
        "placeholder_label": round(random.Random(seed + 1).uniform(1, 5) * 2) / 2,
    }


def build_synthetic_dataset(
    output_dir: str | Path = "data/synthetic",
    n_train: int = 769,
    n_test: int = 216,
    duration: tuple[float, float] = (45.0, 60.0),
    seed: int = 20260501,
    overwrite: bool = False,
    progress: bool = True,
    n_jobs: int = -1,
):
    """Generate the corpus on disk and return it as a :class:`~grammar_scoring.data.Dataset`."""
    from joblib import Parallel, delayed

    from grammar_scoring.data import load_dataset

    output_dir = Path(output_dir)
    if (output_dir / "train.csv").exists() and not overwrite:
        LOGGER.info("Reusing existing synthetic dataset at %s", output_dir)
        return load_dataset(output_dir, is_synthetic=True)

    if not espeak_available():
        LOGGER.warning(
            "espeak-ng not found: the synthetic corpus will contain non-speech audio and "
            "the ASR stage will return empty transcripts. Install espeak-ng for a "
            "meaningful demo run."
        )

    LOGGER.info("Generating synthetic corpus (%d train / %d test) ...", n_train, n_test)
    master = random.Random(seed)
    ground_truth_rows: list[dict] = []
    test_filenames: list[str] = []

    for split, n_samples in (("train", n_train), ("test", n_test)):
        audio_dir = output_dir / "audios" / split
        audio_dir.mkdir(parents=True, exist_ok=True)
        seeds = [master.randint(0, 2**31 - 1) for _ in range(n_samples)]

        jobs = [
            delayed(_render_to_disk)(i, split, seeds[i], duration, audio_dir)
            for i in range(n_samples)
        ]
        # `return_as="generator"` yields results as workers finish them, so the progress
        # bar tracks completed clips. Wrapping the job list instead would only measure
        # how fast the jobs are dispatched.
        results = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator")(jobs)
        if progress:
            from tqdm.auto import tqdm

            results = tqdm(results, total=n_samples, desc=f"Synthesising {split}", unit="clip")
        records = list(results)

        # The real test.csv ships placeholder labels, so we mirror that exactly.
        rows = [
            {
                "filename": r["filename"],
                "label": r["true_label"] if split == "train" else r["placeholder_label"],
            }
            for r in records
        ]
        pd.DataFrame(rows).to_csv(output_dir / f"{split}.csv", index=False)

        if split == "test":
            test_filenames = [r["filename"] for r in records]
        ground_truth_rows.extend({**r, "split": split} for r in records)

    pd.DataFrame({"filename": test_filenames, "label": 3.0}).to_csv(
        output_dir / "sample_submission.csv", index=False
    )

    # Held-out truth for the test split, so the demo can report honest test metrics.
    pd.DataFrame(ground_truth_rows)[
        ["filename", "split", "level", "true_label", "reference_text"]
    ].to_csv(output_dir / "ground_truth.csv", index=False)

    LOGGER.info("Synthetic corpus written to %s", output_dir.resolve())
    return load_dataset(output_dir, is_synthetic=True)


def load_ground_truth(root: str | Path) -> pd.DataFrame | None:
    """Read the hidden labels of a synthetic corpus, if present."""
    path = Path(root) / "ground_truth.csv"
    return pd.read_csv(path) if path.exists() else None
