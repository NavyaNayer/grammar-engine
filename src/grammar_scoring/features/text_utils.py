"""Tokenisation helpers and closed-class word inventories.

The syntactic features rely on closed-class (function) words rather than a POS tagger.
That is a deliberate trade-off: function words are a finite, stable inventory that needs
no model download, and the constructions the rubric cares about -- subordination,
relativisation, complex verb phrases -- are exactly the ones signalled by function words.
"""

from __future__ import annotations

import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?")

#: Markers of subordinate clauses - the core signal for "complex language structures".
SUBORDINATORS = frozenset(
    """
    although though because since unless whereas while whilst whenever wherever
    if whether after before until once so that provided assuming given lest
    even as when where why how what
    """.split()
)

#: Relative pronouns introduce relative clauses, a level-4/5 construction.
RELATIVIZERS = frozenset("who whom whose which that where when".split())

COORDINATORS = frozenset("and but or nor yet so for".split())

#: Auxiliaries and modals carry tense, aspect, voice and mood.
AUXILIARIES = frozenset(
    """
    be am is are was were been being have has had having do does did doing
    will would shall should can could may might must ought need dare
    """.split()
)

MODALS = frozenset("will would shall should can could may might must ought".split())

PREPOSITIONS = frozenset(
    """
    about above across after against along among around at before behind below beneath
    beside besides between beyond by despite down during except for from in inside into
    like near of off on onto out outside over past since through throughout to toward
    towards under underneath until up upon with within without
    """.split()
)

DETERMINERS = frozenset(
    """
    a an the this that these those my your his her its our their some any each every
    no another such either neither both all most many much few little several
    """.split()
)

PRONOUNS = frozenset(
    """
    i me my mine myself you your yours yourself yourselves he him his himself she her
    hers herself it its itself we us our ours ourselves they them their theirs themselves
    """.split()
)

#: Hesitation and filled-pause tokens. Whisper transcribes many of these verbatim.
FILLERS = frozenset(
    """
    um uh erm er ah eh hmm hm mm mhm uhm umm ahh ohh huh
    """.split()
)

#: Multi-word hedges and discourse fillers that signal planning difficulty when frequent.
FILLER_PHRASES = (
    "you know",
    "i mean",
    "kind of",
    "sort of",
    "like i said",
    "and stuff",
    "or something",
    "or whatever",
    "how to say",
    "what is it called",
    "how do you say",
)

#: Explicit self-repair markers. The rubric rewards speakers who correct themselves.
REPAIR_MARKERS = (
    "i mean",
    "sorry",
    "no wait",
    "rather",
    "actually no",
    "let me rephrase",
    "what i meant",
    "or rather",
    "excuse me",
)

#: A sentence ending in one of these is almost certainly an abandoned fragment.
DANGLING_ENDINGS = (
    COORDINATORS
    | DETERMINERS
    | PREPOSITIONS
    | frozenset("to is are was were am be been very really just also then than".split())
)

#: Words whose presence implies a finite verb somewhere in the clause.
_FINITE_VERB_HINTS = AUXILIARIES | frozenset(
    "said says say goes go went think thinks thought want wants wanted get gets got".split()
)

_VERBAL_SUFFIXES = ("ed", "es", "s", "ing")


def split_sentences(text: str) -> list[str]:
    """Split a transcript into sentences.

    Whisper restores punctuation, so punctuation is the primary cue. When it produces an
    unpunctuated wall of text (common for disfluent speech) we fall back to fixed-width
    20-word windows so that the per-sentence statistics stay meaningful instead of
    collapsing to a single enormous "sentence".
    """
    text = (text or "").strip()
    if not text:
        return []

    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if len(sentences) > 1:
        return sentences

    words = text.split()
    if len(words) <= 25:
        return [text]
    return [" ".join(words[i : i + 20]) for i in range(0, len(words), 20)]


def tokenize(text: str) -> list[str]:
    """Lower-cased alphabetic word tokens (apostrophes preserved inside contractions)."""
    return [m.group(0).lower() for m in _WORD.finditer(text or "")]


def count_phrases(text: str, phrases: tuple[str, ...]) -> int:
    """Count non-overlapping occurrences of each phrase in ``text``."""
    padded = f" {' '.join(tokenize(text))} "
    return sum(padded.count(f" {phrase} ") for phrase in phrases)


def looks_verbless(sentence: str) -> bool:
    """Heuristic test for a clause with no finite verb (an incomplete utterance).

    Treated as verbless when the sentence contains neither a known auxiliary/common verb
    nor any word with a verbal inflectional suffix. It over-triggers on nominal
    fragments, which is exactly the behaviour we want from an "incomplete sentence" cue.
    """
    tokens = tokenize(sentence)
    if len(tokens) < 3:
        return True
    if any(t in _FINITE_VERB_HINTS for t in tokens):
        return False
    return not any(
        len(t) > 3 and t.endswith(_VERBAL_SUFFIXES) and t not in DETERMINERS for t in tokens
    )


def safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide, returning ``default`` for a zero/invalid denominator."""
    if denominator is None or denominator <= 0:
        return default
    return float(numerator) / float(denominator)
