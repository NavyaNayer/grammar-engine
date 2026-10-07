"""Lexical richness and syntactic complexity features.

The rubric separates *accuracy* ("seldom making noticeable mistakes") from *range*
("handle complex language structures well"). Error counts cover accuracy; this module
covers range. A speaker who only ever produces short main clauses can be error-free and
still sit at level 3, so measuring subordination and lexical variety is what
distinguishes the top of the scale.

Prefixes: ``lx_`` for lexical diversity, ``sy_`` for syntactic complexity.
"""

from __future__ import annotations

from collections import Counter

from grammar_scoring.features.text_utils import (
    AUXILIARIES,
    COORDINATORS,
    DETERMINERS,
    MODALS,
    PREPOSITIONS,
    PRONOUNS,
    RELATIVIZERS,
    SUBORDINATORS,
    safe_ratio,
    split_sentences,
    tokenize,
)

#: All closed-class inventories combined, used for the function/content word ratio.
_FUNCTION_WORDS = (
    SUBORDINATORS | COORDINATORS | AUXILIARIES | PREPOSITIONS | DETERMINERS | PRONOUNS
)


def _mattr(tokens: list[str], window: int) -> float:
    """Moving-average type-token ratio.

    Plain TTR falls as texts get longer, so on clips of unequal length it measures
    duration as much as vocabulary. MATTR averages TTR over fixed-size windows and is
    therefore comparable across a 45 s and a 60 s sample.
    """
    if not tokens:
        return 0.0
    if len(tokens) <= window:
        return safe_ratio(len(set(tokens)), len(tokens))

    counts = Counter(tokens[:window])
    total = len(set(counts))
    n_windows = 1
    for i in range(window, len(tokens)):
        outgoing = tokens[i - window]
        counts[outgoing] -= 1
        if counts[outgoing] == 0:
            del counts[outgoing]
        counts[tokens[i]] += 1
        total += len(counts)
        n_windows += 1
    return safe_ratio(total / n_windows, window)


def extract_lexical_features(text: str, mattr_window: int = 50) -> dict[str, float]:
    """Lexical-diversity and syntactic-complexity block for one transcript."""
    tokens = tokenize(text)
    sentences = split_sentences(text)
    n_words = len(tokens)
    n_sentences = len(sentences)

    if n_words == 0:
        keys = [
            "lx_ttr",
            "lx_root_ttr",
            "lx_mattr",
            "lx_hapax_ratio",
            "lx_mean_word_len",
            "lx_long_word_ratio",
            "lx_content_word_ratio",
            "lx_contraction_per_100w",
            "sy_subordinator_per_100w",
            "sy_relativizer_per_100w",
            "sy_coordinator_per_100w",
            "sy_modal_per_100w",
            "sy_auxiliary_per_100w",
            "sy_preposition_per_100w",
            "sy_determiner_per_100w",
            "sy_pronoun_per_100w",
            "sy_clauses_per_sentence",
            "sy_mean_clause_len",
            "sy_complex_sentence_ratio",
            "sy_initial_coordinator_ratio",
        ]
        return dict.fromkeys(keys, 0.0)

    counts = Counter(tokens)
    n_types = len(counts)
    per_100 = lambda n: 100.0 * safe_ratio(n, n_words)  # noqa: E731

    n_subordinators = sum(counts[w] for w in SUBORDINATORS)
    n_relativizers = sum(counts[w] for w in RELATIVIZERS)
    n_coordinators = sum(counts[w] for w in COORDINATORS)

    # Clause count approximated as one main clause per sentence plus one per
    # subordinating/relativising marker.
    n_clauses = n_sentences + n_subordinators + n_relativizers

    complex_sentences = 0
    initial_coordinators = 0
    for sentence in sentences:
        sentence_tokens = tokenize(sentence)
        if not sentence_tokens:
            continue
        if any(t in SUBORDINATORS or t in RELATIVIZERS for t in sentence_tokens[1:]):
            complex_sentences += 1
        if sentence_tokens[0] in COORDINATORS:
            initial_coordinators += 1

    return {
        # --- lexical diversity --------------------------------------------------
        "lx_ttr": safe_ratio(n_types, n_words),
        "lx_root_ttr": safe_ratio(n_types, n_words**0.5),
        "lx_mattr": _mattr(tokens, mattr_window),
        "lx_hapax_ratio": safe_ratio(sum(1 for c in counts.values() if c == 1), n_types),
        "lx_mean_word_len": sum(len(t) for t in tokens) / n_words,
        "lx_long_word_ratio": safe_ratio(sum(1 for t in tokens if len(t) >= 8), n_words),
        "lx_content_word_ratio": safe_ratio(
            sum(1 for t in tokens if t not in _FUNCTION_WORDS), n_words
        ),
        "lx_contraction_per_100w": per_100(sum(1 for t in tokens if "'" in t)),
        # --- syntactic complexity -----------------------------------------------
        "sy_subordinator_per_100w": per_100(n_subordinators),
        "sy_relativizer_per_100w": per_100(n_relativizers),
        "sy_coordinator_per_100w": per_100(n_coordinators),
        "sy_modal_per_100w": per_100(sum(counts[w] for w in MODALS)),
        "sy_auxiliary_per_100w": per_100(sum(counts[w] for w in AUXILIARIES)),
        "sy_preposition_per_100w": per_100(sum(counts[w] for w in PREPOSITIONS)),
        "sy_determiner_per_100w": per_100(sum(counts[w] for w in DETERMINERS)),
        "sy_pronoun_per_100w": per_100(sum(counts[w] for w in PRONOUNS)),
        "sy_clauses_per_sentence": safe_ratio(n_clauses, n_sentences),
        "sy_mean_clause_len": safe_ratio(n_words, n_clauses),
        "sy_complex_sentence_ratio": safe_ratio(complex_sentences, n_sentences),
        "sy_initial_coordinator_ratio": safe_ratio(initial_coordinators, n_sentences),
    }
