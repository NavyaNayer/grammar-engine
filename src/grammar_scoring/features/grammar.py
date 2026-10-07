"""Direct measurements of grammatical quality from the transcript.

Three complementary views, because no single one is sufficient:

1. **Rule-based error detection (LanguageTool).** High precision on the exact error
   types the rubric enumerates -- agreement, articles, verb form, confused words. It
   gives interpretable counts, but it is blind to errors outside its rule set.
2. **Neural acceptability (CoLA).** A RoBERTa classifier fine-tuned on the Corpus of
   Linguistic Acceptability returns P(sentence is grammatical). It generalises to errors
   no rule covers, and its per-sentence spread separates "one bad sentence" from
   "uniformly shaky".
3. **Language-model surprisal (distilGPT-2).** Perplexity is sensitive to unusual word
   sequences in general, which catches awkward-but-parseable constructions that the
   other two miss. Its weakness is that rare *topics* also raise perplexity, so it is
   used alongside, never instead of, the first two.

Two engineering points matter for correctness here:

* ASR output carries **model-generated punctuation and casing**, not the speaker's.
  LanguageTool rule categories that judge punctuation, typography or capitalisation are
  therefore disabled -- counting them would score Whisper, not the candidate.
* The CoLA label index is **calibrated at load time** against probe sentences rather
  than hard-coded, so a checkpoint with flipped label order cannot silently invert the
  entire feature.

All features are prefixed ``gr_``.
"""

from __future__ import annotations

import logging
import math
import re

from grammar_scoring.config import FeatureConfig
from grammar_scoring.features.text_utils import safe_ratio, split_sentences, tokenize

LOGGER = logging.getLogger(__name__)

#: Rule categories that reflect the speaker's grammar rather than Whisper's formatting.
_SCORED_CATEGORIES = {
    "GRAMMAR",
    "TYPOS",
    "CONFUSED_WORDS",
    "COLLOCATIONS",
    "SEMANTICS",
    "MISC",
    "NONSTANDARD_PHRASES",
}

#: Categories driven by punctuation/casing, which the ASR model invents.
_IGNORED_CATEGORIES = {
    "PUNCTUATION",
    "TYPOGRAPHY",
    "CASING",
    "STYLE",
    "REDUNDANCY",
    "COLLOQUIALISMS",
    "PLAIN_ENGLISH",
    "WIKIPEDIA",
    "CREATIVE_WRITING",
}

#: Rule-id substrings grouped into the error types the rubric names. The patterns match
#: LanguageTool's real identifiers (``HE_VERB_AGR``, ``PRP_VBG``, ``MANY_NN`` ...), which
#: encode the part-of-speech pattern that fired rather than a human-readable name.
_ERROR_GROUPS = {
    "agreement": ("AGR", "MANY_NN", "SINGULAR", "PLURAL", "_VBZ", "_VBP", "THERE_RE", "NNS"),
    "verb_form": ("VBG", "VBD", "VBN", "TENSE", "PAST", "GERUND", "INFINITIV", "MD_", "DID_"),
    "determiner": ("DT_", "ARTICLE", "A_VS_AN", "MISSING_DET", "MANY_MUCH", "_DT"),
    "preposition": ("PREPOSITION", "PREP", "IN_ON", "AT_THE", "_TO_"),
    "word_order": ("WORD_ORDER", "ORDER_OF", "INVERSION"),
    "confusion": ("CONFUSION", "CONFUSED", "THEIR_THERE", "ITS_IT_S", "YOUR_YOURE", "_VS_"),
    "spelling": ("MORFOLOGIK", "SPELL", "HUNSPELL"),
}

#: LanguageTool's own ``ruleIssueType``, a coarser but far more reliable label than the
#: rule id. Only the types that reflect the speaker's language are kept.
_ISSUE_TYPES = ("grammar", "misspelling", "duplication", "uncategorized")

_PROBE_GRAMMATICAL = "The results of the experiment were published last year."
_PROBE_UNGRAMMATICAL = "The results of the experiment was publish last year ago."

_LANGUAGE_TOOL_KEYS = (
    "gr_lt_errors_per_100w",
    "gr_lt_error_count",
    "gr_lt_clean_sentence_ratio",
    "gr_lt_errors_per_sentence",
    *(f"gr_lt_{group}_per_100w" for group in _ERROR_GROUPS),
    *(f"gr_lt_issue_{issue}_per_100w" for issue in _ISSUE_TYPES),
    "gr_lt_available",
)

_COLA_KEYS = (
    "gr_cola_mean",
    "gr_cola_min",
    "gr_cola_std",
    "gr_cola_p25",
    "gr_cola_unacceptable_ratio",
    "gr_cola_weighted_mean",
    "gr_cola_available",
)

_PPL_KEYS = (
    "gr_logppl",
    "gr_logppl_sentence_mean",
    "gr_logppl_sentence_std",
    "gr_logppl_sentence_max",
    "gr_surprisal_spike_ratio",
    "gr_ppl_available",
)


def _zeros(keys: tuple[str, ...]) -> dict[str, float]:
    return dict.fromkeys(keys, 0.0)


# --------------------------------------------------------------------------- torch
def _configure_torch(config: FeatureConfig):
    import torch

    torch.set_num_threads(max(1, config.torch_num_threads))
    return torch


# ------------------------------------------------------------------ LanguageTool
class LanguageToolScorer:
    """Rule-based grammatical error counts.

    Requires a Java runtime; when Java or the LanguageTool download is unavailable the
    scorer degrades to zeros and flags itself as unavailable so the model can learn to
    ignore the block rather than crashing the run.
    """

    def __init__(self, language: str = "en-US", version: str | None = "6.6"):
        self.language = language
        self.version = version
        self._tool = None
        self._failed = False

    def _ensure_loaded(self) -> bool:
        if self._tool is not None:
            return True
        if self._failed:
            return False

        import language_tool_python

        # Caching the analysis pipeline matters: a run checks ~1000 documents, and
        # rebuilding the pipeline for each one dominates the cost.
        config = {"cacheSize": 2048, "pipelineCaching": True, "maxSpellingSuggestions": 1}

        # A pinned release is downloaded once and reused; "latest" resolves to a nightly
        # snapshot that is re-fetched (~260 MB) on every start.
        attempts = [self.version, None] if self.version else [None]
        last_error: Exception | None = None
        for version in attempts:
            try:
                LOGGER.info(
                    "Starting LanguageTool %s (%s); the first run downloads the engine.",
                    version or "latest",
                    self.language,
                )
                kwargs = {"config": config}
                if version:
                    kwargs["language_tool_download_version"] = version
                self._tool = language_tool_python.LanguageTool(self.language, **kwargs)
                return True
            except Exception as exc:  # noqa: BLE001 - any failure must degrade, not crash
                last_error = exc
                LOGGER.warning("Could not start LanguageTool %s: %s", version or "latest", exc)

        LOGGER.warning("LanguageTool unavailable (%s); rule-based features disabled.", last_error)
        self._failed = True
        return False

    @staticmethod
    def _classify(rule_id: str) -> list[str]:
        groups = [g for g, patterns in _ERROR_GROUPS.items() if any(p in rule_id for p in patterns)]
        return groups

    def score(self, text: str) -> dict[str, float]:
        features = _zeros(_LANGUAGE_TOOL_KEYS)
        if not text.strip() or not self._ensure_loaded():
            return features

        sentences = split_sentences(text)
        n_words = max(len(tokenize(text)), 1)

        try:
            matches = self._tool.check(text)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("LanguageTool check failed: %s", exc)
            return features

        scored = []
        for match in matches:
            category = (getattr(match, "category", "") or "").upper()
            if category in _IGNORED_CATEGORIES:
                continue
            if _SCORED_CATEGORIES and category and category not in _SCORED_CATEGORIES:
                continue
            scored.append(match)

        group_counts = dict.fromkeys(_ERROR_GROUPS, 0)
        issue_counts = dict.fromkeys(_ISSUE_TYPES, 0)
        for match in scored:
            for group in self._classify((getattr(match, "ruleId", "") or "").upper()):
                group_counts[group] += 1
            issue = (getattr(match, "ruleIssueType", "") or "").lower()
            if issue in issue_counts:
                issue_counts[issue] += 1

        # A sentence is "clean" when no scored match falls inside its character span.
        clean = len(sentences)
        if sentences and scored:
            offsets, cursor = [], 0
            for sentence in sentences:
                start = text.find(sentence, cursor)
                start = cursor if start < 0 else start
                offsets.append((start, start + len(sentence)))
                cursor = start + len(sentence)
            dirty = {
                i
                for i, (start, end) in enumerate(offsets)
                if any(start <= m.offset < end for m in scored)
            }
            clean = len(sentences) - len(dirty)

        features.update(
            {
                "gr_lt_error_count": float(len(scored)),
                "gr_lt_errors_per_100w": 100.0 * len(scored) / n_words,
                "gr_lt_errors_per_sentence": safe_ratio(len(scored), len(sentences)),
                "gr_lt_clean_sentence_ratio": safe_ratio(clean, len(sentences), 1.0),
                "gr_lt_available": 1.0,
            }
        )
        for group, count in group_counts.items():
            features[f"gr_lt_{group}_per_100w"] = 100.0 * count / n_words
        for issue, count in issue_counts.items():
            features[f"gr_lt_issue_{issue}_per_100w"] = 100.0 * count / n_words
        return features

    def close(self) -> None:
        if self._tool is not None:
            try:
                self._tool.close()
            except Exception:  # noqa: BLE001
                pass
            self._tool = None


# ------------------------------------------------------------------------- CoLA
class AcceptabilityScorer:
    """Per-sentence P(grammatically acceptable) from a CoLA-finetuned classifier."""

    def __init__(self, model_name: str, config: FeatureConfig):
        self.model_name = model_name
        self.config = config
        self._model = None
        self._tokenizer = None
        self._acceptable_index = 1
        self._failed = False

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if self._failed:
            return False
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            _configure_torch(self.config)
            LOGGER.info("Loading acceptability model %s ...", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.eval()
            self._calibrate_label_index()
            return True
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Acceptability model unavailable (%s); CoLA features disabled.", exc)
            self._failed = True
            return False

    def _calibrate_label_index(self) -> None:
        """Decide which logit index means "acceptable" by probing known sentences.

        Checkpoints disagree on label order, and a silently flipped index would invert
        one of the strongest features in the model. Probing costs one forward pass.
        """
        probs = self._forward([_PROBE_GRAMMATICAL, _PROBE_UNGRAMMATICAL])
        if probs is None:
            return
        # The index whose probability drops most from the good to the bad probe is the
        # one that encodes acceptability.
        deltas = probs[0] - probs[1]
        self._acceptable_index = int(deltas.argmax())
        LOGGER.info(
            "CoLA acceptable-class index calibrated to %d (probe delta %.3f).",
            self._acceptable_index,
            float(deltas.max()),
        )

    def _forward(self, sentences: list[str]):
        import torch

        if not sentences:
            return None
        batch = self._tokenizer(
            sentences, return_tensors="pt", padding=True, truncation=True, max_length=128
        )
        with torch.no_grad():
            logits = self._model(**batch).logits
        return torch.softmax(logits, dim=-1).numpy()

    def score(self, text: str) -> dict[str, float]:
        import numpy as np

        features = _zeros(_COLA_KEYS)
        sentences = [s for s in split_sentences(text) if len(tokenize(s)) >= 3]
        if not sentences or not self._ensure_loaded():
            return features

        try:
            probs = self._forward(sentences)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Acceptability scoring failed: %s", exc)
            return features
        if probs is None:
            return features

        acceptability = probs[:, self._acceptable_index]
        lengths = np.array([len(tokenize(s)) for s in sentences], dtype=float)

        features.update(
            {
                "gr_cola_mean": float(acceptability.mean()),
                "gr_cola_min": float(acceptability.min()),
                "gr_cola_std": float(acceptability.std()),
                "gr_cola_p25": float(np.percentile(acceptability, 25)),
                # Longer sentences carry more grammatical commitment, so weight by length.
                "gr_cola_weighted_mean": float(np.average(acceptability, weights=lengths)),
                "gr_cola_unacceptable_ratio": float((acceptability < 0.5).mean()),
                "gr_cola_available": 1.0,
            }
        )
        return features


# ------------------------------------------------------------------- perplexity
class PerplexityScorer:
    """Token-level surprisal statistics from a small causal language model."""

    def __init__(self, model_name: str, config: FeatureConfig):
        self.model_name = model_name
        self.config = config
        self._model = None
        self._tokenizer = None
        self._failed = False

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if self._failed:
            return False
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            _configure_torch(self.config)
            LOGGER.info("Loading perplexity model %s ...", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self._model.eval()
            return True
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Perplexity model unavailable (%s); features disabled.", exc)
            self._failed = True
            return False

    def _token_nll(self, text: str):
        """Negative log-likelihood of every token in ``text`` (nats)."""
        import torch

        ids = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.config.perplexity_max_tokens
        ).input_ids
        if ids.shape[1] < 2:
            return None

        with torch.no_grad():
            logits = self._model(ids).logits
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        targets = ids[:, 1:]
        return -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).squeeze(0)

    def score(self, text: str) -> dict[str, float]:
        import numpy as np

        features = _zeros(_PPL_KEYS)
        if len(tokenize(text)) < 5 or not self._ensure_loaded():
            return features

        try:
            nll = self._token_nll(text)
            if nll is None:
                return features
            nll_np = nll.numpy()

            sentence_logppl = []
            for sentence in split_sentences(text):
                if len(tokenize(sentence)) < 4:
                    continue
                sentence_nll = self._token_nll(sentence)
                if sentence_nll is not None:
                    sentence_logppl.append(float(sentence_nll.mean()))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Perplexity scoring failed: %s", exc)
            return features

        features["gr_logppl"] = float(nll_np.mean())
        # Tokens above ~7 nats (p < 1e-3) are places the language model found the
        # sequence genuinely surprising - a better localiser of errors than the mean.
        features["gr_surprisal_spike_ratio"] = float((nll_np > 7.0).mean())
        features["gr_ppl_available"] = 1.0

        if sentence_logppl:
            values = np.asarray(sentence_logppl)
            features["gr_logppl_sentence_mean"] = float(values.mean())
            features["gr_logppl_sentence_std"] = float(values.std())
            features["gr_logppl_sentence_max"] = float(values.max())
        else:
            for key in (
                "gr_logppl_sentence_mean",
                "gr_logppl_sentence_std",
                "gr_logppl_sentence_max",
            ):
                features[key] = features["gr_logppl"]
        return features


# ------------------------------------------------------------------- orchestration
class GrammarFeatureExtractor:
    """Runs whichever grammar back-ends are enabled and available."""

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
        self._language_tool = (
            LanguageToolScorer(self.config.language_tool_lang, self.config.language_tool_version)
            if self.config.use_language_tool
            else None
        )
        self._cola = (
            AcceptabilityScorer(self.config.cola_model_name, self.config)
            if self.config.use_cola
            else None
        )
        self._ppl = (
            PerplexityScorer(self.config.perplexity_model_name, self.config)
            if self.config.use_perplexity
            else None
        )

    @property
    def feature_names(self) -> list[str]:
        names: list[str] = []
        if self._language_tool is not None:
            names += list(_LANGUAGE_TOOL_KEYS)
        if self._cola is not None:
            names += list(_COLA_KEYS)
        if self._ppl is not None:
            names += list(_PPL_KEYS)
        return names + ["gr_empty_transcript"]

    def extract(self, text: str) -> dict[str, float]:
        text = (text or "").strip()
        features: dict[str, float] = {"gr_empty_transcript": float(not text)}
        if self._language_tool is not None:
            features.update(self._language_tool.score(text))
        if self._cola is not None:
            features.update(self._cola.score(text))
        if self._ppl is not None:
            features.update(self._ppl.score(text))
        return features

    def close(self) -> None:
        if self._language_tool is not None:
            self._language_tool.close()


def normalized_error_density(text: str, error_count: float) -> float:
    """Errors per 100 words, guarding against empty transcripts."""
    return 100.0 * safe_ratio(error_count, len(tokenize(text)))


def estimate_reading_ease(text: str) -> float:
    """Flesch reading ease over the transcript (syllables approximated by vowel runs)."""
    words = tokenize(text)
    sentences = split_sentences(text)
    if not words or not sentences:
        return 0.0
    syllables = sum(max(1, len(re.findall(r"[aeiouy]+", word))) for word in words)
    return (
        206.835
        - 1.015 * (len(words) / len(sentences))
        - 84.6 * (syllables / len(words))
    )


def log_ppl_to_ppl(log_ppl: float) -> float:
    """Convert mean NLL (nats/token) back to perplexity."""
    return float(math.exp(min(log_ppl, 20.0)))
