"""Feature extraction for the grammar scoring engine.

Six complementary blocks, each with its own column prefix:

==========  =======================================================================
``gr_``     Direct grammatical evidence: rule-based error counts, neural
            acceptability, language-model surprisal.
``fl_``     Fluency, disfluency, self-repair and ASR-confidence features.
``sy_``     Syntactic complexity (subordination, clause density, verb phrases).
``lx_``     Lexical diversity and sophistication.
``ac_``     Acoustic and prosodic features (timing, pauses, spectrum, pitch).
``wv_``     Frozen wav2vec2 speech embeddings, PCA-reduced (off by default).
``wl_``     Frozen WavLM speech embeddings, PCA-reduced (off by default; the
            strongest single accuracy lever found -- see the notebook).
==========  =======================================================================
"""

from grammar_scoring.features.acoustic import extract_acoustic_features
from grammar_scoring.features.assemble import (
    FeatureBundle,
    FeaturePipeline,
    align_features,
    order_columns,
)
from grammar_scoring.features.embeddings import (
    Wav2Vec2EmbeddingExtractor,
    reduce_wav2vec2_embeddings,
)
from grammar_scoring.features.fluency import extract_fluency_features
from grammar_scoring.features.grammar import GrammarFeatureExtractor
from grammar_scoring.features.lexical import extract_lexical_features
from grammar_scoring.features.wavlm import WavLMEmbeddingExtractor, reduce_wavlm_embeddings

__all__ = [
    "FeatureBundle",
    "FeaturePipeline",
    "GrammarFeatureExtractor",
    "Wav2Vec2EmbeddingExtractor",
    "WavLMEmbeddingExtractor",
    "align_features",
    "extract_acoustic_features",
    "extract_fluency_features",
    "extract_lexical_features",
    "order_columns",
    "reduce_wav2vec2_embeddings",
    "reduce_wavlm_embeddings",
]
