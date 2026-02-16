"""Lightweight sparse vector generator for hybrid retrieval.

Generates BM25-style sparse vectors from text without requiring external
ML models (no SPLADE/ColBERT dependency).  Tokens are mapped to integer
indices via CRC32 hashing, and weights are computed using BM25 term
frequency saturation.

These sparse vectors are stored alongside dense OpenAI embeddings in
Qdrant's named vector architecture, enabling server-side RRF fusion
of dense (semantic) + sparse (lexical) retrieval.

Usage:
    from utils.sparse_vectors import compute_sparse_vector

    sv = compute_sparse_vector("הסכם גירושין בין דוד ומירי")
    # sv = SparseVector(indices=[...], values=[...])
"""

import re
import unicodedata
from binascii import crc32
from collections import Counter
from math import log
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Tokenizer (aligned with LlamaIndexRAG._tokenize_query)
# ---------------------------------------------------------------------------

_HE_RE = re.compile(r'[\u0590-\u05FF]')

# Unicode categories to strip (invisible formatting characters)
_STRIP_CATEGORIES = {"Cf"}


def tokenize(text: str) -> List[str]:
    """Tokenize text for sparse vector generation.

    Aligns with LlamaIndexRAG._tokenize_query() for consistency between
    ingestion-time and query-time tokenization.  Keeps Hebrew 2-char
    tokens (בן, בת, אב) and Latin 3+ char tokens.

    Args:
        text: Input text (any language)

    Returns:
        List of lowercase tokens (may contain duplicates for TF counting)
    """
    # Strip Unicode format characters (RTL/LTR marks, zero-width joiners)
    clean = "".join(
        ch for ch in text if unicodedata.category(ch) not in _STRIP_CATEGORIES
    )

    # Extract tokens: ≥2 chars, keep 2-char only if Hebrew
    raw_tokens = re.findall(r"[\w]{2,}", clean, re.UNICODE)
    tokens = [
        t.lower()
        for t in raw_tokens
        if len(t) >= 3 or _HE_RE.search(t)
    ]
    return tokens


# ---------------------------------------------------------------------------
# BM25-style sparse vector computation
# ---------------------------------------------------------------------------

# BM25 parameters (standard defaults)
_BM25_K1 = 1.2   # Term frequency saturation parameter
_BM25_B = 0.75   # Length normalization parameter
_AVG_DOC_LENGTH = 100.0  # Assumed average document length in tokens


def _token_to_index(token: str) -> int:
    """Map a token string to a uint32 sparse vector index via CRC32.

    CRC32 provides a fast, deterministic hash with good distribution
    across the uint32 space.  Collisions are rare enough for
    retrieval purposes (not classification).

    Args:
        token: Lowercased token string

    Returns:
        Unsigned 32-bit integer index
    """
    return crc32(token.encode("utf-8")) & 0xFFFFFFFF


def compute_sparse_vector(
    text: str,
    boost_tokens: Optional[Dict[str, float]] = None,
) -> Tuple[List[int], List[float]]:
    """Compute a BM25-style sparse vector from text.

    Returns (indices, values) suitable for Qdrant's SparseVector.
    Each unique token gets a weight based on BM25 term frequency
    saturation: ``(tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl/avgdl))``

    This gives diminishing returns for repeated tokens (saturation)
    and normalizes for document length, producing better ranking than
    raw term frequency.

    Args:
        text: Input text to vectorize
        boost_tokens: Optional dict of token → boost multiplier
            (e.g., {"sender_name": 2.0} to boost sender matches)

    Returns:
        Tuple of (indices, values) for SparseVector construction
    """
    tokens = tokenize(text)
    if not tokens:
        return ([], [])

    # Count term frequencies
    tf_counts = Counter(tokens)
    doc_length = len(tokens)

    indices: List[int] = []
    values: List[float] = []

    for token, tf in tf_counts.items():
        # BM25 term frequency with saturation and length normalization
        numerator = tf * (_BM25_K1 + 1)
        denominator = tf + _BM25_K1 * (
            1 - _BM25_B + _BM25_B * (doc_length / _AVG_DOC_LENGTH)
        )
        weight = numerator / denominator

        # Apply optional boost
        if boost_tokens and token in boost_tokens:
            weight *= boost_tokens[token]

        idx = _token_to_index(token)
        indices.append(idx)
        values.append(weight)

    return (indices, values)


def compute_query_sparse_vector(
    query: str,
) -> Tuple[List[int], List[float]]:
    """Compute a sparse vector for a search query.

    Query vectors use simpler weighting (just presence, weight=1.0)
    since queries are short and term frequency is less meaningful.
    This matches how BM25 traditionally treats queries.

    Args:
        query: Search query text

    Returns:
        Tuple of (indices, values) for SparseVector construction
    """
    tokens = tokenize(query)
    if not tokens:
        return ([], [])

    # Deduplicate — each query term gets weight 1.0
    seen: Dict[int, str] = {}
    indices: List[int] = []
    values: List[float] = []

    for token in tokens:
        idx = _token_to_index(token)
        if idx not in seen:
            seen[idx] = token
            indices.append(idx)
            values.append(1.0)

    return (indices, values)
