"""Shared text processing utilities for RAG ingestion.

Provides common text sanitization, chunking, and quality filtering
functions used by multiple plugin sync modules (Paperless, Gmail, etc.).

Centralises logic to avoid code duplication between sync modules.
"""

import logging
import re
import unicodedata
from html.parser import HTMLParser
from io import StringIO
from typing import List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (defaults — sync modules may override via settings)
# ---------------------------------------------------------------------------

# Maximum characters per chunk for embedding.
# text-embedding-3-large has an 8191 token limit.
# Hebrew text tokenizes at ~1.5 tokens/char (vs ~0.25 for English),
# so we must size for the worst-case multilingual scenario:
#   4500 chars × 1.55 tok/char ≈ 6975 tokens — safe margin under 8191.
# The ~100-char header added by sync modules brings this to ~7130 tokens.
MAX_CHUNK_CHARS = 4_500
CHUNK_OVERLAP_CHARS = 200

# Minimum useful content length after sanitization (characters).
MIN_CONTENT_CHARS = 50

# Minimum ratio of word-like characters in a chunk for it to be useful.
MIN_WORD_CHAR_RATIO = 0.40

# Unicode categories to strip from document content.
# Category "Cf" (Format) covers invisible formatting characters:
# RTL/LTR marks, zero-width joiners, directional overrides, BOM, etc.
_STRIP_UNICODE_CATEGORIES = {"Cf"}


# ---------------------------------------------------------------------------
# Unicode control character stripping
# ---------------------------------------------------------------------------


def strip_unicode_control(text: str) -> str:
    """Remove Unicode format characters (category Cf) from text.

    Uses :mod:`unicodedata` to identify characters by category rather
    than maintaining a manual list of code points.  Category ``Cf``
    (Format) covers all invisible formatting characters such as
    RTL/LTR marks, zero-width joiners, directional overrides, BOM,
    soft hyphens, etc.

    Args:
        text: Input string potentially containing control characters

    Returns:
        Cleaned string with format characters removed
    """
    return "".join(
        ch for ch in text
        if unicodedata.category(ch) not in _STRIP_UNICODE_CATEGORIES
    )


# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------


class _HTMLTextExtractor(HTMLParser):
    """Minimal HTML-to-text extractor.

    Strips all tags and returns concatenated text content.
    """

    def __init__(self):
        super().__init__()
        self._buf = StringIO()

    def handle_data(self, data: str) -> None:
        self._buf.write(data)

    def get_text(self) -> str:
        return self._buf.getvalue()


def strip_html(html: str) -> str:
    """Remove HTML tags and return plain text.

    Args:
        html: String potentially containing HTML markup

    Returns:
        Plain text with tags removed
    """
    extractor = _HTMLTextExtractor()
    try:
        extractor.feed(html)
        return extractor.get_text()
    except Exception:
        # If parsing fails, fall back to simple regex strip
        return re.sub(r"<[^>]+>", " ", html)


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------


def split_text(
    text: str,
    max_chars: int = MAX_CHUNK_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> List[str]:
    """Split text into chunks that fit within the embedding model's token limit.

    Tries to split on paragraph boundaries (double newline) for cleaner chunks.
    Falls back to sentence boundaries, then hard character splits with overlap.

    Args:
        text: Full document text
        max_chars: Maximum characters per chunk
        overlap: Character overlap between consecutive chunks

    Returns:
        List of text chunks (at least one element)
    """
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            chunks.append(text[start:])
            break

        # Try to break at a paragraph boundary
        boundary = text.rfind("\n\n", start, end)
        if boundary == -1 or boundary <= start:
            # Fall back to sentence boundary
            boundary = text.rfind(". ", start, end)
        if boundary == -1 or boundary <= start:
            # Hard split
            boundary = end
        else:
            boundary += 1  # Include the delimiter character

        chunks.append(text[start:boundary])
        start = max(boundary - overlap, boundary)  # overlap only when hard-splitting
        if boundary == end:
            start = boundary - overlap  # Apply overlap on hard splits

    return chunks


# ---------------------------------------------------------------------------
# Quality filtering
# ---------------------------------------------------------------------------


def is_quality_chunk(
    chunk: str,
    min_word_char_ratio: float = MIN_WORD_CHAR_RATIO,
    min_length: int = 20,
) -> bool:
    """Check whether a text chunk contains enough meaningful content.

    Rejects chunks that are predominantly non-word characters (base64
    residue, encoded data, random symbols) or too short to be useful.

    Args:
        chunk: A single text chunk
        min_word_char_ratio: Minimum ratio of word-like characters (0.0-1.0)
        min_length: Minimum stripped length to be considered useful

    Returns:
        True if the chunk passes quality checks
    """
    stripped = chunk.strip()
    if len(stripped) < min_length:
        return False

    # Count word-like characters (letters, digits, common punctuation, spaces)
    # Hebrew/Arabic/Cyrillic etc. are included via \w
    word_chars = len(re.findall(r"[\w\s.,;:!?'\"-]", stripped, re.UNICODE))
    ratio = word_chars / len(stripped) if stripped else 0

    if ratio < min_word_char_ratio:
        logger.debug(
            "Rejecting low-quality chunk (%.0f%% word chars, %d chars): %.60s...",
            ratio * 100,
            len(stripped),
            stripped,
        )
        return False

    return True
