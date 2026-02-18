"""Rule-based query intent classifier for gated graph expansion.

Classifies user queries to determine which graph expansion strategies
should be applied during retrieval.  Uses lightweight keyword/pattern
matching (no LLM call) to keep latency near-zero.

Intent types control expansion behavior in ArchiveRetriever._retrieve():
- PERSON_FACTS     → inject entity facts only (no extra search)
- PERSON_HISTORY   → person-scoped search on resolved IDs
- FAMILY_CONTEXT   → expand identity graph (family edges) before search
- ASSET_THREAD     → expand thread_id neighborhood
- ASSET_ATTACHMENT  → expand parent_asset_id / attachment_of edges
- CROSS_CHANNEL    → full asset neighborhood expansion
- GENERAL          → no special expansion (basic retrieval)
"""

import re
from enum import Enum
from typing import List, Set


class QueryIntent(Enum):
    """Query intent types that control graph expansion strategies."""

    PERSON_FACTS = "person_facts"
    PERSON_HISTORY = "person_history"
    FAMILY_CONTEXT = "family_context"
    ASSET_THREAD = "asset_thread"
    ASSET_ATTACHMENT = "asset_attachment"
    CROSS_CHANNEL = "cross_channel"
    GENERAL = "general"


# ---------------------------------------------------------------------------
# Pattern sets for intent detection
# ---------------------------------------------------------------------------

# Hebrew + English patterns for family/relationship queries
_FAMILY_PATTERNS = [
    re.compile(r"(?:family|families|spouse|wife|husband|child|children|son|daughter|parent|mother|father|brother|sister|kid|kids)", re.IGNORECASE),
    re.compile(r"(?:משפחה|בן זוג|אישה|בעל|ילד|ילדים|בן|בת|הורה|אמא|אבא|אח|אחות)", re.UNICODE),
    re.compile(r"(?:'s\s+family|של\s+(?:ה)?משפחה)", re.IGNORECASE | re.UNICODE),
]

# Patterns indicating factual questions about a person
_FACT_PATTERNS = [
    re.compile(r"(?:how old|age|birthday|birth date|born|where.*live|city|job|work|employer|id number|phone|email)", re.IGNORECASE),
    re.compile(r"(?:בן כמה|בת כמה|גיל|יום הולדת|תאריך לידה|נולד|גר ב|עיר|עבודה|מספר תעודת|טלפון|מייל)", re.UNICODE),
]

# Patterns indicating thread/conversation context requests
_THREAD_PATTERNS = [
    re.compile(r"(?:thread|conversation|context|surrounding|before and after|full (?:chat|discussion|exchange))", re.IGNORECASE),
    re.compile(r"(?:שרשור|שיחה|הקשר|מסביב|לפני ואחרי|כל ה(?:שיחה|דיון))", re.UNICODE),
]

# Patterns indicating attachment/document requests
_ATTACHMENT_PATTERNS = [
    re.compile(r"(?:attachment|attached|file|document|pdf|contract|invoice|receipt)", re.IGNORECASE),
    re.compile(r"(?:קובץ|מצורף|מסמך|חוזה|חשבונית|קבלה)", re.UNICODE),
]

# Patterns indicating cross-channel queries
_CROSS_CHANNEL_PATTERNS = [
    re.compile(r"(?:also.*(?:call|email|whatsapp|message)|(?:call|email|whatsapp|message).*too|across|both.*and)", re.IGNORECASE),
    re.compile(r"(?:גם.*(?:שיחה|מייל|הודעה)|(?:שיחה|מייל|הודעה).*גם|בכל ה)", re.UNICODE),
]

# Patterns indicating person-directed queries
_PERSON_QUERY_PATTERNS = [
    re.compile(r"(?:what did \w+ (?:say|tell|ask|write|send|mention))", re.IGNORECASE),
    re.compile(r"(?:מה \w+ (?:אמר|שאל|כתב|שלח|ציין|סיפר))", re.UNICODE),
    re.compile(r"(?:tell me about|show me.*from|everything about|summarize.*about)", re.IGNORECASE),
    re.compile(r"(?:ספר לי על|תראה לי.*מ|הכל על|סכם.*על)", re.UNICODE),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_query_intent(
    query: str,
    has_resolved_persons: bool = False,
    has_resolved_assets: bool = False,
) -> List[QueryIntent]:
    """Classify a query into one or more intents.

    Uses rule-based pattern matching for near-zero latency.
    Multiple intents can be returned (e.g., FAMILY_CONTEXT + PERSON_HISTORY).

    Args:
        query: The user's query string
        has_resolved_persons: Whether entity linking found person IDs
        has_resolved_assets: Whether asset linking found asset IDs

    Returns:
        List of QueryIntent values (never empty — at least GENERAL)
    """
    intents: Set[QueryIntent] = set()

    # Check each pattern set
    if _matches_any(query, _FAMILY_PATTERNS):
        intents.add(QueryIntent.FAMILY_CONTEXT)

    if _matches_any(query, _FACT_PATTERNS):
        intents.add(QueryIntent.PERSON_FACTS)

    if _matches_any(query, _THREAD_PATTERNS):
        intents.add(QueryIntent.ASSET_THREAD)

    if _matches_any(query, _ATTACHMENT_PATTERNS):
        intents.add(QueryIntent.ASSET_ATTACHMENT)

    if _matches_any(query, _CROSS_CHANNEL_PATTERNS):
        intents.add(QueryIntent.CROSS_CHANNEL)

    # Person-directed queries (when we have resolved persons)
    if has_resolved_persons and _matches_any(query, _PERSON_QUERY_PATTERNS):
        intents.add(QueryIntent.PERSON_HISTORY)

    # If we resolved persons but no specific intent detected, default to PERSON_HISTORY
    if has_resolved_persons and not intents:
        intents.add(QueryIntent.PERSON_HISTORY)

    # Default: GENERAL if nothing matched
    if not intents:
        intents.add(QueryIntent.GENERAL)

    return list(intents)


def should_expand_relationships(intents: List[QueryIntent]) -> bool:
    """Check if identity relationship expansion should run.

    Only expands the identity graph (spouse, parent, child edges)
    when the query intent explicitly calls for family/team context.

    Args:
        intents: List of classified intents

    Returns:
        True if relationship expansion should be performed
    """
    return QueryIntent.FAMILY_CONTEXT in intents


def should_expand_asset_neighborhood(intents: List[QueryIntent]) -> bool:
    """Check if asset neighborhood expansion should run.

    Enables thread, attachment, and cross-channel expansion when
    the query is about threads, attachments, or cross-channel content.

    Args:
        intents: List of classified intents

    Returns:
        True if asset neighborhood expansion should be performed
    """
    return bool(
        {QueryIntent.ASSET_THREAD, QueryIntent.ASSET_ATTACHMENT, QueryIntent.CROSS_CHANNEL}
        & set(intents)
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _matches_any(text: str, patterns: List[re.Pattern]) -> bool:
    """Check if text matches any of the given patterns."""
    for pattern in patterns:
        if pattern.search(text):
            return True
    return False
