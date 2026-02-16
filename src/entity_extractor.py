"""LLM-based entity extraction from messages and documents.

Extracts person facts (birth dates, ages, cities, jobs, relationships, etc.)
from WhatsApp messages and Paperless documents using GPT-4o-mini, then stores
them in the Entity Store (entity_db).

Extraction is designed to be:
- **Cheap**: Uses GPT-4o-mini (~$0.15/1M tokens), not the main LLM
- **Selective**: Smart filtering skips low-value messages (short, emoji-only, etc.)
- **Async**: Runs in background threads, never blocks message processing
- **Incremental**: Higher-confidence facts overwrite lower-confidence ones

Usage:
    # From WhatsApp webhook (after RAG storage):
    maybe_extract_entities(sender="Shiran Waintrob", chat_name="Shiran Waintrob",
                           message="I'm turning 32 next week!", ...)
    
    # From Paperless sync:
    extract_entities_from_document(doc_title="פרטים אישיים", doc_text="...",
                                   source_ref="paperless:42")
"""

import json
import re
from typing import Any, Dict, List, Optional

from config import settings
from utils.logger import logger


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM_PROMPT = """You are a structured entity extraction system. Given a message or document, extract factual information about PEOPLE mentioned or implied.

RULES:
- Extract ONLY permanent, time-invariant facts — NOT temporary states or opinions
- IMPORTANT: Do NOT extract "age" — it changes over time. Instead extract "birth_date" which is permanent. The system will calculate age from birth_date + current date at query time.
- Focus on: birth dates, locations, jobs, phone numbers, email, ID numbers, family relationships, gender
- Do NOT extract: age, mood, recent_topic, temporary states, opinions, emotions
- If a person's name is mentioned with a fact, extract it
- If the sender is talking about themselves, the sender IS the entity
- Return valid JSON only — no markdown, no explanation
- If nothing extractable, return {"entities": []}
- For dates, use ISO format (YYYY-MM-DD) when possible
- Names should be in their original script (Hebrew/English as written)

RESPONSE FORMAT:
{
  "entities": [
    {
      "name": "Full Person Name",
      "facts": {
        "birth_date": "1994-03-15",
        "city": "Tel Aviv",
        "job_title": "Product Manager",
        "employer": "Wix",
        "id_number": "038041612",
        "gender": "female",
        "marital_status": "married",
        "email": "name@example.com"
      },
      "relationships": [
        {"related_to": "Other Person Name", "type": "spouse"}
      ]
    }
  ]
}

Only include facts that are EXPLICITLY stated or very clearly implied. Do NOT guess."""


def _build_extraction_user_prompt(
    sender: str,
    chat_name: str,
    message: str,
    timestamp: str = "",
    is_document: bool = False,
) -> str:
    """Build the user prompt for entity extraction."""
    source_type = "Document" if is_document else "WhatsApp message"
    return (
        f"Source: {source_type}\n"
        f"Chat/Document: {chat_name}\n"
        f"Sender/Author: {sender}\n"
        f"Timestamp: {timestamp}\n"
        f"---\n"
        f"{message}\n"
        f"---\n"
        f"Extract person facts from the above."
    )


# ---------------------------------------------------------------------------
# Smart filtering — skip low-value messages
# ---------------------------------------------------------------------------

# Minimum message length to consider for extraction
_MIN_LENGTH = int(settings.get("entity_extraction_min_message_length", "15"))

# Patterns that indicate extractable content
_FACT_PATTERNS = [
    re.compile(r'\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}'),  # Date patterns
    re.compile(r'\b\d{5,}\b'),  # Long numbers (IDs, phone numbers)
    re.compile(r'@\w+\.\w+'),  # Email-like patterns
    re.compile(r'בן\s*\d|בת\s*\d|גיל\s*\d', re.UNICODE),  # Hebrew age patterns
    re.compile(r'נולד|birthday|born|birth', re.UNICODE | re.IGNORECASE),
    re.compile(r'גר\s+ב|living in|lives in|from\s+\w+', re.UNICODE | re.IGNORECASE),
    re.compile(r'עובד|עובדת|works at|working at|job', re.UNICODE | re.IGNORECASE),
    re.compile(r'נשוי|נשואה|married|divorced|גרוש|single', re.UNICODE | re.IGNORECASE),
    re.compile(r'אבא|אמא|אח\b|אחות|בן\b|בת\b|ילד|father|mother|brother|sister|son|daughter|child', re.UNICODE | re.IGNORECASE),
]

# Patterns that indicate low-value content (skip)
_SKIP_PATTERNS = [
    re.compile(r'^[\U0001F600-\U0001F9FF\s]+$'),  # Pure emoji
    re.compile(r'^\[sticker\]$', re.IGNORECASE),
    re.compile(r'^\[Image:', re.IGNORECASE),
]


def _should_extract(message: str, is_document: bool = False) -> bool:
    """Determine if a message warrants entity extraction.

    Documents always get extracted (high fact density).
    Messages are filtered by length and content patterns.

    Args:
        message: The message text
        is_document: Whether this is a Paperless document

    Returns:
        True if extraction should proceed
    """
    if is_document:
        return True  # Documents are always worth extracting

    if not message or len(message) < _MIN_LENGTH:
        return False

    # Skip patterns
    for pattern in _SKIP_PATTERNS:
        if pattern.match(message):
            return False

    # Check for fact-indicating patterns (fast pre-filter before LLM call)
    for pattern in _FACT_PATTERNS:
        if pattern.search(message):
            return True

    # For longer messages (>100 chars), extract even without pattern match
    if len(message) > 100:
        return True

    return False


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------

def _call_extraction_llm(user_prompt: str) -> Optional[Dict[str, Any]]:
    """Call GPT-4o-mini for entity extraction.

    Args:
        user_prompt: The formatted extraction prompt

    Returns:
        Parsed JSON response, or None on failure
    """
    try:
        from openai import OpenAI

        model = settings.get("entity_extraction_model", "gpt-4o-mini")
        client = OpenAI(api_key=settings.openai_api_key)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Low temp for factual extraction
            max_tokens=1000,
            response_format={"type": "json_object"},
        )

        # Track cost
        try:
            from cost_meter import METER
            usage = response.usage
            if usage:
                METER.record_chat(
                    provider="openai",
                    model=getattr(response, "model", model) or model,
                    in_tokens=usage.prompt_tokens or 0,
                    out_tokens=usage.completion_tokens or 0,
                    request_context="entity_extraction",
                )
        except Exception:
            pass  # Non-fatal

        content = response.choices[0].message.content
        if content:
            return json.loads(content)
        return None

    except json.JSONDecodeError as e:
        logger.warning(f"Entity extraction returned invalid JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Entity extraction LLM call failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Entity storage
# ---------------------------------------------------------------------------

def _store_extracted_entities(
    extraction_result: Dict[str, Any],
    source_type: str = "extracted",
    source_ref: Optional[str] = None,
    sender_whatsapp_id: Optional[str] = None,
) -> int:
    """Store extracted entities in the Entity Store.

    Args:
        extraction_result: Parsed JSON from LLM with 'entities' list
        source_type: Where this was extracted from ("whatsapp", "paperless")
        source_ref: Reference string (e.g., "chat:972501234567@c.us:1708012345")
        sender_whatsapp_id: WhatsApp ID of the message sender (for auto-matching)

    Returns:
        Number of facts stored
    """
    import entity_db

    entities = extraction_result.get("entities", [])
    if not entities:
        return 0

    facts_stored = 0

    for entity in entities:
        name = entity.get("name")
        if not name or not isinstance(name, str):
            continue

        # Extract email from facts for identifier-based dedup
        facts = entity.get("facts", {})
        extracted_email = None
        if isinstance(facts, dict):
            extracted_email = facts.get("email")
            if extracted_email and isinstance(extracted_email, str):
                extracted_email = extracted_email.strip()
            else:
                extracted_email = None

        # Get or create person — uses phone→email→name cascade
        person_id = entity_db.get_or_create_person(
            canonical_name=name,
            whatsapp_id=sender_whatsapp_id if len(entities) == 1 else None,
            email=extracted_email,
        )

        # Store facts
        if isinstance(facts, dict):
            for key, value in facts.items():
                if value and isinstance(value, str) and value.strip():
                    entity_db.set_fact(
                        person_id=person_id,
                        key=key,
                        value=value.strip(),
                        confidence=0.6,  # LLM extraction confidence
                        source_type=source_type,
                        source_ref=source_ref,
                    )
                    facts_stored += 1

        # Store relationships
        relationships = entity.get("relationships", [])
        if isinstance(relationships, list):
            for rel in relationships:
                related_name = rel.get("related_to")
                rel_type = rel.get("type")
                if related_name and rel_type:
                    related_id = entity_db.get_or_create_person(
                        canonical_name=related_name
                    )
                    entity_db.add_relationship(
                        person_id=person_id,
                        related_person_id=related_id,
                        relationship_type=rel_type,
                        confidence=0.5,
                        source_ref=source_ref,
                    )

    return facts_stored


# ---------------------------------------------------------------------------
# Public API — called from plugin pipelines
# ---------------------------------------------------------------------------

def maybe_extract_entities(
    sender: str,
    chat_name: str,
    message: str,
    timestamp: str = "",
    chat_id: str = "",
    whatsapp_id: Optional[str] = None,
) -> int:
    """Extract entities from a WhatsApp message (if it passes filtering).

    This is the main entry point called from the WhatsApp webhook pipeline.
    It filters low-value messages, runs LLM extraction, and stores results.

    Args:
        sender: Message sender name
        chat_name: Chat/group name
        message: Message text
        timestamp: Unix timestamp as string
        chat_id: WhatsApp chat ID
        whatsapp_id: WhatsApp ID of the sender

    Returns:
        Number of facts stored (0 if skipped or failed)
    """
    if not settings.get("entity_extraction_enabled", "true").lower() == "true":
        return 0

    if not _should_extract(message):
        return 0

    # Skip group chats if configured
    if settings.get("entity_extraction_skip_groups", "true").lower() == "true":
        if chat_id and chat_id.endswith("@g.us"):
            return 0

    user_prompt = _build_extraction_user_prompt(
        sender=sender,
        chat_name=chat_name,
        message=message,
        timestamp=timestamp,
    )

    result = _call_extraction_llm(user_prompt)
    if not result:
        return 0

    source_ref = f"chat:{chat_id}:{timestamp}" if chat_id else None
    facts_stored = _store_extracted_entities(
        extraction_result=result,
        source_type="whatsapp",
        source_ref=source_ref,
        sender_whatsapp_id=whatsapp_id,
    )

    if facts_stored > 0:
        logger.info(
            f"Entity extraction: {facts_stored} facts from {sender} in {chat_name}"
        )

    return facts_stored


def extract_entities_from_document(
    doc_title: str,
    doc_text: str,
    source_ref: str = "",
    sender: str = "",
) -> int:
    """Extract entities from a Paperless document.

    Documents always get extracted (no filtering).
    For long documents, only the first 4000 chars are sent to the LLM
    to stay within token limits while capturing the most fact-dense parts
    (headers, preambles, personal details sections).

    Args:
        doc_title: Document title
        doc_text: Full document text
        source_ref: Reference string (e.g., "paperless:42")
        sender: Document author/owner if known

    Returns:
        Number of facts stored
    """
    if not settings.get("entity_extraction_enabled", "true").lower() == "true":
        return 0

    if not doc_text or len(doc_text.strip()) < 20:
        return 0

    # Truncate long documents — first 4000 chars contain most facts
    text_for_extraction = doc_text[:4000] if len(doc_text) > 4000 else doc_text

    user_prompt = _build_extraction_user_prompt(
        sender=sender or "Unknown",
        chat_name=doc_title,
        message=text_for_extraction,
        is_document=True,
    )

    result = _call_extraction_llm(user_prompt)
    if not result:
        return 0

    facts_stored = _store_extracted_entities(
        extraction_result=result,
        source_type="paperless",
        source_ref=source_ref,
    )

    if facts_stored > 0:
        logger.info(
            f"Entity extraction from document '{doc_title}': {facts_stored} facts"
        )

    return facts_stored
