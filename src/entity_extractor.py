"""LLM-based identity extraction from messages and documents.

Extracts person facts (birth dates, ages, cities, jobs, relationships, etc.)
from WhatsApp messages and Paperless documents using GPT-4o-mini, then stores
them in the Identity Store (entity_db).

Extraction is designed to be:
- **Cheap**: Uses GPT-4o-mini (~$0.15/1M tokens), not the main LLM
- **Selective**: Smart filtering skips low-value messages (short, emoji-only, etc.)
- **Async**: Runs in background threads, never blocks message processing
- **Incremental**: Higher-confidence facts overwrite lower-confidence ones

Usage:
    # From WhatsApp webhook (after RAG storage):
    maybe_extract_identities(sender="Shiran Waintrob", chat_name="Shiran Waintrob",
                           message="I'm turning 32 next week!", ...)
    
    # From Paperless sync:
    extract_identities_from_document(doc_title="פרטים אישיים", doc_text="...",
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

_EXTRACTION_SYSTEM_PROMPT = """You are a structured identity extraction system. Given a message or document, extract factual information about PEOPLE mentioned or implied.

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
- For each fact, include a "quote" field with the exact short snippet from the source text that supports this fact

RESPONSE FORMAT:
{
  "entities": [
    {
      "name": "Full Person Name",
      "facts": {
        "birth_date": {"value": "1994-03-15", "quote": "I'm turning 30 on March 15th"},
        "city": {"value": "Tel Aviv", "quote": "I live in Tel Aviv"},
        "job_title": {"value": "Product Manager", "quote": "started my new PM role"},
        "employer": {"value": "Wix", "quote": "working at Wix now"},
        "id_number": {"value": "038041612", "quote": "my ID is 038041612"},
        "gender": {"value": "female", "quote": "as a woman in tech"},
        "marital_status": {"value": "married", "quote": "my husband and I"},
        "email": {"value": "name@example.com", "quote": "reach me at name@example.com"}
      },
      "relationships": [
        {"related_to": "Other Person Name", "type": "spouse"}
      ]
    }
  ]
}

NOTE: Each fact value can be either a simple string OR an object with "value" and "quote" fields.
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
    """Call GPT-4o-mini for identity extraction.

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
        logger.warning(f"Identity extraction returned invalid JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Identity extraction LLM call failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Identity storage
# ---------------------------------------------------------------------------

def _store_extracted_identities(
    extraction_result: Dict[str, Any],
    source_type: str = "extracted",
    source_ref: Optional[str] = None,
    sender_whatsapp_id: Optional[str] = None,
    confidence: float = 0.6,
) -> int:
    """Store extracted identities in the Identity Store.

    Also creates person-asset links for mentioned persons so that the
    person-asset graph tracks which messages/documents mention which people.

    Args:
        extraction_result: Parsed JSON from LLM with 'entities' list
        source_type: Where this was extracted from ("whatsapp", "paperless", "user_correction")
        source_ref: Reference string (e.g., "chat:972501234567@c.us:1708012345")
        sender_whatsapp_id: WhatsApp ID of the message sender (for auto-matching)
        confidence: Confidence score for stored facts (default 0.6 for auto, 0.8 for user corrections)

    Returns:
        Number of facts stored
    """
    import entity_db
    from identity import Identity

    entities = extraction_result.get("entities", [])
    if not entities:
        return 0

    facts_stored = 0

    # Derive the Qdrant asset_ref from source_ref.
    # source_ref format: "chat:{chat_id}:{timestamp}" → asset_ref: "{chat_id}:{timestamp}"
    # For paperless: "paperless:{doc_id}" → kept as-is
    asset_ref = None
    asset_type = "whatsapp_msg"
    if source_ref:
        if source_ref.startswith("chat:"):
            # "chat:972501234567@c.us:1708012345" → "972501234567@c.us:1708012345"
            asset_ref = source_ref[len("chat:"):]
            asset_type = "whatsapp_msg"
        elif source_ref.startswith("paperless:"):
            asset_ref = source_ref
            asset_type = "document"
        else:
            asset_ref = source_ref

    for entity in entities:
        name = entity.get("name")
        if not name or not isinstance(name, str):
            continue

        # Extract email from facts for identifier-based dedup
        facts = entity.get("facts", {})
        extracted_email = None
        if isinstance(facts, dict):
            email_val = facts.get("email")
            # Handle both string and {value, quote} formats
            if isinstance(email_val, dict):
                extracted_email = email_val.get("value", "")
            elif isinstance(email_val, str):
                extracted_email = email_val
            if extracted_email:
                extracted_email = extracted_email.strip() or None
            else:
                extracted_email = None

        # Get or create person — uses phone→email→name cascade
        person = Identity.find_or_create(
            name,
            whatsapp_id=sender_whatsapp_id if len(entities) == 1 else None,
            email=extracted_email,
        )

        # Store facts — handle both string and {value, quote} formats
        if isinstance(facts, dict):
            for key, raw_val in facts.items():
                # Parse value and optional quote
                fact_value: str = ""
                fact_quote: str = ""
                if isinstance(raw_val, dict):
                    fact_value = str(raw_val.get("value", "")).strip()
                    fact_quote = str(raw_val.get("quote", "")).strip()
                elif isinstance(raw_val, str):
                    fact_value = raw_val.strip()

                if fact_value:
                    person.set_fact(
                        key=key,
                        value=fact_value,
                        confidence=confidence,
                        source_type=source_type,
                        source_ref=source_ref,
                        source_quote=fact_quote or None,
                    )
                    facts_stored += 1

        # Store relationships
        relationships = entity.get("relationships", [])
        if isinstance(relationships, list):
            for rel in relationships:
                related_name = rel.get("related_to")
                rel_type = rel.get("type")
                if related_name and rel_type:
                    related = Identity.find_or_create(related_name)
                    person.add_relationship(
                        related=related,
                        rel_type=rel_type,
                        confidence=confidence,
                        source_ref=source_ref,
                    )

        # Person-asset graph: link extracted person as "mentioned" on the source asset
        if asset_ref and person.id:
            try:
                entity_db.link_person_asset(
                    person_id=person.id,
                    asset_type=asset_type,
                    asset_ref=asset_ref,
                    role="mentioned",
                    confidence=confidence,
                )
            except Exception:
                pass  # Non-critical

    return facts_stored


# ---------------------------------------------------------------------------
# Public API — called from plugin pipelines
# ---------------------------------------------------------------------------

def maybe_extract_identities(
    sender: str,
    chat_name: str,
    message: str,
    timestamp: str = "",
    chat_id: str = "",
    whatsapp_id: Optional[str] = None,
) -> int:
    """Extract identities from a WhatsApp message (if it passes filtering).

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
    facts_stored = _store_extracted_identities(
        extraction_result=result,
        source_type="whatsapp",
        source_ref=source_ref,
        sender_whatsapp_id=whatsapp_id,
    )

    if facts_stored > 0:
        logger.info(
            f"Identity extraction: {facts_stored} facts from {sender} in {chat_name}"
        )

    return facts_stored


def extract_identities_from_document(
    doc_title: str,
    doc_text: str,
    source_ref: str = "",
    sender: str = "",
) -> int:
    """Extract identities from a Paperless document.

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

    facts_stored = _store_extracted_identities(
        extraction_result=result,
        source_type="paperless",
        source_ref=source_ref,
    )

    if facts_stored > 0:
        logger.info(
            f"Identity extraction from document '{doc_title}': {facts_stored} facts"
        )

    return facts_stored


def extract_from_chat_message(
    user_message: str,
    llm_answer: str = "",
    conversation_id: str = "",
) -> int:
    """Extract identities from a user's chat message (corrections, facts shared in conversation).

    When a user tells the assistant something like "David has a son named Ben
    and a daughter named Mia", this function extracts those relationships and
    stores them in the Identity Store.

    Uses higher confidence (0.8) than auto-extraction (0.6) because the user
    is explicitly providing factual information.

    The LLM answer is included as context so the extractor can resolve
    pronouns and references (e.g., "he" refers to the person discussed).

    Args:
        user_message: The user's message text
        llm_answer: The assistant's response (provides context for pronoun resolution)
        conversation_id: Conversation ID for source tracking

    Returns:
        Number of facts stored (0 if skipped or failed)
    """
    if not settings.get("entity_extraction_enabled", "true").lower() == "true":
        return 0

    if not settings.get("chat_entity_extraction_enabled", "true").lower() == "true":
        return 0

    if not _should_extract(user_message):
        return 0

    # Build a context-aware prompt that includes the LLM's response
    # so the extractor can resolve "him", "her", "he" etc.
    context_block = ""
    if llm_answer:
        # Truncate long answers — only need recent context for pronoun resolution
        answer_snippet = llm_answer[:500]
        context_block = (
            f"\nConversation context (assistant's previous response):\n"
            f"{answer_snippet}\n"
            f"---\n"
        )

    user_prompt = (
        f"Source: User correction in chat conversation\n"
        f"Sender/Author: User (app owner)\n"
        f"{context_block}"
        f"---\n"
        f"{user_message}\n"
        f"---\n"
        f"Extract person facts from the above. The user is correcting or providing "
        f"new information about people. Pay special attention to:\n"
        f"- Family relationships (son, daughter, parent, spouse)\n"
        f"- Personal facts (name, birth date, city, job)\n"
        f"- Name corrections (actual name, nickname)\n"
        f"Use the conversation context to resolve pronouns (he/she/him/her → the person being discussed)."
    )

    result = _call_extraction_llm(user_prompt)
    if not result:
        return 0

    source_ref = f"chat_correction:{conversation_id}" if conversation_id else "chat_correction"
    facts_stored = _store_extracted_identities(
        extraction_result=result,
        source_type="user_correction",
        source_ref=source_ref,
        confidence=0.8,  # Higher confidence for user-provided facts
    )

    if facts_stored > 0:
        logger.info(
            f"Chat identity extraction: {facts_stored} facts from user message "
            f"(conversation={conversation_id})"
        )

    return facts_stored
