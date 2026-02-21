"""Unified identity extraction service.

Single point for gathering facts and identities from all content sources.
All callers should use the singleton ``get_extractor()`` service and its
``submit()`` method to fire-and-forget content for LLM-based extraction,
or ``set_fact()`` for direct fact storage without LLM.

The service handles:
- **Smart filtering** — skips low-value content (short, emoji-only, etc.)
- **Deduplication** — tracks extracted source_refs to avoid re-extracting
- **LLM extraction** — single GPT-4o-mini prompt with structured JSON output
- **Storage** — facts, relationships, and person-asset links
- **Cost tracking** — centralized meter integration
- **Source-specific behavior** — confidence, filtering, truncation per source

Usage::

    from identity_extractor import get_extractor, ExtractionSource

    extractor = get_extractor()

    # Fire-and-forget from any source
    extractor.submit(
        content="I'm turning 32 next week!",
        source=ExtractionSource.WHATSAPP_MESSAGE,
        source_ref="chat:972501234567@c.us:1708012345",
        sender="Shiran Waintrob",
        chat_name="Shiran Waintrob",
    )

    # Direct fact storage — no LLM
    extractor.set_fact(person_id=42, key="city", value="Tel Aviv")

Backward-compatible module-level functions are provided for existing callers
but will be removed in a future cleanup pass.
"""

import json
import re
from enum import Enum
from typing import Any, Dict, List, Optional

from config import settings
from utils.logger import logger


# ---------------------------------------------------------------------------
# Extraction source enum
# ---------------------------------------------------------------------------

class ExtractionSource(str, Enum):
    """Content source types for identity extraction."""

    WHATSAPP_MESSAGE = "whatsapp"
    CHAT_CORRECTION = "chat_correction"
    PAPERLESS_DOCUMENT = "paperless"
    GMAIL_EMAIL = "gmail"
    CALL_RECORDING = "call_recording"
    RAG_PIPELINE = "rag_pipeline"
    MANUAL = "manual"


# ---------------------------------------------------------------------------
# Source-specific configuration
# ---------------------------------------------------------------------------

# Default confidence per source type
_SOURCE_CONFIDENCE: Dict[str, float] = {
    ExtractionSource.WHATSAPP_MESSAGE: 0.6,
    ExtractionSource.CHAT_CORRECTION: 0.8,
    ExtractionSource.PAPERLESS_DOCUMENT: 0.6,
    ExtractionSource.GMAIL_EMAIL: 0.6,
    ExtractionSource.CALL_RECORDING: 0.6,
    ExtractionSource.RAG_PIPELINE: 0.6,
    ExtractionSource.MANUAL: 0.8,
}

# Sources that are always worth extracting (documents, not short messages)
_ALWAYS_EXTRACT_SOURCES = frozenset({
    ExtractionSource.PAPERLESS_DOCUMENT,
    ExtractionSource.GMAIL_EMAIL,
    ExtractionSource.CALL_RECORDING,
})

# Maximum content length sent to LLM for document sources
_DOC_TRUNCATE_CHARS = 4000


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


# ---------------------------------------------------------------------------
# IdentityExtractor service class
# ---------------------------------------------------------------------------

class IdentityExtractor:
    """Singleton service for all identity/fact extraction.

    Provides two entry points:

    - ``submit()`` — fire-and-forget content for LLM-based extraction
    - ``set_fact()`` — direct fact storage without LLM (manual edits, known data)

    The service owns smart filtering, deduplication, LLM calls, storage,
    and cost tracking. Callers should never call the LLM or identity_db
    directly for extraction purposes.
    """

    # =====================================================================
    # Public API
    # =====================================================================

    def submit(
        self,
        content: str,
        source: ExtractionSource,
        source_ref: str = "",
        *,
        sender: str = "",
        chat_name: str = "",
        timestamp: str = "",
        sender_whatsapp_id: Optional[str] = None,
        chat_id: str = "",
        llm_context: str = "",
        confidence: Optional[float] = None,
    ) -> int:
        """Submit content for identity extraction.

        The service decides whether to extract based on:
        1. Global enabled setting
        2. Source-specific enabled settings
        3. Dedup check against source_ref
        4. Smart content filtering (length, patterns, etc.)

        Args:
            content: The text to extract identities from.
            source: Where this content came from (enum).
            source_ref: Unique reference for dedup (e.g. "chat:972...:170...").
            sender: Content author/sender name.
            chat_name: Chat or document title.
            timestamp: Unix timestamp as string.
            sender_whatsapp_id: WhatsApp ID of the sender (for auto-matching).
            chat_id: WhatsApp chat ID (for group filtering).
            llm_context: Additional context for pronoun resolution
                (e.g. assistant response for chat corrections).
            confidence: Override default per-source confidence.

        Returns:
            Number of facts stored (0 if skipped/filtered/deduped).
        """
        # 1. Global kill switch
        if not settings.get("entity_extraction_enabled", "true").lower() == "true":
            return 0

        # 2. Source-specific kill switches
        if not self._is_source_enabled(source, chat_id):
            return 0

        # 3. Dedup check — skip if we already extracted from this source_ref
        if source_ref:
            import identity_db
            cached = identity_db.check_extracted(source_ref)
            if cached is not None:
                logger.debug(
                    f"Identity extraction skipped (already extracted): {source_ref}"
                )
                return cached

        # 4. Smart content filtering
        is_document = source in _ALWAYS_EXTRACT_SOURCES
        if not self._should_extract(content, is_document):
            return 0

        # 5. Build LLM prompt
        user_prompt = self._build_prompt(
            content=content,
            source=source,
            sender=sender,
            chat_name=chat_name,
            timestamp=timestamp,
            llm_context=llm_context,
        )

        # 6. Call LLM
        result = self._call_extraction_llm(user_prompt)
        if not result:
            # Mark as extracted with 0 facts so we don't retry
            if source_ref:
                import identity_db
                identity_db.mark_extracted(source_ref, source.value, 0)
            return 0

        # 7. Store extracted identities
        effective_confidence = confidence if confidence is not None else _SOURCE_CONFIDENCE.get(source, 0.6)
        facts_stored = self._store_extracted_identities(
            extraction_result=result,
            source_type=source.value,
            source_ref=source_ref or None,
            sender_whatsapp_id=sender_whatsapp_id,
            confidence=effective_confidence,
        )

        # 8. Record in dedup log
        if source_ref:
            import identity_db
            identity_db.mark_extracted(source_ref, source.value, facts_stored)

        # 9. Log
        if facts_stored > 0:
            logger.info(
                f"Identity extraction [{source.value}]: {facts_stored} facts "
                f"from {sender or chat_name or source_ref}"
            )

        return facts_stored

    def set_fact(
        self,
        person_id: int,
        key: str,
        value: str,
        confidence: float = 0.8,
        source_type: str = "manual",
        source_ref: Optional[str] = None,
        source_quote: Optional[str] = None,
    ) -> None:
        """Store a fact directly without LLM extraction.

        Used for:
        - Manual edits via REST API
        - Known structured data (is_business flag, etc.)

        Args:
            person_id: The person's database ID.
            key: Fact key (e.g. "city", "birth_date").
            value: Fact value.
            confidence: Confidence score (0.0-1.0).
            source_type: Source type string.
            source_ref: Reference to source.
            source_quote: Original text snippet.
        """
        from identity import Identity

        person = Identity.get(person_id)
        if person is None:
            logger.warning(f"set_fact: person_id={person_id} not found")
            return

        person.set_fact(
            key=key,
            value=value,
            confidence=confidence,
            source_type=source_type,
            source_ref=source_ref,
            source_quote=source_quote,
        )

    # =====================================================================
    # Internal — filtering
    # =====================================================================

    @staticmethod
    def _is_source_enabled(source: ExtractionSource, chat_id: str = "") -> bool:
        """Check source-specific enabled settings.

        Args:
            source: The extraction source type.
            chat_id: WhatsApp chat ID (used for group filtering).

        Returns:
            True if extraction is enabled for this source.
        """
        if source == ExtractionSource.CHAT_CORRECTION:
            if not settings.get("chat_identity_extraction_enabled", "true").lower() == "true":
                return False

        if source == ExtractionSource.WHATSAPP_MESSAGE:
            # Skip group chats if configured
            if settings.get("entity_extraction_skip_groups", "true").lower() == "true":
                if chat_id and chat_id.endswith("@g.us"):
                    return False

        if source == ExtractionSource.RAG_PIPELINE:
            if settings.get("rag_entity_extraction_in_pipeline", "false").lower() != "true":
                return False

        return True

    @staticmethod
    def _should_extract(content: str, is_document: bool = False) -> bool:
        """Determine if content warrants entity extraction.

        Documents always get extracted (high fact density).
        Messages are filtered by length and content patterns.

        Args:
            content: The text content.
            is_document: Whether this is a document (vs. short message).

        Returns:
            True if extraction should proceed.
        """
        if is_document:
            # Documents need at least some content
            return bool(content) and len(content.strip()) >= 20

        if not content or len(content) < _MIN_LENGTH:
            return False

        # Skip patterns
        for pattern in _SKIP_PATTERNS:
            if pattern.match(content):
                return False

        # Check for fact-indicating patterns (fast pre-filter before LLM call)
        for pattern in _FACT_PATTERNS:
            if pattern.search(content):
                return True

        # For longer messages (>100 chars), extract even without pattern match
        if len(content) > 100:
            return True

        return False

    # =====================================================================
    # Internal — prompt building
    # =====================================================================

    @staticmethod
    def _build_prompt(
        content: str,
        source: ExtractionSource,
        sender: str = "",
        chat_name: str = "",
        timestamp: str = "",
        llm_context: str = "",
    ) -> str:
        """Build the user prompt for the extraction LLM.

        Args:
            content: Text to extract from.
            source: Source type (affects prompt framing).
            sender: Content author/sender.
            chat_name: Chat or document title.
            timestamp: Unix timestamp string.
            llm_context: Additional context (e.g. assistant response).

        Returns:
            Formatted prompt string.
        """
        # Truncate document content to stay within token limits
        text = content
        if source in _ALWAYS_EXTRACT_SOURCES and len(content) > _DOC_TRUNCATE_CHARS:
            text = content[:_DOC_TRUNCATE_CHARS]

        # Chat correction prompt with context for pronoun resolution
        if source == ExtractionSource.CHAT_CORRECTION:
            context_block = ""
            if llm_context:
                answer_snippet = llm_context[:500]
                context_block = (
                    f"\nConversation context (assistant's previous response):\n"
                    f"{answer_snippet}\n"
                    f"---\n"
                )

            return (
                f"Source: User correction in chat conversation\n"
                f"Sender/Author: User (app owner)\n"
                f"{context_block}"
                f"---\n"
                f"{text}\n"
                f"---\n"
                f"Extract person facts from the above. The user is correcting or providing "
                f"new information about people. Pay special attention to:\n"
                f"- Family relationships (son, daughter, parent, spouse)\n"
                f"- Personal facts (name, birth date, city, job)\n"
                f"- Name corrections (actual name, nickname)\n"
                f"Use the conversation context to resolve pronouns (he/she/him/her → the person being discussed)."
            )

        # Standard prompt for messages and documents
        is_document = source in _ALWAYS_EXTRACT_SOURCES
        source_type_label = "Document" if is_document else "WhatsApp message"

        return (
            f"Source: {source_type_label}\n"
            f"Chat/Document: {chat_name}\n"
            f"Sender/Author: {sender}\n"
            f"Timestamp: {timestamp}\n"
            f"---\n"
            f"{text}\n"
            f"---\n"
            f"Extract person facts from the above."
        )

    # =====================================================================
    # Internal — LLM call
    # =====================================================================

    @staticmethod
    def _call_extraction_llm(user_prompt: str) -> Optional[Dict[str, Any]]:
        """Call GPT-4o-mini for identity extraction.

        Args:
            user_prompt: The formatted extraction prompt.

        Returns:
            Parsed JSON response, or None on failure.
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

    # =====================================================================
    # Internal — storage
    # =====================================================================

    @staticmethod
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
            extraction_result: Parsed JSON from LLM with 'entities' list.
            source_type: Where this was extracted from.
            source_ref: Reference string (e.g., "chat:972501234567@c.us:1708012345").
            sender_whatsapp_id: WhatsApp ID of the sender (for auto-matching).
            confidence: Confidence score for stored facts.

        Returns:
            Number of facts stored.
        """
        import identity_db
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
                asset_ref = source_ref[len("chat:"):]
                asset_type = "whatsapp_msg"
            elif source_ref.startswith("paperless:"):
                asset_ref = source_ref
                asset_type = "document"
            elif source_ref.startswith("gmail:"):
                asset_ref = source_ref
                asset_type = "gmail"
            elif source_ref.startswith("call_recording:"):
                asset_ref = source_ref
                asset_type = "call_recording"
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
                    identity_db.link_person_asset(
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
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[IdentityExtractor] = None


def get_extractor() -> IdentityExtractor:
    """Get or create the singleton IdentityExtractor service.

    Returns:
        The global IdentityExtractor instance.
    """
    global _instance
    if _instance is None:
        _instance = IdentityExtractor()
    return _instance


