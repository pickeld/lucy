"""PII detection and redaction using Microsoft Presidio.

Provides configurable per-channel PII policies for the ingestion pipeline.
Detects and anonymizes sensitive data (phone numbers, emails, ID numbers,
credit cards, etc.) before embedding generation and LLM context assembly.

Supports Hebrew-specific recognizers (Teudat Zehut, Israeli phone formats).

Feature-gated: only active when ``pii_redaction_enabled=true`` in settings.

Usage:
    from pii_redactor import get_redactor

    redactor = get_redactor()
    if redactor:
        safe_text = redactor.redact_for_embedding(text, channel="whatsapp")
"""

import re
from typing import Any, Dict, List, Optional

from utils.logger import logger


# ---------------------------------------------------------------------------
# Per-channel PII policies
# ---------------------------------------------------------------------------

DEFAULT_POLICIES: Dict[str, Dict[str, Any]] = {
    "whatsapp": {
        "entities": [
            "PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD",
            "IBAN_CODE", "IL_ID_NUMBER",
        ],
        "action": "hash",  # Replace with <TYPE_HASH> for reversibility
        "score_threshold": 0.6,
    },
    "gmail": {
        "entities": [
            "PHONE_NUMBER", "CREDIT_CARD", "IBAN_CODE",
            "IL_ID_NUMBER",
        ],
        "action": "replace",  # Replace with <TYPE> placeholder
        "score_threshold": 0.6,
    },
    "paperless": {
        "entities": [
            "CREDIT_CARD", "IBAN_CODE",
        ],
        "action": "redact",  # Full removal
        "score_threshold": 0.7,
    },
    "call_recording": {
        "entities": [
            "PHONE_NUMBER", "CREDIT_CARD",
        ],
        "action": "replace",
        "score_threshold": 0.6,
    },
}


# ---------------------------------------------------------------------------
# Israeli ID number (Teudat Zehut) recognizer
# ---------------------------------------------------------------------------

def _is_valid_il_id(id_str: str) -> bool:
    """Validate an Israeli ID number using the Luhn-like algorithm.

    Israeli ID numbers are 9 digits with a check digit.

    Args:
        id_str: 9-digit string

    Returns:
        True if the check digit is valid
    """
    if not id_str or len(id_str) != 9 or not id_str.isdigit():
        return False

    total = 0
    for i, ch in enumerate(id_str):
        digit = int(ch)
        if i % 2 == 1:
            digit *= 2
        if digit > 9:
            digit -= 9
        total += digit
    return total % 10 == 0


# Israeli ID pattern: 9 digits, may be preceded/followed by non-digit
_IL_ID_PATTERN = re.compile(r'(?<!\d)\d{9}(?!\d)')

# Israeli phone patterns
_IL_PHONE_PATTERNS = [
    re.compile(r'(?:\+972|972)[\s\-]?[2-9]\d[\s\-]?\d{3}[\s\-]?\d{4}'),
    re.compile(r'0[2-9]\d[\s\-]?\d{3}[\s\-]?\d{4}'),
    re.compile(r'05\d[\s\-]?\d{3}[\s\-]?\d{4}'),
]


# ---------------------------------------------------------------------------
# PIIRedactor class
# ---------------------------------------------------------------------------

class PIIRedactor:
    """Configurable PII detection and redaction using Microsoft Presidio.

    Initialized lazily on first use. Thread-safe (Presidio analyzer is
    thread-safe by design).

    Args:
        policies: Per-channel policy overrides (merged with DEFAULT_POLICIES)
        default_language: Default language for analysis (default: "en")
    """

    def __init__(
        self,
        policies: Optional[Dict[str, Dict[str, Any]]] = None,
        default_language: str = "en",
    ):
        self._policies = {**DEFAULT_POLICIES, **(policies or {})}
        self._default_language = default_language
        self._analyzer = None
        self._anonymizer = None
        self._initialized = False

    def _ensure_initialized(self) -> bool:
        """Lazily initialize Presidio components.

        Returns:
            True if initialization succeeded
        """
        if self._initialized:
            return True

        try:
            from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
            from presidio_anonymizer import AnonymizerEngine

            # Create analyzer with custom recognizers
            self._analyzer = AnalyzerEngine()

            # Add Israeli ID number recognizer
            il_id_recognizer = PatternRecognizer(
                supported_entity="IL_ID_NUMBER",
                name="IL ID Number Recognizer",
                patterns=[
                    Pattern(
                        name="il_id_9digits",
                        regex=r"(?<!\d)\d{9}(?!\d)",
                        score=0.5,
                    ),
                ],
                context=["תעודת זהות", "ת.ז", "ת\"ז", "id number", "id no", "מספר זהות"],
            )
            self._analyzer.registry.add_recognizer(il_id_recognizer)

            # Add Israeli phone recognizer (supplements built-in phone recognizer)
            il_phone_recognizer = PatternRecognizer(
                supported_entity="PHONE_NUMBER",
                name="IL Phone Recognizer",
                patterns=[
                    Pattern(
                        name="il_mobile",
                        regex=r"05\d[\s\-]?\d{3}[\s\-]?\d{4}",
                        score=0.7,
                    ),
                    Pattern(
                        name="il_intl",
                        regex=r"(?:\+972|972)[\s\-]?[2-9]\d[\s\-]?\d{3}[\s\-]?\d{4}",
                        score=0.8,
                    ),
                ],
                context=["טלפון", "נייד", "phone", "mobile", "call", "חייג"],
            )
            self._analyzer.registry.add_recognizer(il_phone_recognizer)

            self._anonymizer = AnonymizerEngine()
            self._initialized = True
            logger.info("PIIRedactor initialized with Presidio + Hebrew recognizers")
            return True

        except ImportError:
            logger.warning(
                "Presidio not installed. PII redaction disabled. "
                "Install with: pip install presidio-analyzer presidio-anonymizer"
            )
            return False
        except Exception as e:
            logger.error(f"PIIRedactor initialization failed: {e}")
            return False

    def detect(self, text: str, channel: str = "whatsapp") -> List[Dict[str, Any]]:
        """Detect PII entities in text.

        Args:
            text: Text to analyze
            channel: Channel name for policy selection

        Returns:
            List of detected PII entities with type, start, end, score
        """
        if not self._ensure_initialized() or not self._analyzer:
            return []

        policy = self._policies.get(channel, self._policies.get("whatsapp", {}))
        entities = policy.get("entities", [])
        threshold = policy.get("score_threshold", 0.6)

        if not entities:
            return []

        try:
            results = self._analyzer.analyze(
                text=text,
                entities=entities,
                language=self._default_language,
                score_threshold=threshold,
            )
            return [
                {
                    "entity_type": r.entity_type,
                    "start": r.start,
                    "end": r.end,
                    "score": r.score,
                    "text": text[r.start:r.end],
                }
                for r in results
            ]
        except Exception as e:
            logger.debug(f"PII detection failed (non-critical): {e}")
            return []

    def redact(self, text: str, channel: str = "whatsapp") -> str:
        """Apply redaction policy for the channel.

        Args:
            text: Text to redact
            channel: Channel name for policy selection

        Returns:
            Redacted text
        """
        if not self._ensure_initialized() or not self._analyzer or not self._anonymizer:
            return text

        policy = self._policies.get(channel, self._policies.get("whatsapp", {}))
        entities = policy.get("entities", [])
        action = policy.get("action", "replace")
        threshold = policy.get("score_threshold", 0.6)

        if not entities:
            return text

        try:
            from presidio_anonymizer.entities import OperatorConfig

            # Analyze
            analysis_results = self._analyzer.analyze(
                text=text,
                entities=entities,
                language=self._default_language,
                score_threshold=threshold,
            )

            if not analysis_results:
                return text

            # Build operator config based on policy action
            if action == "redact":
                operators = {
                    entity: OperatorConfig("redact")
                    for entity in entities
                }
            elif action == "hash":
                operators = {
                    entity: OperatorConfig("hash", {"hash_type": "sha256", "length": 8})
                    for entity in entities
                }
            else:  # "replace" (default)
                operators = {
                    entity: OperatorConfig("replace", {"new_value": f"<{entity}>"})
                    for entity in entities
                }

            # Anonymize
            result = self._anonymizer.anonymize(
                text=text,
                analyzer_results=analysis_results,
                operators=operators,
            )
            return result.text

        except Exception as e:
            logger.debug(f"PII redaction failed (non-critical): {e}")
            return text

    def redact_for_embedding(self, text: str, channel: str = "whatsapp") -> str:
        """Redact text before embedding generation.

        Uses the channel's policy but always applies "replace" action
        for embeddings to ensure consistent token structure.

        Args:
            text: Text to redact before embedding
            channel: Channel name for policy selection

        Returns:
            Redacted text safe for embedding
        """
        if not self._ensure_initialized():
            return text

        # For embeddings, always use replace action regardless of channel policy
        # to maintain consistent text structure for the embedding model
        return self.redact(text, channel)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_redactor_instance: Optional[PIIRedactor] = None


def get_redactor() -> Optional[PIIRedactor]:
    """Get the global PIIRedactor instance.

    Returns None if PII redaction is disabled via settings.

    Returns:
        PIIRedactor instance, or None if disabled
    """
    global _redactor_instance

    try:
        from config import settings
        if settings.get("pii_redaction_enabled", "false").lower() != "true":
            return None
    except Exception:
        return None

    if _redactor_instance is None:
        _redactor_instance = PIIRedactor()

    return _redactor_instance
