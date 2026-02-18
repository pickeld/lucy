"""Celery tasks for WhatsApp message processing.

Replaces the in-process ThreadPoolExecutor with durable, retryable tasks.
Each task is idempotent — re-delivery produces the same result because
``add_message()`` deduplicates by ``source_id = chat_id:timestamp``.

Tasks:
    process_whatsapp_message  — Parse payload, enrich media, store in RAG
"""

import traceback

from celery.utils.log import get_task_logger

from tasks import app

logger = get_task_logger(__name__)


@app.task(
    bind=True,
    name="tasks.whatsapp.process_whatsapp_message",
    max_retries=3,
    default_retry_delay=30,
    acks_late=True,
    reject_on_worker_lost=True,
    # Soft limit: 3 min (Vision + Whisper can be slow on large media)
    soft_time_limit=180,
    time_limit=300,
)
def process_whatsapp_message(self, payload: dict) -> dict:
    """Process a WhatsApp webhook payload in a Celery worker.

    This is the durable equivalent of ``WhatsAppPlugin._process_webhook_payload()``.
    The webhook handler enqueues this task and returns 200 immediately.

    Steps:
        1. Parse payload into a WhatsappMSG subclass (may trigger Vision/Whisper)
        2. Resolve chat identification (group, contact, recipient)
        3. Store message in RAG vector store via ``rag.add_message()``
        4. Run entity extraction (non-blocking, failures logged)

    Args:
        payload: The WAHA webhook ``payload`` dict (not the outer envelope).

    Returns:
        Dict with processing result metadata.

    Raises:
        celery.exceptions.Retry: On transient failures (network, API rate limits).
    """
    try:
        from plugins.whatsapp.handler import create_whatsapp_message
        from llamaindex_rag import get_rag

        rag = get_rag()
        msg = create_whatsapp_message(payload)

        # Determine chat identification
        if msg.is_group:
            chat_id = msg.group.id
            chat_name = msg.group.name
        elif msg.from_me and msg.recipient:
            chat_id = msg.recipient.number or msg.recipient.id
            chat_name = msg.recipient.name
        else:
            chat_id = msg.contact.number
            chat_name = msg.contact.name

        sender = str(msg.contact.name or "Unknown")
        logger.info(f"[task] Processing message: {chat_name} ({chat_id}) - {msg.message}")

        # Store message in RAG vector store
        if msg.message and rag:
            has_media = getattr(msg, "has_media", False)
            media_type = getattr(msg, "media_type", None)
            media_url = getattr(msg, "media_url", None)
            media_path = getattr(msg, "saved_path", None)

            handler_content_type = getattr(msg, "content_type", None)
            ct_value = handler_content_type.value if handler_content_type else None

            rag.add_message(
                thread_id=chat_id or "UNKNOWN",
                chat_id=chat_id or "UNKNOWN",
                chat_name=chat_name or "UNKNOWN",
                is_group=msg.is_group,
                sender=sender,
                message=msg.message,
                timestamp=str(msg.timestamp) if msg.timestamp else "0",
                has_media=has_media,
                media_type=media_type,
                media_url=media_url,
                media_path=media_path,
                message_content_type=ct_value,
            )
            logger.debug(f"[task] Stored message: {chat_name} || {msg}")

            # Entity extraction (non-blocking — failures are logged and ignored)
            try:
                from entity_extractor import maybe_extract_entities
                maybe_extract_entities(
                    sender=sender,
                    chat_name=chat_name or "Unknown",
                    message=msg.message,
                    timestamp=str(msg.timestamp) if msg.timestamp else "0",
                    chat_id=chat_id or "",
                    whatsapp_id=msg.contact.id,
                )
            except Exception as ee:
                logger.debug(f"[task] Entity extraction failed (non-critical): {ee}")

        return {
            "status": "ok",
            "chat_id": chat_id,
            "chat_name": chat_name,
            "sender": sender,
            "has_message": bool(msg.message),
        }

    except Exception as exc:
        trace = traceback.format_exc()
        logger.error(f"[task] WhatsApp message processing failed: {exc}\n{trace}")

        # Retry on transient errors (network issues, API rate limits)
        # Non-transient errors (bad payload, parse errors) should NOT retry
        transient_indicators = [
            "ConnectionError",
            "Timeout",
            "rate_limit",
            "429",
            "503",
            "502",
        ]
        is_transient = any(ind in str(exc) for ind in transient_indicators)

        if is_transient and self.request.retries < self.max_retries:
            # Exponential backoff: 30s, 60s, 120s
            backoff = self.default_retry_delay * (2 ** self.request.retries)
            logger.warning(
                f"[task] Transient error, retrying in {backoff}s "
                f"(attempt {self.request.retries + 1}/{self.max_retries})"
            )
            raise self.retry(exc=exc, countdown=backoff)

        # Non-transient or max retries exhausted — log as dead letter
        logger.error(
            f"[task] DEAD LETTER: WhatsApp message processing permanently failed "
            f"after {self.request.retries} retries. Payload keys: {list(payload.keys())}"
        )
        return {
            "status": "failed",
            "error": str(exc),
            "retries": self.request.retries,
        }
