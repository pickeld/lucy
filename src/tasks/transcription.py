"""Celery tasks for call recording transcription.

Replaces the in-process ThreadPoolExecutor in CallRecordingSyncer with
durable, retryable tasks on the ``heavy`` queue.

The heavy queue should run with ``--concurrency=1`` because:
- Whisper is CPU/GPU-bound and concurrent runs would OOM or thrash
- This matches the previous ``_POOL_WORKERS = 1`` behavior

Tasks:
    transcribe_recording  — Transcribe a single audio file by content hash
"""

import traceback

from celery import shared_task
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)


def _get_syncer():
    """Lazily get the CallRecordingSyncer from the plugin registry.

    The syncer holds the transcriber (Whisper/AssemblyAI) and RAG references.
    On first call in a worker process the plugin will initialize, loading
    the Whisper model.  Subsequent calls reuse the cached instance.
    """
    from plugins.registry import plugin_registry

    plugin = plugin_registry.get_plugin("call_recordings")
    if plugin is None:
        raise RuntimeError(
            "call_recordings plugin not found in registry. "
            "Ensure it is enabled before dispatching transcription tasks."
        )

    syncer = getattr(plugin, "_syncer", None)
    if syncer is None:
        raise RuntimeError(
            "call_recordings plugin syncer not initialized. "
            "The plugin must be initialized before transcription tasks run."
        )
    return syncer


@shared_task(
    bind=True,
    name="tasks.transcription.transcribe_recording",
    max_retries=2,
    default_retry_delay=60,
    acks_late=True,
    reject_on_worker_lost=True,
    # Transcription can be very slow for long recordings
    soft_time_limit=1800,   # 30 min soft limit
    time_limit=3600,        # 60 min hard kill
)
def transcribe_recording(self, content_hash: str) -> dict:
    """Transcribe a single audio file by content hash.

    This is the durable equivalent of
    ``CallRecordingSyncer.transcribe_file_async()`` +
    ``CallRecordingSyncer._transcribe_worker()``.

    The task:
        1. Gets the syncer from the plugin registry
        2. Delegates to ``syncer.transcribe_file(content_hash)``
        3. Returns the result dict

    Idempotency: re-delivery is safe because ``transcribe_file()`` checks
    the DB status before processing and the result is stored in SQLite.

    Args:
        content_hash: SHA256 content hash identifying the audio file.

    Returns:
        Dict with transcription result.
    """
    try:
        syncer = _get_syncer()

        logger.info(f"[task] Starting transcription for {content_hash}")
        result = syncer.transcribe_file(content_hash)

        status = result.get("status", "unknown")
        logger.info(
            f"[task] Transcription complete for {content_hash}: status={status}"
        )
        return result

    except Exception as exc:
        trace = traceback.format_exc()
        logger.error(
            f"[task] Transcription failed for {content_hash}: {exc}\n{trace}"
        )

        # Update DB status on failure
        try:
            from plugins.call_recordings import db as recording_db
            recording_db.update_status(content_hash, "error", str(exc))
        except Exception:
            pass

        # Retry on transient errors (file locked, network issues)
        transient_indicators = [
            "EDEADLK",
            "file_locked",
            "ConnectionError",
            "Timeout",
            "rate_limit",
        ]
        is_transient = any(ind in str(exc) for ind in transient_indicators)

        if is_transient and self.request.retries < self.max_retries:
            backoff = self.default_retry_delay * (2 ** self.request.retries)
            logger.warning(
                f"[task] Transient error, retrying in {backoff}s "
                f"(attempt {self.request.retries + 1}/{self.max_retries})"
            )
            raise self.retry(exc=exc, countdown=backoff)

        logger.error(
            f"[task] DEAD LETTER: Transcription permanently failed for "
            f"{content_hash} after {self.request.retries} retries"
        )
        return {"status": "error", "error": str(exc), "retries": self.request.retries}
