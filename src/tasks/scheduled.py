"""Celery tasks for Scheduled Insights execution.

Two tasks:
    check_scheduled_insights  — Periodic (every 60s via Beat): polls SQLite for
                                due tasks and dispatches individual executions.
    execute_scheduled_insight — On-demand: runs a single insight query through
                                the RAG pipeline and stores the result.

Task routing: tasks.scheduled.* → default queue (lightweight LLM calls).
"""

import json
import time
import traceback
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from celery.utils.log import get_task_logger

from tasks import app

logger = get_task_logger(__name__)


def _build_insight_prompt(task_prompt: str, timezone: str) -> str:
    """Wrap the user's prompt with temporal context.

    Injects the current date/time so the RAG system can ground its
    answer in "today" when the user asks about upcoming events,
    deadlines, or time-sensitive commitments.

    Args:
        task_prompt: The user-defined prompt text
        timezone: IANA timezone for the current time

    Returns:
        The enriched prompt string
    """
    try:
        tz = ZoneInfo(timezone)
    except Exception:
        tz = ZoneInfo("Asia/Jerusalem")

    now = datetime.now(tz)

    return (
        f"Today is {now.strftime('%A, %B %d, %Y')} "
        f"({now.strftime('%d/%m/%Y')}, {now.strftime('%H:%M')} {timezone}).\n\n"
        f"{task_prompt}\n\n"
        f"Important: Base your answer ONLY on the retrieved messages and documents. "
        f"Be specific — cite dates, people, and exact quotes when possible. "
        f"If you find nothing relevant, say so clearly."
    )


@app.task(
    bind=True,
    name="tasks.scheduled.check_scheduled_insights",
    max_retries=0,
    acks_late=False,
    soft_time_limit=30,
    time_limit=60,
)
def check_scheduled_insights(self) -> dict:
    """Poll for due scheduled tasks and dispatch individual executions.

    This task runs every 60 seconds via Celery Beat.  It checks the
    SQLite database for any enabled tasks whose ``next_run_at`` is in
    the past and dispatches ``execute_scheduled_insight`` for each one.

    Returns:
        Dict with count of dispatched tasks.
    """
    try:
        import scheduled_tasks_db

        due_tasks = scheduled_tasks_db.get_due_tasks()

        if not due_tasks:
            return {"status": "ok", "dispatched": 0}

        dispatched = 0
        for task in due_tasks:
            task_id = task["id"]
            try:
                # Advance next_run_at BEFORE dispatching to prevent
                # double-dispatch if the checker runs again before the
                # task completes.
                scheduled_tasks_db.advance_next_run(task_id)

                # Dispatch execution
                execute_scheduled_insight.delay(task_id)
                dispatched += 1
                logger.info(
                    f"[beat] Dispatched insight task #{task_id}: '{task['name']}'"
                )
            except Exception as e:
                logger.error(
                    f"[beat] Failed to dispatch task #{task_id}: {e}"
                )

        return {"status": "ok", "dispatched": dispatched}

    except Exception as exc:
        trace = traceback.format_exc()
        logger.error(f"[beat] check_scheduled_insights failed: {exc}\n{trace}")
        return {"status": "error", "error": str(exc)}


@app.task(
    bind=True,
    name="tasks.scheduled.execute_scheduled_insight",
    max_retries=2,
    default_retry_delay=60,
    acks_late=True,
    reject_on_worker_lost=True,
    soft_time_limit=120,
    time_limit=180,
)
def execute_scheduled_insight(self, task_id: int) -> dict:
    """Execute a single scheduled insight query through the RAG pipeline.

    Steps:
        1. Load task definition from SQLite
        2. Build the effective prompt with date/time context
        3. Create a RAG chat engine with the task's filters
        4. Execute the query and capture the answer + sources
        5. Record cost via METER snapshot delta
        6. Store result in task_results table

    Args:
        task_id: The scheduled_tasks row ID to execute.

    Returns:
        Dict with execution result metadata.
    """
    start_ms = int(time.time() * 1000)

    try:
        import scheduled_tasks_db
        from llamaindex_rag import get_rag
        from cost_meter import METER

        # 1. Load task definition
        task = scheduled_tasks_db.get_task(task_id)
        if not task:
            logger.warning(f"[insight] Task #{task_id} not found, skipping")
            return {"status": "skipped", "reason": "task_not_found"}

        if not task["enabled"]:
            logger.info(f"[insight] Task #{task_id} is disabled, skipping")
            return {"status": "skipped", "reason": "disabled"}

        logger.info(
            f"[insight] Executing task #{task_id}: '{task['name']}'"
        )

        # 2. Build the effective prompt
        prompt = _build_insight_prompt(
            task["prompt"],
            task.get("timezone", "Asia/Jerusalem"),
        )

        # 3. Parse filters
        filters = task.get("filters", {})
        if isinstance(filters, str):
            try:
                filters = json.loads(filters)
            except (json.JSONDecodeError, TypeError):
                filters = {}

        sources_list = filters.get("sources")
        content_types_list = filters.get("content_types")

        # 4. Create RAG chat engine with filters
        rag = get_rag()

        # Use a unique conversation_id so insights don't pollute
        # user chat history
        conversation_id = f"insight-{task_id}-{uuid.uuid4().hex[:8]}"

        chat_engine = rag.create_chat_engine(
            conversation_id=conversation_id,
            filter_chat_name=filters.get("chat_name"),
            filter_sender=filters.get("sender"),
            filter_days=int(filters["days"]) if filters.get("days") else None,
            filter_sources=sources_list,
            filter_date_from=filters.get("date_from"),
            filter_date_to=filters.get("date_to"),
            filter_content_types=content_types_list,
            k=int(filters.get("k", 10)),
        )

        # 5. Execute query with cost tracking
        cost_snapshot = METER.snapshot()
        response = chat_engine.chat(prompt)
        answer = str(response)
        query_cost = METER.session_total - cost_snapshot

        # Extract sources
        sources = []
        if hasattr(response, "source_nodes") and response.source_nodes:
            for node_with_score in response.source_nodes:
                node = node_with_score.node
                metadata = getattr(node, "metadata", {})
                if metadata.get("source") == "system":
                    continue
                sources.append({
                    "content": getattr(node, "text", "")[:300],
                    "score": node_with_score.score,
                    "sender": metadata.get("sender", ""),
                    "chat_name": metadata.get("chat_name", ""),
                    "timestamp": metadata.get("timestamp"),
                })

        duration_ms = int(time.time() * 1000) - start_ms

        # 6. Determine status
        status = "success"
        if not answer or answer.strip() == "Empty Response":
            status = "no_results"
            answer = "No relevant information found for this query."

        # 7. Store result
        scheduled_tasks_db.add_result(
            task_id=task_id,
            answer=answer,
            prompt_used=prompt,
            sources=sources,
            cost_usd=query_cost,
            duration_ms=duration_ms,
            status=status,
        )

        logger.info(
            f"[insight] Task #{task_id} completed: "
            f"status={status}, cost=${query_cost:.4f}, "
            f"duration={duration_ms}ms, sources={len(sources)}"
        )

        return {
            "status": status,
            "task_id": task_id,
            "task_name": task["name"],
            "cost_usd": round(query_cost, 6),
            "duration_ms": duration_ms,
            "source_count": len(sources),
        }

    except Exception as exc:
        duration_ms = int(time.time() * 1000) - start_ms
        trace = traceback.format_exc()
        logger.error(
            f"[insight] Task #{task_id} failed: {exc}\n{trace}"
        )

        # Store error result
        try:
            import scheduled_tasks_db
            scheduled_tasks_db.add_result(
                task_id=task_id,
                answer=f"Insight execution failed: {exc}",
                prompt_used="",
                status="error",
                error_message=str(exc),
                duration_ms=duration_ms,
            )
        except Exception:
            pass  # Don't mask the original error

        # Retry on transient errors
        transient_indicators = [
            "ConnectionError", "Timeout", "rate_limit", "429", "503",
        ]
        is_transient = any(ind in str(exc) for ind in transient_indicators)

        if is_transient and self.request.retries < self.max_retries:
            backoff = self.default_retry_delay * (2 ** self.request.retries)
            logger.warning(
                f"[insight] Transient error, retrying in {backoff}s "
                f"(attempt {self.request.retries + 1}/{self.max_retries})"
            )
            raise self.retry(exc=exc, countdown=backoff)

        return {
            "status": "error",
            "task_id": task_id,
            "error": str(exc),
            "duration_ms": duration_ms,
        }
