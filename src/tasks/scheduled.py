"""Celery tasks for Scheduled Insights execution.

Two tasks:
    check_scheduled_insights  — Periodic (every 60s via Beat): polls SQLite for
                                due tasks and dispatches individual executions.
    execute_scheduled_insight — On-demand: runs a single insight query through
                                the multi-query RAG pipeline and stores the result.

Quality pipeline (see plans/insights-quality-improvements.md):
    1. Decompose prompt into targeted sub-queries (template or LLM-based)
    2. Multi-pass retrieval with higher k per sub-query
    3. Deduplicate + Cohere rerank merged results
    4. Direct LLM completion with insight-specific system prompt
    5. Compute quality metrics (source coverage, confidence, etc.)

Task routing: tasks.scheduled.* → default queue (lightweight LLM calls).
"""

import json
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from celery.utils.log import get_task_logger

from tasks import app

logger = get_task_logger(__name__)


# =========================================================================
# Insight System Prompt — dedicated for analytical insight generation
# =========================================================================

_INSIGHT_SYSTEM_PROMPT_TEMPLATE = """\
You are an analytical intelligence assistant that produces comprehensive, \
actionable briefings from a personal knowledge base of messages, documents, \
emails, and call recordings.

Current Date/Time: {current_datetime}
תאריך ושעה נוכחיים: {hebrew_date}

YOUR ROLE:
1. ANALYZE all retrieved messages and documents THOROUGHLY — scan every item
2. EXTRACT every relevant piece of information — do NOT skip or summarize away details
3. ORGANIZE findings into clear, structured sections with headers
4. CITE specifics: exact dates, full names, chat names, and direct quotes when helpful
5. PRIORITIZE: flag urgent or time-sensitive items FIRST
6. Be EXHAUSTIVE rather than concise — it is better to include a marginal finding than miss something important
7. Answer in the SAME LANGUAGE as the query (Hebrew → Hebrew, English → English)
8. When no information is found for a section, explicitly state "Nothing found" — do NOT skip sections
9. Do NOT fabricate information — only report what is found in the retrieved context

QUALITY CHECKLIST (verify before responding):
✓ Did I address every aspect/section of the query?
✓ Did I cite specific people, dates, and messages for each finding?
✓ Did I review ALL retrieved items, not just the first few?
✓ Are my findings organized by priority/urgency?
✓ Would the user find this actionable and specific (not vague)?
✓ Did I include "Nothing found" for empty sections instead of skipping them?"""


def _build_insight_system_prompt(timezone: str) -> str:
    """Build the insight system prompt with current date/time.

    Uses a dedicated prompt optimized for analytical insight generation —
    no disambiguation rules, no calendar creation, no image handling,
    no chat follow-up instructions.

    Args:
        timezone: IANA timezone for the current time

    Returns:
        The insight system prompt string
    """
    try:
        tz = ZoneInfo(timezone)
    except Exception:
        tz = ZoneInfo("Asia/Jerusalem")

    now = datetime.now(tz)
    current_datetime = now.strftime("%A, %B %d, %Y at %H:%M")

    # Build locale-aware local date string
    try:
        import locale as _locale
        saved = _locale.getlocale(_locale.LC_TIME)
        try:
            _locale.setlocale(_locale.LC_TIME, "")
            local_day = now.strftime("%A")
        finally:
            _locale.setlocale(_locale.LC_TIME, saved)
    except Exception:
        local_day = now.strftime("%A")

    hebrew_date = f"{local_day}, {now.day}/{now.month}/{now.year} {now.strftime('%H:%M')}"

    return _INSIGHT_SYSTEM_PROMPT_TEMPLATE.format(
        current_datetime=current_datetime,
        hebrew_date=hebrew_date,
    )


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
        f"{task_prompt}"
    )


# =========================================================================
# LLM-based prompt decomposition for custom (non-template) prompts
# =========================================================================

def _decompose_prompt_with_llm(prompt: str) -> List[str]:
    """Use a fast LLM to decompose a complex insight prompt into targeted sub-queries.

    Only called for custom prompts that don't have predefined sub_queries
    in their template. Uses gpt-4o-mini for speed and cost.

    Args:
        prompt: The user-defined insight prompt

    Returns:
        List of 3-6 targeted search sub-queries, or [prompt] on failure
    """
    try:
        from config import settings
        from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI

        decomposition_prompt = (
            "You are a search query planner. Given an insight query, break it down "
            "into 3-6 specific, targeted search queries that a knowledge-base "
            "retriever can use to find relevant messages and documents.\n\n"
            "Each sub-query should:\n"
            "- Focus on ONE specific aspect of the original query\n"
            "- Use concrete keywords a message search would match\n"
            "- Be in the same language as the original query\n"
            "- Avoid vague terms — prefer specific actions, names, topics\n\n"
            f"Insight query:\n{prompt}\n\n"
            "Return ONLY a JSON array of strings, nothing else. Example:\n"
            '["meetings appointments scheduled for today or tomorrow",\n'
            ' "promises commitments I agreed to do",\n'
            ' "deadlines tasks due soon"]'
        )

        # Use a fast, cheap model for decomposition
        decompose_model = settings.get("insight_decompose_model", "gpt-4o-mini")
        llm = LlamaIndexOpenAI(
            api_key=settings.openai_api_key,
            model=decompose_model,
            temperature=0.0,
        )

        response = llm.complete(decomposition_prompt)
        text = str(response).strip()

        # Parse JSON array from response
        # Handle cases where the response might have markdown code fences
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        sub_queries = json.loads(text)
        if isinstance(sub_queries, list) and len(sub_queries) >= 2:
            logger.info(
                f"[insight] LLM decomposed prompt into {len(sub_queries)} sub-queries: "
                f"{sub_queries}"
            )
            return sub_queries[:8]  # Cap at 8 sub-queries

    except Exception as e:
        logger.warning(f"[insight] LLM decomposition failed, using original prompt: {e}")

    return [prompt]


# =========================================================================
# Reasoning model support
# =========================================================================

def _get_insight_llm():
    """Get the LLM configured for insight analysis.

    Checks the insight_llm_model setting — if set, creates a separate
    LLM instance for insights (e.g., o3-mini for reasoning). Otherwise
    falls back to the main LLM.

    Reasoning models (o3-mini, o4-mini) don't support temperature and
    use internal chain-of-thought automatically.

    Returns:
        LLM instance for insight execution
    """
    from config import settings
    from llama_index.core import Settings as LISettings

    insight_model = settings.get("insight_llm_model", "")
    if not insight_model:
        return LISettings.llm

    try:
        temperature = float(settings.get("insight_llm_temperature", "0.1"))

        llm_provider = settings.get("llm_provider", "openai").lower()

        if llm_provider == "gemini":
            try:
                from llama_index.llms.gemini import Gemini
                return Gemini(
                    api_key=settings.google_api_key,
                    model=insight_model,
                    temperature=temperature,
                )
            except ImportError:
                logger.warning("Gemini LLM not available for insights, falling back to OpenAI")

        from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI

        # Reasoning models (o1, o3, o4 family) don't support temperature
        is_reasoning = any(
            insight_model.startswith(prefix)
            for prefix in ("o1", "o3", "o4")
        )

        if is_reasoning:
            llm = LlamaIndexOpenAI(
                api_key=settings.openai_api_key,
                model=insight_model,
            )
            logger.info(f"[insight] Using reasoning model: {insight_model}")
        else:
            llm = LlamaIndexOpenAI(
                api_key=settings.openai_api_key,
                model=insight_model,
                temperature=temperature,
            )
            logger.info(f"[insight] Using model: {insight_model} (temp={temperature})")

        return llm

    except Exception as e:
        logger.warning(f"[insight] Failed to create insight LLM ({insight_model}): {e}")
        return LISettings.llm


# =========================================================================
# Template sub-query lookup
# =========================================================================

def _get_template_sub_queries(task_name: str) -> Optional[List[str]]:
    """Look up predefined sub-queries for a built-in template by name.

    Args:
        task_name: The task name to match against templates

    Returns:
        List of sub-query strings, or None if no template matches
    """
    try:
        import scheduled_tasks_db
        templates = scheduled_tasks_db.get_templates()
        for tpl in templates:
            if tpl["name"] == task_name:
                return tpl.get("sub_queries")
    except Exception:
        pass
    return None


# =========================================================================
# Celery Tasks
# =========================================================================

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
    soft_time_limit=180,
    time_limit=240,
)
def execute_scheduled_insight(self, task_id: int) -> dict:
    """Execute a single scheduled insight query through the multi-query RAG pipeline.

    Quality pipeline:
        1. Load task definition and resolve sub-queries
        2. Build insight-specific system prompt (no chat noise)
        3. Resolve sub-queries: template-defined → LLM decomposition → fallback
        4. Execute multi-query retrieval via rag.execute_insight_query()
        5. Use insight LLM (potentially a reasoning model like o3-mini)
        6. Record cost via METER snapshot delta
        7. Store result with quality metrics

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
        from config import settings

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

        # 2. Build the insight-specific system prompt
        timezone = task.get("timezone", "Asia/Jerusalem")
        system_prompt = _build_insight_system_prompt(timezone)

        # 3. Build the effective user prompt with date context
        prompt = _build_insight_prompt(task["prompt"], timezone)

        # 4. Parse filters
        filters = task.get("filters", {})
        if isinstance(filters, str):
            try:
                filters = json.loads(filters)
            except (json.JSONDecodeError, TypeError):
                filters = {}

        # 5. Resolve sub-queries
        #    Priority: template sub-queries > LLM decomposition > single prompt
        sub_queries = _get_template_sub_queries(task["name"])

        if not sub_queries:
            # Check if LLM decomposition is enabled
            decompose_enabled = settings.get(
                "insight_decompose_with_llm", "true"
            ).lower() == "true"

            if decompose_enabled and len(task["prompt"]) > 50:
                sub_queries = _decompose_prompt_with_llm(task["prompt"])
            else:
                sub_queries = None  # Will use prompt as single query

        # 6. Get insight-specific settings
        k = int(filters.get("k", settings.get("insight_default_k", "20")))
        max_context_tokens = int(
            settings.get("insight_max_context_tokens", "8000")
        )

        # 7. Get the insight LLM (may be a reasoning model)
        insight_llm = _get_insight_llm()

        # 8. Execute multi-query insight via RAG
        rag = get_rag()

        cost_snapshot = METER.snapshot()

        result = rag.execute_insight_query(
            prompt=prompt,
            system_prompt=system_prompt,
            sub_queries=sub_queries,
            filter_chat_name=filters.get("chat_name"),
            filter_sender=filters.get("sender"),
            filter_days=int(filters["days"]) if filters.get("days") else None,
            filter_sources=filters.get("sources"),
            filter_date_from=filters.get("date_from"),
            filter_date_to=filters.get("date_to"),
            filter_content_types=filters.get("content_types"),
            k=k,
            max_context_tokens=max_context_tokens,
            llm_override=insight_llm,
        )

        answer = result["answer"]
        sources = result["sources"]
        quality_metrics = result["quality_metrics"]

        query_cost = METER.session_total - cost_snapshot
        duration_ms = int(time.time() * 1000) - start_ms

        # Add execution metadata to quality metrics
        quality_metrics["duration_ms"] = duration_ms
        quality_metrics["cost_usd"] = round(query_cost, 6)

        # 9. Determine status
        status = "success"
        if not answer or answer.strip() == "Empty Response":
            status = "no_results"
            answer = "No relevant information found for this query."

        # 10. Store result with quality metrics
        scheduled_tasks_db.add_result(
            task_id=task_id,
            answer=answer,
            prompt_used=prompt,
            sources=sources,
            cost_usd=query_cost,
            duration_ms=duration_ms,
            status=status,
            quality_metrics=quality_metrics,
        )

        logger.info(
            f"[insight] Task #{task_id} completed: "
            f"status={status}, cost=${query_cost:.4f}, "
            f"duration={duration_ms}ms, sources={len(sources)}, "
            f"sub_queries={quality_metrics.get('sub_queries_used', 1)}, "
            f"model={quality_metrics.get('model_used', 'default')}"
        )

        return {
            "status": status,
            "task_id": task_id,
            "task_name": task["name"],
            "cost_usd": round(query_cost, 6),
            "duration_ms": duration_ms,
            "source_count": len(sources),
            "quality_metrics": quality_metrics,
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
            "context_length_exceeded",
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
