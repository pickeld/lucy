"""SQLite-backed database for Scheduled Insights tasks and results.

Stores task definitions (prompt, schedule, filters) and their execution
results (answer, sources, cost).  Uses the same data/ directory as
settings_db for consistency.

Database location: data/scheduled_tasks.db
"""

import json
import re
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from utils.logger import logger


# ---------------------------------------------------------------------------
# Database path resolution (mirrors settings_db pattern)
# ---------------------------------------------------------------------------

def _resolve_db_path() -> str:
    """Resolve the SQLite database file path.

    Uses the same data/ directory as settings_db.

    Returns:
        Absolute path to scheduled_tasks.db
    """
    import os

    explicit = os.environ.get("SCHEDULED_TASKS_DB_PATH")
    if explicit:
        Path(explicit).parent.mkdir(parents=True, exist_ok=True)
        return explicit

    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "docker-compose.yml").exists() or (parent / ".git").exists():
            db_dir = parent / "data"
            db_dir.mkdir(parents=True, exist_ok=True)
            return str(db_dir / "scheduled_tasks.db")

    docker_data = Path("/app/data")
    if docker_data.exists() or Path("/app/src").exists():
        docker_data.mkdir(parents=True, exist_ok=True)
        return str(docker_data / "scheduled_tasks.db")

    db_dir = current / "data"
    db_dir.mkdir(parents=True, exist_ok=True)
    return str(db_dir / "scheduled_tasks.db")


DB_PATH = _resolve_db_path()

# Thread-local connection reuse (same pattern as settings_db)
_local = threading.local()


def _get_connection() -> sqlite3.Connection:
    """Get a thread-local SQLite connection.

    Returns:
        sqlite3.Connection with row_factory set to sqlite3.Row
    """
    conn = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.execute("SELECT 1")
            return conn
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            _local.conn = None

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _local.conn = conn
    return conn


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create tables and indexes if they don't exist."""
    conn = _get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS scheduled_tasks (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            name            TEXT NOT NULL,
            description     TEXT DEFAULT '',
            prompt          TEXT NOT NULL,
            schedule_type   TEXT NOT NULL DEFAULT 'daily',
            schedule_value  TEXT NOT NULL DEFAULT '08:00',
            timezone        TEXT DEFAULT 'Asia/Jerusalem',
            enabled         INTEGER DEFAULT 1,
            filters         TEXT DEFAULT '{}',
            delivery_channel TEXT DEFAULT 'ui',
            created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at      TEXT DEFAULT CURRENT_TIMESTAMP,
            last_run_at     TEXT,
            next_run_at     TEXT
        );

        CREATE TABLE IF NOT EXISTS task_results (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id         INTEGER NOT NULL REFERENCES scheduled_tasks(id) ON DELETE CASCADE,
            answer          TEXT NOT NULL,
            prompt_used     TEXT NOT NULL DEFAULT '',
            sources         TEXT DEFAULT '[]',
            cost_usd        REAL DEFAULT 0,
            duration_ms     INTEGER DEFAULT 0,
            status          TEXT DEFAULT 'success',
            error_message   TEXT,
            executed_at     TEXT DEFAULT CURRENT_TIMESTAMP,
            quality_metrics TEXT DEFAULT '{}',
            rating          INTEGER DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_task_results_task_id
            ON task_results(task_id);
        CREATE INDEX IF NOT EXISTS idx_task_results_executed_at
            ON task_results(executed_at);
        CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_enabled
            ON scheduled_tasks(enabled);
        CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_next_run
            ON scheduled_tasks(next_run_at);
    """)
    # Migrate: add quality_metrics and rating columns if missing
    _migrate_add_column(conn, "task_results", "quality_metrics", "TEXT DEFAULT '{}'")
    _migrate_add_column(conn, "task_results", "rating", "INTEGER DEFAULT 0")
    conn.commit()
    logger.info(f"Scheduled tasks DB initialized at {DB_PATH}")


def _migrate_add_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    """Add a column to an existing table if it doesn't exist (safe migration)."""
    try:
        cursor = conn.execute(f"PRAGMA table_info({table})")
        columns = {row[1] for row in cursor.fetchall()}
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
            logger.info(f"Migrated: added {column} to {table}")
    except Exception as e:
        logger.debug(f"Migration check for {table}.{column} failed: {e}")


# Auto-init on first import
init_db()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert a sqlite3.Row to a plain dict."""
    return dict(row)


def _now_iso() -> str:
    """Current UTC time as ISO string."""
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Schedule computation
# ---------------------------------------------------------------------------

# Weekday name → Python weekday int mapping
_WEEKDAY_MAP = {
    "mon": 0, "tue": 1, "wed": 2, "thu": 3,
    "fri": 4, "sat": 5, "sun": 6,
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


def compute_next_run(
    schedule_type: str,
    schedule_value: str,
    timezone: str = "Asia/Jerusalem",
    from_time: Optional[datetime] = None,
) -> Optional[str]:
    """Compute the next run time for a task.

    Args:
        schedule_type: One of 'daily', 'weekly', 'monthly', 'interval', 'cron'
        schedule_value: Schedule-specific value (see plan docs)
        timezone: IANA timezone string
        from_time: Base time to compute from (default: now)

    Returns:
        ISO datetime string for next run, or None on parse error
    """
    try:
        tz = ZoneInfo(timezone)
        now = from_time or datetime.now(tz)
        if now.tzinfo is None:
            now = now.replace(tzinfo=tz)

        if schedule_type == "daily":
            return _next_daily(schedule_value, now, tz)
        elif schedule_type == "weekly":
            return _next_weekly(schedule_value, now, tz)
        elif schedule_type == "monthly":
            return _next_monthly(schedule_value, now, tz)
        elif schedule_type == "interval":
            return _next_interval(schedule_value, now, tz)
        elif schedule_type == "cron":
            return _next_cron(schedule_value, now, tz)
        else:
            logger.warning(f"Unknown schedule_type: {schedule_type}")
            return None
    except Exception as e:
        logger.error(f"Failed to compute next run ({schedule_type}={schedule_value}): {e}")
        return None


def _next_daily(value: str, now: datetime, tz: ZoneInfo) -> str:
    """Compute next daily run. value = 'HH:MM'."""
    hour, minute = map(int, value.strip().split(":"))
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate.strftime("%Y-%m-%d %H:%M:%S")


def _next_weekly(value: str, now: datetime, tz: ZoneInfo) -> str:
    """Compute next weekly run. value = 'day1,day2 HH:MM'."""
    parts = value.strip().rsplit(" ", 1)
    if len(parts) == 2:
        days_str, time_str = parts
    else:
        days_str = parts[0]
        time_str = "08:00"

    hour, minute = map(int, time_str.split(":"))
    target_weekdays = []
    for d in days_str.split(","):
        d = d.strip().lower()
        if d in _WEEKDAY_MAP:
            target_weekdays.append(_WEEKDAY_MAP[d])

    if not target_weekdays:
        target_weekdays = [0]  # Default to Monday

    # Find the next matching weekday
    best = None
    for offset in range(8):  # Check next 7 days + today
        candidate = now + timedelta(days=offset)
        if candidate.weekday() in target_weekdays:
            candidate = candidate.replace(
                hour=hour, minute=minute, second=0, microsecond=0,
            )
            if candidate > now:
                if best is None or candidate < best:
                    best = candidate
                break  # First future match wins

    if best is None:
        # Fallback: tomorrow
        best = now + timedelta(days=1)
        best = best.replace(hour=hour, minute=minute, second=0, microsecond=0)

    return best.strftime("%Y-%m-%d %H:%M:%S")


def _next_monthly(value: str, now: datetime, tz: ZoneInfo) -> str:
    """Compute next monthly run. value = 'DD HH:MM'."""
    parts = value.strip().split(" ")
    day = int(parts[0])
    time_str = parts[1] if len(parts) > 1 else "08:00"
    hour, minute = map(int, time_str.split(":"))

    # Try this month
    try:
        candidate = now.replace(
            day=day, hour=hour, minute=minute, second=0, microsecond=0,
        )
        if candidate > now:
            return candidate.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        pass  # Day doesn't exist in current month

    # Next month
    if now.month == 12:
        next_month = now.replace(year=now.year + 1, month=1, day=1)
    else:
        next_month = now.replace(month=now.month + 1, day=1)

    try:
        candidate = next_month.replace(
            day=day, hour=hour, minute=minute, second=0, microsecond=0,
        )
    except ValueError:
        # Day doesn't exist in next month either — use last day
        import calendar
        last_day = calendar.monthrange(next_month.year, next_month.month)[1]
        candidate = next_month.replace(
            day=last_day, hour=hour, minute=minute, second=0, microsecond=0,
        )

    return candidate.strftime("%Y-%m-%d %H:%M:%S")


def _next_interval(value: str, now: datetime, tz: ZoneInfo) -> str:
    """Compute next interval run. value = 'Nm', 'Nh', or 'Nd'."""
    match = re.match(r"^(\d+)([mhd])$", value.strip().lower())
    if not match:
        # Fallback: 30 minutes
        return (now + timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S")

    amount = int(match.group(1))
    unit = match.group(2)

    if unit == "m":
        delta = timedelta(minutes=amount)
    elif unit == "h":
        delta = timedelta(hours=amount)
    elif unit == "d":
        delta = timedelta(days=amount)
    else:
        delta = timedelta(minutes=30)

    return (now + delta).strftime("%Y-%m-%d %H:%M:%S")


def _next_cron(value: str, now: datetime, tz: ZoneInfo) -> str:
    """Compute next cron run. value = standard 5-field cron expression.

    Uses a simple brute-force approach: iterate minute-by-minute for
    up to 2 days to find the next match. For most cron expressions
    this finds a match within seconds.
    """
    fields = value.strip().split()
    if len(fields) != 5:
        logger.warning(f"Invalid cron expression (expected 5 fields): {value}")
        return (now + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")

    minute_spec, hour_spec, dom_spec, month_spec, dow_spec = fields

    def _matches_field(spec: str, val: int, max_val: int) -> bool:
        """Check if a value matches a cron field spec."""
        if spec == "*":
            return True
        for part in spec.split(","):
            if "-" in part:
                lo, hi = part.split("-", 1)
                if int(lo) <= val <= int(hi):
                    return True
            elif "/" in part:
                base, step = part.split("/", 1)
                base_val = 0 if base == "*" else int(base)
                if (val - base_val) % int(step) == 0 and val >= base_val:
                    return True
            else:
                if val == int(part):
                    return True
        return False

    # Start from next minute
    candidate = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
    # Search up to 2 days ahead
    limit = now + timedelta(days=2)

    while candidate < limit:
        # cron dow: 0=Sunday, Python weekday: 0=Monday
        cron_dow = (candidate.weekday() + 1) % 7  # Convert to cron convention

        if (
            _matches_field(minute_spec, candidate.minute, 59)
            and _matches_field(hour_spec, candidate.hour, 23)
            and _matches_field(dom_spec, candidate.day, 31)
            and _matches_field(month_spec, candidate.month, 12)
            and _matches_field(dow_spec, cron_dow, 6)
        ):
            return candidate.strftime("%Y-%m-%d %H:%M:%S")

        candidate += timedelta(minutes=1)

    # Fallback if no match found
    return (now + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# CRUD — Tasks
# ---------------------------------------------------------------------------

def create_task(
    name: str,
    prompt: str,
    schedule_type: str = "daily",
    schedule_value: str = "08:00",
    timezone: str = "Asia/Jerusalem",
    description: str = "",
    filters: Optional[Dict[str, Any]] = None,
    delivery_channel: str = "ui",
    enabled: bool = True,
) -> Dict[str, Any]:
    """Create a new scheduled task.

    Args:
        name: Human-readable task name
        prompt: The query/prompt to run against the RAG system
        schedule_type: 'daily', 'weekly', 'monthly', 'interval', 'cron'
        schedule_value: Schedule-specific value
        timezone: IANA timezone
        description: Optional longer description
        filters: Optional RAG filters (chat_name, sender, days, sources, etc.)
        delivery_channel: 'ui' for now (future: 'whatsapp', 'email', etc.)
        enabled: Whether the task starts enabled

    Returns:
        The created task as a dict
    """
    conn = _get_connection()
    now = _now_iso()
    filters_json = json.dumps(filters or {})

    next_run = compute_next_run(schedule_type, schedule_value, timezone) if enabled else None

    cursor = conn.execute(
        """
        INSERT INTO scheduled_tasks
            (name, description, prompt, schedule_type, schedule_value,
             timezone, enabled, filters, delivery_channel,
             created_at, updated_at, next_run_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            name, description, prompt, schedule_type, schedule_value,
            timezone, int(enabled), filters_json, delivery_channel,
            now, now, next_run,
        ),
    )
    conn.commit()

    task_id = cursor.lastrowid
    logger.info(f"Created scheduled task #{task_id}: '{name}' ({schedule_type}={schedule_value})")
    task = get_task(task_id)  # type: ignore[arg-type]
    assert task is not None, f"Task #{task_id} not found immediately after creation"
    return task


def get_task(task_id: int) -> Optional[Dict[str, Any]]:
    """Get a task by ID, including its latest result.

    Returns:
        Task dict with 'latest_result' key, or None if not found
    """
    conn = _get_connection()
    row = conn.execute(
        "SELECT * FROM scheduled_tasks WHERE id = ?", (task_id,)
    ).fetchone()

    if not row:
        return None

    task = _row_to_dict(row)
    task["enabled"] = bool(task["enabled"])

    # Parse filters JSON
    try:
        task["filters"] = json.loads(task.get("filters") or "{}")
    except (json.JSONDecodeError, TypeError):
        task["filters"] = {}

    # Attach latest result
    result_row = conn.execute(
        """
        SELECT * FROM task_results
        WHERE task_id = ?
        ORDER BY executed_at DESC
        LIMIT 1
        """,
        (task_id,),
    ).fetchone()

    if result_row:
        result = _row_to_dict(result_row)
        try:
            result["sources"] = json.loads(result.get("sources") or "[]")
        except (json.JSONDecodeError, TypeError):
            result["sources"] = []
        task["latest_result"] = result
    else:
        task["latest_result"] = None

    return task


def list_tasks(include_disabled: bool = True) -> List[Dict[str, Any]]:
    """List all tasks with their latest result.

    Args:
        include_disabled: If False, only return enabled tasks

    Returns:
        List of task dicts, sorted by created_at descending
    """
    conn = _get_connection()
    if include_disabled:
        rows = conn.execute(
            "SELECT * FROM scheduled_tasks ORDER BY created_at DESC"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM scheduled_tasks WHERE enabled = 1 ORDER BY created_at DESC"
        ).fetchall()

    tasks = []
    for row in rows:
        task = _row_to_dict(row)
        task["enabled"] = bool(task["enabled"])
        try:
            task["filters"] = json.loads(task.get("filters") or "{}")
        except (json.JSONDecodeError, TypeError):
            task["filters"] = {}

        # Attach latest result
        result_row = conn.execute(
            """
            SELECT * FROM task_results
            WHERE task_id = ?
            ORDER BY executed_at DESC
            LIMIT 1
            """,
            (task["id"],),
        ).fetchone()

        if result_row:
            result = _row_to_dict(result_row)
            try:
                result["sources"] = json.loads(result.get("sources") or "[]")
            except (json.JSONDecodeError, TypeError):
                result["sources"] = []
            task["latest_result"] = result
        else:
            task["latest_result"] = None

        tasks.append(task)

    return tasks


def update_task(task_id: int, **fields) -> Optional[Dict[str, Any]]:
    """Update a task's fields.

    Allowed fields: name, description, prompt, schedule_type, schedule_value,
    timezone, enabled, filters, delivery_channel

    Returns:
        Updated task dict, or None if not found
    """
    allowed = {
        "name", "description", "prompt", "schedule_type", "schedule_value",
        "timezone", "enabled", "filters", "delivery_channel",
    }
    updates = {k: v for k, v in fields.items() if k in allowed}

    if not updates:
        return get_task(task_id)

    # Serialize filters to JSON
    if "filters" in updates and isinstance(updates["filters"], dict):
        updates["filters"] = json.dumps(updates["filters"])

    # Convert enabled bool to int
    if "enabled" in updates:
        updates["enabled"] = int(updates["enabled"])

    conn = _get_connection()

    # Check existence
    existing = conn.execute(
        "SELECT * FROM scheduled_tasks WHERE id = ?", (task_id,)
    ).fetchone()
    if not existing:
        return None

    # Build SET clause
    set_parts = [f"{k} = ?" for k in updates.keys()]
    set_parts.append("updated_at = ?")
    values = list(updates.values()) + [_now_iso(), task_id]

    conn.execute(
        f"UPDATE scheduled_tasks SET {', '.join(set_parts)} WHERE id = ?",
        values,
    )
    conn.commit()

    # Recompute next_run if schedule changed
    task = get_task(task_id)
    if task and any(k in fields for k in ("schedule_type", "schedule_value", "timezone", "enabled")):
        if task["enabled"]:
            next_run = compute_next_run(
                task["schedule_type"], task["schedule_value"], task["timezone"],
            )
            update_next_run(task_id, next_run)
            task["next_run_at"] = next_run
        else:
            update_next_run(task_id, None)
            task["next_run_at"] = None

    logger.info(f"Updated scheduled task #{task_id}: {list(updates.keys())}")
    return task


def delete_task(task_id: int) -> bool:
    """Delete a task and all its results (cascading).

    Returns:
        True if a task was deleted
    """
    conn = _get_connection()
    cursor = conn.execute(
        "DELETE FROM scheduled_tasks WHERE id = ?", (task_id,)
    )
    conn.commit()
    deleted = cursor.rowcount > 0
    if deleted:
        logger.info(f"Deleted scheduled task #{task_id}")
    return deleted


def toggle_task(task_id: int) -> Optional[bool]:
    """Toggle a task's enabled state.

    Returns:
        New enabled state, or None if task not found
    """
    conn = _get_connection()
    row = conn.execute(
        "SELECT enabled, schedule_type, schedule_value, timezone FROM scheduled_tasks WHERE id = ?",
        (task_id,),
    ).fetchone()

    if not row:
        return None

    new_enabled = not bool(row["enabled"])
    now = _now_iso()

    if new_enabled:
        next_run = compute_next_run(
            row["schedule_type"], row["schedule_value"], row["timezone"],
        )
    else:
        next_run = None

    conn.execute(
        "UPDATE scheduled_tasks SET enabled = ?, next_run_at = ?, updated_at = ? WHERE id = ?",
        (int(new_enabled), next_run, now, task_id),
    )
    conn.commit()

    logger.info(f"Toggled scheduled task #{task_id}: enabled={new_enabled}")
    return new_enabled


# ---------------------------------------------------------------------------
# CRUD — Results
# ---------------------------------------------------------------------------

def add_result(
    task_id: int,
    answer: str,
    prompt_used: str = "",
    sources: Optional[List[Dict[str, Any]]] = None,
    cost_usd: float = 0.0,
    duration_ms: int = 0,
    status: str = "success",
    error_message: Optional[str] = None,
    quality_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Record a task execution result.

    Args:
        task_id: The scheduled task ID
        answer: The RAG answer text
        prompt_used: The effective prompt that was sent
        sources: List of source dicts from RAG
        cost_usd: Query cost in USD
        duration_ms: Execution duration in milliseconds
        status: 'success', 'error', or 'no_results'
        error_message: Error details if status='error'
        quality_metrics: Optional quality metrics dict (source_count,
            unique_chats, avg_score, sub_queries_used, model_used, etc.)

    Returns:
        The created result as a dict
    """
    conn = _get_connection()
    sources_json = json.dumps(sources or [])
    metrics_json = json.dumps(quality_metrics or {})
    now = _now_iso()

    cursor = conn.execute(
        """
        INSERT INTO task_results
            (task_id, answer, prompt_used, sources, cost_usd,
             duration_ms, status, error_message, executed_at,
             quality_metrics)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            task_id, answer, prompt_used, sources_json,
            cost_usd, duration_ms, status, error_message, now,
            metrics_json,
        ),
    )

    # Update task's last_run_at
    conn.execute(
        "UPDATE scheduled_tasks SET last_run_at = ?, updated_at = ? WHERE id = ?",
        (now, now, task_id),
    )
    conn.commit()

    result_id = cursor.lastrowid
    result = get_result(result_id)  # type: ignore[arg-type]
    assert result is not None, f"Result #{result_id} not found immediately after creation"
    return result


def get_result(result_id: int) -> Optional[Dict[str, Any]]:
    """Get a single result by ID."""
    conn = _get_connection()
    row = conn.execute(
        "SELECT * FROM task_results WHERE id = ?", (result_id,)
    ).fetchone()

    if not row:
        return None

    result = _row_to_dict(row)
    try:
        result["sources"] = json.loads(result.get("sources") or "[]")
    except (json.JSONDecodeError, TypeError):
        result["sources"] = []
    try:
        result["quality_metrics"] = json.loads(result.get("quality_metrics") or "{}")
    except (json.JSONDecodeError, TypeError):
        result["quality_metrics"] = {}

    return result


def get_results(
    task_id: int,
    limit: int = 20,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Get paginated results for a task, newest first.

    Returns:
        List of result dicts
    """
    conn = _get_connection()
    rows = conn.execute(
        """
        SELECT * FROM task_results
        WHERE task_id = ?
        ORDER BY executed_at DESC
        LIMIT ? OFFSET ?
        """,
        (task_id, limit, offset),
    ).fetchall()

    results = []
    for row in rows:
        result = _row_to_dict(row)
        try:
            result["sources"] = json.loads(result.get("sources") or "[]")
        except (json.JSONDecodeError, TypeError):
            result["sources"] = []
        try:
            result["quality_metrics"] = json.loads(result.get("quality_metrics") or "{}")
        except (json.JSONDecodeError, TypeError):
            result["quality_metrics"] = {}
        results.append(result)

    return results


def rate_result(result_id: int, rating: int) -> bool:
    """Rate a result (thumbs up/down).

    Args:
        result_id: The task result ID
        rating: 1 for thumbs up, -1 for thumbs down, 0 to clear

    Returns:
        True if the result was found and updated
    """
    conn = _get_connection()
    cursor = conn.execute(
        "UPDATE task_results SET rating = ? WHERE id = ?",
        (rating, result_id),
    )
    conn.commit()
    updated = cursor.rowcount > 0
    if updated:
        logger.info(f"Rated result #{result_id}: {rating}")
    return updated


def get_result_count(task_id: int) -> int:
    """Get total number of results for a task."""
    conn = _get_connection()
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM task_results WHERE task_id = ?",
        (task_id,),
    ).fetchone()
    return row["cnt"] if row else 0


# ---------------------------------------------------------------------------
# Scheduling helpers
# ---------------------------------------------------------------------------

def update_next_run(task_id: int, next_run_at: Optional[str]) -> None:
    """Update a task's next_run_at timestamp."""
    conn = _get_connection()
    conn.execute(
        "UPDATE scheduled_tasks SET next_run_at = ? WHERE id = ?",
        (next_run_at, task_id),
    )
    conn.commit()


def get_due_tasks() -> List[Dict[str, Any]]:
    """Get all enabled tasks whose next_run_at is in the past.

    These tasks are due for execution.

    Returns:
        List of task dicts ready to be executed
    """
    conn = _get_connection()
    now = _now_iso()
    rows = conn.execute(
        """
        SELECT * FROM scheduled_tasks
        WHERE enabled = 1
          AND next_run_at IS NOT NULL
          AND next_run_at <= ?
        ORDER BY next_run_at ASC
        """,
        (now,),
    ).fetchall()

    tasks = []
    for row in rows:
        task = _row_to_dict(row)
        task["enabled"] = bool(task["enabled"])
        try:
            task["filters"] = json.loads(task.get("filters") or "{}")
        except (json.JSONDecodeError, TypeError):
            task["filters"] = {}
        tasks.append(task)

    return tasks


def advance_next_run(task_id: int) -> Optional[str]:
    """Recompute and store next_run_at for a task after execution.

    Returns:
        The new next_run_at value, or None if task not found
    """
    conn = _get_connection()
    row = conn.execute(
        "SELECT schedule_type, schedule_value, timezone FROM scheduled_tasks WHERE id = ?",
        (task_id,),
    ).fetchone()

    if not row:
        return None

    next_run = compute_next_run(
        row["schedule_type"], row["schedule_value"], row["timezone"],
    )
    update_next_run(task_id, next_run)
    return next_run


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

INSIGHT_TEMPLATES: List[Dict[str, Any]] = [
    {
        "name": "Daily Briefing",
        "icon": "☀️",
        "description": "Morning overview of your day — meetings, commitments, deadlines",
        "prompt": (
            "Produce a comprehensive daily briefing based on my messages and documents.\n\n"
            "ANALYZE each section thoroughly — list EVERY relevant item found:\n\n"
            "## 📅 Today's Schedule\n"
            "Meetings, appointments, events, calls — anything with a specific time.\n"
            "Include: what, when, with whom, any preparation needed.\n\n"
            "## ✅ Open Commitments\n"
            "Things I promised, agreed to, or said I would do.\n"
            "Include: what I committed to, who I told, when I said it, is it overdue?\n\n"
            "## ⏰ Upcoming Deadlines\n"
            "Tasks with deadlines, payment due dates, expiring documents.\n"
            "Include: what, when it's due, how urgent.\n\n"
            "## 🎂 People & Occasions\n"
            "Birthdays, anniversaries, special events for people I know.\n\n"
            "## 💬 Pending Conversations\n"
            "Messages I haven't responded to, conversations that need follow-up.\n\n"
            "For each item, cite: who mentioned it, when, in which chat.\n"
            "If a section has no relevant items, write 'Nothing found' — "
            "do NOT skip the section."
        ),
        "sub_queries": [
            "meetings appointments events calls scheduled for today or tomorrow",
            "promises commitments I said I would do agreed to follow up on",
            "deadlines due dates tasks that need to be completed soon",
            "birthdays anniversaries celebrations special occasions upcoming",
            "unanswered messages conversations needing response or follow-up",
        ],
        "schedule_type": "daily",
        "schedule_value": "08:00",
        "filters": {"days": 30},
    },
    {
        "name": "Weekly Summary",
        "icon": "📊",
        "description": "End-of-week interaction summary and highlights",
        "prompt": (
            "Produce a comprehensive weekly summary based on my messages and documents.\n\n"
            "ANALYZE each section thoroughly:\n\n"
            "## 👥 Communication Overview\n"
            "Who did I communicate with the most this week?\n"
            "List the top contacts with approximate message counts and topics.\n\n"
            "## 📝 Key Topics & Discussions\n"
            "What were the main subjects discussed across all chats?\n"
            "Group by topic, cite specific conversations.\n\n"
            "## ❓ Unresolved Conversations\n"
            "Conversations that ended without resolution or clear next steps.\n"
            "Include: who, what topic, when the conversation happened.\n\n"
            "## 🎯 Decisions Made\n"
            "Important decisions, agreements, or plans that were finalized.\n"
            "Include: what was decided, who was involved, any action items.\n\n"
            "## 📋 Open Action Items\n"
            "Things I or others agreed to do that may still be pending.\n\n"
            "For each item, cite: who, when, which chat.\n"
            "If a section has no relevant items, write 'Nothing found'."
        ),
        "sub_queries": [
            "most frequent conversations and contacts this week",
            "important topics discussions decisions made this week",
            "unresolved conversations unanswered questions pending topics",
            "agreements promises action items from this week",
            "plans meetings events scheduled for next week",
        ],
        "schedule_type": "weekly",
        "schedule_value": "fri 17:00",
        "filters": {"days": 7},
    },
    {
        "name": "Follow-up Tracker",
        "icon": "📋",
        "description": "Find things you promised to do",
        "prompt": (
            "Search my recent messages exhaustively for anything I promised, "
            "agreed to, or said I would do.\n\n"
            "Look for ALL of these patterns (in both English and Hebrew):\n"
            "- Direct promises: 'I will', 'I'll send', 'I'll check', 'let me'\n"
            "- Obligations: 'I need to', 'I should', 'I have to', 'I must'\n"
            "- Agreements: 'OK I'll do it', 'sure', 'no problem', 'I'm on it'\n"
            "- Hebrew: 'אני אעשה', 'אני צריך', 'בוא נעשה', 'אני אשלח', "
            "'אני אבדוק', 'אין בעיה', 'אני אטפל', 'מבטיח'\n\n"
            "For EACH commitment found, provide:\n"
            "1. **What** I committed to do\n"
            "2. **Who** I made the promise to\n"
            "3. **When** I said it (exact date/time from the message)\n"
            "4. **Status**: Is it likely still open or was it resolved?\n"
            "5. **Urgency**: Was a deadline mentioned?\n\n"
            "Sort by urgency (overdue first, then soonest deadline).\n"
            "Be exhaustive — better to include a marginal commitment than miss one."
        ),
        "sub_queries": [
            "I will I'll let me I need to I should I promised",
            "אני אעשה אני צריך אני אשלח אני אבדוק מבטיח אטפל",
            "sure no problem OK I'll do agreed to follow up",
            "אין בעיה בוא נעשה בסדר אני אטפל מסכים",
            "deadline due by end of week tomorrow must send",
        ],
        "schedule_type": "daily",
        "schedule_value": "09:00",
        "filters": {"days": 14},
    },
    {
        "name": "People Check-in",
        "icon": "👥",
        "description": "Who have you not talked to recently?",
        "prompt": (
            "Analyze my communication patterns to find people I may be losing touch with.\n\n"
            "## 🔍 Analysis Method\n"
            "1. Identify my regular contacts from the past 3 months\n"
            "2. Check which of these I have NOT communicated with in the past 2 weeks\n"
            "3. Flag any contacts where I usually communicate weekly but went silent\n\n"
            "## 📊 Report Format\n"
            "For each person found:\n"
            "- **Name**: Contact name\n"
            "- **Last contact**: When we last messaged (date and chat)\n"
            "- **Usual frequency**: How often we typically communicate\n"
            "- **Gap**: How long since last contact vs usual frequency\n\n"
            "Sort by gap severity (longest gap relative to usual frequency first).\n"
            "Only include people I genuinely communicate with regularly — "
            "not group chats or automated messages."
        ),
        "sub_queries": [
            "recent conversations messages from the past two weeks",
            "regular contacts I frequently message or chat with",
            "conversations and contacts from 1-3 months ago",
            "close friends family important contacts communication",
        ],
        "schedule_type": "weekly",
        "schedule_value": "sun 10:00",
        "filters": {"days": 90},
    },
]


def get_templates() -> List[Dict[str, Any]]:
    """Return the built-in insight templates."""
    return INSIGHT_TEMPLATES
