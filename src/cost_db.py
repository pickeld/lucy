"""SQLite persistence for LLM cost events.

Stores cost events in the shared settings.db database. Provides
functions for inserting events and querying aggregates (daily totals,
breakdowns by kind/model, etc.).

Database location: data/settings.db (shared with settings_db and conversations_db)
"""

import sqlite3
from typing import Any, Dict, List, Optional

from settings_db import DB_PATH


# ---------------------------------------------------------------------------
# Database connection helper
# ---------------------------------------------------------------------------

def _get_connection() -> sqlite3.Connection:
    """Get a new SQLite connection (connection-per-request for thread safety).

    Returns:
        sqlite3.Connection with row_factory set to sqlite3.Row
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_cost_db() -> None:
    """Create the cost_events table if it doesn't exist."""
    conn = _get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cost_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                kind TEXT NOT NULL,
                in_tokens INTEGER DEFAULT 0,
                out_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                cost_usd REAL DEFAULT 0.0,
                conversation_id TEXT DEFAULT '',
                request_context TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cost_events_ts
                ON cost_events(ts)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cost_events_conv
                ON cost_events(conversation_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cost_events_kind
                ON cost_events(kind)
        """)
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Insert
# ---------------------------------------------------------------------------

def insert_cost_event(
    ts: float,
    provider: str,
    model: str,
    kind: str,
    in_tokens: int = 0,
    out_tokens: int = 0,
    total_tokens: int = 0,
    cost_usd: float = 0.0,
    conversation_id: str = "",
    request_context: str = "",
) -> int:
    """Insert a single cost event.

    Args:
        ts: Unix timestamp of the event
        provider: Provider name (e.g., "openai", "gemini")
        model: Model name (e.g., "gpt-4o", "text-embedding-3-large")
        kind: Event kind ("chat", "embed", "whisper", "image")
        in_tokens: Input/prompt token count
        out_tokens: Output/completion token count
        total_tokens: Total tokens (for embeddings)
        cost_usd: Calculated cost in USD
        conversation_id: Associated conversation UUID (if any)
        request_context: Context label (e.g., "rag_query", "image_describe")

    Returns:
        Row ID of the inserted event
    """
    conn = _get_connection()
    try:
        cursor = conn.execute(
            """INSERT INTO cost_events
               (ts, provider, model, kind, in_tokens, out_tokens, total_tokens,
                cost_usd, conversation_id, request_context)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ts, provider, model, kind, in_tokens, out_tokens, total_tokens,
             cost_usd, conversation_id, request_context),
        )
        conn.commit()
        return cursor.lastrowid or 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def get_events(
    limit: int = 50,
    offset: int = 0,
    conversation_id: Optional[str] = None,
    kind: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Get cost events with optional filters, most recent first.

    Args:
        limit: Maximum events to return
        offset: Number of events to skip
        conversation_id: Filter by conversation (if provided)
        kind: Filter by event kind (if provided)

    Returns:
        List of event dicts
    """
    conn = _get_connection()
    try:
        conditions = []
        params: list = []

        if conversation_id:
            conditions.append("conversation_id = ?")
            params.append(conversation_id)
        if kind:
            conditions.append("kind = ?")
            params.append(kind)

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        params.extend([limit, offset])
        rows = conn.execute(
            f"""SELECT * FROM cost_events
                {where}
                ORDER BY ts DESC
                LIMIT ? OFFSET ?""",
            params,
        ).fetchall()

        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_total_cost(days: Optional[int] = None) -> float:
    """Get the total cost, optionally limited to the last N days.

    Args:
        days: If provided, only count events from the last N days

    Returns:
        Total cost in USD
    """
    conn = _get_connection()
    try:
        if days is not None and days > 0:
            import time
            min_ts = time.time() - (days * 86400)
            row = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) as total FROM cost_events WHERE ts >= ?",
                (min_ts,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) as total FROM cost_events"
            ).fetchone()

        return float(row["total"]) if row else 0.0
    finally:
        conn.close()


def get_daily_summary(days: int = 7) -> List[Dict[str, Any]]:
    """Get daily cost totals for the last N days.

    Args:
        days: Number of days to include

    Returns:
        List of dicts: {date, total_cost, event_count, total_tokens}
    """
    import time

    conn = _get_connection()
    try:
        min_ts = time.time() - (days * 86400)
        rows = conn.execute(
            """SELECT
                DATE(created_at) as date,
                SUM(cost_usd) as total_cost,
                COUNT(*) as event_count,
                SUM(in_tokens + out_tokens + total_tokens) as total_tokens
               FROM cost_events
               WHERE ts >= ?
               GROUP BY DATE(created_at)
               ORDER BY date DESC""",
            (min_ts,),
        ).fetchall()

        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_cost_by_kind(days: Optional[int] = None) -> Dict[str, float]:
    """Get cost breakdown by event kind (chat, embed, whisper, image).

    Args:
        days: If provided, only count events from the last N days

    Returns:
        Dict of kind -> total cost USD
    """
    import time

    conn = _get_connection()
    try:
        if days is not None and days > 0:
            min_ts = time.time() - (days * 86400)
            rows = conn.execute(
                """SELECT kind, SUM(cost_usd) as total
                   FROM cost_events WHERE ts >= ?
                   GROUP BY kind""",
                (min_ts,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT kind, SUM(cost_usd) as total FROM cost_events GROUP BY kind"
            ).fetchall()

        return {row["kind"]: float(row["total"]) for row in rows}
    finally:
        conn.close()


def get_cost_by_model(days: Optional[int] = None) -> List[Dict[str, Any]]:
    """Get cost breakdown by provider:model.

    Args:
        days: If provided, only count events from the last N days

    Returns:
        List of dicts: {provider, model, kind, total_cost, event_count}
    """
    import time

    conn = _get_connection()
    try:
        if days is not None and days > 0:
            min_ts = time.time() - (days * 86400)
            rows = conn.execute(
                """SELECT provider, model, kind,
                          SUM(cost_usd) as total_cost,
                          COUNT(*) as event_count
                   FROM cost_events WHERE ts >= ?
                   GROUP BY provider, model, kind
                   ORDER BY total_cost DESC""",
                (min_ts,),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT provider, model, kind,
                          SUM(cost_usd) as total_cost,
                          COUNT(*) as event_count
                   FROM cost_events
                   GROUP BY provider, model, kind
                   ORDER BY total_cost DESC"""
            ).fetchall()

        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_conversation_cost(conversation_id: str) -> float:
    """Get the total cost for a specific conversation.

    Args:
        conversation_id: The conversation UUID

    Returns:
        Total cost in USD for that conversation
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) as total FROM cost_events WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        return float(row["total"]) if row else 0.0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Module-level initialization
# ---------------------------------------------------------------------------

# Auto-initialize on import
init_cost_db()
