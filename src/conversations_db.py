"""SQLite-backed conversation persistence for Lucy.

Stores conversation metadata and message history in SQLite for the
"previous chats" feature. SQLite is the source of truth — if Redis
chat memory expires, it is automatically restored from here.

Database location: data/settings.db (shared with settings_db module)
"""

import json
import sqlite3
from datetime import datetime
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
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_conversations_db() -> None:
    """Create the conversations and conversation_messages tables if they don't exist."""
    conn = _get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                filters TEXT DEFAULT '{}',
                message_count INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conv_messages_conv_id
                ON conversation_messages(conversation_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_updated
                ON conversations(updated_at DESC)
        """)
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------

def conversation_exists(conversation_id: str) -> bool:
    """Check if a conversation exists in SQLite.

    Args:
        conversation_id: The conversation UUID

    Returns:
        True if the conversation exists
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def create_conversation(
    conversation_id: str,
    title: str = "",
    filters: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Create a new conversation record.

    Args:
        conversation_id: UUID for the conversation
        title: Display title (auto-generated from first message if empty)
        filters: Optional active filters dict

    Returns:
        Dict with the created conversation data
    """
    filters_json = json.dumps(filters or {})
    conn = _get_connection()
    try:
        conn.execute(
            """INSERT INTO conversations (id, title, filters)
               VALUES (?, ?, ?)""",
            (conversation_id, title, filters_json),
        )
        conn.commit()
        return {
            "id": conversation_id,
            "title": title,
            "filters": filters or {},
            "message_count": 0,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
    finally:
        conn.close()


def get_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
    """Get a conversation with all its messages.

    Args:
        conversation_id: The conversation UUID

    Returns:
        Dict with conversation metadata and messages list, or None
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if not row:
            return None

        messages = conn.execute(
            """SELECT role, content, created_at
               FROM conversation_messages
               WHERE conversation_id = ?
               ORDER BY id ASC""",
            (conversation_id,),
        ).fetchall()

        return {
            "id": row["id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "filters": json.loads(row["filters"]) if row["filters"] else {},
            "message_count": row["message_count"],
            "messages": [
                {
                    "role": m["role"],
                    "content": m["content"],
                    "created_at": m["created_at"],
                }
                for m in messages
            ],
        }
    finally:
        conn.close()


def list_conversations(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    """List conversations sorted by most recently updated.

    Args:
        limit: Maximum number of conversations to return
        offset: Number of conversations to skip

    Returns:
        List of conversation summary dicts (without messages)
    """
    conn = _get_connection()
    try:
        rows = conn.execute(
            """SELECT id, title, created_at, updated_at, filters, message_count
               FROM conversations
               ORDER BY updated_at DESC
               LIMIT ? OFFSET ?""",
            (limit, offset),
        ).fetchall()

        return [
            {
                "id": row["id"],
                "title": row["title"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "filters": json.loads(row["filters"]) if row["filters"] else {},
                "message_count": row["message_count"],
            }
            for row in rows
        ]
    finally:
        conn.close()


def update_conversation_title(conversation_id: str, title: str) -> bool:
    """Update a conversation's title.

    Args:
        conversation_id: The conversation UUID
        title: New title string

    Returns:
        True if the conversation was found and updated
    """
    conn = _get_connection()
    try:
        cursor = conn.execute(
            """UPDATE conversations
               SET title = ?, updated_at = CURRENT_TIMESTAMP
               WHERE id = ?""",
            (title, conversation_id),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def update_conversation_filters(conversation_id: str, filters: Dict[str, str]) -> bool:
    """Update a conversation's stored filters.

    Args:
        conversation_id: The conversation UUID
        filters: New filters dict

    Returns:
        True if the conversation was found and updated
    """
    conn = _get_connection()
    try:
        cursor = conn.execute(
            """UPDATE conversations
               SET filters = ?, updated_at = CURRENT_TIMESTAMP
               WHERE id = ?""",
            (json.dumps(filters), conversation_id),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def add_message(conversation_id: str, role: str, content: str) -> Dict[str, Any]:
    """Add a message to a conversation and update metadata.

    Args:
        conversation_id: The conversation UUID
        role: 'user' or 'assistant'
        content: The message content

    Returns:
        Dict with the created message data
    """
    conn = _get_connection()
    try:
        conn.execute(
            """INSERT INTO conversation_messages (conversation_id, role, content)
               VALUES (?, ?, ?)""",
            (conversation_id, role, content),
        )
        conn.execute(
            """UPDATE conversations
               SET message_count = message_count + 1,
                   updated_at = CURRENT_TIMESTAMP
               WHERE id = ?""",
            (conversation_id,),
        )
        conn.commit()
        return {
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "created_at": datetime.utcnow().isoformat(),
        }
    finally:
        conn.close()


def get_messages(
    conversation_id: str, limit: Optional[int] = None
) -> List[Dict[str, str]]:
    """Get messages for a conversation, optionally limited to the last N.

    Args:
        conversation_id: The conversation UUID
        limit: If set, return only the last N messages

    Returns:
        List of message dicts with role and content
    """
    conn = _get_connection()
    try:
        if limit:
            rows = conn.execute(
                """SELECT role, content FROM (
                       SELECT role, content, id FROM conversation_messages
                       WHERE conversation_id = ?
                       ORDER BY id DESC
                       LIMIT ?
                   ) sub ORDER BY id ASC""",
                (conversation_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT role, content
                   FROM conversation_messages
                   WHERE conversation_id = ?
                   ORDER BY id ASC""",
                (conversation_id,),
            ).fetchall()

        return [{"role": r["role"], "content": r["content"]} for r in rows]
    finally:
        conn.close()


def delete_conversation(conversation_id: str) -> bool:
    """Delete a conversation and all its messages.

    Args:
        conversation_id: The conversation UUID

    Returns:
        True if the conversation was found and deleted
    """
    conn = _get_connection()
    try:
        # Messages are cascade-deleted via FK constraint
        cursor = conn.execute(
            "DELETE FROM conversations WHERE id = ?", (conversation_id,)
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def _generate_title(first_message: str, max_length: int = 60) -> str:
    """Generate a conversation title from the first user message.

    Truncates to max_length characters with ellipsis if needed.

    Args:
        first_message: The first user message text
        max_length: Maximum title length

    Returns:
        Title string
    """
    title = first_message.strip()
    # Remove leading/trailing quotes if present
    if title and title[0] in ('"', "'") and title[-1] == title[0]:
        title = title[1:-1]
    if len(title) > max_length:
        title = title[:max_length - 1] + "…"
    return title or "New Chat"


# ---------------------------------------------------------------------------
# Chat Memory Restoration
# ---------------------------------------------------------------------------

def restore_chat_memory_if_needed(
    conversation_id: str,
    chat_store: Any,
    max_messages: int = 40,
) -> bool:
    """Restore chat memory from SQLite into RedisChatStore if Redis is empty.

    Checks if the RedisChatStore has messages for this conversation.
    If not, loads the last N messages from SQLite and re-injects them.

    Args:
        conversation_id: The conversation UUID (used as chat_store_key)
        chat_store: A LlamaIndex RedisChatStore instance
        max_messages: Maximum number of messages to restore

    Returns:
        True if memory was restored, False if already present or nothing to restore
    """
    try:
        # Check if Redis already has messages for this conversation
        existing = chat_store.get_messages(conversation_id)
        if existing:
            return False  # Memory is still alive in Redis

        # Load from SQLite
        messages = get_messages(conversation_id, limit=max_messages)
        if not messages:
            return False  # No history to restore

        # Re-inject into RedisChatStore as ChatMessage objects
        from llama_index.core.llms import ChatMessage, MessageRole

        chat_messages = []
        for msg in messages:
            role = MessageRole.USER if msg["role"] == "user" else MessageRole.ASSISTANT
            chat_messages.append(ChatMessage(role=role, content=msg["content"]))

        chat_store.set_messages(conversation_id, chat_messages)
        return True

    except Exception as e:
        # Non-fatal: if restoration fails, the conversation continues without history
        import logging
        logging.getLogger(__name__).warning(
            f"Failed to restore chat memory for {conversation_id}: {e}"
        )
        return False


# ---------------------------------------------------------------------------
# Module-level initialization
# ---------------------------------------------------------------------------

# Auto-initialize on import
init_conversations_db()
