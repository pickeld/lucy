"""SQLite persistence for call recording file tracking.

Manages the lifecycle of audio files from discovery through transcription
to approval (indexing into Qdrant).  Uses the same SQLite database as
settings_db but in a dedicated ``call_recording_files`` table.

Status flow: pending → transcribing → transcribed → approved | error
"""

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DB_PATH: Optional[str] = None

# Thread-local storage for connection reuse — avoids opening/closing a fresh
# SQLite connection on every single DB call.
import threading as _threading
_local = _threading.local()


def _resolve_db_path() -> str:
    """Resolve the database path — same location as settings_db."""
    global _DB_PATH
    if _DB_PATH:
        return _DB_PATH
    _DB_PATH = os.environ.get("SETTINGS_DB_PATH", "/app/data/settings.db")
    return _DB_PATH


def _get_connection() -> sqlite3.Connection:
    """Get a thread-local SQLite connection (reused within the same thread).

    Uses threading.local() so each gunicorn/Celery worker thread gets its
    own persistent connection, eliminating the open/close overhead.
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

    conn = sqlite3.connect(_resolve_db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _local.conn = conn
    return conn


def init_table() -> None:
    """Create the call_recording_files table if it doesn't exist."""
    conn = _get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS call_recording_files (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            content_hash    TEXT UNIQUE NOT NULL,
            filename        TEXT NOT NULL,
            file_path       TEXT NOT NULL,
            file_size       INTEGER DEFAULT 0,
            extension       TEXT DEFAULT '',
            modified_at     TEXT DEFAULT '',
            status          TEXT DEFAULT 'pending',
            transcript_text TEXT DEFAULT '',
            language        TEXT DEFAULT '',
            duration_seconds INTEGER DEFAULT 0,
            confidence      REAL DEFAULT 0.0,
            participants    TEXT DEFAULT '[]',
            contact_name    TEXT DEFAULT '',
            phone_number    TEXT DEFAULT '',
            error_message   TEXT DEFAULT '',
            source_id       TEXT DEFAULT '',
            transcription_started_at TEXT DEFAULT '',
            transcription_progress   TEXT DEFAULT '',
            created_at      TEXT DEFAULT (datetime('now')),
            updated_at      TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_crf_status
        ON call_recording_files(status)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_crf_hash
        ON call_recording_files(content_hash)
    """)

    # Migrations for existing databases
    _migrate_progress_columns(conn)

    conn.commit()
    logger.info("call_recording_files table initialized")


def _migrate_progress_columns(conn: sqlite3.Connection) -> None:
    """Add transcription progress columns if missing (migration)."""
    try:
        conn.execute("SELECT transcription_started_at FROM call_recording_files LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute(
            "ALTER TABLE call_recording_files ADD COLUMN transcription_started_at TEXT DEFAULT ''"
        )
        logger.info("Migration: added transcription_started_at column")

    try:
        conn.execute("SELECT transcription_progress FROM call_recording_files LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute(
            "ALTER TABLE call_recording_files ADD COLUMN transcription_progress TEXT DEFAULT ''"
        )
        logger.info("Migration: added transcription_progress column")


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------


def upsert_file(
    content_hash: str,
    filename: str,
    file_path: str,
    file_size: int = 0,
    extension: str = "",
    modified_at: str = "",
    participants: Optional[List[str]] = None,
    contact_name: str = "",
    phone_number: str = "",
) -> Dict[str, Any]:
    """Insert a new file record or return existing one (idempotent).

    If the content_hash already exists, the existing row is returned
    unchanged (preserving any transcription results or edits).

    Returns:
        The row as a dict.
    """
    conn = _get_connection()
    # Check if already tracked
    row = conn.execute(
        "SELECT * FROM call_recording_files WHERE content_hash = ?",
        (content_hash,),
    ).fetchone()
    if row:
        return dict(row)

    participants_json = json.dumps(participants or [])
    conn.execute(
        """
        INSERT INTO call_recording_files
            (content_hash, filename, file_path, file_size, extension,
             modified_at, status, participants, contact_name, phone_number)
        VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)
        """,
        (
            content_hash,
            filename,
            file_path,
            file_size,
            extension,
            modified_at,
            participants_json,
            contact_name,
            phone_number,
        ),
    )
    conn.commit()

    row = conn.execute(
        "SELECT * FROM call_recording_files WHERE content_hash = ?",
        (content_hash,),
    ).fetchone()
    return dict(row) if row else {}


def get_file(content_hash: str) -> Optional[Dict[str, Any]]:
    """Get a single file record by content hash."""
    conn = _get_connection()
    row = conn.execute(
        "SELECT * FROM call_recording_files WHERE content_hash = ?",
        (content_hash,),
    ).fetchone()
    return dict(row) if row else None


def list_files(
    status: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """List all tracked files, optionally filtered by status.

    Returns:
        List of row dicts, ordered by created_at descending.
    """
    conn = _get_connection()
    if status:
        rows = conn.execute(
            "SELECT * FROM call_recording_files WHERE status = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (status, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM call_recording_files "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def update_status(
    content_hash: str,
    status: str,
    error_message: str = "",
) -> bool:
    """Update the status of a file record.

    When transitioning to 'transcribing', records the start timestamp.
    When transitioning away from 'transcribing', clears progress fields.

    Args:
        content_hash: File content hash
        status: New status (pending, transcribing, transcribed, approved, error)
        error_message: Error details if status=error

    Returns:
        True if row was updated.
    """
    conn = _get_connection()
    now_utc = datetime.now(timezone.utc).isoformat()

    if status == "transcribing":
        # Record when transcription started; reset progress
        cursor = conn.execute(
            """
            UPDATE call_recording_files
            SET status = ?, error_message = ?,
                transcription_started_at = ?,
                transcription_progress = 'Starting…',
                updated_at = datetime('now')
            WHERE content_hash = ?
            """,
            (status, error_message, now_utc, content_hash),
        )
    else:
        # Clear progress fields when leaving transcribing state
        cursor = conn.execute(
            """
            UPDATE call_recording_files
            SET status = ?, error_message = ?,
                transcription_progress = '',
                updated_at = datetime('now')
            WHERE content_hash = ?
            """,
            (status, error_message, content_hash),
        )
    conn.commit()
    return cursor.rowcount > 0


def update_progress(
    content_hash: str,
    progress: str,
) -> bool:
    """Update the transcription progress message for a file.

    Called periodically during transcription to report live progress
    (e.g., "Transcribing: 45s / 120s" or "Loading model…").

    Args:
        content_hash: File content hash
        progress: Human-readable progress string

    Returns:
        True if row was updated.
    """
    conn = _get_connection()
    cursor = conn.execute(
        """
        UPDATE call_recording_files
        SET transcription_progress = ?, updated_at = datetime('now')
        WHERE content_hash = ? AND status = 'transcribing'
        """,
        (progress, content_hash),
    )
    conn.commit()
    return cursor.rowcount > 0


def update_transcription(
    content_hash: str,
    transcript_text: str,
    language: str = "",
    duration_seconds: int = 0,
    confidence: float = 0.0,
    participants: Optional[List[str]] = None,
    contact_name: str = "",
) -> bool:
    """Store transcription results and set status to 'transcribed'.

    Args:
        content_hash: File content hash
        transcript_text: Full transcription text
        language: Detected language code
        duration_seconds: Audio duration
        confidence: Whisper confidence score
        participants: Auto-detected participant names
        contact_name: Auto-detected contact name

    Returns:
        True if row was updated.
    """
    participants_json = json.dumps(participants or [])
    conn = _get_connection()
    cursor = conn.execute(
        """
        UPDATE call_recording_files
        SET status = 'transcribed',
            transcript_text = ?,
            language = ?,
            duration_seconds = ?,
            confidence = ?,
            participants = ?,
            contact_name = CASE WHEN contact_name = '' THEN ? ELSE contact_name END,
            error_message = '',
            transcription_progress = '',
            updated_at = datetime('now')
        WHERE content_hash = ?
        """,
        (
            transcript_text,
            language,
            duration_seconds,
            confidence,
            participants_json,
            contact_name,
            content_hash,
        ),
    )
    conn.commit()
    return cursor.rowcount > 0


def update_metadata(
    content_hash: str,
    contact_name: Optional[str] = None,
    phone_number: Optional[str] = None,
    participants: Optional[List[str]] = None,
) -> bool:
    """Update user-editable metadata fields.

    Only updates fields that are provided (not None).

    Returns:
        True if row was updated.
    """
    updates: List[str] = []
    params: List[Any] = []

    if contact_name is not None:
        updates.append("contact_name = ?")
        params.append(contact_name)
    if phone_number is not None:
        updates.append("phone_number = ?")
        params.append(phone_number)
    if participants is not None:
        updates.append("participants = ?")
        params.append(json.dumps(participants))

    if not updates:
        return False

    updates.append("updated_at = datetime('now')")
    params.append(content_hash)

    conn = _get_connection()
    sql = f"UPDATE call_recording_files SET {', '.join(updates)} WHERE content_hash = ?"
    cursor = conn.execute(sql, params)
    conn.commit()
    return cursor.rowcount > 0


def mark_approved(content_hash: str, source_id: str) -> bool:
    """Mark a file as approved (indexed in Qdrant).

    Args:
        content_hash: File content hash
        source_id: The Qdrant source_id assigned during indexing

    Returns:
        True if row was updated.
    """
    conn = _get_connection()
    cursor = conn.execute(
        """
        UPDATE call_recording_files
        SET status = 'approved', source_id = ?, updated_at = datetime('now')
        WHERE content_hash = ?
        """,
        (source_id, content_hash),
    )
    conn.commit()
    return cursor.rowcount > 0


def delete_file(content_hash: str) -> bool:
    """Delete a file record from the tracking table.

    Returns:
        True if a row was deleted.
    """
    conn = _get_connection()
    cursor = conn.execute(
        "DELETE FROM call_recording_files WHERE content_hash = ?",
        (content_hash,),
    )
    conn.commit()
    return cursor.rowcount > 0


def get_counts() -> Dict[str, int]:
    """Get counts of files by status.

    Returns:
        Dict with status -> count, e.g. {"pending": 3, "transcribed": 5, ...}
    """
    conn = _get_connection()
    rows = conn.execute(
        "SELECT status, COUNT(*) as cnt FROM call_recording_files GROUP BY status"
    ).fetchall()
    return {r["status"]: r["cnt"] for r in rows}


def get_known_hashes() -> set:
    """Return a set of all content_hash values already tracked in the DB.

    Used by scan_and_register() to skip files that are already known
    without running expensive metadata lookups or entity resolution.
    """
    conn = _get_connection()
    rows = conn.execute(
        "SELECT content_hash FROM call_recording_files"
    ).fetchall()
    return {r["content_hash"] for r in rows}


def reset_stale_transcribing(stale_minutes: int = 30) -> int:
    """Reset recordings stuck in 'transcribing' state back to 'pending'.

    A recording is considered stuck if it has been in 'transcribing' state
    for longer than *stale_minutes* (default 30).  This typically happens
    when the transcription process was killed or crashed without updating
    the DB.

    Args:
        stale_minutes: Minutes after which a transcribing job is considered stale.

    Returns:
        Number of rows reset.
    """
    conn = _get_connection()
    cutoff = datetime.now(timezone.utc).timestamp() - stale_minutes * 60
    cutoff_iso = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat()

    cursor = conn.execute(
        """
        UPDATE call_recording_files
        SET status = 'pending',
            error_message = 'Reset: transcription appeared stuck (no progress for >' || ? || ' min)',
            transcription_progress = '',
            updated_at = datetime('now')
        WHERE status = 'transcribing'
          AND transcription_started_at != ''
          AND transcription_started_at < ?
        """,
        (str(stale_minutes), cutoff_iso),
    )
    conn.commit()
    count = cursor.rowcount
    if count:
        logger.info(
            f"Reset {count} stale transcribing recording(s) "
            f"(started > {stale_minutes} min ago)"
        )
    return count
