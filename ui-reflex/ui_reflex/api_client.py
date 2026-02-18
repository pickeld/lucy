"""Async API client for the RAG Assistant Flask backend.

All calls use httpx.AsyncClient for non-blocking Reflex event handlers.
Mirrors the existing ui/utils/api.py but async.
"""

import logging
import os
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

API_URL = os.environ.get("API_URL", "http://localhost:8765")

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            base_url=API_URL,
            # Default timeout — overridden per-request for fast endpoints.
            # Long default accommodates RAG queries and transcription syncs.
            timeout=120,
            limits=httpx.Limits(
                max_keepalive_connections=5,
                max_connections=10,
                keepalive_expiry=30,  # recycle idle connections after 30s
            ),
        )
    return _client


def _reset_client() -> None:
    """Close and discard the current client so the next call creates a fresh one."""
    global _client
    if _client is not None and not _client.is_closed:
        try:
            import asyncio
            asyncio.get_event_loop().create_task(_client.aclose())
        except Exception:
            pass
    _client = None


# =========================================================================
# HEALTH
# =========================================================================

async def check_health() -> dict[str, Any]:
    try:
        resp = await _get_client().get("/health", timeout=5)
        if resp.status_code in (200, 503):
            return resp.json()
    except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError) as e:
        logger.warning(f"Health check connection error, resetting client: {e}")
        _reset_client()
    except Exception as e:
        logger.warning(f"Health check error: {e}")
        _reset_client()
    return {"status": "unreachable", "dependencies": {}}


# =========================================================================
# CONVERSATIONS
# =========================================================================

async def fetch_conversations(limit: int = 50) -> list[dict[str, Any]]:
    try:
        resp = await _get_client().get("/conversations", params={"limit": limit}, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("conversations", [])
    except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError):
        logger.warning("Connection error fetching conversations — resetting client")
        _reset_client()
    except Exception as e:
        logger.error(f"Error fetching conversations: {e}")
        _reset_client()
    return []


async def fetch_conversation(conversation_id: str) -> dict[str, Any] | None:
    try:
        resp = await _get_client().get(f"/conversations/{conversation_id}", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.error(f"Error fetching conversation {conversation_id}: {e}")
    return None


async def delete_conversation(conversation_id: str) -> bool:
    try:
        resp = await _get_client().delete(f"/conversations/{conversation_id}", timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


async def delete_all_conversations() -> dict[str, Any]:
    """Delete all conversations and chat history."""
    try:
        resp = await _get_client().delete("/conversations", timeout=30)
        return resp.json()
    except Exception as e:
        logger.error(f"Error deleting all conversations: {e}")
        return {"error": str(e)}


async def export_conversation(conversation_id: str) -> dict[str, Any]:
    """Export a conversation as Markdown content for download."""
    try:
        resp = await _get_client().get(
            f"/conversations/{conversation_id}/export", timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 404:
            return {"error": "Conversation not found"}
        else:
            data = resp.json()
            return {"error": data.get("error", f"HTTP {resp.status_code}")}
    except Exception as e:
        logger.error(f"Error exporting conversation {conversation_id}: {e}")
        return {"error": str(e)}


async def rename_conversation(conversation_id: str, title: str) -> bool:
    try:
        resp = await _get_client().put(
            f"/conversations/{conversation_id}",
            json={"title": title},
        )
        return resp.status_code == 200
    except Exception:
        return False


# =========================================================================
# RAG QUERY
# =========================================================================

async def rag_query(
    question: str,
    conversation_id: str | None = None,
    k: int = 10,
    filter_chat_name: str | None = None,
    filter_sender: str | None = None,
    filter_days: int | None = None,
    filter_sources: list[str] | None = None,
    filter_date_from: str | None = None,
    filter_date_to: str | None = None,
    filter_content_types: list[str] | None = None,
    sort_order: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"question": question, "k": k}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    if filter_chat_name:
        payload["filter_chat_name"] = filter_chat_name
    if filter_sender:
        payload["filter_sender"] = filter_sender
    if filter_days is not None:
        payload["filter_days"] = filter_days
    if filter_sources:
        payload["filter_sources"] = filter_sources
    if filter_date_from:
        payload["filter_date_from"] = filter_date_from
    if filter_date_to:
        payload["filter_date_to"] = filter_date_to
    if filter_content_types:
        payload["filter_content_types"] = filter_content_types
    if sort_order and sort_order != "relevance":
        payload["sort_order"] = sort_order

    try:
        resp = await _get_client().post("/rag/query", json=payload, timeout=120)
        if resp.status_code == 200:
            return resp.json()
        else:
            data = resp.json()
            return {"error": data.get("error", f"HTTP {resp.status_code}")}
    except httpx.ConnectError:
        return {"error": "Connection error — is the API running?"}
    except httpx.ReadTimeout:
        return {"error": "Request timed out"}
    except Exception as e:
        return {"error": str(e)}


# =========================================================================
# FILTERS DATA
# =========================================================================

async def get_chat_list() -> list[str]:
    try:
        resp = await _get_client().get("/rag/chats", timeout=10)
        if resp.status_code == 200:
            return resp.json().get("chats", [])
    except Exception:
        pass
    return []


async def get_sender_list() -> list[str]:
    try:
        resp = await _get_client().get("/rag/senders", timeout=10)
        if resp.status_code == 200:
            return resp.json().get("senders", [])
    except Exception:
        pass
    return []


# =========================================================================
# STATS
# =========================================================================

async def get_rag_stats() -> dict[str, Any]:
    try:
        resp = await _get_client().get("/rag/stats", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


# =========================================================================
# CONFIGURATION
# =========================================================================

async def fetch_config(unmask: bool = False) -> dict[str, Any]:
    try:
        params = {"unmask": "true"} if unmask else {}
        resp = await _get_client().get("/config", params=params, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


async def fetch_config_meta() -> dict[str, Any]:
    try:
        resp = await _get_client().get("/config/meta", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


async def save_config(updates: dict[str, str]) -> dict[str, Any]:
    try:
        resp = await _get_client().put("/config", json={"settings": updates})
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


async def reset_config(category: str | None = None) -> dict[str, Any]:
    try:
        payload = {"category": category} if category else {}
        resp = await _get_client().post("/config/reset", json=payload)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


async def fetch_plugins() -> dict[str, Any]:
    try:
        resp = await _get_client().get("/plugins", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


# =========================================================================
# SETTINGS EXPORT/IMPORT
# =========================================================================

async def export_config() -> dict[str, Any]:
    """Export all settings as JSON (with secrets unmasked)."""
    try:
        resp = await _get_client().get("/config/export", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.error(f"Error exporting config: {e}")
    return {"error": "Failed to export settings"}


async def import_config(settings_data: dict[str, Any]) -> dict[str, Any]:
    """Import settings from JSON data."""
    try:
        resp = await _get_client().post("/config/import", json=settings_data, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        else:
            data = resp.json()
            return {"error": data.get("error", f"HTTP {resp.status_code}")}
    except Exception as e:
        logger.error(f"Error importing config: {e}")
        return {"error": str(e)}


async def fetch_secret_value(key: str) -> dict[str, Any]:
    """Fetch the unmasked value of a single secret setting."""
    try:
        resp = await _get_client().get(f"/config/secret/{key}", timeout=10)
        if resp.status_code == 200:
            return resp.json()
        else:
            data = resp.json()
            return {"error": data.get("error", f"HTTP {resp.status_code}")}
    except Exception as e:
        logger.error(f"Error fetching secret {key}: {e}")
        return {"error": str(e)}


# =========================================================================
# COST TRACKING
# =========================================================================

async def get_cost_session(n: int = 20) -> dict[str, Any]:
    """Get current session cost total and recent events."""
    try:
        resp = await _get_client().get("/costs/session", params={"n": n}, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {"session_total_usd": 0.0, "recent_events": []}


async def get_cost_summary(days: int = 7) -> dict[str, Any]:
    """Get daily cost summary for the last N days."""
    try:
        resp = await _get_client().get("/costs/summary", params={"days": days}, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {"total_cost_usd": 0.0, "by_kind": {}, "daily": []}


async def get_cost_breakdown(days: int = 7) -> dict[str, Any]:
    """Get cost breakdown by provider and model."""
    try:
        resp = await _get_client().get("/costs/breakdown", params={"days": days}, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {"total_cost_usd": 0.0, "by_kind": {}, "by_model": []}


# =========================================================================
# PLUGINS — PAPERLESS
# =========================================================================

async def test_paperless_connection() -> dict[str, Any]:
    """Test Paperless-NGX connection."""
    try:
        resp = await _get_client().get("/plugins/paperless/test", timeout=10)
        data = resp.json()
        if resp.status_code == 200:
            return data
        else:
            # The test endpoint returns {"status": "error", "message": "..."}
            msg = data.get("message") or data.get("error") or "Connection failed"
            return {"error": msg}
    except httpx.ConnectError:
        return {"error": "Cannot reach API server"}
    except Exception as e:
        logger.error(f"Error testing Paperless connection: {e}")
        return {"error": str(e)}


async def fetch_paperless_tags() -> list[dict[str, Any]]:
    """Fetch all tags from Paperless-NGX via the backend.

    Uses a long timeout because the backend paginates through all tags
    from the Paperless server (can be thousands).
    """
    try:
        resp = await _get_client().get("/plugins/paperless/tags", timeout=120)
        if resp.status_code == 200:
            return resp.json().get("tags", [])
        else:
            data = resp.json()
            logger.warning(f"Failed to fetch paperless tags: {data.get('error', '')}")
    except httpx.ConnectError:
        logger.warning("Cannot reach API server for paperless tags")
    except httpx.ReadTimeout:
        logger.warning("Timeout fetching paperless tags — large tag collection?")
    except Exception as e:
        logger.error(f"Error fetching paperless tags: {e}")
    return []


async def start_paperless_sync(force: bool = False) -> dict[str, Any]:
    """Trigger Paperless-NGX document sync to RAG vector store.

    Args:
        force: If True, skip processed-tag exclusion and dedup checks.
               Required after deleting/recreating the Qdrant collection.
    """
    try:
        params = {"force": "true"} if force else {}
        resp = await _get_client().post(
            "/plugins/paperless/sync", params=params, timeout=120,
        )
        data = resp.json()
        if resp.status_code == 200:
            return data
        else:
            msg = data.get("error") or f"HTTP {resp.status_code}"
            return {"error": msg}
    except httpx.ConnectError:
        return {"error": "Cannot reach API server"}
    except httpx.ReadTimeout:
        return {"error": "Sync timed out — it may still be running in the background"}
    except Exception as e:
        logger.error(f"Error starting Paperless sync: {e}")
        return {"error": str(e)}


# =========================================================================
# PLUGINS — GMAIL
# =========================================================================

async def gmail_get_auth_url() -> dict[str, Any]:
    """Get Gmail OAuth2 authorization URL."""
    try:
        resp = await _get_client().get("/plugins/gmail/auth/url", timeout=10)
        data = resp.json()
        if resp.status_code == 200:
            return data
        else:
            msg = data.get("message") or data.get("error") or "Failed to generate auth URL"
            return {"error": msg}
    except httpx.ConnectError:
        return {"error": "Cannot reach API server"}
    except Exception as e:
        logger.error(f"Error getting Gmail auth URL: {e}")
        return {"error": str(e)}


async def gmail_submit_auth_code(code: str) -> dict[str, Any]:
    """Submit Gmail OAuth2 authorization code for token exchange."""
    try:
        resp = await _get_client().post(
            "/plugins/gmail/auth/callback",
            json={"code": code},
            timeout=30,
        )
        data = resp.json()
        if resp.status_code == 200:
            return data
        else:
            msg = data.get("message") or data.get("error") or "Authorization failed"
            return {"error": msg}
    except httpx.ConnectError:
        return {"error": "Cannot reach API server"}
    except Exception as e:
        logger.error(f"Error submitting Gmail auth code: {e}")
        return {"error": str(e)}


async def gmail_test_connection() -> dict[str, Any]:
    """Test Gmail connection with current credentials."""
    try:
        resp = await _get_client().get("/plugins/gmail/test", timeout=15)
        data = resp.json()
        if resp.status_code == 200:
            return data
        else:
            msg = data.get("message") or data.get("error") or "Connection failed"
            return {"error": msg}
    except httpx.ConnectError:
        return {"error": "Cannot reach API server"}
    except Exception as e:
        logger.error(f"Error testing Gmail connection: {e}")
        return {"error": str(e)}


async def fetch_gmail_folders() -> list[dict[str, Any]]:
    """Fetch all Gmail labels/folders via the backend."""
    try:
        resp = await _get_client().get("/plugins/gmail/folders", timeout=30)
        if resp.status_code == 200:
            return resp.json().get("folders", [])
        else:
            data = resp.json()
            logger.warning(f"Failed to fetch Gmail folders: {data.get('error', '')}")
    except httpx.ConnectError:
        logger.warning("Cannot reach API server for Gmail folders")
    except httpx.ReadTimeout:
        logger.warning("Timeout fetching Gmail folders")
    except Exception as e:
        logger.error(f"Error fetching Gmail folders: {e}")
    return []


async def start_gmail_sync(force: bool = False) -> dict[str, Any]:
    """Trigger Gmail email sync to RAG vector store.

    Args:
        force: If True, skip processed-label exclusion and dedup checks.
    """
    try:
        params = {"force": "true"} if force else {}
        resp = await _get_client().post(
            "/plugins/gmail/sync", params=params, timeout=300,
        )
        data = resp.json()
        if resp.status_code == 200:
            return data
        else:
            msg = data.get("error") or f"HTTP {resp.status_code}"
            return {"error": msg}
    except httpx.ConnectError:
        return {"error": "Cannot reach API server"}
    except httpx.ReadTimeout:
        return {"error": "Sync timed out — it may still be running in the background"}
    except Exception as e:
        logger.error(f"Error starting Gmail sync: {e}")
        return {"error": str(e)}


# =========================================================================
# PLUGINS — CALL RECORDINGS
# =========================================================================

async def test_call_recordings_connection() -> dict[str, Any]:
    """Test call recordings source connectivity."""
    try:
        resp = await _get_client().get("/plugins/call_recordings/test", timeout=10)
        data = resp.json()
        if resp.status_code == 200:
            return data
        else:
            msg = data.get("message") or data.get("error") or "Connection failed"
            return {"error": msg}
    except httpx.ConnectError:
        return {"error": "Cannot reach API server"}
    except Exception as e:
        logger.error(f"Error testing call recordings connection: {e}")
        return {"error": str(e)}


async def start_call_recordings_sync(force: bool = False) -> dict[str, Any]:
    """Trigger call recordings sync (scan → transcribe → index).

    Args:
        force: If True, skip dedup checks and re-index everything.
    """
    try:
        params = {"force": "true"} if force else {}
        resp = await _get_client().post(
            "/plugins/call_recordings/sync", params=params, timeout=600,
        )
        data = resp.json()
        if resp.status_code == 200:
            return data
        else:
            msg = data.get("error") or f"HTTP {resp.status_code}"
            return {"error": msg}
    except httpx.ConnectError:
        return {"error": "Cannot reach API server"}
    except httpx.ReadTimeout:
        return {"error": "Sync timed out — it may still be running in the background"}
    except Exception as e:
        logger.error(f"Error starting call recordings sync: {e}")
        return {"error": str(e)}


async def upload_call_recordings(file_data: list[tuple[str, bytes]]) -> dict[str, Any]:
    """Upload audio files to the call recordings plugin.

    Args:
        file_data: List of (filename, file_bytes) tuples.

    Returns:
        Dict with saved count, filenames, and errors.
    """
    try:
        files = [
            ("files", (name, data))
            for name, data in file_data
        ]
        resp = await _get_client().post(
            "/plugins/call_recordings/upload",
            files=files,
            timeout=120,
        )
        # Guard against empty / non-JSON responses
        if resp.status_code == 404:
            return {
                "error": "Upload endpoint not found — "
                "is the Call Recordings plugin enabled?"
            }
        try:
            data = resp.json()
        except Exception:
            if resp.status_code == 200:
                return {"error": "Server returned an empty response"}
            return {"error": f"HTTP {resp.status_code} — non-JSON response"}
        if resp.status_code == 200:
            return data
        else:
            msg = data.get("error") or f"HTTP {resp.status_code}"
            return {"error": msg}
    except httpx.ConnectError:
        return {"error": "Cannot reach API server"}
    except Exception as e:
        logger.error(f"Error uploading call recordings: {e}")
        return {"error": str(e)}


async def fetch_call_recording_files(status: str | None = None) -> dict[str, Any]:
    """Fetch all tracked recording files with status and metadata.

    Args:
        status: Optional status filter (pending, transcribing, transcribed, approved, error)
    """
    try:
        params: dict[str, Any] = {}
        if status:
            params["status"] = status
        resp = await _get_client().get(
            "/plugins/call_recordings/files", params=params, timeout=15,
        )
        if resp.status_code == 200:
            return resp.json()
        return {"files": [], "counts": {}, "error": f"HTTP {resp.status_code}"}
    except httpx.ConnectError:
        return {"files": [], "counts": {}, "error": "Cannot reach API server"}
    except Exception as e:
        logger.error(f"Error fetching recording files: {e}")
        return {"files": [], "counts": {}, "error": str(e)}


async def update_recording_metadata(
    content_hash: str,
    contact_name: str | None = None,
    phone_number: str | None = None,
) -> dict[str, Any]:
    """Update metadata for a recording file."""
    try:
        payload: dict[str, Any] = {}
        if contact_name is not None:
            payload["contact_name"] = contact_name
        if phone_number is not None:
            payload["phone_number"] = phone_number
        resp = await _get_client().put(
            f"/plugins/call_recordings/files/{content_hash}/metadata",
            json=payload,
            timeout=10,
        )
        return resp.json()
    except Exception as e:
        logger.error(f"Error updating recording metadata: {e}")
        return {"error": str(e)}


async def transcribe_recording(content_hash: str) -> dict[str, Any]:
    """Trigger transcription for a single recording."""
    try:
        resp = await _get_client().post(
            f"/plugins/call_recordings/files/{content_hash}/transcribe",
            timeout=30,  # endpoint returns 202 immediately; actual work is async
        )
        return resp.json()
    except httpx.ReadTimeout:
        return {"error": "Transcription timed out — it may still be running"}
    except Exception as e:
        logger.error(f"Error transcribing recording: {e}")
        return {"error": str(e)}



async def wait_for_transcription(
    content_hash: str, timeout: int = 300,
) -> dict[str, Any]:
    """Wait for a transcription to complete (push-based via Redis BLPOP).

    The Flask endpoint blocks until the Celery worker pushes a completion
    notification to Redis.  Returns immediately if the file is already
    in a terminal state.

    Args:
        content_hash: File content hash
        timeout: Max seconds to wait (server caps at 600)

    Returns:
        Dict with status (transcribed/error/timeout)
    """
    try:
        resp = await _get_client().get(
            f"/plugins/call_recordings/files/{content_hash}/wait",
            params={"timeout": min(timeout, 600)},
            # HTTP timeout slightly longer than the server BLPOP timeout
            # to avoid the client timing out before the server responds
            timeout=timeout + 10,
        )
        return resp.json()
    except httpx.ReadTimeout:
        return {"status": "timeout", "content_hash": content_hash}
    except Exception as e:
        logger.error(f"Error waiting for transcription: {e}")
        return {"status": "error", "error": str(e)}


async def restart_recording(content_hash: str) -> dict[str, Any]:
    """Restart a stuck transcription by resetting status and re-queuing."""
    try:
        resp = await _get_client().post(
            f"/plugins/call_recordings/files/{content_hash}/restart",
            timeout=30,  # endpoint returns 202 immediately; actual work is async
        )
        return resp.json()
    except httpx.ReadTimeout:
        return {"error": "Restart timed out — it may still be running"}
    except Exception as e:
        logger.error(f"Error restarting recording: {e}")
        return {"error": str(e)}

async def approve_recording(content_hash: str) -> dict[str, Any]:
    """Approve a transcribed recording and index it into Qdrant."""
    try:
        resp = await _get_client().post(
            f"/plugins/call_recordings/files/{content_hash}/approve",
            timeout=120,
        )
        return resp.json()
    except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError) as e:
        logger.warning(f"Connection error approving recording — resetting client: {e}")
        _reset_client()
        return {"error": f"Connection error: {e}"}
    except httpx.ReadTimeout:
        return {"error": "Approve timed out — the indexing may still be running. Refresh to check."}
    except Exception as e:
        logger.error(f"Error approving recording: {e}")
        _reset_client()
        return {"error": str(e)}


async def delete_recording(content_hash: str, remove_disk: bool = True) -> dict[str, Any]:
    """Delete a recording file from tracking and optionally from disk."""
    try:
        params = {"disk": "true" if remove_disk else "false"}
        resp = await _get_client().delete(
            f"/plugins/call_recordings/files/{content_hash}",
            params=params,
            timeout=10,
        )
        return resp.json()
    except Exception as e:
        logger.error(f"Error deleting recording: {e}")
        return {"error": str(e)}


async def scan_call_recordings(auto_transcribe: bool = True) -> dict[str, Any]:
    """Scan for new recording files and optionally auto-transcribe."""
    try:
        params = {"auto_transcribe": "true" if auto_transcribe else "false"}
        resp = await _get_client().post(
            "/plugins/call_recordings/scan",
            params=params,
            timeout=600,
        )
        return resp.json()
    except httpx.ReadTimeout:
        return {"error": "Scan timed out — it may still be running"}
    except Exception as e:
        logger.error(f"Error scanning recordings: {e}")
        return {"error": str(e)}


async def update_speaker_labels(
    content_hash: str,
    speakers: list[dict[str, str]] | None = None,
    speaker_a: str = "",
    speaker_b: str = "",
) -> dict[str, Any]:
    """Update speaker labels and rename in transcript text.

    Args:
        content_hash: File content hash
        speakers: List of {"old": "Speaker A", "new": "David"} mappings
        speaker_a: (Legacy) Display name for Speaker A
        speaker_b: (Legacy) Display name for Speaker B

    Returns:
        Dict with status and updated labels.
    """
    try:
        payload: dict[str, Any] = {}
        if speakers:
            payload["speakers"] = speakers
        else:
            # Legacy 2-speaker format
            if speaker_a:
                payload["speaker_a"] = speaker_a
            if speaker_b:
                payload["speaker_b"] = speaker_b
        resp = await _get_client().put(
            f"/plugins/call_recordings/files/{content_hash}/speakers",
            json=payload,
            timeout=15,
        )
        return resp.json()
    except Exception as e:
        logger.error(f"Error updating speaker labels: {e}")
        return {"error": str(e)}


# =========================================================================
# ENTITY STORE
# =========================================================================

async def fetch_entities(query: str | None = None) -> list[dict[str, Any]]:
    """Fetch all person entities, optionally filtered by search query."""
    try:
        params: dict[str, Any] = {}
        if query:
            params["q"] = query
        resp = await _get_client().get("/entities", params=params, timeout=15)
        if resp.status_code == 200:
            return resp.json().get("persons", [])
    except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError):
        logger.warning("Connection error fetching entities — resetting client")
        _reset_client()
    except Exception as e:
        logger.error(f"Error fetching entities: {e}")
    return []


async def fetch_entity_stats() -> dict[str, Any]:
    """Fetch entity store statistics (persons, aliases, facts, relationships)."""
    try:
        resp = await _get_client().get("/entities/stats", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.error(f"Error fetching entity stats: {e}")
    return {}


async def fetch_entity(person_id: int) -> dict[str, Any]:
    """Fetch a single person entity with all facts, aliases, and relationships."""
    try:
        resp = await _get_client().get(f"/entities/{person_id}", timeout=10)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 404:
            return {"error": "Person not found"}
        else:
            data = resp.json()
            return {"error": data.get("error", f"HTTP {resp.status_code}")}
    except Exception as e:
        logger.error(f"Error fetching entity {person_id}: {e}")
        return {"error": str(e)}


async def delete_entity(person_id: int) -> dict[str, Any]:
    """Delete a person entity and all associated data."""
    try:
        resp = await _get_client().delete(f"/entities/{person_id}", timeout=10)
        return resp.json()
    except Exception as e:
        logger.error(f"Error deleting entity {person_id}: {e}")
        return {"error": str(e)}


async def add_entity_fact(
    person_id: int, key: str, value: str,
) -> dict[str, Any]:
    """Add or update a fact for a person entity."""
    try:
        resp = await _get_client().post(
            f"/entities/{person_id}/facts",
            json={"key": key, "value": value},
            timeout=10,
        )
        return resp.json()
    except Exception as e:
        logger.error(f"Error adding entity fact: {e}")
        return {"error": str(e)}


async def delete_entity_fact(person_id: int, fact_key: str) -> dict[str, Any]:
    """Delete a single fact for a person entity."""
    try:
        resp = await _get_client().delete(
            f"/entities/{person_id}/facts/{fact_key}", timeout=10,
        )
        return resp.json()
    except Exception as e:
        logger.error(f"Error deleting entity fact: {e}")
        return {"error": str(e)}


async def add_entity_alias(person_id: int, alias: str) -> dict[str, Any]:
    """Add a name alias to a person entity."""
    try:
        resp = await _get_client().post(
            f"/entities/{person_id}/aliases",
            json={"alias": alias},
            timeout=10,
        )
        return resp.json()
    except Exception as e:
        logger.error(f"Error adding entity alias: {e}")
        return {"error": str(e)}


async def delete_entity_alias(
    person_id: int, alias_id: int,
) -> dict[str, Any]:
    """Delete a single alias for a person entity by alias row ID."""
    try:
        resp = await _get_client().delete(
            f"/entities/{person_id}/aliases/{alias_id}", timeout=10,
        )
        return resp.json()
    except Exception as e:
        logger.error(f"Error deleting entity alias: {e}")
        return {"error": str(e)}


async def seed_entities() -> dict[str, Any]:
    """Seed entity store from WhatsApp contacts."""
    try:
        resp = await _get_client().post(
            "/entities/seed",
            json={"confirm": True},
            timeout=60,
        )
        return resp.json()
    except httpx.ConnectError:
        return {"error": "Cannot reach API server"}
    except httpx.ReadTimeout:
        return {"error": "Seed timed out — it may still be running"}
    except Exception as e:
        logger.error(f"Error seeding entities: {e}")
        return {"error": str(e)}


async def cleanup_entities() -> dict[str, Any]:
    """Remove persons with garbage/invalid names."""
    try:
        resp = await _get_client().post("/entities/cleanup", timeout=30)
        return resp.json()
    except Exception as e:
        logger.error(f"Error cleaning up entities: {e}")
        return {"error": str(e)}


async def merge_entities(
    target_id: int, source_ids: list[int],
) -> dict[str, Any]:
    """Merge multiple person entities into one target.

    Args:
        target_id: The person ID to keep (merge target)
        source_ids: List of person IDs to absorb into the target
    """
    try:
        resp = await _get_client().post(
            "/entities/merge",
            json={"target_id": target_id, "source_ids": source_ids},
            timeout=30,
        )
        return resp.json()
    except Exception as e:
        logger.error(f"Error merging entities: {e}")
        return {"error": str(e)}


async def fetch_merge_candidates(limit: int = 50) -> dict[str, Any]:
    """Fetch potential duplicate persons that could be merged."""
    try:
        resp = await _get_client().get(
            "/entities/merge-candidates",
            params={"limit": limit},
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.error(f"Error fetching merge candidates: {e}")
    return {"candidates": [], "count": 0}


async def update_entity_display_name(person_id: int) -> dict[str, Any]:
    """Recalculate bilingual display name for a person."""
    try:
        resp = await _get_client().post(
            f"/entities/{person_id}/display-name", timeout=10,
        )
        return resp.json()
    except Exception as e:
        logger.error(f"Error updating entity display name: {e}")
        return {"error": str(e)}


async def fetch_all_entity_facts(
    key: str | None = None,
) -> dict[str, Any]:
    """Fetch all facts across all persons, optionally filtered by key."""
    try:
        params: dict[str, Any] = {}
        if key:
            params["key"] = key
        resp = await _get_client().get(
            "/entities/facts/all", params=params, timeout=15,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.error(f"Error fetching all entity facts: {e}")
    return {"facts": [], "available_keys": []}


# =========================================================================
# SCHEDULED INSIGHTS
# =========================================================================

async def fetch_scheduled_tasks(
    include_disabled: bool = True,
) -> dict[str, Any]:
    """Fetch all scheduled insight tasks with their latest result."""
    try:
        resp = await _get_client().get(
            "/scheduled-tasks",
            params={"include_disabled": str(include_disabled).lower()},
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.error(f"Error fetching scheduled tasks: {e}")
    return {"tasks": [], "count": 0}


async def fetch_scheduled_task(task_id: int) -> dict[str, Any] | None:
    """Fetch a single scheduled task with its latest result."""
    try:
        resp = await _get_client().get(
            f"/scheduled-tasks/{task_id}", timeout=15,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.error(f"Error fetching scheduled task {task_id}: {e}")
    return None


async def create_scheduled_task(data: dict[str, Any]) -> dict[str, Any]:
    """Create a new scheduled insight task."""
    try:
        resp = await _get_client().post(
            "/scheduled-tasks", json=data, timeout=15,
        )
        return resp.json()
    except Exception as e:
        logger.error(f"Error creating scheduled task: {e}")
        return {"error": str(e)}


async def update_scheduled_task(
    task_id: int, data: dict[str, Any],
) -> dict[str, Any]:
    """Update a scheduled task's fields."""
    try:
        resp = await _get_client().put(
            f"/scheduled-tasks/{task_id}", json=data, timeout=15,
        )
        return resp.json()
    except Exception as e:
        logger.error(f"Error updating scheduled task {task_id}: {e}")
        return {"error": str(e)}


async def delete_scheduled_task(task_id: int) -> dict[str, Any]:
    """Delete a scheduled task and all its results."""
    try:
        resp = await _get_client().delete(
            f"/scheduled-tasks/{task_id}", timeout=15,
        )
        return resp.json()
    except Exception as e:
        logger.error(f"Error deleting scheduled task {task_id}: {e}")
        return {"error": str(e)}


async def toggle_scheduled_task(task_id: int) -> dict[str, Any]:
    """Toggle a scheduled task's enabled/disabled state."""
    try:
        resp = await _get_client().post(
            f"/scheduled-tasks/{task_id}/toggle", timeout=15,
        )
        return resp.json()
    except Exception as e:
        logger.error(f"Error toggling scheduled task {task_id}: {e}")
        return {"error": str(e)}


async def run_scheduled_task(task_id: int) -> dict[str, Any]:
    """Manually trigger a scheduled task execution."""
    try:
        resp = await _get_client().post(
            f"/scheduled-tasks/{task_id}/run", timeout=15,
        )
        return resp.json()
    except Exception as e:
        logger.error(f"Error running scheduled task {task_id}: {e}")
        return {"error": str(e)}


async def fetch_scheduled_task_results(
    task_id: int,
    limit: int = 20,
    offset: int = 0,
) -> dict[str, Any]:
    """Fetch paginated result history for a scheduled task."""
    try:
        resp = await _get_client().get(
            f"/scheduled-tasks/{task_id}/results",
            params={"limit": limit, "offset": offset},
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.error(f"Error fetching task results for {task_id}: {e}")
    return {"results": [], "count": 0, "total": 0, "has_more": False}


async def fetch_insight_templates() -> dict[str, Any]:
    """Fetch built-in insight prompt templates."""
    try:
        resp = await _get_client().get(
            "/scheduled-tasks/templates", timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.error(f"Error fetching insight templates: {e}")
    return {"templates": []}


async def rate_insight_result(result_id: int, rating: int) -> dict[str, Any]:
    """Rate an insight result (thumbs up/down).

    Args:
        result_id: The task result ID
        rating: 1 for thumbs up, -1 for thumbs down, 0 to clear

    Returns:
        API response dict
    """
    try:
        resp = await _get_client().post(
            f"/scheduled-tasks/results/{result_id}/rate",
            json={"rating": rating},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.error(f"Error rating insight result: {e}")
    return {"error": "Failed to rate result"}
