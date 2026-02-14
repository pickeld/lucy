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
            timeout=300,
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
        resp = await _get_client().get("/conversations", params={"limit": limit})
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
        resp = await _get_client().get(f"/conversations/{conversation_id}")
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.error(f"Error fetching conversation {conversation_id}: {e}")
    return None


async def delete_conversation(conversation_id: str) -> bool:
    try:
        resp = await _get_client().delete(f"/conversations/{conversation_id}")
        return resp.status_code == 200
    except Exception:
        return False


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

    try:
        resp = await _get_client().post("/rag/query", json=payload)
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
