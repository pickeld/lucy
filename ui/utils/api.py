"""API client wrapper for the RAG Assistant backend.

Consolidates all ``requests`` calls to the Flask API into one module.
Every function handles errors gracefully and returns sensible defaults.
"""

import logging
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

logger = logging.getLogger(__name__)

# Default backend URL — can be overridden via session state
DEFAULT_API_URL = "http://localhost:8765"


def _api_url() -> str:
    """Return the configured API base URL."""
    return st.session_state.get("api_url", DEFAULT_API_URL)


# =========================================================================
# HEALTH
# =========================================================================

def check_health() -> Dict[str, Any]:
    """Check API health status."""
    try:
        resp = requests.get(f"{_api_url()}/health", timeout=5)
        if resp.status_code in (200, 503):
            return resp.json()
    except Exception:
        pass
    return {"status": "unreachable", "dependencies": {}}


# =========================================================================
# CONVERSATIONS
# =========================================================================

def fetch_conversations(limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch the list of previous conversations."""
    try:
        resp = requests.get(
            f"{_api_url()}/conversations",
            params={"limit": limit},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json().get("conversations", [])
    except requests.exceptions.ConnectionError:
        logger.warning("Connection error fetching conversations")
    except Exception as e:
        logger.error(f"Error fetching conversations: {e}")
    return []


def fetch_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a single conversation with all messages."""
    try:
        resp = requests.get(
            f"{_api_url()}/conversations/{conversation_id}",
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.error(f"Error fetching conversation {conversation_id}: {e}")
    return None


def delete_conversation(conversation_id: str) -> bool:
    """Delete a conversation and all its data."""
    try:
        resp = requests.delete(
            f"{_api_url()}/conversations/{conversation_id}",
            timeout=10,
        )
        return resp.status_code == 200
    except Exception:
        return False


def rename_conversation(conversation_id: str, title: str) -> bool:
    """Rename a conversation."""
    try:
        resp = requests.put(
            f"{_api_url()}/conversations/{conversation_id}",
            json={"title": title},
            timeout=10,
        )
        return resp.status_code == 200
    except Exception:
        return False


# =========================================================================
# RAG QUERY & SEARCH
# =========================================================================

def rag_query(
    question: str,
    conversation_id: Optional[str] = None,
    k: int = 10,
    filter_chat_name: Optional[str] = None,
    filter_sender: Optional[str] = None,
    filter_days: Optional[int] = None,
) -> Dict[str, Any]:
    """Send a chat query to the RAG endpoint.

    Returns the full response dict on success, or an error dict.
    """
    payload: Dict[str, Any] = {"question": question, "k": k}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    if filter_chat_name:
        payload["filter_chat_name"] = filter_chat_name
    if filter_sender:
        payload["filter_sender"] = filter_sender
    if filter_days is not None:
        payload["filter_days"] = filter_days

    try:
        resp = requests.post(
            f"{_api_url()}/rag/query",
            json=payload,
            timeout=300,
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            data = resp.json()
            return {"error": data.get("error", f"HTTP {resp.status_code}")}
    except requests.exceptions.ConnectionError:
        return {"error": "Connection error — is the API running?"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out"}
    except Exception as e:
        return {"error": str(e)}


def rag_search(
    query: str,
    k: int = 20,
    filter_chat_name: Optional[str] = None,
    filter_sender: Optional[str] = None,
    filter_days: Optional[int] = None,
) -> Dict[str, Any]:
    """Semantic search for messages."""
    payload: Dict[str, Any] = {"query": query, "k": k}
    if filter_chat_name:
        payload["filter_chat_name"] = filter_chat_name
    if filter_sender:
        payload["filter_sender"] = filter_sender
    if filter_days is not None:
        payload["filter_days"] = filter_days

    try:
        resp = requests.post(
            f"{_api_url()}/rag/search",
            json=payload,
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json()
        return {"error": resp.text}
    except Exception as e:
        return {"error": str(e)}


# =========================================================================
# FILTERS DATA
# =========================================================================

@st.cache_data(ttl=300)
def get_chat_list() -> List[str]:
    """Fetch all unique chat names (cached 5 min)."""
    try:
        resp = requests.get(f"{_api_url()}/rag/chats", timeout=10)
        if resp.status_code == 200:
            return resp.json().get("chats", [])
    except Exception:
        pass
    return []


@st.cache_data(ttl=300)
def get_sender_list() -> List[str]:
    """Fetch all unique sender names (cached 5 min)."""
    try:
        resp = requests.get(f"{_api_url()}/rag/senders", timeout=10)
        if resp.status_code == 200:
            return resp.json().get("senders", [])
    except Exception:
        pass
    return []


# =========================================================================
# STATS
# =========================================================================

@st.cache_data(ttl=60)
def get_rag_stats() -> Dict[str, Any]:
    """Fetch RAG vector store statistics (cached 1 min)."""
    try:
        resp = requests.get(f"{_api_url()}/rag/stats", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


# =========================================================================
# CONFIGURATION
# =========================================================================

def fetch_config() -> Dict[str, Any]:
    """Fetch all settings grouped by category."""
    try:
        resp = requests.get(f"{_api_url()}/config", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


def fetch_config_meta() -> Dict[str, Any]:
    """Fetch configuration metadata (category labels, select options).
    
    Returns:
        Dict with 'category_meta' and 'select_options' keys,
        or empty dict on failure.
    """
    try:
        resp = requests.get(f"{_api_url()}/config/meta", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


def save_config(updates: Dict[str, str]) -> Dict[str, Any]:
    """Save settings via PUT /config."""
    try:
        resp = requests.put(
            f"{_api_url()}/config",
            json={"settings": updates},
            timeout=10,
        )
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def reset_config(category: Optional[str] = None) -> Dict[str, Any]:
    """Reset settings to defaults, optionally for a specific category."""
    try:
        payload = {"category": category} if category else {}
        resp = requests.post(f"{_api_url()}/config/reset", json=payload, timeout=10)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def fetch_plugins() -> Dict[str, Any]:
    """Fetch discovered plugins with their enabled/disabled state.
    
    Returns:
        Dict with 'plugins' key mapping plugin names to info dicts,
        or empty dict on failure.
    """
    try:
        resp = requests.get(f"{_api_url()}/plugins", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}
