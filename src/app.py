import base64
import os
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Union

# Force unbuffered output for immediate logging
os.environ['PYTHONUNBUFFERED'] = '1'

print("🚀 Starting WhatsApp-GPT application...", flush=True)

from flask import Flask, jsonify, redirect, render_template_string, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from requests.models import Response

print("✅ Flask imported", flush=True)

from config import settings
print("✅ Config loaded", flush=True)

from llamaindex_rag import get_rag
print("✅ RAG module imported", flush=True)

from utils.globals import send_request
from utils.logger import logger
from utils.redis_conn import get_redis_client
print("✅ Utils imported", flush=True)

import conversations_db
print("✅ Conversations DB imported", flush=True)

from whatsapp import create_whatsapp_message, group_manager
print("✅ WhatsApp module imported", flush=True)

app = Flask(__name__)
print("✅ Flask app created", flush=True)

# Rate limiter — protects LLM-consuming endpoints from abuse
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],  # No default limit — only applied to specific endpoints
    storage_uri=f"redis://{settings.redis_host}:{settings.redis_port}",
)
print("✅ Rate limiter configured", flush=True)

# Thread pool for async webhook processing
# max_workers=4 allows up to 4 messages to be processed concurrently
_webhook_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="webhook")


# Initialize singletons
rag = get_rag()
print("✅ RAG instance initialized", flush=True)


# =============================================================================
# CONVERSATION FILTER STATE (simple Redis hash per conversation)
# =============================================================================

FILTER_KEY_PREFIX = "filters:"
FILTER_TTL = int(settings.session_ttl_minutes) * 60


def get_conversation_filters(conversation_id: str) -> Dict[str, str]:
    """Get stored filters for a conversation from Redis hash.
    
    Args:
        conversation_id: The conversation identifier
        
    Returns:
        Dictionary of filter key-value pairs (may be empty)
    """
    try:
        redis = get_redis_client()
        key = f"{FILTER_KEY_PREFIX}{conversation_id}"
        filters = redis.hgetall(key)
        return filters or {}
    except Exception as e:
        logger.debug(f"Failed to get conversation filters: {e}")
        return {}


def set_conversation_filters(conversation_id: str, filters: Dict[str, str]) -> None:
    """Store filters for a conversation as a Redis hash with TTL.
    
    Args:
        conversation_id: The conversation identifier
        filters: Dictionary of filter key-value pairs to store
    """
    try:
        redis = get_redis_client()
        key = f"{FILTER_KEY_PREFIX}{conversation_id}"
        # Remove empty values before storing
        clean_filters = {k: v for k, v in filters.items() if v}
        if clean_filters:
            redis.hset(key, mapping=clean_filters)
            redis.expire(key, FILTER_TTL)
        else:
            redis.delete(key)
    except Exception as e:
        logger.debug(f"Failed to set conversation filters: {e}")


def delete_conversation_data(conversation_id: str) -> bool:
    """Delete all data for a conversation (Redis filters + chat history + SQLite).
    
    Args:
        conversation_id: The conversation identifier
        
    Returns:
        True if anything was deleted
    """
    deleted_any = False
    try:
        redis = get_redis_client()
        filter_key = f"{FILTER_KEY_PREFIX}{conversation_id}"
        if redis.delete(filter_key):
            deleted_any = True
    except Exception as e:
        logger.debug(f"Failed to delete Redis conversation data: {e}")
    
    # Delete from SQLite (persistent store)
    try:
        if conversations_db.delete_conversation(conversation_id):
            deleted_any = True
    except Exception as e:
        logger.debug(f"Failed to delete SQLite conversation data: {e}")
    
    return deleted_any


# =============================================================================
# WEBHOOK FILTER
# =============================================================================

def pass_filter(payload):
    """Filter out non-message webhook events.
    
    Returns False for events that should be ignored (acks, newsletters,
    broadcasts, system notifications).
    """
    if payload.get('event') == "message_ack":
        return False
    
    from_field = payload.get("from") or ""
    if from_field.endswith("@newsletter") or from_field.endswith("@broadcast"):
        return False
    
    data_type = payload.get("_data", {}).get("type")
    if data_type in ["e2e_notification", "notification_template"]:
        return False

    return True


# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint that verifies connectivity to dependencies."""
    status = {"status": "up", "dependencies": {}}
    overall_healthy = True
    
    # Check Redis
    try:
        redis_client = get_redis_client()
        redis_client.ping()
        status["dependencies"]["redis"] = "connected"
    except Exception as e:
        status["dependencies"]["redis"] = f"error: {str(e)}"
        overall_healthy = False
    
    # Check Qdrant
    try:
        rag.qdrant_client.get_collections()
        status["dependencies"]["qdrant"] = "connected"
    except Exception as e:
        status["dependencies"]["qdrant"] = f"error: {str(e)}"
        overall_healthy = False
    
    # Check WAHA
    try:
        import requests as req
        waha_url = settings.waha_base_url
        resp = req.get(
            f"{waha_url}/api/sessions",
            headers={"X-Api-Key": settings.waha_api_key},
            timeout=5
        )
        if resp.status_code < 500:
            status["dependencies"]["waha"] = "connected"
        else:
            status["dependencies"]["waha"] = f"error: HTTP {resp.status_code}"
            overall_healthy = False
    except Exception as e:
        status["dependencies"]["waha"] = f"error: {str(e)}"
        overall_healthy = False
    
    if not overall_healthy:
        status["status"] = "degraded"
    
    return jsonify(status), 200 if overall_healthy else 503


# =============================================================================
# RAG QUERY & SEARCH ENDPOINTS
# =============================================================================

@app.route("/rag/query", methods=["POST"])
@limiter.limit("20/minute")  # LLM-consuming: 20 requests per minute per IP
def rag_query():
    """Query the RAG system with a natural language question.

    Uses LlamaIndex CondensePlusContextChatEngine for automatic
    conversation management, query reformulation, and context retrieval.

    Request body:
        {
            "question": "who said they would be late?",
            "conversation_id": "uuid",  # optional, for multi-turn conversations
            "k": 10,  # optional, number of context documents
            "filter_chat_name": "Work Group",  # optional
            "filter_sender": "John",  # optional
            "filter_days": 7  # optional (1=24h, 3=3 days, 7=week, 30=month, null=all time)
        }

    Response:
        {
            "answer": "...",
            "question": "...",
            "conversation_id": "...",
            "filters": {"chat_name": "...", "sender": "..."},
            "stats": {"total_documents": 123}
        }
    """
    try:
        data = request.json or {}
        question = data.get("question")

        if not question:
            return jsonify({"error": "Missing 'question' in request body"}), 400

        # Get or generate conversation ID
        conversation_id = data.get("conversation_id") or str(uuid.uuid4())
        k = data.get("k", 10)
        
        # Load persisted filters, then override with any explicit request params
        filters = get_conversation_filters(conversation_id)
        
        if data.get("filter_chat_name") is not None:
            if data["filter_chat_name"]:
                filters["chat_name"] = data["filter_chat_name"]
            else:
                filters.pop("chat_name", None)
        
        if data.get("filter_sender") is not None:
            if data["filter_sender"]:
                filters["sender"] = data["filter_sender"]
            else:
                filters.pop("sender", None)
        
        if data.get("filter_days") is not None:
            if data["filter_days"]:
                filters["days"] = str(data["filter_days"])
            else:
                filters.pop("days", None)
        
        # Persist updated filters
        set_conversation_filters(conversation_id, filters)
        
        # --- Conversation persistence: create if new ---
        is_new_conversation = not conversations_db.conversation_exists(conversation_id)
        if is_new_conversation:
            title = conversations_db._generate_title(question)
            conversations_db.create_conversation(
                conversation_id=conversation_id,
                title=title,
                filters=filters,
            )
            logger.info(f"Created new conversation: {conversation_id} — '{title}'")
        else:
            # Update filters in SQLite if they changed
            conversations_db.update_conversation_filters(conversation_id, filters)
        
        # --- Restore Redis chat memory from SQLite if expired ---
        conversations_db.restore_chat_memory_if_needed(
            conversation_id=conversation_id,
            chat_store=rag.chat_store,
            max_messages=int(settings.session_max_history) * 2,  # user+assistant pairs
        )
        
        # Create chat engine with filters and conversation memory
        chat_engine = rag.create_chat_engine(
            conversation_id=conversation_id,
            filter_chat_name=filters.get("chat_name"),
            filter_sender=filters.get("sender"),
            filter_days=int(filters["days"]) if filters.get("days") else None,
            k=k,
        )
        
        # Single call handles: condense → retrieve → generate → store in memory
        response = chat_engine.chat(question)
        answer = str(response)
        
        # --- Persist messages to SQLite ---
        conversations_db.add_message(conversation_id, "user", question)
        conversations_db.add_message(conversation_id, "assistant", answer)

        # Extract source documents from the response for citations
        sources = []
        if hasattr(response, 'source_nodes') and response.source_nodes:
            for node_with_score in response.source_nodes:
                node = node_with_score.node
                metadata = getattr(node, 'metadata', {})
                # Skip system placeholder nodes
                if metadata.get("source") == "system":
                    continue
                sources.append({
                    "content": getattr(node, 'text', '')[:300],  # Truncate for response size
                    "score": node_with_score.score,
                    "sender": metadata.get("sender", "Unknown"),
                    "chat_name": metadata.get("chat_name", "Unknown"),
                    "timestamp": metadata.get("timestamp"),
                })

        stats = rag.get_stats()

        return jsonify({
            "answer": answer,
            "question": question,
            "conversation_id": conversation_id,
            "filters": filters,
            "sources": sources,
            "stats": stats,
        }), 200

    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"RAG query error: {e}\n{trace}")
        return jsonify({"error": str(e), "traceback": trace}), 500


@app.route("/rag/search", methods=["POST"])
@limiter.limit("30/minute")  # Embedding API call: 30 requests per minute per IP
def rag_search():
    """Search the RAG system for relevant messages.

    Request body:
        {
            "query": "meeting tomorrow",
            "k": 10,  # optional
            "filter_chat_name": "Work Group",  # optional
            "filter_sender": "John",  # optional
            "filter_days": 7  # optional (1=24h, 3=3 days, 7=week, 30=month, null=all time)
        }

    Response:
        {
            "results": [
                {
                    "content": "[2024-01-15 10:30] John in Work Group: meeting tomorrow at 2pm",
                    "metadata": {...}
                }
            ]
        }
    """
    try:
        data = request.json or {}
        query = data.get("query")

        if not query:
            return jsonify({"error": "Missing 'query' in request body"}), 400

        k = data.get("k", 10)
        filter_chat_name = data.get("filter_chat_name")
        filter_sender = data.get("filter_sender")
        filter_days = data.get("filter_days")

        results = rag.search(
            query=query,
            k=k,
            filter_chat_name=filter_chat_name,
            filter_sender=filter_sender,
            filter_days=filter_days
        )

        # Convert LlamaIndex NodeWithScore to dict
        formatted_results = [
            {
                "content": getattr(result.node, 'text', '') or getattr(result.node, 'get_content', lambda: '')(),
                "metadata": getattr(result.node, 'metadata', {}),
                "score": result.score
            }
            for result in results
        ]

        return jsonify({"results": formatted_results}), 200

    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"RAG search error: {e}\n{trace}")
        return jsonify({"error": str(e), "traceback": trace}), 500


@app.route("/rag/stats", methods=["GET"])
def rag_stats():
    """Get RAG vector store statistics."""
    try:
        stats = rag.get_stats()
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/rag/chats", methods=["GET"])
def rag_chats():
    """Get all unique chat names from the RAG vector store.
    
    Response:
        {
            "chats": ["Chat1", "Chat2", "Group1", ...]
        }
    """
    try:
        chats = rag.get_chat_list()
        return jsonify({"chats": chats}), 200
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"RAG chats error: {e}\n{trace}")
        return jsonify({"error": str(e), "traceback": trace}), 500


@app.route("/rag/senders", methods=["GET"])
def rag_senders():
    """Get all unique sender names from the RAG vector store.
    
    Response:
        {
            "senders": ["Alice", "Bob", "Charlie", ...]
        }
    """
    try:
        senders = rag.get_sender_list()
        return jsonify({"senders": senders}), 200
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"RAG senders error: {e}\n{trace}")
        return jsonify({"error": str(e), "traceback": trace}), 500


@app.route("/rag/messages", methods=["GET"])
def rag_messages():
    """Browse messages stored in the RAG vector store with pagination and filters.
    
    Query params:
        chat_name: Filter by chat/group name (optional)
        sender: Filter by sender name (optional)
        days: Filter by recency in days (optional)
        limit: Max results per page (default 50, max 200)
        offset: Pagination offset as Qdrant point ID (optional)
    
    Response:
        {
            "messages": [
                {
                    "sender": "John",
                    "chat_name": "Family Group",
                    "message": "hello everyone",
                    "timestamp": 1707648000,
                    "is_group": true,
                    "source_type": "whatsapp"
                }
            ],
            "count": 50,
            "has_more": true,
            "next_offset": "point-id-string"
        }
    """
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
        from datetime import datetime
        
        chat_name = request.args.get("chat_name")
        sender = request.args.get("sender")
        days = request.args.get("days", type=int)
        limit = min(request.args.get("limit", 50, type=int), 200)
        offset = request.args.get("offset")  # Qdrant point ID for pagination
        
        # Build filter conditions
        must_conditions = []
        if chat_name:
            must_conditions.append(
                FieldCondition(key="chat_name", match=MatchValue(value=chat_name))
            )
        if sender:
            must_conditions.append(
                FieldCondition(key="sender", match=MatchValue(value=sender))
            )
        if days and days > 0:
            min_ts = int(datetime.now().timestamp()) - (days * 86400)
            must_conditions.append(
                FieldCondition(key="timestamp", range=Range(gte=min_ts))
            )
        
        scroll_filter = Filter(must=must_conditions) if must_conditions else None
        
        records, next_offset = rag.qdrant_client.scroll(
            collection_name=rag.COLLECTION_NAME,
            scroll_filter=scroll_filter,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        
        messages = []
        for record in records:
            payload = record.payload or {}
            # Skip internal LlamaIndex fields
            msg = {
                "sender": payload.get("sender", "Unknown"),
                "chat_name": payload.get("chat_name", "Unknown"),
                "message": payload.get("message", ""),
                "timestamp": payload.get("timestamp", 0),
                "is_group": payload.get("is_group", False),
                "source_type": payload.get("source_type", "whatsapp"),
                "has_media": payload.get("has_media", False),
            }
            if msg["message"]:  # Skip empty messages
                messages.append(msg)
        
        return jsonify({
            "messages": messages,
            "count": len(messages),
            "has_more": next_offset is not None,
            "next_offset": str(next_offset) if next_offset else None,
        }), 200
        
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"RAG messages error: {e}\n{trace}")
        return jsonify({"error": str(e), "traceback": trace}), 500


@app.route("/rag/reset", methods=["POST"])
def rag_reset():
    """Drop and recreate the Qdrant collection with fresh vector configuration.
    
    Required when changing embedding models or dimensions (e.g., switching
    from text-embedding-3-small to text-embedding-3-large). All existing
    embeddings are permanently deleted and must be re-ingested.
    
    Request body (optional):
        {
            "confirm": true  # Safety flag to prevent accidental resets
        }
    
    Response:
        {
            "status": "ok",
            "message": "Collection reset successfully",
            "collection_name": "whatsapp_messages",
            "new_vector_size": 1024
        }
    """
    try:
        data = request.json or {}
        if not data.get("confirm", False):
            return jsonify({
                "error": "Safety check: pass {\"confirm\": true} to confirm collection reset. "
                         "This will permanently delete ALL stored embeddings."
            }), 400
        
        success = rag.reset_collection()
        if success:
            return jsonify({
                "status": "ok",
                "message": "Collection reset successfully. All embeddings dropped.",
                "collection_name": rag.COLLECTION_NAME,
                "new_vector_size": rag.VECTOR_SIZE,
            }), 200
        else:
            return jsonify({"error": "Failed to reset collection — check logs"}), 500
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"RAG reset error: {e}\n{trace}")
        return jsonify({"error": str(e), "traceback": trace}), 500


# =============================================================================
# CONVERSATION MANAGEMENT ENDPOINTS
# =============================================================================

@app.route("/conversations", methods=["GET"])
def list_conversations_endpoint():
    """List all conversations sorted by most recently updated.
    
    Query params:
        limit: Max number of conversations (default 50)
        offset: Number to skip for pagination (default 0)
    
    Response:
        {
            "conversations": [
                {
                    "id": "uuid",
                    "title": "What did John say about...",
                    "created_at": "2024-01-15T10:30:00",
                    "updated_at": "2024-01-15T10:35:00",
                    "message_count": 4,
                    "filters": {}
                }
            ]
        }
    """
    try:
        limit = min(request.args.get("limit", 50, type=int), 200)
        offset = request.args.get("offset", 0, type=int)
        convos = conversations_db.list_conversations(limit=limit, offset=offset)
        return jsonify({"conversations": convos}), 200
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"List conversations error: {e}\n{trace}")
        return jsonify({"error": str(e), "traceback": trace}), 500


@app.route("/conversations/<conversation_id>", methods=["GET"])
def get_conversation_endpoint(conversation_id: str):
    """Get a single conversation with all its messages.
    
    Response:
        {
            "id": "uuid",
            "title": "...",
            "messages": [
                {"role": "user", "content": "...", "created_at": "..."},
                {"role": "assistant", "content": "...", "created_at": "..."}
            ],
            ...
        }
    """
    try:
        convo = conversations_db.get_conversation(conversation_id)
        if not convo:
            return jsonify({"error": "Conversation not found"}), 404
        return jsonify(convo), 200
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Get conversation error: {e}\n{trace}")
        return jsonify({"error": str(e), "traceback": trace}), 500


@app.route("/conversations", methods=["POST"])
def create_conversation_endpoint():
    """Create a new empty conversation.
    
    Request body (optional):
        {
            "title": "My conversation",
            "filters": {"chat_name": "Work Group"}
        }
    
    Response:
        {
            "id": "uuid",
            "title": "...",
            ...
        }
    """
    try:
        data = request.json or {}
        conversation_id = str(uuid.uuid4())
        title = data.get("title", "New Chat")
        filters = data.get("filters", {})
        
        convo = conversations_db.create_conversation(
            conversation_id=conversation_id,
            title=title,
            filters=filters,
        )
        return jsonify(convo), 201
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Create conversation error: {e}\n{trace}")
        return jsonify({"error": str(e), "traceback": trace}), 500


@app.route("/conversations/<conversation_id>", methods=["PUT"])
def update_conversation_endpoint(conversation_id: str):
    """Update a conversation (currently: rename title).
    
    Request body:
        {
            "title": "New title"
        }
    
    Response:
        {
            "status": "ok",
            "conversation_id": "..."
        }
    """
    try:
        data = request.json or {}
        title = data.get("title")
        
        if title is None:
            return jsonify({"error": "Missing 'title' in request body"}), 400
        
        if not conversations_db.conversation_exists(conversation_id):
            return jsonify({"error": "Conversation not found"}), 404
        
        conversations_db.update_conversation_title(conversation_id, title)
        return jsonify({
            "status": "ok",
            "conversation_id": conversation_id,
        }), 200
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Update conversation error: {e}\n{trace}")
        return jsonify({"error": str(e), "traceback": trace}), 500


@app.route("/conversations/<conversation_id>", methods=["DELETE"])
@app.route("/conversation/<conversation_id>", methods=["DELETE"])  # backwards compat
def delete_conversation_endpoint(conversation_id: str):
    """Delete a conversation and all its data (SQLite + Redis).
    
    Response:
        {
            "status": "ok",
            "conversation_id": "..."
        }
    """
    try:
        delete_conversation_data(conversation_id)
        return jsonify({
            "status": "ok",
            "conversation_id": conversation_id,
        }), 200
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Conversation delete error: {e}\n{trace}")
        return jsonify({"error": str(e), "traceback": trace}), 500


# =============================================================================
# CONFIGURATION MANAGEMENT ENDPOINTS
# =============================================================================

@app.route("/config", methods=["GET"])
def get_config():
    """Get all settings grouped by category, with secrets masked."""
    try:
        import settings_db
        all_settings = settings_db.get_all_settings_masked()
        return jsonify(all_settings), 200
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Config get error: {e}\n{trace}")
        return jsonify({"error": str(e), "traceback": trace}), 500


@app.route("/config", methods=["PUT"])
def update_config():
    """Update one or more settings.
    
    Request body: {"settings": {"key": "value", ...}}
    """
    try:
        import settings_db
        data = request.json or {}
        updates = data.get("settings", {})
        
        if not updates:
            return jsonify({"error": "No settings provided"}), 400
        
        updated_keys = settings_db.set_settings(updates)
        
        return jsonify({
            "status": "ok",
            "updated": updated_keys
        }), 200
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Config update error: {e}\n{trace}")
        return jsonify({"error": str(e), "traceback": trace}), 500


@app.route("/config/categories", methods=["GET"])
def get_config_categories():
    """Get available setting categories."""
    try:
        import settings_db
        categories = settings_db.get_categories()
        return jsonify({"categories": categories}), 200
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Config categories error: {e}\n{trace}")
        return jsonify({"error": str(e), "traceback": trace}), 500


@app.route("/config/reset", methods=["POST"])
def reset_config():
    """Reset settings to defaults, optionally for a specific category.
    
    Request body (optional): {"category": "llm"}
    """
    try:
        import settings_db
        data = request.json or {}
        category = data.get("category")
        
        count = settings_db.reset_to_defaults(category=category)
        
        return jsonify({
            "status": "ok",
            "reset_count": count
        }), 200
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Config reset error: {e}\n{trace}")
        return jsonify({"error": str(e), "traceback": trace}), 500


# =============================================================================
# CACHE MANAGEMENT ENDPOINTS
# =============================================================================

@app.route("/cache/groups/clear", methods=["POST", "DELETE"])
def clear_groups_cache():
    """Clear all cached group data from Redis.
    
    This forces the system to re-fetch group info from WAHA on next message.
    Use this when group names are showing incorrectly.
    
    Response:
        {
            "status": "ok",
            "deleted_count": 5
        }
    """
    try:
        count = group_manager.clear_all_groups_cache()
        return jsonify({"status": "ok", "deleted_count": count}), 200
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Failed to clear groups cache: {e}\n{trace}")
        return jsonify({"error": str(e), "traceback": trace}), 500


@app.route("/cache/groups/<group_id>/refresh", methods=["POST"])
def refresh_group_cache(group_id: str):
    """Refresh cache for a specific group.
    
    Args:
        group_id: The group ID (e.g., '120363123456789@g.us')
        
    Response:
        {
            "status": "ok",
            "group": {"id": "...", "name": "..."}
        }
    """
    try:
        group = group_manager.refresh_group(group_id)
        if group:
            return jsonify({"status": "ok", "group": group.to_dict()}), 200
        else:
            return jsonify({"status": "error", "message": "Failed to fetch group"}), 404
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Failed to refresh group cache: {e}\n{trace}")
        return jsonify({"error": str(e), "traceback": trace}), 500


# =============================================================================
# TEST & WEBHOOK ENDPOINTS
# =============================================================================

@app.route("/test", methods=["GET"])
def test():
    # send a test message to yourself
    test_message = f"my name is david and i 39 years old, i have two kids, mia which is 4 and ben which is 6 and i work as a software developer."
    send_request(method="POST",
                 endpoint="/api/sendText",
                 payload={
                     "chatId": "972547755011@c.us",
                     "text": test_message,
                     "session": settings.waha_session_name
                 }
                 )
    return jsonify({"status": "test message sent"}), 200


def _process_webhook_payload(payload: Dict[str, Any]) -> None:
    """Process a webhook payload in a background thread.
    
    This function handles the heavy work: creating the message object
    (which may trigger Whisper transcription or GPT-4 Vision), then
    storing the result in the RAG vector store.
    
    Args:
        payload: The webhook payload dictionary
    """
    try:
        msg = create_whatsapp_message(payload)

        # Determine chat identification:
        # - Group messages: use group info
        # - Outgoing DMs (from_me): use recipient as the chat identity
        # - Incoming DMs: use sender contact as the chat identity
        if msg.is_group:
            chat_id = msg.group.id
            chat_name = msg.group.name
        elif msg.from_me and msg.recipient:
            chat_id = msg.recipient.number or msg.recipient.id
            chat_name = msg.recipient.name
        else:
            chat_id = msg.contact.number
            chat_name = msg.contact.name

        sender = str(msg.contact.name or "Unknown")
        logger.info(f"Processing message: {chat_name} ({chat_id}) - {msg.message}")

        # Store message in RAG vector store
        if msg.message:
            rag.add_message(
                thread_id=chat_id or "UNKNOWN",
                chat_id=chat_id or "UNKNOWN",
                chat_name=chat_name or "UNKNOWN",
                is_group=msg.is_group,
                sender=sender,
                message=msg.message,
                timestamp=str(msg.timestamp) if msg.timestamp else "0"
            )
            logger.debug(f"Stored message: {chat_name} || {msg}")
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Background webhook processing error: {e} ::: {payload}\n{trace}")


@app.route("/webhook", methods=["POST"])
def webhook():
    """Receive WhatsApp webhook events from WAHA.
    
    Returns 200 immediately and processes the message asynchronously
    in a background thread. This prevents WAHA timeouts during heavy
    processing (e.g., Whisper transcription, GPT-4 Vision analysis).
    """
    request_data = request.json or {}
    payload = request_data.get("payload", {})
    
    try:
        if not pass_filter(payload):
            return jsonify({"status": "ok"}), 200

        # Submit to background thread pool — return 200 immediately
        _webhook_executor.submit(_process_webhook_payload, payload)
        
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(
            f"Error submitting webhook: {e} ::: {payload}\n{trace}")
        return jsonify({"error": str(e), "traceback": trace}), 500


# =============================================================================
# WAHA SESSION MANAGEMENT (WhatsApp pairing)
# =============================================================================

@app.route("/", methods=["GET"])
def index():
    try:
        response: Union[Dict[str, Any], Response] = send_request(
            "GET", f"/api/sessions/{settings.waha_session_name}")
        logger.debug(f"Session status: {response}")

        # Handle response - could be dict or Response object
        if isinstance(response, dict) and response.get("status") == "WORKING" and response.get("engine", {}).get("state") == "CONNECTED":
            return "<h1>Session 'default' is already connected.</h1>", 200
        elif isinstance(response, dict) and response.get("status") == "SCAN_QR_CODE":
            return redirect("/qr_code")
        else:
            return redirect("/pair")
    # catch 404 error
    except Exception as e:
        logger.error(f"Error checking session status: {e}")
        return redirect("/pair")


@app.route("/qr_code", methods=["GET"])
def qr_code():
    qr_response = send_request(
        "GET", f"/api/{settings.waha_session_name}/auth/qr")
    # qr_response is a Response object for binary content
    qr_image_data = qr_response.content if isinstance(qr_response, Response) else None
    if qr_image_data:
        qr_base64 = base64.b64encode(qr_image_data).decode("utf-8")
        html = f"<h1>Scan to Pair WhatsApp</h1><img src='data:image/png;base64,{qr_base64}'>"
        return render_template_string(html)
    else:
        return "QR code not available yet. Please refresh in a few seconds.", 200


@app.route("/pair", methods=["GET"])
def pair():
    session_name = settings.waha_session_name
    send_request(method="POST", endpoint="/api/sessions/start",
                 payload={"name": session_name})

    send_request("PUT", f"/api/sessions/{session_name}", {
        "config": {
            "webhooks": [
                {
                    "url": settings.webhook_url,
                    "events": ["message.any"]
                }
            ]
        }
    })
    time.sleep(2)
    return redirect("/qr_code")


if __name__ == "__main__":
    os.environ["LOCAL"] = "TRUE"
    is_debug = settings.log_level == "DEBUG"
    app.run(host="0.0.0.0", port=8765,
            debug=is_debug, use_reloader=True)
