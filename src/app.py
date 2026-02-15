import json
import os
import time
import traceback
import uuid
from typing import Any, Dict

# Force unbuffered output for immediate logging
os.environ['PYTHONUNBUFFERED'] = '1'

print("🚀 Starting application...", flush=True)

from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

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

import cost_db
from cost_meter import METER
print("✅ Cost tracking imported", flush=True)

from plugins.registry import plugin_registry
print("✅ Plugin registry imported", flush=True)

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

# Initialize singletons
rag = get_rag()
print("✅ RAG instance initialized", flush=True)

# =============================================================================
# PLUGIN DISCOVERY AND LOADING
# =============================================================================

print("🔌 Discovering plugins...", flush=True)
plugin_registry.discover_plugins()
print("🔌 Loading enabled plugins...", flush=True)
plugin_registry.load_enabled_plugins(app)
print("✅ Plugins loaded", flush=True)


# =============================================================================
# CONVERSATION FILTER STATE (simple Redis hash per conversation)
# =============================================================================

FILTER_KEY_PREFIX = "filters:"
FILTER_TTL = int(settings.session_ttl_minutes) * 60


def get_conversation_filters(conversation_id: str) -> Dict[str, str]:
    """Get stored filters for a conversation from Redis hash."""
    try:
        redis = get_redis_client()
        key = f"{FILTER_KEY_PREFIX}{conversation_id}"
        filters = redis.hgetall(key)
        return filters or {}
    except Exception as e:
        logger.debug(f"Failed to get conversation filters: {e}")
        return {}


def set_conversation_filters(conversation_id: str, filters: Dict[str, str]) -> None:
    """Store filters for a conversation as a Redis hash with TTL."""
    try:
        redis = get_redis_client()
        key = f"{FILTER_KEY_PREFIX}{conversation_id}"
        clean_filters = {k: v for k, v in filters.items() if v}
        if clean_filters:
            redis.hset(key, mapping=clean_filters)
            redis.expire(key, FILTER_TTL)
        else:
            redis.delete(key)
    except Exception as e:
        logger.debug(f"Failed to set conversation filters: {e}")


def delete_conversation_data(conversation_id: str) -> bool:
    """Delete all data for a conversation (Redis filters + chat history + SQLite)."""
    deleted_any = False
    try:
        redis = get_redis_client()
        filter_key = f"{FILTER_KEY_PREFIX}{conversation_id}"
        if redis.delete(filter_key):
            deleted_any = True
    except Exception as e:
        logger.debug(f"Failed to delete Redis conversation data: {e}")
    
    try:
        if conversations_db.delete_conversation(conversation_id):
            deleted_any = True
    except Exception as e:
        logger.debug(f"Failed to delete SQLite conversation data: {e}")
    
    return deleted_any


# =============================================================================
# HEALTH CHECK — delegates to plugin registry
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
    
    # Check all enabled plugins
    plugin_health = plugin_registry.health_check_all()
    for plugin_name, deps in plugin_health.items():
        for dep_name, dep_status in deps.items():
            full_key = f"{plugin_name}.{dep_name}"
            status["dependencies"][full_key] = dep_status
            if dep_status.startswith("error"):
                overall_healthy = False
    
    if not overall_healthy:
        status["status"] = "degraded"
    
    return jsonify(status), 200 if overall_healthy else 503


# =============================================================================
# PLUGIN STATUS ENDPOINT
# =============================================================================

@app.route("/plugins", methods=["GET"])
def list_plugins():
    """List all discovered plugins with their enabled/disabled state."""
    try:
        plugins = plugin_registry.discovered_plugins()
        return jsonify({"plugins": plugins}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =============================================================================
# RAG QUERY & SEARCH ENDPOINTS
# =============================================================================

@app.route("/rag/query", methods=["POST"])
@limiter.limit("20/minute")
def rag_query():
    """Query the RAG system with a natural language question."""
    try:
        data = request.json or {}
        question = data.get("question")

        if not question:
            return jsonify({"error": "Missing 'question' in request body"}), 400

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
        
        # New filters: sources, date range, content types, sort order
        if data.get("filter_sources") is not None:
            if data["filter_sources"]:
                # Store as comma-separated string in Redis hash
                filters["sources"] = ",".join(data["filter_sources"])
            else:
                filters.pop("sources", None)
        
        if data.get("filter_date_from") is not None:
            if data["filter_date_from"]:
                filters["date_from"] = data["filter_date_from"]
            else:
                filters.pop("date_from", None)
        
        if data.get("filter_date_to") is not None:
            if data["filter_date_to"]:
                filters["date_to"] = data["filter_date_to"]
            else:
                filters.pop("date_to", None)
        
        if data.get("filter_content_types") is not None:
            if data["filter_content_types"]:
                filters["content_types"] = ",".join(data["filter_content_types"])
            else:
                filters.pop("content_types", None)
        
        if data.get("sort_order") is not None:
            if data["sort_order"] and data["sort_order"] != "relevance":
                filters["sort_order"] = data["sort_order"]
            else:
                filters.pop("sort_order", None)
        
        set_conversation_filters(conversation_id, filters)
        
        # Conversation persistence
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
            conversations_db.update_conversation_filters(conversation_id, filters)
        
        # Restore Redis chat memory from SQLite if expired
        conversations_db.restore_chat_memory_if_needed(
            conversation_id=conversation_id,
            chat_store=rag.chat_store,
            max_messages=int(settings.session_max_history) * 2,
        )
        
        # Parse list filters from comma-separated strings
        sources_list = [s.strip() for s in filters["sources"].split(",")] if filters.get("sources") else None
        content_types_list = [c.strip() for c in filters["content_types"].split(",")] if filters.get("content_types") else None
        
        # Create chat engine with filters and conversation memory
        chat_engine = rag.create_chat_engine(
            conversation_id=conversation_id,
            filter_chat_name=filters.get("chat_name"),
            filter_sender=filters.get("sender"),
            filter_days=int(filters["days"]) if filters.get("days") else None,
            filter_sources=sources_list,
            filter_date_from=filters.get("date_from"),
            filter_date_to=filters.get("date_to"),
            filter_content_types=content_types_list,
            sort_order=filters.get("sort_order", "relevance"),
            k=k,
        )
        
        # Snapshot cost meter before query to compute per-query cost
        cost_snapshot = METER.snapshot()
        
        response = chat_engine.chat(question)
        answer = str(response)
        
        # Compute per-query cost
        query_cost = METER.session_total - cost_snapshot
        
        # Extract source documents
        sources = []
        if hasattr(response, 'source_nodes') and response.source_nodes:
            for node_with_score in response.source_nodes:
                node = node_with_score.node
                metadata = getattr(node, 'metadata', {})
                if metadata.get("source") == "system":
                    continue
                sources.append({
                    "content": getattr(node, 'text', '')[:300],
                    "score": node_with_score.score,
                    "sender": metadata.get("sender", ""),
                    "chat_name": metadata.get("chat_name", ""),
                    "timestamp": metadata.get("timestamp"),
                })

        # Persist messages to SQLite (include sources JSON for assistant)
        conversations_db.add_message(conversation_id, "user", question)
        conversations_db.add_message(
            conversation_id, "assistant", answer,
            sources=json.dumps(sources) if sources else "",
        )

        stats = rag.get_stats()

        return jsonify({
            "answer": answer,
            "question": question,
            "conversation_id": conversation_id,
            "filters": filters,
            "sources": sources,
            "stats": stats,
            "cost": {
                "query_cost_usd": round(query_cost, 6),
                "session_total_usd": round(METER.session_total, 6),
            },
        }), 200

    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"RAG query error: {e}\n{trace}")
        return jsonify({"error": str(e), "traceback": trace}), 500


@app.route("/rag/search", methods=["POST"])
@limiter.limit("30/minute")
def rag_search():
    """Search the RAG system for relevant messages."""
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
    """Get all unique chat names from the RAG vector store."""
    try:
        chats = rag.get_chat_list()
        return jsonify({"chats": chats}), 200
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"RAG chats error: {e}\n{trace}")
        return jsonify({"error": str(e), "traceback": trace}), 500


@app.route("/rag/senders", methods=["GET"])
def rag_senders():
    """Get all unique sender names from the RAG vector store."""
    try:
        senders = rag.get_sender_list()
        return jsonify({"senders": senders}), 200
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"RAG senders error: {e}\n{trace}")
        return jsonify({"error": str(e), "traceback": trace}), 500


@app.route("/rag/messages", methods=["GET"])
def rag_messages():
    """Browse messages stored in the RAG vector store with pagination and filters."""
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
        from datetime import datetime
        
        chat_name = request.args.get("chat_name")
        sender = request.args.get("sender")
        days = request.args.get("days", type=int)
        limit = min(request.args.get("limit", 50, type=int), 200)
        offset = request.args.get("offset")
        
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
            msg = {
                "sender": payload.get("sender", "Unknown"),
                "chat_name": payload.get("chat_name", "Unknown"),
                "message": payload.get("message", ""),
                "timestamp": payload.get("timestamp", 0),
                "is_group": payload.get("is_group", False),
                "source": payload.get("source", payload.get("source_type", "unknown")),
                "content_type": payload.get("content_type", "text"),
                "has_media": payload.get("has_media", False),
            }
            if msg["message"]:
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
    """Drop and recreate the Qdrant collection."""
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


@app.route("/rag/delete-by-source", methods=["POST"])
def rag_delete_by_source():
    """Delete all RAG vectors matching a specific source type.
    
    Allows selective cleanup — e.g., delete only WhatsApp messages
    or only Paperless documents without dropping the entire collection.
    
    Body: {"source": "whatsapp"|"paperless", "confirm": true}
    """
    try:
        data = request.json or {}
        source_value = data.get("source")
        
        if not source_value:
            return jsonify({"error": "Missing 'source' in request body"}), 400
        
        if not data.get("confirm", False):
            # Dry run: show how many would be deleted
            stats = rag.get_stats()
            source_counts = stats.get("source_counts", {})
            count = source_counts.get(source_value, 0)
            return jsonify({
                "status": "dry_run",
                "source": source_value,
                "would_delete": count,
                "message": f"Would delete {count} vectors with source='{source_value}'. "
                           f"Pass {{\"confirm\": true}} to proceed.",
            }), 200
        
        deleted = rag.delete_by_source(source_value)
        return jsonify({
            "status": "ok",
            "source": source_value,
            "deleted": deleted,
            "message": f"Deleted {deleted} vectors with source='{source_value}'.",
        }), 200
        
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"RAG delete-by-source error: {e}\n{trace}")
        return jsonify({"error": str(e), "traceback": trace}), 500


# =============================================================================
# CONVERSATION MANAGEMENT ENDPOINTS
# =============================================================================

@app.route("/conversations", methods=["GET"])
def list_conversations_endpoint():
    """List all conversations sorted by most recently updated."""
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
    """Get a single conversation with all its messages."""
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
    """Create a new empty conversation."""
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
    """Update a conversation (currently: rename title)."""
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


@app.route("/conversations/<conversation_id>/export", methods=["GET"])
def export_conversation_endpoint(conversation_id: str):
    """Export a conversation as a Markdown file.

    Returns JSON with the conversation title and Markdown-formatted content
    suitable for client-side file download.
    """
    try:
        convo = conversations_db.get_conversation(conversation_id)
        if not convo:
            return jsonify({"error": "Conversation not found"}), 404

        title = convo.get("title", "Untitled")
        created = convo.get("created_at", "")
        messages = convo.get("messages", [])
        filters = convo.get("filters", {})

        # Build Markdown content
        lines: list[str] = []
        lines.append(f"# {title}")
        lines.append("")
        if created:
            lines.append(f"**Created:** {created}")
        if filters:
            filter_parts = []
            if filters.get("chat_name"):
                filter_parts.append(f"Chat: {filters['chat_name']}")
            if filters.get("sender"):
                filter_parts.append(f"Sender: {filters['sender']}")
            if filters.get("days"):
                filter_parts.append(f"Last {filters['days']} days")
            if filter_parts:
                lines.append(f"**Filters:** {', '.join(filter_parts)}")
        lines.append("")
        lines.append("---")
        lines.append("")

        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            timestamp = msg.get("created_at", "")
            sources_json = msg.get("sources", "")
            if role == "user":
                lines.append(f"### 🧑 You")
            else:
                lines.append(f"### 🤖 Lucy")
            if timestamp:
                lines.append(f"*{timestamp}*")
            lines.append("")
            lines.append(content)
            lines.append("")

            # Append sources for assistant messages
            if role == "assistant" and sources_json:
                try:
                    src_list = json.loads(sources_json)
                    if src_list:
                        lines.append("<details>")
                        lines.append("<summary>📚 Sources</summary>")
                        lines.append("")
                        for i, src in enumerate(src_list):
                            src_content = src.get("content", "")[:300]
                            score = src.get("score")
                            score_str = f" ({score:.0%})" if score and 0 < score <= 1 else ""
                            if src_content:
                                lines.append(f"**{i + 1}.** {src_content}{'…' if len(src_content) >= 300 else ''}{score_str}")
                            else:
                                lines.append(f"**{i + 1}.** _(empty source)_")
                            lines.append("")
                        lines.append("</details>")
                        lines.append("")
                except (ValueError, TypeError):
                    pass  # Skip malformed sources JSON

        markdown = "\n".join(lines)

        return jsonify({
            "title": title,
            "markdown": markdown,
            "message_count": len(messages),
        }), 200

    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Export conversation error: {e}\n{trace}")
        return jsonify({"error": str(e), "traceback": trace}), 500


@app.route("/conversations/<conversation_id>", methods=["DELETE"])
@app.route("/conversation/<conversation_id>", methods=["DELETE"])  # backwards compat
def delete_conversation_endpoint(conversation_id: str):
    """Delete a conversation and all its data (SQLite + Redis)."""
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
    """Get all settings grouped by category, with secrets masked by default."""
    try:
        import settings_db
        unmask = request.args.get("unmask", "false").lower() == "true"
        if unmask:
            all_settings = settings_db.get_all_settings()
        else:
            all_settings = settings_db.get_all_settings_masked()
        return jsonify(all_settings), 200
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Config get error: {e}\n{trace}")
        return jsonify({"error": str(e), "traceback": trace}), 500


@app.route("/config/secret/<key>", methods=["GET"])
def get_secret_value(key):
    """Get the unmasked value of a single secret setting."""
    try:
        import settings_db
        row = settings_db.get_setting_row(key)
        if not row:
            return jsonify({"error": "Setting not found"}), 404
        if row["type"] != "secret":
            return jsonify({"error": "Not a secret setting"}), 400
        return jsonify({"key": key, "value": row["value"]}), 200
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Secret fetch error: {e}\n{trace}")
        return jsonify({"error": str(e), "traceback": trace}), 500


@app.route("/config", methods=["PUT"])
def update_config():
    """Update one or more settings."""
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


@app.route("/config/meta", methods=["GET"])
def get_config_meta():
    """Get configuration metadata for UI rendering.
    
    Returns category labels/ordering and select-type option lists
    so the UI can render proper form controls without hardcoding.
    """
    try:
        import settings_db
        return jsonify({
            "category_meta": settings_db.CATEGORY_META,
            "select_options": settings_db.SELECT_OPTIONS,
        }), 200
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Config meta error: {e}\n{trace}")
        return jsonify({"error": str(e), "traceback": trace}), 500


@app.route("/config/reset", methods=["POST"])
def reset_config():
    """Reset settings to defaults, optionally for a specific category."""
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


@app.route("/config/export", methods=["GET"])
def export_config():
    """Export all settings as JSON (with secrets unmasked for backup)."""
    try:
        import settings_db
        # Use get_all_settings (not masked) for export
        all_settings = settings_db.get_all_settings()
        return jsonify(all_settings), 200
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Config export error: {e}\n{trace}")
        return jsonify({"error": str(e), "traceback": trace}), 500


@app.route("/config/import", methods=["POST"])
def import_config():
    """Import settings from JSON."""
    try:
        import settings_db
        data = request.json or {}
        
        if not data:
            return jsonify({"error": "No settings provided"}), 400
        
        # Flatten nested category structure to flat key-value pairs
        flat_settings = {}
        for category, settings in data.items():
            if isinstance(settings, dict):
                for key, info in settings.items():
                    if isinstance(info, dict) and "value" in info:
                        flat_settings[key] = info["value"]
                    else:
                        flat_settings[key] = str(info)
        
        updated_keys = settings_db.set_settings(flat_settings)
        
        return jsonify({
            "status": "ok",
            "updated": updated_keys,
            "count": len(updated_keys)
        }), 200
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Config import error: {e}\n{trace}")
        return jsonify({"error": str(e), "traceback": trace}), 500


# =============================================================================
# COST TRACKING ENDPOINTS
# =============================================================================

@app.route("/costs/session", methods=["GET"])
def costs_session():
    """Get current session cost total and recent events."""
    try:
        n = request.args.get("n", 20, type=int)
        return jsonify({
            "session_total_usd": round(METER.session_total, 6),
            "recent_events": METER.get_recent_events(n),
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/costs/summary", methods=["GET"])
def costs_summary():
    """Get daily cost summary for the last N days."""
    try:
        days = request.args.get("days", 7, type=int)
        daily = cost_db.get_daily_summary(days=days)
        total = cost_db.get_total_cost(days=days)
        by_kind = cost_db.get_cost_by_kind(days=days)
        return jsonify({
            "days": days,
            "total_cost_usd": round(total, 6),
            "by_kind": {k: round(v, 6) for k, v in by_kind.items()},
            "daily": daily,
        }), 200
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Cost summary error: {e}\n{trace}")
        return jsonify({"error": str(e)}), 500


@app.route("/costs/events", methods=["GET"])
def costs_events():
    """Get paginated cost event log with optional filters."""
    try:
        limit = min(request.args.get("limit", 50, type=int), 200)
        offset = request.args.get("offset", 0, type=int)
        conversation_id = request.args.get("conversation_id")
        kind = request.args.get("kind")
        
        events = cost_db.get_events(
            limit=limit,
            offset=offset,
            conversation_id=conversation_id,
            kind=kind,
        )
        return jsonify({"events": events, "count": len(events)}), 200
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Cost events error: {e}\n{trace}")
        return jsonify({"error": str(e)}), 500


@app.route("/costs/breakdown", methods=["GET"])
def costs_breakdown():
    """Get cost breakdown by provider and model."""
    try:
        days = request.args.get("days", 7, type=int)
        by_model = cost_db.get_cost_by_model(days=days)
        by_kind = cost_db.get_cost_by_kind(days=days)
        total = cost_db.get_total_cost(days=days)
        return jsonify({
            "days": days,
            "total_cost_usd": round(total, 6),
            "by_kind": {k: round(v, 6) for k, v in by_kind.items()},
            "by_model": by_model,
        }), 200
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Cost breakdown error: {e}\n{trace}")
        return jsonify({"error": str(e)}), 500


# =============================================================================
# ROOT ENDPOINT
# =============================================================================

@app.route("/", methods=["GET"])
def index():
    """Root endpoint — show app status and enabled plugins."""
    enabled = plugin_registry.discovered_plugins()
    enabled_names = [f"{info['icon']} {info['display_name']}" 
                     for name, info in enabled.items() if info['enabled']]
    plugins_str = ", ".join(enabled_names) if enabled_names else "None"
    return jsonify({
        "status": "running",
        "app": "RAG Assistant",
        "enabled_plugins": plugins_str,
    }), 200


if __name__ == "__main__":
    os.environ["LOCAL"] = "TRUE"
    is_debug = settings.log_level == "DEBUG"
    app.run(host="0.0.0.0", port=8765,
            debug=is_debug, use_reloader=True, exclude_patterns=[".venv/*", "*/.venv/*"])
