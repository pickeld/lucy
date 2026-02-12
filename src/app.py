import base64
import os
import sys
import time
from typing import Any, Dict, Optional, Union

# Force unbuffered output for immediate logging
# Set PYTHONUNBUFFERED equivalent at runtime
os.environ['PYTHONUNBUFFERED'] = '1'

print("🚀 Starting WhatsApp-GPT application...", flush=True)

from flask import Flask, jsonify, redirect, render_template_string, request
from requests.models import Response

print("✅ Flask imported", flush=True)

from config import config
print("✅ Config loaded", flush=True)

from llamaindex_rag import get_rag
print("✅ RAG module imported", flush=True)

from session import get_session_manager, ConversationSession, EntityType
print("✅ Session module imported", flush=True)

from utils.globals import send_request
from utils.logger import logger
print("✅ Utils imported", flush=True)

import traceback

from whatsapp import create_whatsapp_message, group_manager
print("✅ WhatsApp module imported", flush=True)

app = Flask(__name__)
print("✅ Flask app created", flush=True)


# Initialize singletons
rag = get_rag()
print("✅ RAG instance initialized", flush=True)
session_manager = get_session_manager()
print("✅ Session manager initialized", flush=True)


def pass_filter(payload):
    if payload.get('event') == "message_ack" or \
            payload.get("from").endswith("@newsletter") or \
            payload.get("from").endswith("@broadcast") or \
            payload.get("_data", {}).get("type") in ["e2e_notification", "notification_template"]:
        return False

    return True


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "up"}), 200


@app.route("/rag/query", methods=["POST"])
def rag_query():
    """Query the RAG system with a natural language question.

    Request body:
        {
            "question": "who said they would be late?",
            "session_id": "uuid",  # optional, for context-aware conversations
            "k": 10,  # optional, number of context documents
            "filter_chat_name": "Work Group",  # optional (auto-set from session if not provided)
            "filter_sender": "John",  # optional (auto-set from session if not provided)
            "filter_days": 7,  # optional (1=24h, 3=3 days, 7=week, 30=month, null=all time)
            "conversation_history": [  # optional, deprecated - use session_id instead
                {"role": "user", "content": "what did John say?"},
                {"role": "assistant", "content": "John said..."}
            ]
        }

    Response:
        {
            "answer": "...",
            "question": "...",
            "session_id": "...",
            "context": {"chat_filter": "...", "sender_filter": "..."},
            "stats": {"total_documents": 123}
        }
    """
    try:
        data = request.json or {}
        question = data.get("question")

        if not question:
            return jsonify({"error": "Missing 'question' in request body"}), 400

        # Session management
        session_id = data.get("session_id")
        session: Optional[ConversationSession] = None
        
        if session_id:
            session = session_manager.get_session(session_id)
        
        if session is None:
            # Create new session
            session = session_manager.create_session()
        
        k = data.get("k", 10)
        
        # Use filters from request, falling back to session context
        filter_chat_name = data.get("filter_chat_name") or session.active_chat_filter
        filter_sender = data.get("filter_sender") or session.active_sender_filter
        filter_days = data.get("filter_days")
        
        # Update session context if explicit filters provided
        if data.get("filter_chat_name"):
            session.set_chat_context(data.get("filter_chat_name"))
        if data.get("filter_sender"):
            session.set_sender_context(data.get("filter_sender"))
        
        # Auto-detect chat/sender mentions in the question and update context
        chat_list = rag.get_chat_list()
        sender_list = rag.get_sender_list()
        session_manager.extract_and_track_entities(
            session, question, known_chats=chat_list, known_senders=sender_list
        )
        
        # Re-check filters after entity extraction (may have updated)
        filter_chat_name = filter_chat_name or session.active_chat_filter
        
        # Get conversation history from session or request
        conversation_history = data.get("conversation_history")
        if not conversation_history and session.turns:
            conversation_history = session.get_conversation_history(max_turns=10)

        answer = rag.query(
            question=question,
            k=k,
            filter_chat_name=filter_chat_name,
            filter_sender=filter_sender,
            filter_days=filter_days,
            conversation_history=conversation_history
        )
        
        # Record this turn in the session
        session_manager.add_turn_to_session(
            session=session,
            user_query=question,
            assistant_response=answer,
            filters={
                "chat": filter_chat_name,
                "sender": filter_sender,
                "days": filter_days
            }
        )

        stats = rag.get_stats()

        return jsonify({
            "answer": answer,
            "question": question,
            "session_id": session.session_id,
            "context": {
                "chat_filter": session.active_chat_filter,
                "sender_filter": session.active_sender_filter,
                "entities_tracked": list(session.mentioned_entities.keys()),
                "turn_number": session.current_turn_number
            },
            "stats": stats
        }), 200

    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"RAG query error: {e}\n{trace}")
        return jsonify({"error": str(e), "traceback": trace}), 500


@app.route("/rag/search", methods=["POST"])
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


# =============================================================================
# SESSION MANAGEMENT ENDPOINTS
# =============================================================================

@app.route("/session/create", methods=["POST"])
def create_session():
    """Create a new conversation session.
    
    Request body (all optional):
        {
            "initial_chat": "Family Group",  # optional initial chat filter
            "initial_sender": "John"          # optional initial sender filter
        }
    
    Response:
        {
            "session_id": "uuid",
            "created_at": "...",
            "context": {...}
        }
    """
    try:
        data = request.json or {}
        initial_chat = data.get("initial_chat")
        initial_sender = data.get("initial_sender")
        
        session = session_manager.create_session(
            initial_chat=initial_chat,
            initial_sender=initial_sender
        )
        
        return jsonify({
            "session_id": session.session_id,
            "created_at": session.created_at.isoformat(),
            "context": {
                "chat_filter": session.active_chat_filter,
                "sender_filter": session.active_sender_filter,
                "summary": session.get_context_summary()
            }
        }), 201
        
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Session create error: {e}\n{trace}")
        return jsonify({"error": str(e), "traceback": trace}), 500


@app.route("/session/<session_id>", methods=["GET"])
def get_session(session_id: str):
    """Get session state by ID.
    
    Response:
        {
            "session_id": "...",
            "context": {...},
            "history": [...],
            "entities": {...}
        }
    """
    try:
        session = session_manager.get_session(session_id)
        
        if session is None:
            return jsonify({"error": "Session not found or expired"}), 404
        
        return jsonify({
            "session_id": session.session_id,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "context": {
                "chat_filter": session.active_chat_filter,
                "sender_filter": session.active_sender_filter,
                "time_range": session.active_time_range,
                "summary": session.get_context_summary()
            },
            "history": session.get_conversation_history(),
            "entities": {
                name: {
                    "type": info.entity_type.value,
                    "mentions": info.mentions_count,
                    "last_turn": info.last_mentioned_turn
                }
                for name, info in session.mentioned_entities.items()
            },
            "turn_count": session.current_turn_number,
            "facts_count": len(session.established_facts)
        }), 200
        
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Session get error: {e}\n{trace}")
        return jsonify({"error": str(e), "traceback": trace}), 500


@app.route("/session/<session_id>", methods=["DELETE"])
def delete_session(session_id: str):
    """Delete a session.
    
    Response:
        {
            "status": "ok",
            "deleted": true
        }
    """
    try:
        deleted = session_manager.delete_session(session_id)
        
        return jsonify({
            "status": "ok",
            "deleted": deleted
        }), 200
        
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Session delete error: {e}\n{trace}")
        return jsonify({"error": str(e), "traceback": trace}), 500


@app.route("/session/<session_id>/context", methods=["PUT", "PATCH"])
def update_session_context(session_id: str):
    """Update session context (chat/sender filters).
    
    Request body:
        {
            "chat_name": "Family Group",  # set chat filter (null to clear)
            "sender_name": "John"          # set sender filter (null to clear)
        }
    
    Response:
        {
            "status": "ok",
            "context": {...}
        }
    """
    try:
        session = session_manager.get_session(session_id)
        
        if session is None:
            return jsonify({"error": "Session not found or expired"}), 404
        
        data = request.json or {}
        
        # Update context (None means don't change, empty string means clear)
        if "chat_name" in data:
            chat_val = data["chat_name"]
            session.set_chat_context(chat_val if chat_val else None)
        
        if "sender_name" in data:
            sender_val = data["sender_name"]
            session.set_sender_context(sender_val if sender_val else None)
        
        session_manager.save_session(session)
        
        return jsonify({
            "status": "ok",
            "context": {
                "chat_filter": session.active_chat_filter,
                "sender_filter": session.active_sender_filter,
                "summary": session.get_context_summary()
            }
        }), 200
        
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Session context update error: {e}\n{trace}")
        return jsonify({"error": str(e), "traceback": trace}), 500


@app.route("/session/<session_id>/clear", methods=["POST"])
def clear_session_context(session_id: str):
    """Clear all session context (filters, entities, history).
    
    Response:
        {
            "status": "ok"
        }
    """
    try:
        session = session_manager.get_session(session_id)
        
        if session is None:
            return jsonify({"error": "Session not found or expired"}), 404
        
        # Clear all context
        session.active_chat_filter = None
        session.active_sender_filter = None
        session.active_time_range = None
        session.mentioned_entities.clear()
        session.resolved_references.clear()
        session.turns.clear()
        session.retrieved_context.clear()
        session.established_facts.clear()
        
        session_manager.save_session(session)
        
        return jsonify({"status": "ok"}), 200
        
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Session clear error: {e}\n{trace}")
        return jsonify({"error": str(e), "traceback": trace}), 500


@app.route("/session/stats", methods=["GET"])
def session_stats():
    """Get session management statistics.
    
    Response:
        {
            "active_sessions": 5,
            "ttl_minutes": 30,
            "max_history": 20
        }
    """
    try:
        stats = session_manager.get_session_stats()
        return jsonify(stats), 200
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Session stats error: {e}\n{trace}")
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


@app.route("/test", methods=["GET"])
def test():
    # send a test message to yourself
    test_message = f"my name is david and i 39 years old, i have two kids, mia which is 4 and ben which is 6 and i work as a software developer."
    send_request(method="POST",
                 endpoint="/api/sendText",
                 payload={
                     "chatId": "972547755011@c.us",
                     "text": test_message,
                     "session": config.waha_session_name
                 }
                 )
    return jsonify({"status": "test message sent"}), 200


@app.route("/webhook", methods=["POST"])
def webhook():

    request_data = request.json or {}
    payload = request_data.get("payload", {})
    
    
    try:
        if not pass_filter(payload):
            return jsonify({"status": "ok"}), 200

        msg = create_whatsapp_message(payload)

        chat_id = msg.group.id if msg.is_group else msg.contact.number
        chat_name = msg.group.name if msg.is_group else msg.contact.name
        logger.info(f"Received message: {chat_name} ({chat_id}) - {msg.message}")

        # Store message in RAG vector store
        if msg.message:
            rag.add_message(
                thread_id=chat_id or "UNKNOWN",
                chat_id=chat_id or "UNKNOWN",
                chat_name=chat_name or "UNKNOWN",
                is_group=msg.is_group,
                sender=str(msg.contact.name),
                message=msg.message,
                timestamp=str(msg.timestamp) if msg.timestamp else "0"
            )
            logger.debug(f"Processed message: {chat_name} || {msg}")
        
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(
            f"Error processing webhook: {e} ::: {payload}\n{trace}")
        return jsonify({"error": str(e), "traceback": trace}), 500


@app.route("/", methods=["GET"])
def index():
    try:
        response: Union[Dict[str, Any], Response] = send_request(
            "GET", f"/api/sessions/{config.waha_session_name}")
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
        "GET", f"/api/{config.waha_session_name}/auth/qr")
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
    session_name = config.waha_session_name
    send_request(method="POST", endpoint="/api/sessions/start",
                 payload={"name": session_name})

    send_request("PUT", f"/api/sessions/{session_name}", {
        "config": {
            "webhooks": [
                {
                    "url": config.webhook_url,
                    "events": ["message.any"]
                }
            ]
        }
    })
    time.sleep(2)
    return redirect("/qr_code")


if __name__ == "__main__":
    os.environ["LOCAL"] = "TRUE"
    app.run(host="0.0.0.0", port=8765,
            debug=True if config.log_level == "DEBUG" else False)
