"""WhatsApp channel plugin implementation.

Implements the ChannelPlugin interface for WhatsApp integration
using WAHA (WhatsApp HTTP API) as the backend.

Provides:
    - Webhook endpoint for receiving WhatsApp messages
    - WAHA session management (pairing, QR code)
    - Group cache management
    - Health check for WAHA connectivity
"""

import base64
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from flask import Blueprint, Flask, jsonify, redirect, render_template_string, request
from requests.models import Response

from config import settings
from plugins.base import ChannelPlugin
from utils.globals import send_request
from utils.logger import logger


class WhatsAppPlugin(ChannelPlugin):
    """WhatsApp integration via WAHA (WhatsApp HTTP API).
    
    This plugin handles:
    - Receiving WhatsApp messages via webhook
    - Parsing messages (text, image, voice, video, document, etc.)
    - WAHA session management (QR code pairing)
    - Group info caching
    """
    
    def __init__(self):
        self._executor: Optional[ThreadPoolExecutor] = None
        self._rag = None
    
    # -------------------------------------------------------------------------
    # Identity
    # -------------------------------------------------------------------------
    
    @property
    def name(self) -> str:
        return "whatsapp"
    
    @property
    def display_name(self) -> str:
        return "WhatsApp"
    
    @property
    def icon(self) -> str:
        return "💬"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "WhatsApp messaging integration via WAHA API"
    
    # -------------------------------------------------------------------------
    # Settings
    # -------------------------------------------------------------------------
    
    def get_default_settings(self) -> List[Tuple[str, str, str, str, str]]:
        return [
            ("chat_prefix", "??", "whatsapp", "text", "Prefix to trigger AI chat response"),
            ("dalle_prefix", "!!", "whatsapp", "text", "Prefix to trigger DALL-E image generation"),
            ("waha_session_name", "default", "whatsapp", "text", "WAHA WhatsApp session name"),
            ("dalle_model", "dall-e-3", "whatsapp", "text", "DALL-E model version"),
            ("waha_base_url", "http://waha:3000", "whatsapp", "text", "WAHA server URL"),
            ("waha_api_key", "", "whatsapp", "secret", "WAHA API key"),
            ("webhook_url", "http://app:8765/plugins/whatsapp/webhook", "whatsapp", "text", "Webhook callback URL"),
        ]
    
    def get_env_key_map(self) -> Dict[str, str]:
        return {
            "chat_prefix": "CHAT_PREFIX",
            "dalle_prefix": "DALLE_PREFIX",
            "waha_session_name": "WAHA_SESSION_NAME",
            "dalle_model": "DALLE_MODEL",
            "waha_base_url": "WAHA_BASE_URL",
            "waha_api_key": "WAHA_API_KEY",
            "webhook_url": "WEBHOOK_URL",
        }
    
    def get_category_meta(self) -> Dict[str, Dict[str, str]]:
        return {
            "whatsapp": {"label": "💬 WhatsApp", "order": "10"}
        }
    
    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
    
    def initialize(self, app: Flask) -> None:
        """Initialize WhatsApp plugin — set up thread pool and RAG reference."""
        self._executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="wa-webhook"
        )
        
        # Get RAG instance (lazy — don't import at module level)
        from llamaindex_rag import get_rag
        self._rag = get_rag()
        
        logger.info("WhatsApp plugin initialized")
    
    def shutdown(self) -> None:
        """Shutdown WhatsApp plugin — clean up thread pool."""
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None
        self._rag = None
        logger.info("WhatsApp plugin shut down")
    
    # -------------------------------------------------------------------------
    # Flask Blueprint
    # -------------------------------------------------------------------------
    
    def get_blueprint(self) -> Blueprint:
        """Create Flask Blueprint with all WhatsApp routes."""
        bp = Blueprint("whatsapp", __name__, url_prefix="/plugins/whatsapp")
        plugin = self  # Capture for closures
        
        # --- Webhook ---
        
        @bp.route("/webhook", methods=["POST"])
        def webhook():
            """Receive WhatsApp webhook events from WAHA."""
            request_data = request.json or {}
            payload = request_data.get("payload", {})
            
            try:
                if not plugin.should_process(payload):
                    return jsonify({"status": "ok"}), 200
                
                # Submit to background thread pool
                if plugin._executor:
                    plugin._executor.submit(plugin._process_webhook_payload, payload)
                
                return jsonify({"status": "ok"}), 200
            except Exception as e:
                trace = traceback.format_exc()
                logger.error(f"Error submitting webhook: {e} ::: {payload}\n{trace}")
                return jsonify({"error": str(e), "traceback": trace}), 500
        
        # --- WAHA Session Management ---
        
        @bp.route("/status", methods=["GET"])
        def status():
            """Check WAHA session status."""
            try:
                response = send_request(
                    "GET", f"/api/sessions/{settings.waha_session_name}")
                
                if isinstance(response, dict):
                    if (response.get("status") == "WORKING" and 
                            response.get("engine", {}).get("state") == "CONNECTED"):
                        return jsonify({"status": "connected", "session": response}), 200
                    elif response.get("status") == "SCAN_QR_CODE":
                        return jsonify({"status": "needs_pairing", "redirect": "/plugins/whatsapp/qr_code"}), 200
                
                return jsonify({"status": "unknown", "response": str(response)}), 200
            except Exception as e:
                logger.error(f"Error checking session status: {e}")
                return jsonify({"status": "error", "error": str(e)}), 500
        
        @bp.route("/qr_code", methods=["GET"])
        def qr_code():
            """Display QR code for WhatsApp pairing."""
            qr_response = send_request(
                "GET", f"/api/{settings.waha_session_name}/auth/qr")
            qr_image_data = qr_response.content if isinstance(qr_response, Response) else None
            if qr_image_data:
                qr_base64 = base64.b64encode(qr_image_data).decode("utf-8")
                html = f"<h1>Scan to Pair WhatsApp</h1><img src='data:image/png;base64,{qr_base64}'>"
                return render_template_string(html)
            else:
                return "QR code not available yet. Please refresh in a few seconds.", 200
        
        @bp.route("/pair", methods=["GET"])
        def pair():
            """Start WAHA session and configure webhook."""
            session_name = settings.waha_session_name
            send_request(method="POST", endpoint="/api/sessions/start",
                         payload={"name": session_name})
            
            webhook_url = settings.get("webhook_url", "http://app:8765/plugins/whatsapp/webhook")
            send_request("PUT", f"/api/sessions/{session_name}", {
                "config": {
                    "webhooks": [
                        {
                            "url": webhook_url,
                            "events": ["message.any"]
                        }
                    ]
                }
            })
            time.sleep(2)
            return redirect("/plugins/whatsapp/qr_code")
        
        # --- Cache Management ---
        
        @bp.route("/cache/groups/clear", methods=["POST", "DELETE"])
        def clear_groups_cache():
            """Clear all cached group data from Redis."""
            try:
                from plugins.whatsapp.handler import group_manager
                count = group_manager.clear_all_groups_cache()
                return jsonify({"status": "ok", "deleted_count": count}), 200
            except Exception as e:
                trace = traceback.format_exc()
                logger.error(f"Failed to clear groups cache: {e}\n{trace}")
                return jsonify({"error": str(e), "traceback": trace}), 500
        
        @bp.route("/cache/groups/<group_id>/refresh", methods=["POST"])
        def refresh_group_cache(group_id: str):
            """Refresh cache for a specific group."""
            try:
                from plugins.whatsapp.handler import group_manager
                group = group_manager.refresh_group(group_id)
                if group:
                    return jsonify({"status": "ok", "group": group.to_dict()}), 200
                else:
                    return jsonify({"status": "error", "message": "Failed to fetch group"}), 404
            except Exception as e:
                trace = traceback.format_exc()
                logger.error(f"Failed to refresh group cache: {e}\n{trace}")
                return jsonify({"error": str(e), "traceback": trace}), 500
        
        # --- Test ---
        
        @bp.route("/test", methods=["GET"])
        def test():
            """Send a test message."""
            test_message = "Test message from WhatsApp plugin"
            send_request(method="POST",
                         endpoint="/api/sendText",
                         payload={
                             "chatId": "972547755011@c.us",
                             "text": test_message,
                             "session": settings.waha_session_name
                         })
            return jsonify({"status": "test message sent"}), 200
        
        return bp
    
    def get_legacy_routes(self) -> List[Tuple[str, str, str]]:
        """Register legacy /webhook route for backward compat with existing WAHA config."""
        return [
            ("/webhook", "whatsapp.webhook", "POST"),
        ]
    
    # -------------------------------------------------------------------------
    # Health
    # -------------------------------------------------------------------------
    
    def health_check(self) -> Dict[str, str]:
        """Check WAHA connectivity."""
        try:
            import requests as req
            waha_url = settings.waha_base_url
            resp = req.get(
                f"{waha_url}/api/sessions",
                headers={"X-Api-Key": settings.waha_api_key},
                timeout=5
            )
            if resp.status_code < 500:
                return {"waha": "connected"}
            else:
                return {"waha": f"error: HTTP {resp.status_code}"}
        except Exception as e:
            return {"waha": f"error: {str(e)}"}
    
    # -------------------------------------------------------------------------
    # Webhook Processing
    # -------------------------------------------------------------------------
    
    def should_process(self, payload: Dict[str, Any]) -> bool:
        """Filter out non-message webhook events."""
        if payload.get('event') == "message_ack":
            return False
        
        from_field = payload.get("from") or ""
        if from_field.endswith("@newsletter") or from_field.endswith("@broadcast"):
            return False
        
        data_type = payload.get("_data", {}).get("type")
        if data_type in ["e2e_notification", "notification_template"]:
            return False
        
        return True
    
    def process_webhook(self, payload: Dict[str, Any]) -> Optional[Any]:
        """Process a WhatsApp webhook payload and return a RAG document.
        
        This is the high-level interface called by the registry.
        For the actual background processing, see _process_webhook_payload().
        """
        from plugins.whatsapp.handler import create_whatsapp_message
        
        msg = create_whatsapp_message(payload)
        if msg.message:
            thread_id = (msg.group.id if msg.is_group else msg.contact.id) or "unknown"
            return msg.to_rag_document(thread_id=thread_id)
        return None
    
    def _process_webhook_payload(self, payload: Dict[str, Any]) -> None:
        """Process a webhook payload in a background thread.
        
        Handles the heavy work: creating the message object (which may
        trigger Whisper transcription or GPT-4 Vision), then storing
        the result in the RAG vector store.
        """
        try:
            from plugins.whatsapp.handler import create_whatsapp_message
            
            msg = create_whatsapp_message(payload)
            
            # Determine chat identification
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
            if msg.message and self._rag:
                self._rag.add_message(
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
