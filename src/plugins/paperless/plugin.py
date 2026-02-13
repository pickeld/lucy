"""Paperless-NGX plugin for document management integration."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from flask import Blueprint, Flask, jsonify, request

from config import settings
from plugins.base import ChannelPlugin

from .client import PaperlessClient
from .sync import DocumentSyncer

logger = logging.getLogger(__name__)


class PaperlessPlugin(ChannelPlugin):
    """Paperless-NGX document management integration.
    
    Syncs documents from Paperless-NGX and indexes them in the RAG system.
    """
    
    def __init__(self):
        self._client: Optional[PaperlessClient] = None
        self._syncer: Optional[DocumentSyncer] = None
        self._rag = None
    
    # -------------------------------------------------------------------------
    # Identity
    # -------------------------------------------------------------------------
    
    @property
    def name(self) -> str:
        return "paperless"
    
    @property
    def display_name(self) -> str:
        return "Paperless-NGX"
    
    @property
    def icon(self) -> str:
        return "📄"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Document management system integration for RAG indexing"
    
    # -------------------------------------------------------------------------
    # Settings
    # -------------------------------------------------------------------------
    
    def get_default_settings(self) -> List[Tuple[str, str, str, str, str]]:
        return [
            ("paperless_url", "http://paperless:8000", "paperless", "text", "Paperless-NGX server URL"),
            ("paperless_token", "", "paperless", "secret", "Paperless-NGX API token"),
            ("paperless_sync_interval", "3600", "paperless", "int", "Sync interval in seconds (0 = manual only)"),
            ("paperless_sync_tags", "", "paperless", "text", "Comma-separated tag names to sync (empty = all)"),
            ("paperless_max_docs", "1000", "paperless", "int", "Maximum documents to sync per run"),
        ]
    
    def get_env_key_map(self) -> Dict[str, str]:
        return {
            "paperless_url": "PAPERLESS_URL",
            "paperless_token": "PAPERLESS_TOKEN",
            "paperless_sync_interval": "PAPERLESS_SYNC_INTERVAL",
            "paperless_sync_tags": "PAPERLESS_SYNC_TAGS",
            "paperless_max_docs": "PAPERLESS_MAX_DOCS",
        }
    
    def get_category_meta(self) -> Dict[str, Dict[str, str]]:
        return {
            "paperless": {"label": "📄 Paperless-NGX", "order": "11"}
        }
    
    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
    
    def initialize(self, app: Flask) -> None:
        """Initialize Paperless plugin."""
        url = settings.paperless_url
        token = settings.paperless_token
        
        if not token:
            logger.warning("Paperless token not configured, plugin will be inactive")
            return
        
        self._client = PaperlessClient(url, token)
        
        # Get RAG instance
        from llamaindex_rag import get_rag
        self._rag = get_rag()
        
        self._syncer = DocumentSyncer(self._client, self._rag)
        
        logger.info("Paperless-NGX plugin initialized")
    
    def shutdown(self) -> None:
        """Shutdown Paperless plugin."""
        self._client = None
        self._syncer = None
        self._rag = None
        logger.info("Paperless-NGX plugin shut down")
    
    # -------------------------------------------------------------------------
    # Flask Blueprint
    # -------------------------------------------------------------------------
    
    def get_blueprint(self) -> Blueprint:
        """Create Flask Blueprint with Paperless routes."""
        bp = Blueprint("paperless", __name__, url_prefix="/plugins/paperless")
        plugin = self  # Capture for closures
        
        @bp.route("/sync", methods=["POST"])
        def sync():
            """Trigger manual document sync."""
            if not plugin._syncer:
                return jsonify({"error": "Plugin not initialized"}), 500
            
            max_docs = int(settings.get("paperless_max_docs", 1000))
            tags_str = settings.get("paperless_sync_tags", "")
            tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else None
            
            result = plugin._syncer.sync_documents(
                max_docs=max_docs,
                tags_filter=tags,
            )
            
            return jsonify(result), 200
        
        @bp.route("/sync/status", methods=["GET"])
        def sync_status():
            """Get sync status."""
            if not plugin._syncer:
                return jsonify({"error": "Plugin not initialized"}), 500
            
            return jsonify({
                "is_syncing": plugin._syncer.is_syncing,
                "last_sync": plugin._syncer.last_sync_time,
                "synced_count": plugin._syncer.synced_count,
            }), 200
        
        @bp.route("/test", methods=["GET"])
        def test():
            """Test Paperless connection."""
            if not plugin._client:
                return jsonify({"error": "Plugin not initialized"}), 500
            
            if plugin._client.test_connection():
                return jsonify({"status": "connected"}), 200
            else:
                return jsonify({"status": "error", "message": "Connection failed"}), 500
        
        return bp
    
    # -------------------------------------------------------------------------
    # Health
    # -------------------------------------------------------------------------
    
    def health_check(self) -> Dict[str, str]:
        """Check Paperless connectivity."""
        if not self._client:
            return {"paperless": "not initialized"}
        
        if self._client.test_connection():
            return {"paperless": "connected"}
        else:
            return {"paperless": "error: connection failed"}
    
    # -------------------------------------------------------------------------
    # Webhook Processing
    # -------------------------------------------------------------------------
    
    def process_webhook(self, payload: Dict[str, Any]) -> Optional[Any]:
        """Process Paperless post-consumption webhook.
        
        Not implemented yet — for future use when Paperless
        sends webhooks on document creation.
        """
        return None
