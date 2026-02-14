"""Gmail plugin for email integration with the RAG system."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from flask import Blueprint, Flask, jsonify, request

from config import settings
from plugins.base import ChannelPlugin

from .client import GmailClient
from .sync import DEFAULT_PROCESSED_LABEL, EmailSyncer

logger = logging.getLogger(__name__)


class GmailPlugin(ChannelPlugin):
    """Gmail email integration for RAG indexing.

    Syncs emails from selected Gmail folders and indexes them (body +
    text-based attachments) in the RAG vector store.  Processed emails
    are labeled in Gmail with a configurable label (default:
    ``rag-indexed``) so they are automatically excluded from future syncs.
    """

    def __init__(self):
        self._client: Optional[GmailClient] = None
        self._syncer: Optional[EmailSyncer] = None
        self._rag = None

    # -------------------------------------------------------------------------
    # Identity
    # -------------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "gmail"

    @property
    def display_name(self) -> str:
        return "Gmail"

    @property
    def icon(self) -> str:
        return "📧"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Gmail email integration for RAG indexing"

    # -------------------------------------------------------------------------
    # Settings
    # -------------------------------------------------------------------------

    def get_default_settings(self) -> List[Tuple[str, str, str, str, str]]:
        return [
            (
                "gmail_client_id",
                "",
                "gmail",
                "secret",
                "Google OAuth2 Client ID (from Google Cloud Console)",
            ),
            (
                "gmail_client_secret",
                "",
                "gmail",
                "secret",
                "Google OAuth2 Client Secret",
            ),
            (
                "gmail_refresh_token",
                "",
                "gmail",
                "secret",
                "OAuth2 refresh token (obtained after authorization)",
            ),
            (
                "gmail_sync_folders",
                "",
                "gmail",
                "text",
                "Comma-separated folder/label names to sync (empty = INBOX only)",
            ),
            (
                "gmail_sync_interval",
                "3600",
                "gmail",
                "int",
                "Sync interval in seconds (0 = manual only)",
            ),
            (
                "gmail_max_emails",
                "500",
                "gmail",
                "int",
                "Maximum emails to sync per run",
            ),
            (
                "gmail_processed_label",
                DEFAULT_PROCESSED_LABEL,
                "gmail",
                "text",
                "Label applied to emails after RAG indexing (prevents reprocessing)",
            ),
            (
                "gmail_include_attachments",
                "true",
                "gmail",
                "bool",
                "Extract and index text from PDF/DOCX attachments",
            ),
        ]

    def get_env_key_map(self) -> Dict[str, str]:
        return {
            "gmail_client_id": "GMAIL_CLIENT_ID",
            "gmail_client_secret": "GMAIL_CLIENT_SECRET",
            "gmail_refresh_token": "GMAIL_REFRESH_TOKEN",
            "gmail_sync_folders": "GMAIL_SYNC_FOLDERS",
            "gmail_sync_interval": "GMAIL_SYNC_INTERVAL",
            "gmail_max_emails": "GMAIL_MAX_EMAILS",
            "gmail_processed_label": "GMAIL_PROCESSED_LABEL",
            "gmail_include_attachments": "GMAIL_INCLUDE_ATTACHMENTS",
        }

    def get_category_meta(self) -> Dict[str, Dict[str, str]]:
        return {"gmail": {"label": "📧 Gmail", "order": "12"}}

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def initialize(self, app: Flask) -> None:
        """Initialize Gmail plugin."""
        import settings_db

        client_id = settings_db.get_setting_value("gmail_client_id") or ""
        client_secret = settings_db.get_setting_value("gmail_client_secret") or ""
        refresh_token = settings_db.get_setting_value("gmail_refresh_token") or ""

        if not refresh_token:
            logger.warning(
                "Gmail refresh token not configured — plugin will be "
                "inactive until authorization is completed"
            )
            return

        if not client_id or not client_secret:
            logger.warning(
                "Gmail client_id/client_secret not configured — "
                "plugin will be inactive"
            )
            return

        self._client = GmailClient(client_id, client_secret, refresh_token)

        # Get RAG instance
        from llamaindex_rag import get_rag

        self._rag = get_rag()
        self._syncer = EmailSyncer(self._client, self._rag)

        logger.info("Gmail plugin initialized")

    def shutdown(self) -> None:
        """Shutdown Gmail plugin."""
        self._client = None
        self._syncer = None
        self._rag = None
        logger.info("Gmail plugin shut down")

    # -------------------------------------------------------------------------
    # Flask Blueprint
    # -------------------------------------------------------------------------

    def get_blueprint(self) -> Blueprint:
        """Create Flask Blueprint with Gmail routes."""
        bp = Blueprint("gmail", __name__, url_prefix="/plugins/gmail")
        plugin = self  # Capture for closures

        @bp.route("/auth/url", methods=["GET"])
        def auth_url():
            """Generate OAuth2 authorization URL.

            Reads client_id and client_secret from settings_db (not from
            the plugin instance) so newly-saved credentials are picked up
            without restarting the server.
            """
            import settings_db

            client_id = settings_db.get_setting_value("gmail_client_id") or ""
            client_secret = settings_db.get_setting_value("gmail_client_secret") or ""

            if not client_id or not client_secret:
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "Client ID and Client Secret must be "
                            "configured before authorization",
                        }
                    ),
                    400,
                )

            try:
                from .auth import get_auth_url

                url = get_auth_url(client_id, client_secret)
                return jsonify({"status": "ok", "auth_url": url}), 200
            except Exception as e:
                logger.error(f"Failed to generate auth URL: {e}")
                return (
                    jsonify({"status": "error", "message": str(e)}),
                    500,
                )

        @bp.route("/auth/callback", methods=["POST"])
        def auth_callback():
            """Exchange an authorization code for OAuth2 tokens.

            Expects JSON body: {"code": "..."}
            Stores the refresh_token in settings_db.
            """
            import settings_db

            data = request.json or {}
            code = data.get("code", "").strip()

            if not code:
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "Authorization code is required",
                        }
                    ),
                    400,
                )

            client_id = settings_db.get_setting_value("gmail_client_id") or ""
            client_secret = settings_db.get_setting_value("gmail_client_secret") or ""

            if not client_id or not client_secret:
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "Client ID and Client Secret not configured",
                        }
                    ),
                    400,
                )

            try:
                from .auth import exchange_code

                _access_token, refresh_token = exchange_code(
                    client_id, client_secret, code
                )

                if not refresh_token:
                    return (
                        jsonify(
                            {
                                "status": "error",
                                "message": "No refresh token received. Revoke "
                                "access at https://myaccount.google.com/permissions "
                                "and try again.",
                            }
                        ),
                        400,
                    )

                # Store the refresh token
                settings_db.set_setting("gmail_refresh_token", refresh_token)

                # Re-initialize the plugin with new credentials
                plugin._client = GmailClient(client_id, client_secret, refresh_token)
                from llamaindex_rag import get_rag

                plugin._rag = get_rag()
                plugin._syncer = EmailSyncer(plugin._client, plugin._rag)

                return (
                    jsonify(
                        {
                            "status": "authorized",
                            "message": "Gmail authorized successfully",
                        }
                    ),
                    200,
                )
            except Exception as e:
                logger.error(f"OAuth2 callback failed: {e}")
                return (
                    jsonify({"status": "error", "message": str(e)}),
                    500,
                )

        @bp.route("/test", methods=["GET"])
        def test():
            """Test Gmail connection.

            Always reads fresh settings from the database.
            """
            import settings_db

            client_id = settings_db.get_setting_value("gmail_client_id") or ""
            client_secret = settings_db.get_setting_value("gmail_client_secret") or ""
            refresh_token = settings_db.get_setting_value("gmail_refresh_token") or ""

            if not refresh_token:
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "Gmail not authorized — complete "
                            "authorization first",
                        }
                    ),
                    400,
                )
            if not client_id or not client_secret:
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "Client ID and Secret not configured",
                        }
                    ),
                    400,
                )

            try:
                test_client = GmailClient(client_id, client_secret, refresh_token)
                profile = test_client.get_profile()
                email = profile.get("emailAddress", "unknown")
                total = profile.get("messagesTotal", 0)
                return (
                    jsonify(
                        {
                            "status": "connected",
                            "email": email,
                            "total_messages": total,
                        }
                    ),
                    200,
                )
            except Exception as e:
                logger.error(f"Gmail test failed: {e}")
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": f"Connection failed — {e}",
                        }
                    ),
                    500,
                )

        @bp.route("/folders", methods=["GET"])
        def folders():
            """Fetch all Gmail labels/folders.

            Always reads fresh settings from the database.

            Returns:
                JSON list of labels: [{id, name, type}, ...]
            """
            import settings_db

            client_id = settings_db.get_setting_value("gmail_client_id") or ""
            client_secret = settings_db.get_setting_value("gmail_client_secret") or ""
            refresh_token = settings_db.get_setting_value("gmail_refresh_token") or ""

            if not refresh_token:
                return (
                    jsonify(
                        {
                            "error": "Gmail not authorized",
                            "folders": [],
                        }
                    ),
                    400,
                )

            try:
                test_client = GmailClient(client_id, client_secret, refresh_token)
                labels = test_client.get_labels()

                # Map system label IDs to friendly display names
                _FRIENDLY_NAMES = {
                    "INBOX": "Inbox",
                    "SENT": "Sent",
                    "DRAFT": "Drafts",
                    "TRASH": "Trash",
                    "SPAM": "Spam",
                    "STARRED": "Starred",
                    "IMPORTANT": "Important",
                    "UNREAD": "Unread",
                    "CATEGORY_PERSONAL": "Personal",
                    "CATEGORY_SOCIAL": "Social",
                    "CATEGORY_PROMOTIONS": "Promotions",
                    "CATEGORY_UPDATES": "Updates",
                    "CATEGORY_FORUMS": "Forums",
                }

                folder_list = []
                for label in labels:
                    label_id = label.get("id", "")
                    label_name = label.get("name", "")
                    label_type = label.get("type", "user")

                    # Skip internal Gmail labels that aren't useful
                    if label_id in ("CHAT", "UNREAD"):
                        continue

                    display_name = _FRIENDLY_NAMES.get(label_id, label_name)

                    folder_list.append(
                        {
                            "id": label_id,
                            "name": display_name,
                            "type": label_type,
                        }
                    )

                return jsonify({"folders": folder_list}), 200
            except Exception as e:
                logger.error(f"Failed to fetch Gmail folders: {e}", exc_info=True)
                return (
                    jsonify({"error": str(e), "folders": []}),
                    500,
                )

        @bp.route("/sync", methods=["POST"])
        def sync():
            """Trigger manual email sync.

            Query parameters:
                force: If ``true``, skip processed-label exclusion and
                    dedup checks.
            """
            if not plugin._syncer:
                return (
                    jsonify({"error": "Plugin not initialized — authorize first"}),
                    500,
                )

            force = request.args.get("force", "").lower() in ("true", "1", "yes")

            max_emails = int(settings.get("gmail_max_emails", 500))
            folders_str = settings.get("gmail_sync_folders", "")
            processed_label = settings.get(
                "gmail_processed_label", DEFAULT_PROCESSED_LABEL
            )
            include_attachments = (
                settings.get("gmail_include_attachments", "true").lower() == "true"
            )

            # Resolve folder names to label IDs
            label_ids = None
            if folders_str:
                folder_names = [f.strip() for f in folders_str.split(",") if f.strip()]
                if folder_names:
                    label_ids = []
                    for name in folder_names:
                        # System labels can be used directly by name/ID
                        # User labels need to be looked up
                        label_id = plugin._client.get_label_id_by_name(name)
                        if label_id:
                            label_ids.append(label_id)
                            logger.info(f"Folder '{name}' → id={label_id}")
                        else:
                            logger.warning(
                                f"Folder '{name}' not found in Gmail — ignoring"
                            )

            result = plugin._syncer.sync_emails(
                max_emails=max_emails,
                label_ids=label_ids,
                processed_label_name=processed_label,
                include_attachments=include_attachments,
                force=force,
            )

            return jsonify(result), 200

        @bp.route("/sync/status", methods=["GET"])
        def sync_status():
            """Get sync status."""
            if not plugin._syncer:
                return jsonify({"error": "Plugin not initialized"}), 500

            return (
                jsonify(
                    {
                        "is_syncing": plugin._syncer.is_syncing,
                        "last_sync": plugin._syncer.last_sync_time,
                        "synced_count": plugin._syncer.synced_count,
                    }
                ),
                200,
            )

        return bp

    # -------------------------------------------------------------------------
    # Health
    # -------------------------------------------------------------------------

    def health_check(self) -> Dict[str, str]:
        """Check Gmail connectivity."""
        if not self._client:
            return {"gmail": "not initialized"}

        if self._client.test_connection():
            return {"gmail": "connected"}
        else:
            return {"gmail": "error: connection failed"}

    # -------------------------------------------------------------------------
    # Webhook Processing
    # -------------------------------------------------------------------------

    def process_webhook(self, payload: Dict[str, Any]) -> Optional[Any]:
        """Process Gmail push notification webhook.

        Not implemented yet — for future use with Gmail push notifications
        (pub/sub) for real-time email indexing.
        """
        return None
