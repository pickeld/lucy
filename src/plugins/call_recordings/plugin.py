"""Call Recordings plugin for transcribing and indexing audio recordings.

Scans a local directory for audio call recordings, transcribes them
using local OpenAI Whisper, and provides a review-and-approve workflow
before indexing transcriptions into the Qdrant vector store.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from flask import Blueprint, Flask, jsonify, request

from config import settings
from plugins.base import ChannelPlugin

from . import db as recording_db
from .scanner import DEFAULT_AUDIO_EXTENSIONS, LocalFileScanner
from .sync import CallRecordingSyncer
from .transcriber import DEFAULT_MODEL_SIZE, VALID_MODEL_SIZES, WhisperTranscriber

logger = logging.getLogger(__name__)


class CallRecordingsPlugin(ChannelPlugin):
    """Call recordings integration with review-and-approve workflow.

    Pipeline: upload/scan → auto-transcribe → user reviews in table →
    edit metadata → approve (index to Qdrant) or delete.
    """

    def __init__(self):
        self._scanner: Optional[LocalFileScanner] = None
        self._transcriber: Optional[WhisperTranscriber] = None
        self._syncer: Optional[CallRecordingSyncer] = None
        self._rag = None

    # -------------------------------------------------------------------------
    # Identity
    # -------------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "call_recordings"

    @property
    def display_name(self) -> str:
        return "Call Recordings"

    @property
    def icon(self) -> str:
        return "📞"

    @property
    def version(self) -> str:
        return "2.0.0"

    @property
    def description(self) -> str:
        return (
            "Scan a local directory for audio call recordings, "
            "transcribe with Whisper, review, and index for RAG retrieval"
        )

    # -------------------------------------------------------------------------
    # Settings
    # -------------------------------------------------------------------------

    def get_default_settings(self) -> List[Tuple[str, str, str, str, str]]:
        return [
            (
                "call_recordings_source_path",
                "/app/data/call_recordings",
                "call_recordings",
                "text",
                "Local directory path to scan for audio recordings",
            ),
            (
                "call_recordings_whisper_model",
                DEFAULT_MODEL_SIZE,
                "call_recordings",
                "select",
                "Whisper model size: small (fast), medium (balanced), large (accurate)",
            ),
            (
                "call_recordings_file_extensions",
                "mp3,wav,m4a,ogg,flac",
                "call_recordings",
                "text",
                "Comma-separated audio file extensions to process",
            ),
            (
                "call_recordings_max_files",
                "100",
                "call_recordings",
                "int",
                "Maximum files to process per sync run",
            ),
            (
                "call_recordings_sync_interval",
                "3600",
                "call_recordings",
                "int",
                "Sync interval in seconds (0 = manual only)",
            ),
        ]

    def get_select_options(self) -> Dict[str, List[str]]:
        return {
            "call_recordings_whisper_model": ["small", "medium", "large"],
        }

    def get_env_key_map(self) -> Dict[str, str]:
        return {
            "call_recordings_source_path": "CALL_RECORDINGS_SOURCE_PATH",
            "call_recordings_whisper_model": "CALL_RECORDINGS_WHISPER_MODEL",
            "call_recordings_file_extensions": "CALL_RECORDINGS_FILE_EXTENSIONS",
            "call_recordings_max_files": "CALL_RECORDINGS_MAX_FILES",
            "call_recordings_sync_interval": "CALL_RECORDINGS_SYNC_INTERVAL",
        }

    def get_category_meta(self) -> Dict[str, Dict[str, str]]:
        return {"call_recordings": {"label": "📞 Call Recordings", "order": "13"}}

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def initialize(self, app: Flask) -> None:
        """Initialize the call recordings plugin."""
        import settings_db

        # Clean up obsolete settings from prior versions
        settings_db.delete_setting("call_recordings_default_participants")

        # Initialize the file tracking table
        recording_db.init_table()

        source_path = (
            settings_db.get_setting_value("call_recordings_source_path")
            or "/app/data/call_recordings"
        )
        whisper_model = (
            settings_db.get_setting_value("call_recordings_whisper_model")
            or DEFAULT_MODEL_SIZE
        )
        extensions_str = (
            settings_db.get_setting_value("call_recordings_file_extensions")
            or "mp3,wav,m4a,ogg,flac"
        )

        extensions = {
            ext.strip().lower().lstrip(".")
            for ext in extensions_str.split(",")
            if ext.strip()
        }

        self._scanner = LocalFileScanner(
            source_path=source_path,
            extensions=extensions,
        )
        logger.info(f"Call Recordings: Using local source at '{source_path}'")

        self._transcriber = WhisperTranscriber(model_size=whisper_model)
        logger.info(f"Call Recordings: Whisper model configured as '{whisper_model}'")

        from llamaindex_rag import get_rag

        self._rag = get_rag()

        self._syncer = CallRecordingSyncer(
            scanner=self._scanner,
            transcriber=self._transcriber,
            rag=self._rag,
        )

        logger.info("Call Recordings plugin initialized (v2 — review workflow)")

    def shutdown(self) -> None:
        if self._transcriber:
            self._transcriber.unload_model()
        self._scanner = None
        self._transcriber = None
        self._syncer = None
        self._rag = None
        logger.info("Call Recordings plugin shut down")

    # -------------------------------------------------------------------------
    # Flask Blueprint
    # -------------------------------------------------------------------------

    def get_blueprint(self) -> Blueprint:
        bp = Blueprint(
            "call_recordings", __name__, url_prefix="/plugins/call_recordings"
        )
        plugin = self

        # =====================================================================
        # FILES — list, detail, delete
        # =====================================================================

        @bp.route("/files", methods=["GET"])
        def list_files():
            """List all tracked recording files with status and metadata.

            Query parameters:
                status: Filter by status (pending, transcribing, transcribed, approved, error)
            """
            try:
                status_filter = request.args.get("status")
                files = recording_db.list_files(status=status_filter)
                counts = recording_db.get_counts()

                return jsonify({
                    "files": files,
                    "counts": counts,
                    "total": len(files),
                }), 200
            except Exception as e:
                logger.error(f"Failed to list files: {e}")
                return jsonify({"error": str(e), "files": []}), 500

        @bp.route("/files/<content_hash>", methods=["GET"])
        def get_file(content_hash):
            """Get full detail for a single tracked file."""
            record = recording_db.get_file(content_hash)
            if not record:
                return jsonify({"error": "File not found"}), 404
            return jsonify(record), 200

        @bp.route("/files/<content_hash>", methods=["DELETE"])
        def delete_file(content_hash):
            """Delete a file from tracking and optionally from disk."""
            if not plugin._syncer:
                return jsonify({"error": "Plugin not initialized"}), 500

            remove_disk = request.args.get("disk", "true").lower() in ("true", "1")
            result = plugin._syncer.delete_file(content_hash, remove_from_disk=remove_disk)

            if result.get("status") == "error":
                return jsonify(result), 404
            return jsonify(result), 200

        # =====================================================================
        # METADATA — update contact name, phone, participants
        # =====================================================================

        @bp.route("/files/<content_hash>/metadata", methods=["PUT"])
        def update_metadata(content_hash):
            """Update user-editable metadata for a file.

            JSON body (all optional):
                contact_name: str
                phone_number: str
                participants: list[str]
            """
            data = request.get_json(silent=True) or {}
            updated = recording_db.update_metadata(
                content_hash=content_hash,
                contact_name=data.get("contact_name"),
                phone_number=data.get("phone_number"),
                participants=data.get("participants"),
            )

            if not updated:
                return jsonify({"error": "File not found or no changes"}), 404
            return jsonify({"status": "updated"}), 200

        # =====================================================================
        # TRANSCRIBE — trigger transcription for a single file
        # =====================================================================

        @bp.route("/files/<content_hash>/transcribe", methods=["POST"])
        def transcribe_file(content_hash):
            """Transcribe (or re-transcribe) a single audio file."""
            if not plugin._syncer:
                return jsonify({"error": "Plugin not initialized"}), 500

            result = plugin._syncer.transcribe_file(content_hash)

            if result.get("status") == "error":
                return jsonify(result), 400
            return jsonify(result), 200

        # =====================================================================
        # APPROVE — index a transcribed file into Qdrant
        # =====================================================================

        @bp.route("/files/<content_hash>/approve", methods=["POST"])
        def approve_file(content_hash):
            """Approve a transcribed file and index it into Qdrant."""
            if not plugin._syncer:
                return jsonify({"error": "Plugin not initialized"}), 500

            result = plugin._syncer.approve_file(content_hash)

            if result.get("status") == "error":
                return jsonify(result), 400
            return jsonify(result), 200

        # =====================================================================
        # SCAN — discover new files + auto-transcribe
        # =====================================================================

        @bp.route("/scan", methods=["POST"])
        def scan():
            """Scan for new files, register them, and auto-transcribe.

            Query parameters:
                auto_transcribe: If 'false', only register without transcribing
            """
            if not plugin._syncer:
                return jsonify({"error": "Plugin not initialized"}), 500

            auto_transcribe = request.args.get(
                "auto_transcribe", "true"
            ).lower() in ("true", "1", "yes")

            result = plugin._syncer.scan_and_register(
                auto_transcribe=auto_transcribe,
            )

            return jsonify(result), 200

        # =====================================================================
        # LEGACY — full sync (scan + transcribe + auto-approve)
        # =====================================================================

        @bp.route("/sync", methods=["POST"])
        def sync():
            """Legacy full sync: scan → transcribe → auto-approve all."""
            if not plugin._syncer:
                return jsonify({"error": "Plugin not initialized"}), 500

            force = request.args.get("force", "").lower() in ("true", "1", "yes")
            max_files = int(settings.get("call_recordings_max_files", "100"))

            result = plugin._syncer.sync_recordings(
                max_files=max_files,
                force=force,
            )

            return jsonify(result), 200

        @bp.route("/sync/status", methods=["GET"])
        def sync_status():
            if not plugin._syncer:
                return jsonify({"error": "Plugin not initialized"}), 500

            return jsonify({
                "is_syncing": plugin._syncer.is_syncing,
                "last_sync": plugin._syncer.last_sync_time,
                "synced_count": plugin._syncer.synced_count,
            }), 200

        # =====================================================================
        # TEST — source connectivity
        # =====================================================================

        @bp.route("/test", methods=["GET"])
        def test():
            import settings_db

            source_path = (
                settings_db.get_setting_value("call_recordings_source_path")
                or "/app/data/call_recordings"
            )

            test_scanner = LocalFileScanner(source_path=source_path)
            if test_scanner.test_connection():
                return jsonify({
                    "status": "connected",
                    "source": "local",
                    "path": source_path,
                }), 200
            else:
                return jsonify({
                    "status": "error",
                    "message": f"Directory not accessible: {source_path}",
                }), 500

        # =====================================================================
        # UPLOAD — save files + register + auto-transcribe
        # =====================================================================

        @bp.route("/upload", methods=["POST"])
        def upload():
            """Upload audio files, save to disk, register, and auto-transcribe."""
            import settings_db
            import werkzeug.utils

            from .scanner import compute_file_hash

            source_path = (
                settings_db.get_setting_value("call_recordings_source_path")
                or "/app/data/call_recordings"
            )
            extensions_str = (
                settings_db.get_setting_value("call_recordings_file_extensions")
                or "mp3,wav,m4a,ogg,flac"
            )
            allowed_extensions = {
                ext.strip().lower().lstrip(".")
                for ext in extensions_str.split(",")
                if ext.strip()
            }

            os.makedirs(source_path, exist_ok=True)

            files = request.files.getlist("files")
            if not files:
                return jsonify({"error": "No files provided"}), 400

            saved = []
            errors = []

            for f in files:
                if not f.filename:
                    continue

                ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
                if ext not in allowed_extensions:
                    errors.append(
                        f"{f.filename}: unsupported format "
                        f"(allowed: {', '.join(sorted(allowed_extensions))})"
                    )
                    continue

                safe_name = werkzeug.utils.secure_filename(f.filename)
                dest = os.path.join(source_path, safe_name)

                if os.path.exists(dest):
                    base, dot_ext = os.path.splitext(safe_name)
                    counter = 1
                    while os.path.exists(dest):
                        dest = os.path.join(source_path, f"{base}_{counter}{dot_ext}")
                        counter += 1
                    safe_name = os.path.basename(dest)

                try:
                    f.save(dest)
                    saved.append(safe_name)
                    logger.info(f"Uploaded: {safe_name} → {dest}")

                    # Register in tracking DB and auto-transcribe
                    content_hash = compute_file_hash(dest)
                    stat = os.stat(dest)

                    recording_db.upsert_file(
                        content_hash=content_hash,
                        filename=safe_name,
                        file_path=dest,
                        file_size=stat.st_size,
                        extension=ext,
                        modified_at=datetime.fromtimestamp(
                            stat.st_mtime, tz=ZoneInfo("UTC")
                        ).isoformat() if _can_import_zoneinfo() else "",
                    )

                    # Auto-transcribe in background
                    if plugin._syncer:
                        try:
                            plugin._syncer.transcribe_file(content_hash)
                        except Exception as te:
                            logger.warning(f"Auto-transcribe failed for {safe_name}: {te}")

                except Exception as e:
                    errors.append(f"{f.filename}: {str(e)}")

            return jsonify({
                "status": "ok",
                "saved": len(saved),
                "filenames": saved,
                "errors": errors,
            }), 200

        return bp

    # -------------------------------------------------------------------------
    # Health
    # -------------------------------------------------------------------------

    def health_check(self) -> Dict[str, str]:
        if not self._scanner:
            return {"call_recordings": "not initialized"}

        if self._scanner.test_connection():
            return {"call_recordings": "connected"}
        else:
            return {"call_recordings": "error: source not accessible"}

    # -------------------------------------------------------------------------
    # Webhook Processing
    # -------------------------------------------------------------------------

    def process_webhook(self, payload: Dict[str, Any]) -> Optional[Any]:
        return None


def _can_import_zoneinfo() -> bool:
    """Check if zoneinfo is available (Python 3.9+)."""
    try:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        return True
    except ImportError:
        return False


# Make datetime/ZoneInfo available at module level for upload handler
try:
    from datetime import datetime
    from zoneinfo import ZoneInfo
except ImportError:
    pass
