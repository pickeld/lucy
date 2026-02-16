"""Call Recordings plugin for transcribing and indexing audio recordings.

Scans a local directory or Dropbox folder for audio call recordings,
transcribes them using local OpenAI Whisper, and indexes the transcriptions
into the Qdrant vector store for RAG retrieval.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from flask import Blueprint, Flask, jsonify, request

from config import settings
from plugins.base import ChannelPlugin

from .scanner import (
    DEFAULT_AUDIO_EXTENSIONS,
    DropboxFileScanner,
    LocalFileScanner,
)
from .sync import CallRecordingSyncer
from .transcriber import DEFAULT_MODEL_SIZE, VALID_MODEL_SIZES, WhisperTranscriber

logger = logging.getLogger(__name__)


class CallRecordingsPlugin(ChannelPlugin):
    """Call recordings integration for RAG indexing.

    Scans a local directory or Dropbox folder for audio call recordings,
    transcribes them using local OpenAI Whisper (small/medium/large),
    and indexes the transcriptions in the RAG vector store.

    Supports deduplication via SHA256 content hashing, metadata extraction
    from audio file tags (ID3/MP4) and filename patterns.
    """

    def __init__(self):
        self._scanner = None
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
        return "1.0.0"

    @property
    def description(self) -> str:
        return (
            "Scan a local directory or Dropbox folder for audio call "
            "recordings, transcribe with Whisper, and index for RAG retrieval"
        )

    # -------------------------------------------------------------------------
    # Settings
    # -------------------------------------------------------------------------

    def get_default_settings(self) -> List[Tuple[str, str, str, str, str]]:
        return [
            (
                "call_recordings_source_type",
                "local",
                "call_recordings",
                "select",
                "Audio file source: local directory or Dropbox folder",
            ),
            (
                "call_recordings_source_path",
                "/app/data/call_recordings",
                "call_recordings",
                "text",
                "Local directory path or Dropbox folder path to scan for recordings",
            ),
            (
                "call_recordings_dropbox_token",
                "",
                "call_recordings",
                "secret",
                "Dropbox API access token (required when source is Dropbox)",
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
            (
                "call_recordings_default_participants",
                "",
                "call_recordings",
                "text",
                "Default participant names (comma-separated) when not detectable from file",
            ),
        ]

    def get_select_options(self) -> Dict[str, List[str]]:
        return {
            "call_recordings_source_type": ["local", "dropbox"],
            "call_recordings_whisper_model": ["small", "medium", "large"],
        }

    def get_env_key_map(self) -> Dict[str, str]:
        return {
            "call_recordings_source_type": "CALL_RECORDINGS_SOURCE_TYPE",
            "call_recordings_source_path": "CALL_RECORDINGS_SOURCE_PATH",
            "call_recordings_dropbox_token": "CALL_RECORDINGS_DROPBOX_TOKEN",
            "call_recordings_whisper_model": "CALL_RECORDINGS_WHISPER_MODEL",
            "call_recordings_file_extensions": "CALL_RECORDINGS_FILE_EXTENSIONS",
            "call_recordings_max_files": "CALL_RECORDINGS_MAX_FILES",
            "call_recordings_sync_interval": "CALL_RECORDINGS_SYNC_INTERVAL",
            "call_recordings_default_participants": "CALL_RECORDINGS_DEFAULT_PARTICIPANTS",
        }

    def get_category_meta(self) -> Dict[str, Dict[str, str]]:
        return {"call_recordings": {"label": "📞 Call Recordings", "order": "13"}}

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def initialize(self, app: Flask) -> None:
        """Initialize the call recordings plugin.

        Creates the appropriate scanner (local or Dropbox) and transcriber
        based on current settings.  Gets the shared RAG instance.
        """
        import settings_db

        source_type = (
            settings_db.get_setting_value("call_recordings_source_type") or "local"
        )
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

        # Parse extensions
        extensions = {
            ext.strip().lower().lstrip(".")
            for ext in extensions_str.split(",")
            if ext.strip()
        }

        # Create scanner
        if source_type == "dropbox":
            token = (
                settings_db.get_setting_value("call_recordings_dropbox_token") or ""
            )
            if not token:
                logger.warning(
                    "Call Recordings: Dropbox token not configured — "
                    "plugin will be inactive until token is set"
                )
                return
            self._scanner = DropboxFileScanner(
                access_token=token,
                folder_path=source_path,
                extensions=extensions,
            )
            logger.info(
                f"Call Recordings: Using Dropbox source at '{source_path}'"
            )
        else:
            self._scanner = LocalFileScanner(
                source_path=source_path,
                extensions=extensions,
            )
            logger.info(
                f"Call Recordings: Using local source at '{source_path}'"
            )

        # Create transcriber (model loaded lazily on first use)
        self._transcriber = WhisperTranscriber(model_size=whisper_model)
        logger.info(
            f"Call Recordings: Whisper model configured as '{whisper_model}'"
        )

        # Get RAG instance
        from llamaindex_rag import get_rag

        self._rag = get_rag()

        # Create syncer
        self._syncer = CallRecordingSyncer(
            scanner=self._scanner,
            transcriber=self._transcriber,
            rag=self._rag,
        )

        logger.info("Call Recordings plugin initialized")

    def shutdown(self) -> None:
        """Shutdown the call recordings plugin.

        Unloads the Whisper model to free memory.
        """
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
        """Create Flask Blueprint with call recordings routes."""
        bp = Blueprint(
            "call_recordings", __name__, url_prefix="/plugins/call_recordings"
        )
        plugin = self  # Capture for closures

        @bp.route("/sync", methods=["POST"])
        def sync():
            """Trigger manual call recording sync.

            Query parameters:
                force: If ``true``, skip dedup checks (re-index everything).
            """
            if not plugin._syncer:
                return (
                    jsonify({"error": "Plugin not initialized — check settings"}),
                    500,
                )

            force = request.args.get("force", "").lower() in (
                "true",
                "1",
                "yes",
            )

            max_files = int(settings.get("call_recordings_max_files", "100"))
            default_participants = settings.get(
                "call_recordings_default_participants", ""
            )

            result = plugin._syncer.sync_recordings(
                max_files=max_files,
                default_participants=default_participants,
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

        @bp.route("/test", methods=["GET"])
        def test():
            """Test source connectivity.

            Verifies that the configured source (local directory or Dropbox)
            is accessible and readable.  Always reads fresh settings from
            the database.
            """
            import settings_db

            source_type = (
                settings_db.get_setting_value("call_recordings_source_type")
                or "local"
            )
            source_path = (
                settings_db.get_setting_value("call_recordings_source_path")
                or "/app/data/call_recordings"
            )

            if source_type == "dropbox":
                token = (
                    settings_db.get_setting_value("call_recordings_dropbox_token")
                    or ""
                )
                if not token:
                    return (
                        jsonify(
                            {
                                "status": "error",
                                "message": "Dropbox token not configured",
                            }
                        ),
                        400,
                    )
                try:
                    test_scanner = DropboxFileScanner(
                        access_token=token, folder_path=source_path
                    )
                    if test_scanner.test_connection():
                        return (
                            jsonify(
                                {
                                    "status": "connected",
                                    "source": "dropbox",
                                    "path": source_path,
                                }
                            ),
                            200,
                        )
                    else:
                        return (
                            jsonify(
                                {
                                    "status": "error",
                                    "message": "Dropbox connection failed",
                                }
                            ),
                            500,
                        )
                except ImportError:
                    return (
                        jsonify(
                            {
                                "status": "error",
                                "message": "dropbox package not installed",
                            }
                        ),
                        500,
                    )
                except Exception as e:
                    return (
                        jsonify({"status": "error", "message": str(e)}),
                        500,
                    )
            else:
                test_scanner = LocalFileScanner(source_path=source_path)
                if test_scanner.test_connection():
                    return (
                        jsonify(
                            {
                                "status": "connected",
                                "source": "local",
                                "path": source_path,
                            }
                        ),
                        200,
                    )
                else:
                    return (
                        jsonify(
                            {
                                "status": "error",
                                "message": f"Directory not accessible: {source_path}",
                            }
                        ),
                        500,
                    )

        @bp.route("/upload", methods=["POST"])
        def upload():
            """Upload one or more audio files for transcription and indexing.

            Accepts multipart/form-data with one or more 'files' fields.
            Files are saved to the configured source path directory and
            will be picked up by the next sync run automatically.

            Returns:
                JSON with upload results: saved count, filenames, errors.
            """
            import os
            import settings_db

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

            # Ensure the upload directory exists
            os.makedirs(source_path, exist_ok=True)

            files = request.files.getlist("files")
            if not files:
                return jsonify({"error": "No files provided"}), 400

            saved = []
            errors = []

            for f in files:
                if not f.filename:
                    continue

                # Validate extension
                ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
                if ext not in allowed_extensions:
                    errors.append(
                        f"{f.filename}: unsupported format "
                        f"(allowed: {', '.join(sorted(allowed_extensions))})"
                    )
                    continue

                # Save to source path (avoid overwriting existing files)
                import werkzeug.utils
                safe_name = werkzeug.utils.secure_filename(f.filename)
                dest = os.path.join(source_path, safe_name)

                # If file already exists, add a numeric suffix
                if os.path.exists(dest):
                    base, dot_ext = os.path.splitext(safe_name)
                    counter = 1
                    while os.path.exists(dest):
                        dest = os.path.join(
                            source_path, f"{base}_{counter}{dot_ext}"
                        )
                        counter += 1

                try:
                    f.save(dest)
                    saved.append(safe_name)
                    logger.info(f"Uploaded recording: {safe_name} → {dest}")
                except Exception as e:
                    errors.append(f"{f.filename}: {str(e)}")

            return jsonify({
                "status": "ok",
                "saved": len(saved),
                "filenames": saved,
                "errors": errors,
            }), 200

        @bp.route("/files", methods=["GET"])
        def list_files():
            """List discovered audio files with their processing status.

            Returns a JSON array of audio files found in the configured
            source, each annotated with whether it has already been
            indexed (based on content hash dedup check).
            """
            if not plugin._scanner:
                return jsonify({"error": "Plugin not initialized"}), 500

            try:
                audio_files = plugin._scanner.scan()

                file_list = []
                for af in audio_files:
                    # Check if already indexed
                    is_indexed = False
                    if plugin._rag:
                        try:
                            is_indexed = plugin._rag._message_exists(
                                af.source_id
                            )
                        except Exception:
                            pass

                    file_list.append(
                        {
                            "filename": af.filename,
                            "path": af.path,
                            "size": af.size,
                            "extension": af.extension,
                            "modified_at": af.modified_at.isoformat(),
                            "content_hash": af.content_hash[:16] + "...",
                            "indexed": is_indexed,
                            "metadata": {
                                "title": af.file_metadata.title,
                                "artist": af.file_metadata.artist,
                                "duration": (
                                    int(af.file_metadata.duration_seconds)
                                    if af.file_metadata.duration_seconds
                                    else None
                                ),
                            },
                        }
                    )

                return (
                    jsonify(
                        {
                            "total": len(file_list),
                            "indexed": sum(
                                1 for f in file_list if f["indexed"]
                            ),
                            "pending": sum(
                                1 for f in file_list if not f["indexed"]
                            ),
                            "files": file_list,
                        }
                    ),
                    200,
                )
            except Exception as e:
                logger.error(f"Failed to list audio files: {e}")
                return jsonify({"error": str(e), "files": []}), 500

        return bp

    # -------------------------------------------------------------------------
    # Health
    # -------------------------------------------------------------------------

    def health_check(self) -> Dict[str, str]:
        """Check source connectivity."""
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
        """Process webhook payload.

        Not applicable for call recordings — sync is triggered manually
        or on a schedule via the /sync endpoint.
        """
        return None
