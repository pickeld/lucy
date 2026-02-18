"""Call Recordings plugin for transcribing and indexing audio recordings.

Scans a local directory for audio call recordings, transcribes them
using local OpenAI Whisper or remote AssemblyAI, and provides a
review-and-approve workflow before indexing transcriptions into the
Qdrant vector store.
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
from .transcriber import (
    DEFAULT_DIARIZATION_MODEL,
    DEFAULT_MODEL_SIZE,
    VALID_MODEL_SIZES,
    WhisperTranscriber,
)

logger = logging.getLogger(__name__)

# Default AssemblyAI model
DEFAULT_ASSEMBLYAI_MODEL = "universal-2"


class CallRecordingsPlugin(ChannelPlugin):
    """Call recordings integration with review-and-approve workflow.

    Pipeline: upload/scan → auto-transcribe → user reviews in table →
    edit metadata → approve (index to Qdrant) or delete.
    """

    def __init__(self):
        self._scanner: Optional[LocalFileScanner] = None
        self._transcriber = None  # WhisperTranscriber or AssemblyAITranscriber
        self._syncer: Optional[CallRecordingSyncer] = None
        self._rag = None
        self._current_provider: str = "local"  # tracks active transcriber type

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
        return "2.1.0"

    @property
    def description(self) -> str:
        return (
            "Scan a local directory for audio call recordings, "
            "transcribe with Whisper (local) or AssemblyAI (remote), "
            "review, and index for RAG retrieval"
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
            # --- Transcription provider ---
            (
                "call_recordings_transcription_provider",
                "local",
                "call_recordings",
                "select",
                "Transcription engine: 'local' (Whisper on this machine) or 'assemblyai' (remote API with built-in diarization)",
            ),
            # --- Local Whisper settings ---
            (
                "call_recordings_whisper_model",
                DEFAULT_MODEL_SIZE,
                "call_recordings",
                "select",
                "Whisper model size (local only): small (fast), medium (balanced), large (accurate)",
            ),
            (
                "call_recordings_compute_type",
                "auto",
                "call_recordings",
                "select",
                "Compute type for Whisper inference (local only). 'auto' picks the fastest for your hardware",
            ),
            (
                "call_recordings_whisper_language",
                "",
                "call_recordings",
                "text",
                "Force language code (e.g. 'en', 'he'). Leave blank for auto-detection",
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
                "call_recordings_enable_diarization",
                "true",
                "call_recordings",
                "bool",
                "Enable speaker diarization (identify Speaker A vs Speaker B). Local: requires HF token + pyannote. AssemblyAI: built-in.",
            ),
            (
                "call_recordings_diarization_model",
                DEFAULT_DIARIZATION_MODEL,
                "call_recordings",
                "text",
                "Diarization pipeline (local only): pyannote model name or local path",
            ),
            # --- AssemblyAI settings ---
            (
                "call_recordings_assemblyai_model",
                DEFAULT_ASSEMBLYAI_MODEL,
                "call_recordings",
                "select",
                "AssemblyAI speech model: 'universal-2' (fast, $0.015/min) or 'universal-3-pro' (best, $0.12/min)",
            ),
            # --- Speaker identification ---
            (
                "call_recordings_my_name",
                "",
                "call_recordings",
                "text",
                "Your display name — used as default for 'Speaker A' in call transcriptions",
            ),
            # --- API Keys ---
            (
                "assemblyai_api_key",
                "",
                "secrets",
                "secret",
                "AssemblyAI API key for remote transcription (get one at https://www.assemblyai.com/dashboard/signup)",
            ),
            (
                "hf_token",
                "",
                "secrets",
                "secret",
                "Hugging Face token for local model downloads and speaker diarization (get one at https://huggingface.co/settings/tokens)",
            ),
        ]

    def get_select_options(self) -> Dict[str, List[str]]:
        return {
            "call_recordings_transcription_provider": ["local", "assemblyai"],
            "call_recordings_whisper_model": ["small", "medium", "large"],
            "call_recordings_compute_type": ["auto", "int8", "float16", "float32"],
            "call_recordings_assemblyai_model": ["universal-2", "universal-3-pro"],
        }

    def get_env_key_map(self) -> Dict[str, str]:
        return {
            "call_recordings_source_path": "CALL_RECORDINGS_SOURCE_PATH",
            "call_recordings_transcription_provider": "CALL_RECORDINGS_TRANSCRIPTION_PROVIDER",
            "call_recordings_whisper_model": "CALL_RECORDINGS_WHISPER_MODEL",
            "call_recordings_compute_type": "CALL_RECORDINGS_COMPUTE_TYPE",
            "call_recordings_whisper_language": "CALL_RECORDINGS_WHISPER_LANGUAGE",
            "call_recordings_file_extensions": "CALL_RECORDINGS_FILE_EXTENSIONS",
            "call_recordings_max_files": "CALL_RECORDINGS_MAX_FILES",
            "call_recordings_sync_interval": "CALL_RECORDINGS_SYNC_INTERVAL",
            "call_recordings_diarization_model": "CALL_RECORDINGS_DIARIZATION_MODEL",
            "call_recordings_assemblyai_model": "CALL_RECORDINGS_ASSEMBLYAI_MODEL",
            "call_recordings_my_name": "CALL_RECORDINGS_MY_NAME",
            "assemblyai_api_key": "ASSEMBLYAI_API_KEY",
            "hf_token": "HF_TOKEN",
        }

    def get_category_meta(self) -> Dict[str, Dict[str, str]]:
        return {"call_recordings": {"label": "📞 Call Recordings", "order": "13"}}

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def initialize(self, app: Flask) -> None:
        """Initialize the call recordings plugin.

        Reads the ``call_recordings_transcription_provider`` setting and
        instantiates the appropriate transcriber:
        - ``local``: WhisperTranscriber (faster-whisper + optional pyannote)
        - ``assemblyai``: AssemblyAITranscriber (remote API with built-in diarization)
        """
        import settings_db

        # Clean up obsolete settings from prior versions
        settings_db.delete_setting("call_recordings_default_participants")

        # Initialize the file tracking table
        recording_db.init_table()

        source_path = (
            settings_db.get_setting_value("call_recordings_source_path")
            or "/app/data/call_recordings"
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

        # --- Transcription provider selection ---
        provider = (
            settings_db.get_setting_value("call_recordings_transcription_provider")
            or "local"
        ).lower().strip()

        enable_diarization = (
            settings_db.get_setting_value("call_recordings_enable_diarization") or "true"
        ).lower() in ("true", "1", "yes")

        language = (
            settings_db.get_setting_value("call_recordings_whisper_language") or ""
        ).strip() or None

        if provider == "assemblyai":
            self._transcriber = self._create_assemblyai_transcriber(
                settings_db, enable_diarization, language,
            )
        else:
            self._transcriber = self._create_local_transcriber(
                settings_db, enable_diarization,
            )

        self._current_provider = provider

        from llamaindex_rag import get_rag

        self._rag = get_rag()

        self._syncer = CallRecordingSyncer(
            scanner=self._scanner,
            transcriber=self._transcriber,
            rag=self._rag,
        )

        logger.info(
            f"Call Recordings plugin initialized (v2.1 — {provider} transcription)"
        )

    def _create_local_transcriber(self, settings_db, enable_diarization: bool):
        """Create a local WhisperTranscriber."""
        whisper_model = (
            settings_db.get_setting_value("call_recordings_whisper_model")
            or DEFAULT_MODEL_SIZE
        )
        compute_type = (
            settings_db.get_setting_value("call_recordings_compute_type")
            or "auto"
        )
        hf_token = settings_db.get_setting_value("hf_token") or ""
        diarization_model = (
            settings_db.get_setting_value("call_recordings_diarization_model")
            or DEFAULT_DIARIZATION_MODEL
        )

        transcriber = WhisperTranscriber(
            model_size=whisper_model,
            hf_token=hf_token if hf_token else None,
            enable_diarization=enable_diarization,
            compute_type=compute_type,
            diarization_model=diarization_model,
        )
        diar_status = "enabled" if enable_diarization else "disabled"
        logger.info(
            f"Call Recordings: Local Whisper model '{whisper_model}' "
            f"(compute={compute_type}), diarization {diar_status} "
            f"(pipeline={diarization_model})"
        )
        return transcriber

    def _create_assemblyai_transcriber(
        self, settings_db, enable_diarization: bool, language: Optional[str],
    ):
        """Create a remote AssemblyAITranscriber."""
        from .remote_transcriber import AssemblyAITranscriber

        api_key = settings_db.get_setting_value("assemblyai_api_key") or ""
        if not api_key.strip():
            logger.warning(
                "AssemblyAI selected but no API key configured — "
                "set it in Settings → API Keys. Falling back to local Whisper."
            )
            return self._create_local_transcriber(settings_db, enable_diarization)

        model = (
            settings_db.get_setting_value("call_recordings_assemblyai_model")
            or DEFAULT_ASSEMBLYAI_MODEL
        )

        transcriber = AssemblyAITranscriber(
            api_key=api_key.strip(),
            model=model,
            language=language,
            enable_diarization=enable_diarization,
        )
        diar_status = "enabled" if enable_diarization else "disabled"
        logger.info(
            f"Call Recordings: AssemblyAI model '{model}', "
            f"diarization {diar_status}"
        )
        return transcriber

    # -------------------------------------------------------------------------
    # Hot-swap transcriber when provider setting changes
    # -------------------------------------------------------------------------

    def _ensure_correct_transcriber(self) -> None:
        """Re-read the transcription provider setting and swap the transcriber
        if it has changed since the last call.

        Called before every transcription-related endpoint so that UI
        settings changes take effect without requiring an app restart.
        """
        import settings_db

        provider = (
            settings_db.get_setting_value("call_recordings_transcription_provider")
            or "local"
        ).lower().strip()

        if provider == self._current_provider:
            return  # no change — nothing to do

        logger.info(
            f"Transcription provider changed: {self._current_provider} → {provider}"
        )

        enable_diarization = (
            settings_db.get_setting_value("call_recordings_enable_diarization") or "true"
        ).lower() in ("true", "1", "yes")

        language = (
            settings_db.get_setting_value("call_recordings_whisper_language") or ""
        ).strip() or None

        # Unload the old transcriber model (frees GPU/CPU memory)
        if self._transcriber:
            try:
                self._transcriber.unload_model()
            except Exception:
                pass

        # Create the new transcriber
        if provider == "assemblyai":
            self._transcriber = self._create_assemblyai_transcriber(
                settings_db, enable_diarization, language,
            )
        else:
            self._transcriber = self._create_local_transcriber(
                settings_db, enable_diarization,
            )

        self._current_provider = provider

        # Update the syncer's reference so it uses the new transcriber
        if self._syncer:
            self._syncer.transcriber = self._transcriber

        logger.info(f"Transcriber hot-swapped to '{provider}'")

    def shutdown(self) -> None:
        if self._syncer:
            self._syncer.shutdown()
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
        # SPEAKERS — rename speaker labels in transcript text
        # =====================================================================

        @bp.route("/files/<content_hash>/speakers", methods=["PUT"])
        def update_speakers(content_hash):
            """Rename speaker labels in the transcript text.

            Performs find-and-replace on the transcript:
            - "Speaker A:" → "<speaker_a>:"
            - "Speaker B:" → "<speaker_b>:"

            Also stores the label assignments in the DB so the UI
            can pre-populate dropdowns on subsequent views.

            JSON body:
                speaker_a: str — display name for Speaker A
                speaker_b: str — display name for Speaker B
            """
            import re as _re

            data = request.get_json(silent=True) or {}
            speaker_a = (data.get("speaker_a") or "").strip()
            speaker_b = (data.get("speaker_b") or "").strip()

            if not speaker_a and not speaker_b:
                return jsonify({"error": "At least one speaker name required"}), 400

            record = recording_db.get_file(content_hash)
            if not record:
                return jsonify({"error": "File not found"}), 404

            transcript = record.get("transcript_text", "")
            if not transcript:
                return jsonify({"error": "No transcript text to update"}), 400

            # Detect current speaker labels in the transcript.
            # Diarised transcripts use "Speaker A:", "Speaker B:" or
            # previously renamed labels stored in speaker_a_label/speaker_b_label.
            old_a = record.get("speaker_a_label") or "Speaker A"
            old_b = record.get("speaker_b_label") or "Speaker B"

            updated_transcript = transcript
            if speaker_a:
                # Replace "OldLabel:" at the start of lines / after newlines
                updated_transcript = _re.sub(
                    rf"(?m)^{_re.escape(old_a)}(\s*:)",
                    f"{speaker_a}\\1",
                    updated_transcript,
                )
            if speaker_b:
                updated_transcript = _re.sub(
                    rf"(?m)^{_re.escape(old_b)}(\s*:)",
                    f"{speaker_b}\\1",
                    updated_transcript,
                )

            recording_db.update_speaker_labels(
                content_hash=content_hash,
                speaker_a=speaker_a or old_a,
                speaker_b=speaker_b or old_b,
                updated_transcript=updated_transcript,
            )

            # If the file is already approved, we should note it needs re-indexing
            needs_reindex = record.get("status") == "approved"

            return jsonify({
                "status": "updated",
                "speaker_a": speaker_a or old_a,
                "speaker_b": speaker_b or old_b,
                "needs_reindex": needs_reindex,
            }), 200

        # =====================================================================
        # TRANSCRIBE — trigger transcription for a single file
        # =====================================================================

        @bp.route("/files/<content_hash>/transcribe", methods=["POST"])
        def transcribe_file(content_hash):
            """Transcribe (or re-transcribe) a single audio file.

            Runs transcription in a background thread and returns
            immediately with status 'queued'.  Poll ``GET /files``
            or ``GET /files/<hash>`` to track progress.
            """
            if not plugin._syncer:
                return jsonify({"error": "Plugin not initialized"}), 500

            # Hot-swap transcriber if the provider setting changed
            plugin._ensure_correct_transcriber()

            # Verify file exists before queuing
            record = recording_db.get_file(content_hash)
            if not record:
                return jsonify({"status": "error", "error": "File not found"}), 404

            # Mark as transcribing immediately so UI shows spinner
            recording_db.update_status(content_hash, "transcribing")

            future = plugin._syncer.transcribe_file_async(content_hash)

            return jsonify({
                "status": "queued",
                "content_hash": content_hash,
                "message": "Transcription started in background",
            }), 202

        # =====================================================================
        # RESTART — reset a stuck transcribing file and re-queue
        # =====================================================================

        @bp.route("/files/<content_hash>/restart", methods=["POST"])
        def restart_transcription(content_hash):
            """Restart a stuck transcription.

            Resets the file status from 'transcribing' to 'pending',
            then immediately re-queues it for transcription.
            """
            if not plugin._syncer:
                return jsonify({"error": "Plugin not initialized"}), 500

            # Hot-swap transcriber if the provider setting changed
            plugin._ensure_correct_transcriber()

            record = recording_db.get_file(content_hash)
            if not record:
                return jsonify({"status": "error", "error": "File not found"}), 404

            # Reset status to pending (clears progress fields)
            recording_db.update_status(
                content_hash, "pending",
                error_message="Manually restarted by user",
            )

            # Re-mark as transcribing and queue
            recording_db.update_status(content_hash, "transcribing")
            future = plugin._syncer.transcribe_file_async(content_hash)

            return jsonify({
                "status": "restarted",
                "content_hash": content_hash,
                "message": "Transcription restarted",
            }), 202

        # =====================================================================
        # WAIT — block until transcription completes (push-based)
        # =====================================================================

        @bp.route("/files/<content_hash>/wait", methods=["GET"])
        def wait_for_transcription(content_hash):
            """Block until transcription for this file completes.

            Uses Redis BLPOP on a notification key that the Celery task
            pushes to when it finishes.  Returns immediately if the file
            is already in a terminal state (transcribed, approved, error).

            Query parameters:
                timeout: Max seconds to wait (default 300, max 600)

            Returns:
                200 with final status when transcription completes
                408 if timeout is reached before completion
            """
            import json as _json

            from utils.redis_conn import get_redis_client

            # Quick check: if file is already done, return immediately
            record = recording_db.get_file(content_hash)
            if not record:
                return jsonify({"status": "error", "error": "File not found"}), 404

            current_status = record.get("status", "")
            if current_status in ("transcribed", "approved", "error"):
                return jsonify({
                    "status": current_status,
                    "content_hash": content_hash,
                }), 200

            # Block on Redis notification from the Celery worker
            timeout = min(int(request.args.get("timeout", 300)), 600)
            notify_key = f"transcription:done:{content_hash}"

            try:
                client = get_redis_client()
                result = client.blpop([notify_key], timeout=timeout)

                if result is None:
                    # Timeout — check DB one more time (task may have finished
                    # just before we started waiting)
                    record = recording_db.get_file(content_hash)
                    final_status = record.get("status", "unknown") if record else "unknown"
                    if final_status in ("transcribed", "approved", "error"):
                        return jsonify({
                            "status": final_status,
                            "content_hash": content_hash,
                        }), 200
                    return jsonify({
                        "status": "timeout",
                        "content_hash": content_hash,
                        "message": f"Transcription not complete after {timeout}s",
                    }), 408

                # result is (key, value) tuple
                _, payload = result
                try:
                    data = _json.loads(payload)
                except (TypeError, _json.JSONDecodeError):
                    data = {"status": "unknown"}

                return jsonify({
                    "status": data.get("status", "unknown"),
                    "content_hash": content_hash,
                }), 200

            except Exception as e:
                logger.error(f"Wait endpoint error for {content_hash}: {e}")
                return jsonify({
                    "status": "error",
                    "error": f"Wait failed: {str(e)}",
                }), 500

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

            # Hot-swap transcriber if the provider setting changed
            plugin._ensure_correct_transcriber()

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

            # Hot-swap transcriber if the provider setting changed
            plugin._ensure_correct_transcriber()

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

            # Hot-swap transcriber if the provider setting changed
            plugin._ensure_correct_transcriber()

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

                    # Queue background transcription (non-blocking)
                    if plugin._syncer:
                        try:
                            plugin._syncer.transcribe_file_async(content_hash)
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
