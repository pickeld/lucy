"""Call recording synchronization logic.

Orchestrates the review-and-approve pipeline:
    scan → register in DB → auto-transcribe → user reviews → approve → index

Composable steps:
    scan_and_register()   — Discover files, insert into tracking DB as pending
    transcribe_file()     — Transcribe one file with Whisper, store result in DB
    approve_file()        — Build nodes from transcription, index in Qdrant
    delete_file()         — Remove file from disk and tracking DB
    sync_recordings()     — Legacy: scan + transcribe + auto-approve (all-in-one)

Transcription runs in a background thread pool so Flask endpoints return
immediately.  Progress is tracked via the ``status`` field in the recording
DB (pending → transcribing → transcribed / error).
"""

import errno
import json
import logging
import os
import re
import shutil
import tempfile
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from llama_index.core.schema import TextNode

from utils.text_processing import (
    MAX_CHUNK_CHARS,
    CHUNK_OVERLAP_CHARS,
    MIN_CONTENT_CHARS,
    is_quality_chunk,
    split_text,
)

from . import db as recording_db
from .scanner import AudioFile, LocalFileScanner, _parse_filename_metadata
from .transcriber import TranscriptionResult

logger = logging.getLogger(__name__)

# Minutes after which a "transcribing" job is considered stuck
_STALE_TRANSCRIBING_MINUTES = 30


class CallRecordingSyncer:
    """Handles the call recordings review-and-approve pipeline.

    Composable steps:
        scan_and_register() — Discover new files, insert into DB, auto-transcribe
        transcribe_file()   — Transcribe a single file by content hash
        approve_file()      — Index a transcribed file into Qdrant
        delete_file()       — Remove file from disk + DB

    Args:
        scanner: File scanner instance
        transcriber: Any object with ``transcribe()``, ``unload_model()``,
            and ``model_size`` — either WhisperTranscriber or
            AssemblyAITranscriber.
        rag: LlamaIndexRAG instance
    """

    # One transcription at a time — Whisper is GPU/CPU-bound and concurrent
    # runs would OOM or thrash.  The single worker thread is enough to keep
    # the Flask event loop unblocked.
    _POOL_WORKERS = 1

    def __init__(
        self,
        scanner: LocalFileScanner,
        transcriber,  # WhisperTranscriber or AssemblyAITranscriber (duck-typed)
        rag,
    ):
        self.scanner = scanner
        self.transcriber = transcriber
        self.rag = rag
        self._syncing = False
        self._last_sync = 0
        self._recording_count = 0
        self._pool: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(
            max_workers=self._POOL_WORKERS,
            thread_name_prefix="whisper",
        )

    @property
    def is_syncing(self) -> bool:
        return self._syncing

    @property
    def last_sync_time(self) -> int:
        return self._last_sync

    @property
    def synced_count(self) -> int:
        return self._recording_count

    # -------------------------------------------------------------------------
    # Step 1: Scan and register
    # -------------------------------------------------------------------------

    def shutdown(self) -> None:
        """Shut down the background thread pool.

        Waits for any in-progress transcription to finish, then releases
        the pool.  Called by the plugin's ``shutdown()`` hook.
        """
        if self._pool:
            logger.info("Shutting down transcription thread pool...")
            self._pool.shutdown(wait=True)
            self._pool = None
            logger.info("Transcription thread pool shut down")

    # -------------------------------------------------------------------------
    # Async transcription helpers
    # -------------------------------------------------------------------------

    def transcribe_file_async(self, content_hash: str) -> Optional[Future]:
        """Submit a transcription job to the background thread pool.

        Returns immediately.  The job updates the recording DB status
        as it progresses (transcribing → transcribed / error).

        Args:
            content_hash: SHA256 content hash identifying the file

        Returns:
            A ``Future`` for the background job, or ``None`` if the pool
            is not available.
        """
        if not self._pool:
            logger.warning("Thread pool not available — running transcription synchronously")
            self.transcribe_file(content_hash)
            return None

        # Pre-validate the record exists before submitting
        record = recording_db.get_file(content_hash)
        if not record:
            logger.warning(f"Cannot queue transcription — file not found: {content_hash}")
            return None

        logger.info(f"Queuing background transcription for {record.get('filename', content_hash)}")

        future = self._pool.submit(self._transcribe_worker, content_hash)
        return future

    def _transcribe_worker(self, content_hash: str) -> Dict:
        """Background worker that runs ``transcribe_file`` in the thread pool.

        Catches all exceptions so the thread pool stays healthy.
        """
        try:
            return self.transcribe_file(content_hash)
        except Exception as e:
            logger.error(f"Background transcription failed for {content_hash}: {e}")
            try:
                recording_db.update_status(content_hash, "error", str(e))
            except Exception:
                pass
            return {"status": "error", "error": str(e)}

    def scan_and_register(self, auto_transcribe: bool = True) -> Dict:
        """Discover audio files and register them in the tracking DB.

        New files are inserted with status 'pending'.  Already-tracked files
        (transcribed, approved, etc.) are skipped entirely — no hash lookup,
        no metadata resolution.

        Stale 'transcribing' jobs (started > 30 min ago) are automatically
        reset to 'pending' before scanning.

        When *auto_transcribe* is ``True``, new files are queued for
        background transcription (non-blocking).

        Args:
            auto_transcribe: If True, queue newly discovered files for
                background transcription.

        Returns:
            Dict with counts: discovered, new, queued, skipped, errors
        """
        if self._syncing:
            return {"status": "already_running"}

        self._syncing = True
        discovered = 0
        new_files = 0
        queued = 0
        skipped = 0
        errors = 0

        try:
            # Reset recordings stuck in 'transcribing' state
            stale_count = recording_db.reset_stale_transcribing(
                stale_minutes=_STALE_TRANSCRIBING_MINUTES
            )
            if stale_count:
                logger.info(f"Reset {stale_count} stale transcribing job(s)")

            # Pre-fetch all known content hashes so we can skip already-tracked
            # files without running expensive metadata lookups or entity resolution
            known_hashes = recording_db.get_known_hashes()

            audio_files = self.scanner.scan()
            discovered = len(audio_files)

            if not audio_files:
                logger.info("No audio files found in source directory")
                return {
                    "status": "complete",
                    "discovered": 0,
                    "new": 0,
                    "queued": 0,
                    "skipped": 0,
                    "errors": 0,
                    "stale_reset": stale_count,
                }

            logger.info(f"Scan found {discovered} audio files")

            for af in audio_files:
                # Fast path: skip files already tracked in the DB
                if af.content_hash in known_hashes:
                    skipped += 1
                    continue

                # Resolve metadata from file tags + filename
                filename_meta = _parse_filename_metadata(af.filename)

                # Extract phone number and participants from filename
                phone_number = filename_meta.get("phone_number") or ""
                fn_participants = filename_meta.get("participants") or ""

                # Resolve contact name + phone from entity store:
                # 1. If phone detected → look up entity store for display name
                # 2. If name detected from filename → look up entity store for phone
                contact_name = ""
                if phone_number:
                    contact_name = self._lookup_contact_by_phone(phone_number)
                elif fn_participants and not fn_participants.replace(" ", "").isdigit():
                    # Filename had a name (not a phone number) — try to
                    # find the entity and fill in the phone number
                    contact_name = fn_participants
                    entity_info = self._lookup_entity_by_name(fn_participants)
                    if entity_info:
                        # Prefer entity display name over raw filename parse
                        contact_name = entity_info.get("display_name") or contact_name
                        if not phone_number and entity_info.get("phone"):
                            phone_number = entity_info["phone"]

                # Auto-detect participants from tags or filename
                participants = self._resolve_participants(af)
                if contact_name and participants == ["Unknown"]:
                    participants = [contact_name]

                # Use date+time from filename if available
                modified_at = af.modified_at.isoformat()
                if filename_meta.get("date_str"):
                    ts = filename_meta["date_str"]
                    if filename_meta.get("time_str"):
                        ts += f"T{filename_meta['time_str']}"
                    modified_at = ts

                # Register in DB (idempotent — skips if already tracked)
                row = recording_db.upsert_file(
                    content_hash=af.content_hash,
                    filename=af.filename,
                    file_path=af.path,
                    file_size=af.size,
                    extension=af.extension,
                    modified_at=modified_at,
                    participants=participants,
                    contact_name=contact_name or (participants[0] if participants[0] != "Unknown" else ""),
                    phone_number=phone_number,
                )

                if row.get("status") == "pending":
                    new_files += 1

                    # Queue background transcription for new files
                    if auto_transcribe:
                        try:
                            self.transcribe_file_async(af.content_hash)
                            queued += 1
                        except Exception as e:
                            logger.error(f"Failed to queue transcription for {af.filename}: {e}")
                            errors += 1

            self._last_sync = int(time.time())

            logger.info(
                f"Scan complete: {discovered} found, {new_files} new, "
                f"{skipped} already tracked, "
                f"{queued} queued for transcription, {errors} errors"
            )

            return {
                "status": "complete",
                "discovered": discovered,
                "new": new_files,
                "queued": queued,
                "skipped": skipped,
                "errors": errors,
                "stale_reset": stale_count,
            }

        except Exception as e:
            logger.error(f"Scan failed: {e}")
            return {"status": "error", "error": str(e)}
        finally:
            self._syncing = False

    # -------------------------------------------------------------------------
    # Step 2: Transcribe a single file
    # -------------------------------------------------------------------------

    def transcribe_file(self, content_hash: str) -> Dict:
        """Transcribe a single audio file by content hash.

        Reads the file record from the DB, runs Whisper, and stores
        the transcription result back in the DB with status 'transcribed'.

        If the source file is locked (e.g. by Dropbox cloud sync), the file
        is copied to a temporary location before passing to ffmpeg/Whisper.

        Args:
            content_hash: SHA256 content hash identifying the file

        Returns:
            Dict with status and transcription summary.
            On lock errors, includes ``"error_type": "file_locked"``.
        """
        record = recording_db.get_file(content_hash)
        if not record:
            return {"status": "error", "error": "File not found in tracking DB"}

        file_path = record["file_path"]
        filename = record["filename"]

        if not os.path.exists(file_path):
            recording_db.update_status(content_hash, "error", "File not found on disk")
            return {"status": "error", "error": f"File not found: {file_path}"}

        # Pre-check: can we actually read the file?
        # Dropbox CloudStorage mounts may return Errno 35 (EDEADLK) for
        # files still being synced.  If so, copy to /tmp first.
        transcribe_path = Path(file_path)
        tmp_path: Optional[Path] = None

        try:
            with open(file_path, "rb") as f:
                f.read(1)  # probe readability
        except OSError as e:
            if e.errno == errno.EDEADLK:
                # Try copying to a temp file to work around the lock
                try:
                    ext = Path(filename).suffix
                    tmp_fd, tmp_str = tempfile.mkstemp(
                        suffix=ext, prefix=f"call_rec_{content_hash[:12]}_"
                    )
                    os.close(tmp_fd)
                    shutil.copy2(file_path, tmp_str)
                    tmp_path = Path(tmp_str)
                    transcribe_path = tmp_path
                    logger.info(
                        f"File locked by cloud sync, copied to temp: {filename}"
                    )
                except OSError as copy_err:
                    logger.warning(
                        f"Cannot read or copy locked file {filename}: {copy_err}"
                    )
                    error_msg = (
                        "File is locked by cloud sync (Dropbox) — "
                        "try again when sync completes"
                    )
                    recording_db.update_status(content_hash, "error", error_msg)
                    return {
                        "status": "error",
                        "error": error_msg,
                        "error_type": "file_locked",
                    }
            else:
                logger.warning(f"Cannot read file {filename}: {e}")
                recording_db.update_status(content_hash, "error", str(e))
                return {"status": "error", "error": str(e)}

        # Mark as transcribing (records start time + resets progress)
        recording_db.update_status(content_hash, "transcribing")

        # Build a progress callback that writes to the DB
        def _on_progress(message: str) -> None:
            try:
                recording_db.update_progress(content_hash, message)
            except Exception:
                pass  # Non-critical — don't let DB writes break transcription

        try:
            logger.info(f"Transcribing: {filename}")
            transcription = self.transcriber.transcribe(
                transcribe_path,
                on_progress=_on_progress,
            )

            if not transcription.text or len(transcription.text.strip()) < MIN_CONTENT_CHARS:
                recording_db.update_status(
                    content_hash, "error",
                    f"Transcription too short ({len(transcription.text)} chars)",
                )
                return {
                    "status": "error",
                    "error": "Transcription too short",
                }

            # Resolve participants from tags + filename (for auto-fill)
            # Build a minimal AudioFile for metadata resolution
            participants = json.loads(record.get("participants", "[]")) or []
            contact_name = participants[0] if participants and participants[0] != "Unknown" else ""

            # Store transcription in DB
            recording_db.update_transcription(
                content_hash=content_hash,
                transcript_text=transcription.text,
                language=transcription.language or "",
                duration_seconds=transcription.duration_seconds,
                confidence=transcription.confidence,
                participants=participants,
                contact_name=contact_name,
            )

            logger.info(
                f"Transcribed: {filename} — "
                f"{len(transcription.text)} chars, {transcription.duration_seconds}s, "
                f"lang={transcription.language}"
            )

            return {
                "status": "transcribed",
                "content_hash": content_hash,
                "duration_seconds": transcription.duration_seconds,
                "language": transcription.language,
                "text_length": len(transcription.text),
            }

        except ValueError as e:
            # Raised by _validate_audio or our reshape-error handler —
            # the audio file is corrupt, empty, or undecodable.
            error_str = str(e)
            logger.warning(f"Transcription failed for {filename}: {error_str}")
            recording_db.update_status(content_hash, "error", error_str)
            return {
                "status": "error",
                "error": error_str,
                "error_type": "bad_audio",
            }
        except Exception as e:
            error_str = str(e)
            # Detect ffmpeg "Resource deadlock avoided" buried in Whisper errors
            if "Resource deadlock avoided" in error_str:
                clean_msg = (
                    "File is locked by cloud sync (Dropbox) — "
                    "try again when sync completes"
                )
                logger.warning(f"Transcription hit file lock for {filename}")
                recording_db.update_status(content_hash, "error", clean_msg)
                return {
                    "status": "error",
                    "error": clean_msg,
                    "error_type": "file_locked",
                }
            # Catch tensor reshape errors that might slip through
            if "cannot reshape tensor of 0 elements" in error_str:
                clean_msg = (
                    f"Audio file contains no processable audio data: {filename}. "
                    f"The file may be corrupt, empty, or in an unsupported format."
                )
                logger.warning(f"Empty audio tensor for {filename}")
                recording_db.update_status(content_hash, "error", clean_msg)
                return {
                    "status": "error",
                    "error": clean_msg,
                    "error_type": "bad_audio",
                }
            logger.error(f"Transcription failed for {filename}: {e}")
            recording_db.update_status(content_hash, "error", error_str)
            return {"status": "error", "error": error_str}
        finally:
            # Clean up temp file if we created one
            if tmp_path and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    # -------------------------------------------------------------------------
    # Step 3: Approve (index into Qdrant)
    # -------------------------------------------------------------------------

    def approve_file(self, content_hash: str) -> Dict:
        """Approve a transcribed file and index it into Qdrant.

        Reads the transcription from the DB, builds LlamaIndex nodes,
        and ingests them into the vector store.

        Args:
            content_hash: SHA256 content hash identifying the file

        Returns:
            Dict with status and indexing results
        """
        record = recording_db.get_file(content_hash)
        if not record:
            return {"status": "error", "error": "File not found"}

        if record["status"] not in ("transcribed", "error"):
            return {
                "status": "error",
                "error": f"Cannot approve file with status '{record['status']}' — "
                         f"must be 'transcribed'",
            }

        transcript_text = record.get("transcript_text", "")
        if not transcript_text or len(transcript_text.strip()) < MIN_CONTENT_CHARS:
            return {"status": "error", "error": "No transcription available"}

        source_id = f"call_recording:{content_hash}"

        try:
            # Build metadata from DB record
            participants = json.loads(record.get("participants", "[]")) or []
            # Override with user-edited contact name if set
            if record.get("contact_name"):
                participants = [record["contact_name"]] + [
                    p for p in participants if p != record["contact_name"]
                ]
            if not participants:
                participants = ["Unknown"]

            participants_str = ", ".join(participants)

            # Parse recorded_at from modified_at
            recorded_at = None
            modified_at_str = record.get("modified_at", "")
            if modified_at_str:
                try:
                    recorded_at = datetime.fromisoformat(modified_at_str)
                    if recorded_at.tzinfo is None:
                        recorded_at = recorded_at.replace(tzinfo=ZoneInfo("UTC"))
                except (ValueError, TypeError):
                    pass
            if not recorded_at:
                recorded_at = datetime.now(tz=ZoneInfo("UTC"))

            duration_seconds = record.get("duration_seconds", 0) or 0
            language = record.get("language", "")
            confidence = record.get("confidence", 0.0) or 0.0
            filename = record.get("filename", "")
            extension = record.get("extension", "")
            file_path = record.get("file_path", "")
            title = Path(filename).stem if filename else "Unknown"

            # Build nodes
            sync_run_id = str(uuid.uuid4())
            ts = int(recorded_at.timestamp()) if recorded_at else int(time.time())

            # Format duration
            hours, remainder = divmod(duration_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            duration_display = (
                f"{hours}:{minutes:02d}:{seconds:02d}" if hours > 0
                else f"{minutes}:{seconds:02d}"
            )

            call_chat_name = f"Call with {participants_str}"

            base_metadata = {
                "source": "call_recording",
                "source_id": source_id,
                "content_type": "call_recording",
                "source_type": "call_recording",
                "chat_name": call_chat_name,
                "sender": participants[0] if participants else "Unknown",
                "timestamp": ts,
                "recording_id": content_hash[:16],
                "duration_seconds": duration_seconds,
                "participants": participants,
                "call_type": "unknown",
                "confidence_score": confidence,
                "audio_format": extension,
                "audio_file_path": file_path,
                "transcription_provider": f"whisper-{self.transcriber.model_size}",
                "language_detected": language,
                "filename": filename,
                "sync_run_id": sync_run_id,
                "indexed_at": int(time.time()),
            }

            # Add phone number if available
            if record.get("phone_number"):
                base_metadata["phone_number"] = record["phone_number"]

            # Chunk the transcript
            transcript = transcript_text.strip()
            chunks = split_text(transcript, MAX_CHUNK_CHARS, CHUNK_OVERLAP_CHARS)
            chunks = [c for c in chunks if is_quality_chunk(c)]

            if not chunks:
                return {"status": "error", "error": "No quality chunks from transcript"}

            from llamaindex_rag import deterministic_node_id

            nodes = []
            for idx, chunk in enumerate(chunks):
                chunk_meta = dict(base_metadata)
                chunk_meta["message"] = chunk[:2000]

                if len(chunks) > 1:
                    chunk_meta["chunk_index"] = str(idx)
                    chunk_meta["chunk_total"] = str(len(chunks))

                embedding_text = (
                    f"Call Recording: {title}\n"
                    f"Participants: {participants_str}\n"
                    f"Duration: {duration_display}\n\n"
                    f"Transcript:\n{chunk}"
                )

                node = TextNode(
                    text=embedding_text,
                    metadata=chunk_meta,
                    id_=deterministic_node_id("call_recording", source_id, idx),
                )
                nodes.append(node)

            # Ingest into Qdrant
            added = self.rag.ingest_nodes(nodes)
            if added != len(nodes):
                logger.warning(
                    f"Partial ingest for {filename}: {added}/{len(nodes)} nodes"
                )

            # Mark as approved in DB
            recording_db.mark_approved(content_hash, source_id)

            logger.info(
                f"Approved and indexed: {filename} "
                f"({len(nodes)} chunk{'s' if len(nodes) > 1 else ''})"
            )

            # Entity extraction (non-critical)
            try:
                from entity_extractor import extract_entities_from_document

                extract_entities_from_document(
                    doc_title=f"Call Recording: {filename}",
                    doc_text=transcript_text,
                    source_ref=source_id,
                    sender=participants_str,
                )
            except Exception as ee:
                logger.debug(f"Entity extraction failed (non-critical): {ee}")

            return {
                "status": "approved",
                "content_hash": content_hash,
                "source_id": source_id,
                "chunks": len(nodes),
            }

        except Exception as e:
            logger.error(f"Approval failed for {content_hash}: {e}")
            return {"status": "error", "error": str(e)}

    # -------------------------------------------------------------------------
    # Step 4: Delete file
    # -------------------------------------------------------------------------

    def delete_file(self, content_hash: str, remove_from_disk: bool = True) -> Dict:
        """Delete a file from tracking and optionally from disk.

        Args:
            content_hash: SHA256 content hash identifying the file
            remove_from_disk: If True, also delete the file from disk

        Returns:
            Dict with status
        """
        record = recording_db.get_file(content_hash)
        if not record:
            return {"status": "error", "error": "File not found"}

        file_path = record.get("file_path", "")

        # Remove from disk if requested
        if remove_from_disk and file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Deleted file from disk: {file_path}")
            except Exception as e:
                logger.warning(f"Could not delete file from disk: {e}")

        # Remove from tracking DB
        recording_db.delete_file(content_hash)
        logger.info(f"Deleted tracking record: {record.get('filename', content_hash)}")

        return {"status": "deleted", "filename": record.get("filename", "")}

    # -------------------------------------------------------------------------
    # Legacy: Full sync (scan + transcribe + auto-approve)
    # -------------------------------------------------------------------------

    def sync_recordings(
        self,
        max_files: int = 100,
        force: bool = False,
    ) -> dict:
        """Legacy full sync: scan, transcribe, and auto-approve all files.

        Kept for backward compatibility with the 'Sync Now' button.

        Args:
            max_files: Maximum files to process per sync run
            force: If True, skip dedup checks

        Returns:
            Dict with sync results
        """
        if self._syncing:
            return {"status": "already_running"}

        self._syncing = True
        self._sync_run_id = str(uuid.uuid4())
        synced = 0
        skipped = 0
        errors = 0

        try:
            # Reset stale transcribing jobs first
            recording_db.reset_stale_transcribing(
                stale_minutes=_STALE_TRANSCRIBING_MINUTES
            )

            # Auto-detect empty collection → force mode
            if not force:
                try:
                    info = self.rag.qdrant_client.get_collection(
                        self.rag.COLLECTION_NAME
                    )
                    if (info.points_count or 0) == 0:
                        logger.info("Qdrant empty — enabling force mode")
                        force = True
                except Exception:
                    pass

            # Pre-fetch known hashes to skip already-tracked files
            known_hashes = recording_db.get_known_hashes()

            audio_files = self.scanner.scan()
            if not audio_files:
                return {"status": "complete", "synced": 0, "skipped": 0, "errors": 0}

            for af in audio_files[:max_files]:
                source_id = af.source_id

                # Dedup: skip files already approved in Qdrant
                if not force and self.rag._message_exists(source_id):
                    skipped += 1
                    continue

                # Skip files already tracked as transcribed/approved
                if af.content_hash in known_hashes:
                    existing = recording_db.get_file(af.content_hash)
                    if existing and existing.get("status") in ("transcribed", "approved"):
                        skipped += 1
                        continue

                # Register + transcribe
                participants = self._resolve_participants(af)
                recording_db.upsert_file(
                    content_hash=af.content_hash,
                    filename=af.filename,
                    file_path=af.path,
                    file_size=af.size,
                    extension=af.extension,
                    modified_at=af.modified_at.isoformat(),
                    participants=participants,
                )

                result = self.transcribe_file(af.content_hash)
                if result.get("status") != "transcribed":
                    errors += 1
                    continue

                # Auto-approve
                approve_result = self.approve_file(af.content_hash)
                if approve_result.get("status") == "approved":
                    synced += 1
                else:
                    errors += 1

            self._last_sync = int(time.time())
            self._recording_count = synced

            return {
                "status": "complete",
                "synced": synced,
                "skipped": skipped,
                "errors": errors,
            }

        except Exception as e:
            logger.error(f"Sync failed: {e}")
            return {"status": "error", "error": str(e)}
        finally:
            self._syncing = False

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _lookup_contact_by_phone(self, phone: str) -> str:
        """Look up a contact name from the entity store by phone number.

        Args:
            phone: Phone number to search for

        Returns:
            Contact display name, or empty string if not found
        """
        if not phone:
            return ""

        try:
            import entity_db

            person_id = entity_db.find_person_by_phone(phone)
            if person_id:
                person = entity_db.get_person(person_id)
                if person:
                    display = person.get("display_name") or person.get("canonical_name") or ""
                    if display:
                        logger.info(
                            f"Resolved phone {phone} → entity '{display}' (id={person_id})"
                        )
                        return str(display)
        except Exception as e:
            logger.debug(f"Entity lookup by phone failed for {phone}: {e}")

        return ""

    def _lookup_entity_by_name(self, name: str) -> Optional[Dict[str, str]]:
        """Look up a person in the entity store by name or alias.

        Used when a filename contains a contact name (not a phone number)
        to retrieve the phone number and canonical display name from the
        entity store.

        Args:
            name: Contact name parsed from filename

        Returns:
            Dict with ``display_name`` and ``phone`` keys, or ``None``
            if not found.
        """
        if not name:
            return None

        try:
            import entity_db

            person = entity_db.get_person_by_name(name)
            if person:
                display = (
                    person.get("display_name")
                    or person.get("canonical_name")
                    or name
                )
                phone = person.get("phone") or ""
                logger.info(
                    f"Resolved name '{name}' → entity '{display}' "
                    f"(id={person.get('id')}, phone={phone or 'N/A'})"
                )
                return {"display_name": str(display), "phone": str(phone)}
        except Exception as e:
            logger.debug(f"Entity lookup by name failed for '{name}': {e}")

        return None

    def _resolve_participants(self, audio_file: AudioFile) -> List[str]:
        """Extract participant names from audio tags and filename."""
        file_meta = audio_file.file_metadata
        filename_meta = _parse_filename_metadata(audio_file.filename)

        participants: List[str] = []

        # From audio tags
        if file_meta.artist:
            participants = [
                p.strip()
                for p in re.split(r"[,;&/]", file_meta.artist)
                if p.strip()
            ]

        # From filename
        fn_participants = filename_meta.get("participants") or ""
        if not participants and fn_participants:
            participants = [
                p.strip()
                for p in fn_participants.split(",")
                if p.strip()
            ]

        if not participants:
            participants = ["Unknown"]

        return participants
