"""Call recording synchronization logic.

Orchestrates the full pipeline: scan audio files → dedup check →
transcribe with Whisper → create CallRecordingDocument → index in Qdrant.

Follows the same sync pattern as EmailSyncer and DocumentSyncer.
"""

import logging
import re
import time
import uuid
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

from .scanner import AudioFile, LocalFileScanner, _parse_filename_metadata
from .transcriber import TranscriptionResult, WhisperTranscriber

logger = logging.getLogger(__name__)


class CallRecordingSyncer:
    """Handles syncing call recordings from a file source to the RAG vector store.

    Pipeline for each audio file:
    1. Scanner discovers audio files (local dir or Dropbox)
    2. Dedup check: SHA256 content hash against Qdrant source_id index
    3. Download (for Dropbox; no-op for local)
    4. Transcribe with local Whisper
    5. Parse metadata from file tags + filename patterns
    6. Create CallRecordingDocument
    7. Chunk long transcripts if needed
    8. Index via rag.ingest_nodes()
    9. Entity extraction post-hook
    10. Cleanup temp files (Dropbox only)

    Args:
        scanner: File scanner instance (local or Dropbox)
        transcriber: WhisperTranscriber instance
        rag: LlamaIndexRAG instance
    """

    def __init__(
        self,
        scanner: LocalFileScanner,
        transcriber: WhisperTranscriber,
        rag,
    ):
        self.scanner = scanner
        self.transcriber = transcriber
        self.rag = rag
        self._syncing = False
        self._last_sync = 0
        self._recording_count = 0

    @property
    def is_syncing(self) -> bool:
        """Check if sync is currently running."""
        return self._syncing

    @property
    def last_sync_time(self) -> int:
        """Get last sync timestamp."""
        return self._last_sync

    @property
    def synced_count(self) -> int:
        """Get count of synced recordings."""
        return self._recording_count

    def sync_recordings(
        self,
        max_files: int = 100,
        force: bool = False,
    ) -> dict:
        """Sync call recordings from the configured source to RAG.

        Args:
            max_files: Maximum files to process per sync run
            force: If True, skip dedup checks (re-index everything)

        Returns:
            Dict with sync results: status, synced, skipped, errors
        """
        if self._syncing:
            return {"status": "already_running"}

        self._syncing = True
        self._sync_run_id = str(uuid.uuid4())
        synced = 0
        skipped = 0
        errors = 0

        try:
            # Auto-detect empty collection → force mode
            if not force:
                try:
                    info = self.rag.qdrant_client.get_collection(
                        self.rag.COLLECTION_NAME
                    )
                    if (info.points_count or 0) == 0:
                        logger.info(
                            "Qdrant collection is empty — automatically "
                            "enabling force mode for full re-sync"
                        )
                        force = True
                except Exception as e:
                    logger.debug(f"Could not check collection point count: {e}")

            if force:
                logger.info(
                    "Starting call recordings FORCE re-sync "
                    "(ignoring dedup checks)..."
                )
            else:
                logger.info("Starting call recordings sync...")

            # Step 1: Scan for audio files
            audio_files = self.scanner.scan()

            if not audio_files:
                logger.info("No audio files found in source")
                return {
                    "status": "complete",
                    "synced": 0,
                    "skipped": 0,
                    "errors": 0,
                }

            logger.info(f"Found {len(audio_files)} audio files to process")

            # Step 2: Process each file
            for audio_file in audio_files:
                if synced >= max_files:
                    logger.info(
                        f"Reached max_files limit ({max_files}), stopping"
                    )
                    break

                try:
                    result = self._process_single_file(
                        audio_file=audio_file,
                        force=force,
                    )

                    if result == "synced":
                        synced += 1
                    elif result == "skipped":
                        skipped += 1
                    elif result == "error":
                        errors += 1

                except Exception as e:
                    logger.error(
                        f"Error processing {audio_file.filename}: {e}"
                    )
                    errors += 1

            self._last_sync = int(time.time())
            self._recording_count = synced

            logger.info(
                f"Call recordings sync complete: {synced} indexed, "
                f"{skipped} skipped, {errors} errors"
            )

            return {
                "status": "complete",
                "synced": synced,
                "skipped": skipped,
                "errors": errors,
            }

        except Exception as e:
            logger.error(f"Call recordings sync failed: {e}")
            return {"status": "error", "error": str(e)}
        finally:
            self._syncing = False

    def _process_single_file(
        self,
        audio_file: AudioFile,
        force: bool,
    ) -> str:
        """Process a single audio file through the full pipeline.

        Args:
            audio_file: The audio file to process
            force: Skip dedup checks

        Returns:
            "synced", "skipped", or "error"
        """
        source_id = audio_file.source_id
        local_path = None

        try:
            # Dedup check
            if not force and self.rag._message_exists(source_id):
                logger.debug(f"Skipping already-indexed: {audio_file.filename}")
                return "skipped"

            # Download (no-op for local files)
            local_path = self.scanner.download(audio_file)

            # Transcribe
            logger.info(f"Transcribing: {audio_file.filename}")
            transcription = self.transcriber.transcribe(local_path)

            if not transcription.text or len(transcription.text.strip()) < MIN_CONTENT_CHARS:
                logger.info(
                    f"Skipping {audio_file.filename}: "
                    f"transcription too short ({len(transcription.text)} chars)"
                )
                return "skipped"

            # Resolve metadata
            metadata = self._resolve_metadata(
                audio_file=audio_file,
                transcription=transcription,
            )

            # Create nodes and index
            nodes = self._build_nodes(
                audio_file=audio_file,
                transcription=transcription,
                metadata=metadata,
            )

            if not nodes:
                return "skipped"

            # Batch ingest via IngestionPipeline (with embedding cache)
            added = self.rag.ingest_nodes(nodes)
            if added != len(nodes):
                logger.warning(
                    f"Partial ingest for {audio_file.filename}: "
                    f"{added}/{len(nodes)} nodes"
                )

            logger.info(
                f"Indexed: {audio_file.filename} "
                f"({len(nodes)} chunk{'s' if len(nodes) > 1 else ''}, "
                f"{transcription.duration_seconds}s, "
                f"lang={transcription.language})"
            )

            # Entity extraction
            try:
                from entity_extractor import extract_entities_from_document

                participants_str = ", ".join(metadata["participants"])
                extract_entities_from_document(
                    doc_title=f"Call Recording: {audio_file.filename}",
                    doc_text=transcription.text,
                    source_ref=source_id,
                    sender=participants_str,
                )
            except Exception as ee:
                logger.debug(
                    f"Entity extraction failed for "
                    f"'{audio_file.filename}' (non-critical): {ee}"
                )

            return "synced"

        except Exception as e:
            logger.error(f"Failed to process {audio_file.filename}: {e}")
            return "error"

        finally:
            # Cleanup temp files (Dropbox downloads)
            if local_path:
                self.scanner.cleanup(audio_file, local_path)

    def _resolve_metadata(
        self,
        audio_file: AudioFile,
        transcription: TranscriptionResult,
    ) -> Dict:
        """Resolve recording metadata from multiple sources.

        Priority order for each field:
        1. Audio file embedded tags (ID3/MP4 via mutagen)
        2. Filename pattern parsing
        3. Transcription result (language, duration)

        Args:
            audio_file: The audio file with file_metadata
            transcription: The Whisper transcription result

        Returns:
            Dict with resolved metadata fields
        """
        file_meta = audio_file.file_metadata
        filename_meta = _parse_filename_metadata(audio_file.filename)

        # Resolve participants
        participants: List[str] = []

        # 1. From audio tags (artist field often has caller name)
        if file_meta.artist:
            participants = [
                p.strip()
                for p in re.split(r"[,;&/]", file_meta.artist)
                if p.strip()
            ]

        # 2. From filename parsing
        fn_participants = filename_meta.get("participants") or ""
        if not participants and fn_participants:
            participants = [
                p.strip()
                for p in fn_participants.split(",")
                if p.strip()
            ]

        # 3. Ultimate fallback
        if not participants:
            participants = ["Unknown"]

        # Resolve timestamp
        recorded_at = None

        # 1. From audio tags date field
        if file_meta.date:
            try:
                # Try ISO format first
                recorded_at = datetime.fromisoformat(file_meta.date)
                if recorded_at.tzinfo is None:
                    recorded_at = recorded_at.replace(tzinfo=ZoneInfo("UTC"))
            except (ValueError, TypeError):
                # Try just year
                try:
                    year = int(file_meta.date[:4])
                    recorded_at = datetime(year, 1, 1, tzinfo=ZoneInfo("UTC"))
                except (ValueError, TypeError, IndexError):
                    pass

        # 2. From filename date pattern
        fn_date_str = filename_meta.get("date_str") or ""
        if not recorded_at and fn_date_str:
            try:
                recorded_at = datetime.fromisoformat(fn_date_str)
                if recorded_at.tzinfo is None:
                    recorded_at = recorded_at.replace(tzinfo=ZoneInfo("UTC"))
            except (ValueError, TypeError):
                pass

        # 3. From file modification time
        if not recorded_at:
            recorded_at = audio_file.modified_at

        # Resolve duration
        duration_seconds = transcription.duration_seconds
        if not duration_seconds and file_meta.duration_seconds:
            duration_seconds = int(file_meta.duration_seconds)

        # Resolve title
        title = file_meta.title or Path(audio_file.filename).stem

        return {
            "participants": participants,
            "recorded_at": recorded_at,
            "duration_seconds": duration_seconds,
            "title": title,
            "language": transcription.language,
            "confidence": transcription.confidence,
        }

    def _build_nodes(
        self,
        audio_file: AudioFile,
        transcription: TranscriptionResult,
        metadata: Dict,
    ) -> List[TextNode]:
        """Build LlamaIndex TextNodes from transcription result.

        Handles chunking for long transcripts that exceed the embedding
        model's token limit.

        Args:
            audio_file: The source audio file
            transcription: Whisper transcription result
            metadata: Resolved metadata dict

        Returns:
            List of TextNode instances ready for ingestion
        """
        from llamaindex_rag import deterministic_node_id

        source_id = audio_file.source_id
        participants = metadata["participants"]
        participants_str = ", ".join(participants)
        recorded_at = metadata["recorded_at"]
        duration_seconds = metadata["duration_seconds"]
        title = metadata["title"]

        # Compute Unix timestamp
        ts = int(recorded_at.timestamp()) if recorded_at else int(time.time())

        # Format duration for display
        hours, remainder = divmod(duration_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            duration_display = f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            duration_display = f"{minutes}:{seconds:02d}"

        # Build call chat name for consistent retrieval
        call_chat_name = f"Call with {participants_str}"

        # Base metadata for all chunks
        base_metadata = {
            "source": "call_recording",
            "source_id": source_id,
            "content_type": "call_recording",
            "source_type": "call_recording",
            "chat_name": call_chat_name,
            "sender": participants[0] if participants else "Unknown",
            "timestamp": ts,
            "recording_id": audio_file.content_hash[:16],
            "duration_seconds": duration_seconds,
            "participants": participants,
            "call_type": "unknown",
            "confidence_score": metadata["confidence"],
            "audio_format": audio_file.extension,
            "audio_file_path": audio_file.path,
            "transcription_provider": f"whisper-{self.transcriber.model_size}",
            "language_detected": metadata["language"],
            "filename": audio_file.filename,
            # Sync metadata
            "sync_run_id": self._sync_run_id,
            "indexed_at": int(time.time()),
        }

        # Chunk the transcript if it's too long for a single embedding
        transcript = transcription.text.strip()
        chunks = split_text(transcript, MAX_CHUNK_CHARS, CHUNK_OVERLAP_CHARS)
        chunks = [c for c in chunks if is_quality_chunk(c)]

        if not chunks:
            logger.info(
                f"No quality chunks for {audio_file.filename} — skipping"
            )
            return []

        nodes = []
        for idx, chunk in enumerate(chunks):
            chunk_meta = dict(base_metadata)
            chunk_meta["message"] = chunk[:2000]  # Truncate for fulltext search field

            if len(chunks) > 1:
                chunk_meta["chunk_index"] = str(idx)
                chunk_meta["chunk_total"] = str(len(chunks))

            # Build embedding text with context header
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

        return nodes
