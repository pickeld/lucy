"""Document synchronization logic for Paperless-NGX."""

import email
import logging
import re
import time
import uuid
from datetime import datetime
from email.policy import default as default_email_policy
from typing import List, Optional

from llama_index.core.schema import TextNode

from utils.text_processing import (
    MAX_CHUNK_CHARS,
    CHUNK_OVERLAP_CHARS,
    MIN_CONTENT_CHARS,
    MIN_WORD_CHAR_RATIO,
    is_quality_chunk,
    split_text,
    strip_html,
    strip_unicode_control,
)

from .client import PaperlessClient

logger = logging.getLogger(__name__)

# Default tag name applied to documents after RAG indexing
DEFAULT_PROCESSED_TAG = "rag-indexed"

# Regex to extract numeric sequences (≥5 digits) from document content.
# Used to populate a 'numbers' metadata field for reverse ID lookups.
_RE_NUMERIC_SEQUENCES = re.compile(r"\b\d{5,}\b")

# Regex patterns for content sanitization
# Matches contiguous base64 blocks (3+ lines of base64 characters)
_RE_BASE64_BLOCK = re.compile(
    r"(?:^[A-Za-z0-9+/=]{40,}\s*$\n?){3,}",
    re.MULTILINE,
)

# Matches MIME boundary markers like --0000000000000c3639060523d905
_RE_MIME_BOUNDARY = re.compile(
    r"^--[A-Za-z0-9_=.-]{10,}(?:--)?$",
    re.MULTILINE,
)

# Matches common MIME/email headers
_RE_MIME_HEADERS = re.compile(
    r"^(?:Content-Type|Content-Disposition|Content-Transfer-Encoding|"
    r"Content-ID|X-Attachment-Id|MIME-Version|X-Mailer|"
    r"X-Google-DKIM-Signature|X-Gm-Message-State|X-Google-Smtp-Source|"
    r"ARC-Seal|ARC-Message-Signature|ARC-Authentication-Results|"
    r"Return-Path|Received|DKIM-Signature|Message-ID|Date|From|To|"
    r"Subject|In-Reply-To|References):.*$",
    re.MULTILINE | re.IGNORECASE,
)

# Matches header continuation lines (start with whitespace after a header)
_RE_HEADER_CONTINUATION = re.compile(
    r"(?<=\n)[ \t]+\S.*$",
    re.MULTILINE,
)


def _extract_mime_text_parts(raw: str) -> Optional[str]:
    """Try to parse *raw* as a MIME message and extract text parts.

    Uses Python's :mod:`email` library.  Returns the concatenated text
    from all ``text/plain`` and ``text/html`` parts (HTML is stripped to
    plain text).  Returns ``None`` if *raw* does not look like a MIME
    message or contains no text parts.

    Args:
        raw: Raw document content that may be a MIME email

    Returns:
        Extracted plain text, or None if not a parseable MIME message
    """
    # Quick heuristic: only attempt MIME parsing if the content contains
    # a MIME boundary marker or Content-Type header near the start.
    head = raw[:2000]
    if not (
        "Content-Type:" in head
        or _RE_MIME_BOUNDARY.search(head)
        or "MIME-Version:" in head
    ):
        return None

    try:
        msg = email.message_from_string(raw, policy=default_email_policy)
    except Exception:
        return None

    text_parts: List[str] = []
    for part in msg.walk():
        ct = part.get_content_type()
        if ct == "text/plain":
            payload = part.get_content()
            if isinstance(payload, str) and payload.strip():
                text_parts.append(payload.strip())
        elif ct == "text/html":
            payload = part.get_content()
            if isinstance(payload, str):
                plain = strip_html(payload).strip()
                if plain:
                    text_parts.append(plain)

    return "\n\n".join(text_parts) if text_parts else None


def _sanitize_content(raw: str) -> str:
    """Clean raw Paperless document content for RAG embedding.

    Handles the common case where Paperless returns raw MIME email data
    (including base64-encoded attachments, MIME headers, boundary markers)
    as the document ``content`` field.

    Processing pipeline:
    1. Attempt full MIME parsing — if successful, extract only text parts
    2. Otherwise, apply regex-based stripping of base64 blocks, MIME
       headers, and boundary markers
    3. Strip residual HTML tags
    4. Normalise whitespace

    Args:
        raw: Raw content string from Paperless API

    Returns:
        Cleaned text suitable for embedding (may be empty)
    """
    if not raw:
        return ""

    # --- Step 0: Strip Unicode control characters (RTL/LTR marks, etc.) ---
    # OCR engines (especially for Hebrew/Arabic) insert these characters
    # which break Qdrant's multilingual tokenizer and prevent fulltext
    # search from matching words like "דוד" wrapped in RTL marks (‫דוד‬).
    raw = strip_unicode_control(raw)

    # --- Step 1: Try structured MIME parsing first ---
    mime_text = _extract_mime_text_parts(raw)
    if mime_text:
        text = mime_text
        logger.debug("Extracted text via MIME parsing (%d chars)", len(text))
    else:
        text = raw

    # --- Step 2: Regex-based cleanup (catches residual noise) ---
    # Remove base64 blocks (must come before header stripping so we don't
    # accidentally break multi-line base64 detection)
    text = _RE_BASE64_BLOCK.sub("", text)

    # Remove MIME headers and their continuation lines
    text = _RE_MIME_HEADERS.sub("", text)
    # Clean up orphaned continuation lines (indented lines after removed headers)
    # Only remove if they follow a blank or removed line
    text = re.sub(r"(?:^[ \t]+\S[^\n]*$\n?){1,}", "", text, flags=re.MULTILINE)

    # Remove MIME boundary markers
    text = _RE_MIME_BOUNDARY.sub("", text)

    # Remove standalone Content-* fragments that may remain
    text = re.sub(
        r'^(?:name|filename|charset|boundary)\s*=\s*"[^"]*".*$',
        "",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    # --- Step 3: Strip residual HTML ---
    if "<" in text and ">" in text:
        text = strip_html(text)

    # --- Step 4: Normalise whitespace ---
    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of spaces/tabs (but not newlines)
    text = re.sub(r"[^\S\n]{2,}", " ", text)
    text = text.strip()

    return text




class DocumentSyncer:
    """Handles syncing documents from Paperless-NGX to RAG.
    
    Tags processed documents in Paperless with a custom tag (default:
    ``rag-indexed``) so they are automatically excluded from future sync
    runs.  The tag is created in Paperless on first use.
    """
    
    def __init__(self, client: PaperlessClient, rag):
        """Initialize syncer.
        
        Args:
            client: PaperlessClient instance
            rag: LlamaIndexRAG instance
        """
        self.client = client
        self.rag = rag
        self._syncing = False
        self._last_sync = 0
        self._doc_count = 0
        # Resolved tag ID — populated lazily on first sync
        self._processed_tag_id: Optional[int] = None
    
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
        """Get count of synced documents."""
        return self._doc_count
    
    def _ensure_processed_tag(self, tag_name: str) -> Optional[int]:
        """Ensure the processed-documents tag exists in Paperless.
        
        Creates the tag if it doesn't exist yet.  Caches the tag ID for
        the lifetime of this syncer instance.
        
        Args:
            tag_name: Name of the tag to use (e.g. ``rag-indexed``)
            
        Returns:
            Tag ID, or None if tag creation failed
        """
        if self._processed_tag_id is not None:
            return self._processed_tag_id
        
        tag_id = self.client.get_or_create_tag(
            name=tag_name,
            color="#17a2b8",  # teal / info-blue
        )
        if tag_id is not None:
            self._processed_tag_id = tag_id
            logger.info(
                f"Using Paperless tag '{tag_name}' (id={tag_id}) "
                "for processed documents"
            )
        else:
            logger.warning(
                f"Could not get/create Paperless tag '{tag_name}'. "
                "Documents will still be synced but not tagged."
            )
        return tag_id
    
    def sync_documents(
        self,
        max_docs: int = 1000,
        tags_filter: Optional[list] = None,
        processed_tag_name: str = DEFAULT_PROCESSED_TAG,
        force: bool = False,
    ) -> dict:
        """Sync documents from Paperless to RAG.
        
        Documents that already carry the ``processed_tag_name`` tag in
        Paperless are automatically excluded from the query, so they
        won't be fetched or re-processed.  After successful indexing
        each document is tagged in Paperless.
        
        When ``force=True``, the processed-tag exclusion filter and the
        Qdrant deduplication check are both skipped.  This is required
        after deleting/recreating the Qdrant collection, because the
        documents in Paperless still carry the tag from the previous
        sync run but the vectors are gone.
        
        Args:
            max_docs: Maximum documents to sync
            tags_filter: Optional list of tag names to include
            processed_tag_name: Tag name to mark processed docs
                (default: ``rag-indexed``)
            force: If True, skip tag exclusion and dedup checks
                (re-index everything)
            
        Returns:
            Dict with sync results
        """
        if self._syncing:
            return {"status": "already_running"}
        
        self._syncing = True
        synced = 0
        skipped = 0
        errors = 0
        tagged = 0
        
        try:
            # Auto-detect empty collection → force mode
            # After deleting/recreating the Qdrant collection the vectors
            # are gone but the Paperless docs still carry the processed tag,
            # so a normal sync returns 0 results.  Detect this and switch
            # to force mode automatically.
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
                logger.info("Starting Paperless FORCE re-sync (ignoring processed tag)...")
            else:
                logger.info("Starting Paperless document sync...")
            
            # Pre-fetch correspondent id→name mapping for sender resolution
            correspondents = self.client.get_correspondents()
            logger.info(f"Loaded {len(correspondents)} correspondents from Paperless")
            
            # Ensure the processed tag exists and get its ID
            processed_tag_id = self._ensure_processed_tag(processed_tag_name)
            
            # Build exclusion list — skip docs already tagged as processed
            # When force=True, don't exclude anything so ALL docs are re-fetched
            if force:
                exclude_tag_ids = []
            else:
                exclude_tag_ids = [processed_tag_id] if processed_tag_id else []
            
            # Resolve tag names to IDs for the include filter
            include_tag_ids: Optional[List[int]] = None
            if tags_filter:
                include_tag_ids = []
                for tag_name in tags_filter:
                    tag = self.client.get_tag_by_name(tag_name)
                    if tag:
                        include_tag_ids.append(tag["id"])
                        logger.info(f"Filter tag '{tag_name}' → id={tag['id']}")
                    else:
                        logger.warning(
                            f"Filter tag '{tag_name}' not found in Paperless — "
                            "ignoring"
                        )
                if not include_tag_ids:
                    logger.warning(
                        "None of the configured sync tags were found. "
                        "No documents will be synced."
                    )
                    return {
                        "status": "complete",
                        "synced": 0,
                        "tagged": 0,
                        "skipped": 0,
                        "errors": 0,
                        "warning": "No matching tags found in Paperless",
                    }
            
            # Fetch documents from Paperless API
            page = 1
            while synced < max_docs:
                logger.info(f"Fetching page {page}...")
                resp = self.client.get_documents(
                    page=page,
                    page_size=50,
                    tags=include_tag_ids,
                    exclude_tags=exclude_tag_ids,
                )
                
                docs = resp.get("results", [])
                if not docs:
                    break
                
                for doc in docs:
                    if synced >= max_docs:
                        break
                    
                    try:
                        doc_id = doc["id"]
                        source_id = f"paperless:{doc_id}"
                        
                        # Check if already indexed in RAG (belt-and-suspenders)
                        # Skip this check when force=True (collection was reset)
                        if not force and self.rag._message_exists(source_id):
                            skipped += 1
                            # Still tag it in Paperless if not tagged yet
                            if processed_tag_id:
                                self.client.add_tag_to_document(doc_id, processed_tag_id)
                            continue
                        
                        # Fetch full content and sanitize
                        raw_content = self.client.get_document_content(doc_id)
                        if not raw_content:
                            skipped += 1
                            continue
                        
                        title = doc.get("title", f"Document {doc_id}")
                        content = _sanitize_content(raw_content)
                        if len(content) < MIN_CONTENT_CHARS:
                            logger.info(
                                f"Skipping '{title}' (id={doc_id}): "
                                f"only {len(content)} chars after sanitization "
                                f"(raw was {len(raw_content)} chars)"
                            )
                            skipped += 1
                            continue
                        
                        # Split large documents into chunks to stay within
                        # the embedding model's token limit
                        chunks = split_text(content, MAX_CHUNK_CHARS, CHUNK_OVERLAP_CHARS)
                        
                        # Quality-gate: drop chunks that are mostly noise
                        pre_filter = len(chunks)
                        chunks = [c for c in chunks if is_quality_chunk(c)]
                        if pre_filter > len(chunks):
                            logger.info(
                                f"Quality filter dropped {pre_filter - len(chunks)}/{pre_filter} "
                                f"chunks for '{title}'"
                            )
                        if not chunks:
                            logger.info(
                                f"Skipping '{title}' (id={doc_id}): "
                                "no chunks passed quality filter"
                            )
                            skipped += 1
                            continue
                        
                        # Resolve correspondent name from pre-fetched mapping
                        correspondent_id = doc.get("correspondent")
                        sender = correspondents.get(correspondent_id, "") if correspondent_id else ""
                        
                        # Parse document creation date from Paperless API.
                        # The 'created' field is an ISO 8601 string like
                        # "2023-03-15T00:00:00+02:00".  Use it as the primary
                        # timestamp so the date shown in Lucy's response is the
                        # actual document date, not the sync/indexing time.
                        created_str = doc.get("created", "")
                        if created_str:
                            try:
                                created_dt = datetime.fromisoformat(created_str)
                                doc_timestamp = int(created_dt.timestamp())
                            except (ValueError, TypeError):
                                doc_timestamp = int(time.time())
                        else:
                            doc_timestamp = int(time.time())
                        
                        base_metadata = {
                            "source": "paperless",
                            "source_id": source_id,
                            "content_type": "document",
                            "chat_name": title,
                            "sender": sender,
                            "timestamp": doc_timestamp,
                            "tags": ",".join(
                                str(t) for t in doc.get("tags", [])
                            ),
                            "document_type": doc.get("document_type_name", ""),
                            "created": created_str,
                            "modified": doc.get("modified", ""),
                        }
                        
                        # Extract all numeric sequences (≥5 digits) from the
                        # full document content for reverse ID/number lookups.
                        # Stored as space-separated string in 'numbers' metadata
                        # field which has a fulltext index in Qdrant.
                        all_numbers = sorted(set(
                            _RE_NUMERIC_SEQUENCES.findall(content)
                        ))
                        numbers_str = " ".join(all_numbers) if all_numbers else ""
                        
                        # Build all chunk nodes, then batch-embed in one API call
                        chunk_nodes = []
                        for idx, chunk in enumerate(chunks):
                            chunk_meta = dict(base_metadata)
                            chunk_meta["message"] = chunk
                            if numbers_str:
                                chunk_meta["numbers"] = numbers_str
                            if len(chunks) > 1:
                                chunk_meta["chunk_index"] = str(idx)
                                chunk_meta["chunk_total"] = str(len(chunks))
                            
                            embedding_text = f"Document: {title}\n\n{chunk}"
                            
                            chunk_nodes.append(TextNode(
                                text=embedding_text,
                                metadata=chunk_meta,
                                id_=str(uuid.uuid4()),
                            ))
                        
                        # Batch insert: single embedding API call + Qdrant upsert
                        added = self.rag.add_nodes(chunk_nodes)
                        chunk_ok = added == len(chunk_nodes)
                        
                        if chunk_ok:
                            synced += 1
                            if len(chunks) > 1:
                                logger.info(
                                    f"Indexed: {title} ({len(chunks)} chunks)"
                                )
                            else:
                                logger.info(f"Indexed: {title}")
                            
                            # Entity extraction from document content
                            try:
                                from entity_extractor import extract_entities_from_document
                                extract_entities_from_document(
                                    doc_title=title,
                                    doc_text=content,
                                    source_ref=source_id,
                                    sender=sender,
                                )
                            except Exception as ee:
                                logger.debug(f"Entity extraction failed for '{title}' (non-critical): {ee}")
                        else:
                            errors += 1
                            logger.warning(f"Partially failed: {title}")
                        
                        # Tag the document in Paperless as processed
                        if processed_tag_id:
                            if self.client.add_tag_to_document(doc_id, processed_tag_id):
                                tagged += 1
                        
                    except Exception as e:
                        logger.error(f"Error syncing document {doc.get('id')}: {e}")
                        errors += 1
                
                # Check if there are more pages
                if not resp.get("next"):
                    break
                page += 1
            
            self._last_sync = int(time.time())
            self._doc_count = synced
            
            logger.info(
                f"Paperless sync complete: {synced} indexed, "
                f"{tagged} tagged, {skipped} skipped, {errors} errors"
            )
            
            return {
                "status": "complete",
                "synced": synced,
                "tagged": tagged,
                "skipped": skipped,
                "errors": errors,
            }
            
        except Exception as e:
            logger.error(f"Paperless sync failed: {e}")
            return {"status": "error", "error": str(e)}
        finally:
            self._syncing = False
