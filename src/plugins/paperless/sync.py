"""Document synchronization logic for Paperless-NGX."""

import logging
import time
import uuid
from typing import List, Optional

from llama_index.core.schema import TextNode

from .client import PaperlessClient

logger = logging.getLogger(__name__)

# Default tag name applied to documents after RAG indexing
DEFAULT_PROCESSED_TAG = "rag-indexed"

# Maximum characters per chunk for embedding.
# text-embedding-3-large has an 8191 token limit.
# Many Paperless documents contain raw HTML, base64, or quoted-printable
# encoded content where the char-to-token ratio is much worse than plain
# text (~1 char/token for encoded data vs ~4 for English prose).
# Using 6000 chars keeps us safely under the 8191 token limit even for
# worst-case encoded content.
MAX_CHUNK_CHARS = 6_000
CHUNK_OVERLAP_CHARS = 200


def _split_text(
    text: str,
    max_chars: int = MAX_CHUNK_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> List[str]:
    """Split text into chunks that fit within the embedding model's token limit.
    
    Tries to split on paragraph boundaries (double newline) for cleaner chunks.
    Falls back to hard character splits with overlap if paragraphs are too large.
    
    Args:
        text: Full document text
        max_chars: Maximum characters per chunk
        overlap: Character overlap between consecutive chunks
        
    Returns:
        List of text chunks (at least one element)
    """
    if len(text) <= max_chars:
        return [text]
    
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        # Try to break at a paragraph boundary
        boundary = text.rfind("\n\n", start, end)
        if boundary == -1 or boundary <= start:
            # Fall back to sentence boundary
            boundary = text.rfind(". ", start, end)
        if boundary == -1 or boundary <= start:
            # Hard split
            boundary = end
        else:
            boundary += 1  # Include the delimiter character
        
        chunks.append(text[start:boundary])
        start = max(boundary - overlap, boundary)  # overlap only when hard-splitting
        if boundary == end:
            start = boundary - overlap  # Apply overlap on hard splits
    
    return chunks


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
    ) -> dict:
        """Sync documents from Paperless to RAG.
        
        Documents that already carry the ``processed_tag_name`` tag in
        Paperless are automatically excluded from the query, so they
        won't be fetched or re-processed.  After successful indexing
        each document is tagged in Paperless.
        
        Args:
            max_docs: Maximum documents to sync
            tags_filter: Optional list of tag names to include
            processed_tag_name: Tag name to mark processed docs
                (default: ``rag-indexed``)
            
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
            logger.info("Starting Paperless document sync...")
            
            # Ensure the processed tag exists and get its ID
            processed_tag_id = self._ensure_processed_tag(processed_tag_name)
            
            # Build exclusion list — skip docs already tagged as processed
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
                        if self.rag._message_exists(source_id):
                            skipped += 1
                            # Still tag it in Paperless if not tagged yet
                            if processed_tag_id:
                                self.client.add_tag_to_document(doc_id, processed_tag_id)
                            continue
                        
                        # Fetch full content
                        content = self.client.get_document_content(doc_id)
                        if not content:
                            skipped += 1
                            continue
                        
                        # Split large documents into chunks to stay within
                        # the embedding model's token limit
                        chunks = _split_text(content, MAX_CHUNK_CHARS, CHUNK_OVERLAP_CHARS)
                        title = doc.get("title", f"Document {doc_id}")
                        
                        base_metadata = {
                            "source": "paperless",
                            "source_id": source_id,
                            "content_type": "document",
                            "chat_name": title,
                            "sender": doc.get("correspondent_name", "Unknown"),
                            "timestamp": int(time.time()),
                            "tags": ",".join(
                                str(t) for t in doc.get("tags", [])
                            ),
                            "document_type": doc.get("document_type_name", ""),
                            "created": doc.get("created", ""),
                            "modified": doc.get("modified", ""),
                        }
                        
                        chunk_ok = True
                        for idx, chunk in enumerate(chunks):
                            chunk_meta = dict(base_metadata)
                            # Store chunk text in 'message' metadata so fulltext
                            # search on the 'message' field can find documents
                            # (truncated to 500 chars to keep payload reasonable)
                            chunk_meta["message"] = chunk[:500]
                            if len(chunks) > 1:
                                chunk_meta["chunk_index"] = str(idx)
                                chunk_meta["chunk_total"] = str(len(chunks))
                            
                            node = TextNode(
                                text=chunk,
                                metadata=chunk_meta,
                                id_=str(uuid.uuid4()),
                            )
                            
                            if not self.rag.add_node(node):
                                chunk_ok = False
                        
                        if chunk_ok:
                            synced += 1
                            if len(chunks) > 1:
                                logger.info(
                                    f"Indexed: {title} ({len(chunks)} chunks)"
                                )
                            else:
                                logger.info(f"Indexed: {title}")
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
