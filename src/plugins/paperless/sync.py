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
            
            # Fetch documents from Paperless API
            page = 1
            while synced < max_docs:
                logger.info(f"Fetching page {page}...")
                resp = self.client.get_documents(
                    page=page,
                    page_size=50,
                    tags=tags_filter,
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
                        
                        # Create TextNode
                        node = TextNode(
                            text=content,
                            metadata={
                                "source": "paperless",
                                "source_id": source_id,
                                "content_type": "document",
                                "chat_name": doc.get("title", f"Document {doc_id}"),
                                "sender": doc.get("correspondent_name", "Unknown"),
                                "timestamp": int(time.time()),
                                "tags": ",".join(
                                    str(t) for t in doc.get("tags", [])
                                ),
                                "document_type": doc.get("document_type_name", ""),
                                "created": doc.get("created", ""),
                                "modified": doc.get("modified", ""),
                            },
                            id_=source_id,
                        )
                        
                        # Index in RAG
                        self.rag.add_node(node)
                        synced += 1
                        logger.info(f"Indexed: {doc.get('title', doc_id)}")
                        
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
