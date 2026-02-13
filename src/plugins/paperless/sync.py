"""Document synchronization logic for Paperless-NGX."""

import logging
import time
from typing import Optional

from llama_index.core.schema import TextNode

from .client import PaperlessClient

logger = logging.getLogger(__name__)


class DocumentSyncer:
    """Handles syncing documents from Paperless-NGX to RAG.
    
    Tracks sync status in Redis to avoid re-indexing.
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
    
    def sync_documents(
        self,
        max_docs: int = 1000,
        tags_filter: Optional[list] = None,
    ) -> dict:
        """Sync documents from Paperless to RAG.
        
        Args:
            max_docs: Maximum documents to sync
            tags_filter: Optional list of tag names to filter
            
        Returns:
            Dict with sync results
        """
        if self._syncing:
            return {"status": "already_running"}
        
        self._syncing = True
        synced = 0
        skipped = 0
        errors = 0
        
        try:
            logger.info("Starting Paperless document sync...")
            
            # Fetch documents from Paperless API
            page = 1
            while synced < max_docs:
                logger.info(f"Fetching page {page}...")
                resp = self.client.get_documents(
                    page=page,
                    page_size=50,
                    tags=tags_filter,
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
                        
                        # Check if already indexed
                        if self.rag._message_exists(source_id):
                            skipped += 1
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
                                "tags": ",".join([t["name"] for t in doc.get("tags", [])]),
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
                f"{skipped} skipped, {errors} errors"
            )
            
            return {
                "status": "complete",
                "synced": synced,
                "skipped": skipped,
                "errors": errors,
            }
            
        except Exception as e:
            logger.error(f"Paperless sync failed: {e}")
            return {"status": "error", "error": str(e)}
        finally:
            self._syncing = False
