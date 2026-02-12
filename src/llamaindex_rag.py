"""LlamaIndex RAG (Retrieval Augmented Generation) for WhatsApp messages.

Uses Qdrant as vector store and OpenAI embeddings for semantic search.
Replaces the previous LangChain-based RAG implementation.

Qdrant Dashboard: http://localhost:6333/dashboard
"""

import json
import os
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from models.base import BaseRAGDocument

from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchText,
    MatchValue,
    PayloadSchemaType,
    Range,
    TextIndexParams,
    TextIndexType,
    TokenizerType,
    VectorParams,
)

from config import config
from utils.logger import logger
from utils.redis_conn import get_redis_client


def format_timestamp(timestamp: str, timezone: str = "Asia/Jerusalem") -> str:
    """Convert Unix timestamp to human-readable format.
    
    Args:
        timestamp: Unix timestamp as string or int
        timezone: Timezone for display (default: Asia/Jerusalem)
        
    Returns:
        Formatted datetime string (e.g., "31/12/2024 10:30")
    """
    try:
        ts = int(timestamp)
        tz = ZoneInfo(timezone)
        dt = datetime.fromtimestamp(ts, tz=tz)
        return dt.strftime("%d/%m/%Y %H:%M")
    except (ValueError, TypeError, KeyError):
        return str(timestamp)


class LlamaIndexRAG:
    """LlamaIndex-based RAG for WhatsApp message search and retrieval.
    
    Uses Qdrant server as vector store and OpenAI embeddings.
    Connects to Qdrant server at QDRANT_HOST:QDRANT_PORT (default: localhost:6333).
    
    Dashboard available at: http://localhost:6333/dashboard
    """
    
    _instance = None
    _index = None
    _qdrant_client = None
    _vector_store = None
    
    COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "whatsapp_messages")
    VECTOR_SIZE = 1536  # OpenAI embedding dimension
    MINIMUM_SIMILARITY_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.5"))
    MAX_CONTEXT_TOKENS = int(os.getenv("RAG_MAX_CONTEXT_TOKENS", "3000"))
    RRF_K = 60  # Reciprocal Rank Fusion constant (standard default)
    
    def __new__(cls):
        """Singleton pattern to ensure one RAG instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the LlamaIndex RAG system."""
        if self._initialized:
            return
        
        logger.info("Starting LlamaIndex RAG initialization...")
        
        # Get Qdrant server config from environment
        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        logger.info(f"Qdrant config: {self.qdrant_host}:{self.qdrant_port}")
        
        # Configure embedding model only (LLM is configured lazily on first use)
        logger.info("Configuring embedding model...")
        self._configure_embedding()
        logger.info("Embedding model configured")
        
        self._initialized = True
        self._llm_configured = False
        
        # Ensure collection exists (connects to Qdrant)
        logger.info("Ensuring Qdrant collection exists...")
        self._ensure_collection()
        logger.info(f"LlamaIndex RAG initialized with Qdrant at {self.qdrant_host}:{self.qdrant_port}")
    
    def _configure_embedding(self):
        """Configure embedding model (fast, required at startup).
        
        Uses text-embedding-3-small which is 5x cheaper and higher quality
        than the legacy text-embedding-ada-002. Same 1536 dimensions.
        """
        logger.debug("Setting up OpenAI embedding model...")
        Settings.embed_model = OpenAIEmbedding(
            api_key=config.OPENAI_API_KEY,
            model="text-embedding-3-small"
        )
        logger.debug("OpenAI embedding model configured (text-embedding-3-small)")
    
    def _ensure_llm_configured(self):
        """Lazily configure LLM on first use (Gemini import is slow)."""
        if self._llm_configured:
            return
        
        llm_provider = os.getenv('LLM_PROVIDER', 'openai').lower()
        logger.info(f"Configuring LLM provider: {llm_provider} (lazy init)...")
        
        if llm_provider == 'gemini':
            try:
                from llama_index.llms.gemini import Gemini
                Settings.llm = Gemini(
                    api_key=config.GOOGLE_API_KEY,
                    model=getattr(config, 'GEMINI_MODEL', 'gemini-pro'),
                    temperature=0.3
                )
                logger.info("Gemini LLM configured")
            except ImportError:
                logger.warning("Gemini LLM not available, falling back to OpenAI")
                Settings.llm = LlamaIndexOpenAI(
                    api_key=config.OPENAI_API_KEY,
                    model=config.OPENAI_MODEL,
                    temperature=0.3
                )
        else:
            Settings.llm = LlamaIndexOpenAI(
                api_key=config.OPENAI_API_KEY,
                model=config.OPENAI_MODEL,
                temperature=0.3
            )
            logger.info("OpenAI LLM configured")
        
        self._llm_configured = True
    
    @property
    def qdrant_client(self) -> QdrantClient:
        """Get or create the Qdrant client."""
        if LlamaIndexRAG._qdrant_client is None:
            logger.info(f"Connecting to Qdrant at {self.qdrant_host}:{self.qdrant_port}...")
            LlamaIndexRAG._qdrant_client = QdrantClient(
                host=self.qdrant_host,
                port=self.qdrant_port,
                timeout=10,  # Connection timeout in seconds
                prefer_grpc=False,  # Use HTTP for more reliable connections
            )
            logger.info("Qdrant client created")
        return LlamaIndexRAG._qdrant_client
    
    @property
    def vector_store(self) -> QdrantVectorStore:
        """Get or create the Qdrant vector store."""
        if LlamaIndexRAG._vector_store is None:
            LlamaIndexRAG._vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.COLLECTION_NAME,
            )
        return LlamaIndexRAG._vector_store
    
    @property
    def index(self) -> VectorStoreIndex:
        """Get or create the vector store index."""
        if LlamaIndexRAG._index is None:
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            LlamaIndexRAG._index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                storage_context=storage_context,
            )
        return LlamaIndexRAG._index
    
    def _ensure_collection(self):
        """Ensure the collection exists in Qdrant with text indexes for metadata search."""
        try:
            logger.debug("Fetching existing collections from Qdrant...")
            collections = self.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]
            logger.debug(f"Found collections: {collection_names}")
            
            if self.COLLECTION_NAME not in collection_names:
                logger.info(f"Creating Qdrant collection: {self.COLLECTION_NAME}")
                self.qdrant_client.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=self.VECTOR_SIZE,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.COLLECTION_NAME}")
            
            # Ensure text indexes exist for full-text search on metadata fields
            self._ensure_text_indexes()
            # Ensure payload indexes for efficient filtering
            self._ensure_payload_indexes()
            
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise  # Re-raise to surface connection issues during init
    
    def _ensure_text_indexes(self):
        """Create text indexes on sender and chat_name fields for full-text search."""
        try:
            # Create text index on 'sender' field for searching by sender name
            self.qdrant_client.create_payload_index(
                collection_name=self.COLLECTION_NAME,
                field_name="sender",
                field_schema=TextIndexParams(
                    type=TextIndexType.TEXT,
                    tokenizer=TokenizerType.MULTILINGUAL,
                    min_token_len=2,
                    max_token_len=20,
                    lowercase=True
                )
            )
            logger.info("Created text index on 'sender' field")
        except Exception as e:
            # Index might already exist
            logger.debug(f"Could not create sender index (may exist): {e}")
        
        try:
            # Create text index on 'chat_name' field for searching by chat/group name
            self.qdrant_client.create_payload_index(
                collection_name=self.COLLECTION_NAME,
                field_name="chat_name",
                field_schema=TextIndexParams(
                    type=TextIndexType.TEXT,
                    tokenizer=TokenizerType.MULTILINGUAL,
                    min_token_len=2,
                    max_token_len=30,
                    lowercase=True
                )
            )
            logger.info("Created text index on 'chat_name' field")
        except Exception as e:
            logger.debug(f"Could not create chat_name index (may exist): {e}")
        
        try:
            # Create text index on 'message' field for full-text message search
            self.qdrant_client.create_payload_index(
                collection_name=self.COLLECTION_NAME,
                field_name="message",
                field_schema=TextIndexParams(
                    type=TextIndexType.TEXT,
                    tokenizer=TokenizerType.MULTILINGUAL,
                    min_token_len=2,
                    max_token_len=40,
                    lowercase=True
                )
            )
            logger.info("Created text index on 'message' field")
        except Exception as e:
            logger.debug(f"Could not create message index (may exist): {e}")
    
    def _ensure_payload_indexes(self):
        """Create payload indexes for efficient filtering on non-text fields.
        
        Indexes timestamp (integer range queries), source_type (keyword filter),
        is_group (boolean filter), and source_id (deduplication lookups).
        """
        index_configs = [
            ("timestamp", PayloadSchemaType.INTEGER, "timestamp integer index"),
            ("source_type", PayloadSchemaType.KEYWORD, "source_type keyword index"),
            ("is_group", PayloadSchemaType.BOOL, "is_group bool index"),
            ("source_id", PayloadSchemaType.KEYWORD, "source_id keyword index"),
        ]
        
        for field_name, schema_type, description in index_configs:
            try:
                self.qdrant_client.create_payload_index(
                    collection_name=self.COLLECTION_NAME,
                    field_name=field_name,
                    field_schema=schema_type
                )
                logger.info(f"Created {description}")
            except Exception as e:
                logger.debug(f"Could not create {description} (may exist): {e}")
    
    def _message_exists(self, source_id: str) -> bool:
        """Check if a message with the given source_id already exists in Qdrant.
        
        Used for deduplication to prevent duplicate messages from webhook retries.
        Requires a keyword index on the source_id field for efficient lookups.
        
        Args:
            source_id: The source identifier (format: '{chat_id}:{timestamp}')
            
        Returns:
            True if a document with this source_id already exists
        """
        try:
            results, _ = self.qdrant_client.scroll(
                collection_name=self.COLLECTION_NAME,
                scroll_filter=Filter(must=[
                    FieldCondition(key="source_id", match=MatchValue(value=source_id))
                ]),
                limit=1,
                with_payload=False,
                with_vectors=False
            )
            return len(results) > 0
        except Exception as e:
            logger.debug(f"Dedup check failed (proceeding with insert): {e}")
            return False
    
    def add_message(
        self,
        thread_id: str,
        chat_id: str,
        chat_name: str,
        is_group: bool,
        sender: str,
        message: str,
        timestamp: str,
        has_media: bool = False,
        media_type: Optional[str] = None,
        media_url: Optional[str] = None
    ) -> bool:
        """Add a WhatsApp message to the vector store.
        
        Uses WhatsAppMessageDocument model for consistent schema across all entries.
        Also incrementally updates the Redis-cached chat and sender lists.
        
        Args:
            thread_id: The conversation thread ID
            chat_id: The WhatsApp chat ID
            chat_name: Name of the chat/group
            is_group: Whether this is a group chat
            sender: The message sender
            message: The message content
            timestamp: The message timestamp (Unix timestamp as string)
            has_media: Whether message has media attachment
            media_type: MIME type of media if present
            media_url: URL to media file if present
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from models import WhatsAppMessageDocument
            
            # Deduplication: skip if message already exists
            source_id = f"{chat_id}:{timestamp}"
            if self._message_exists(source_id):
                logger.debug(f"Skipping duplicate message: {source_id}")
                return True  # Not an error, just already stored
            
            # Create document using standardized model
            doc = WhatsAppMessageDocument.from_webhook_payload(
                thread_id=thread_id,
                chat_id=chat_id,
                chat_name=chat_name,
                is_group=is_group,
                sender=sender,
                message=message,
                timestamp=timestamp,
                has_media=has_media,
                media_type=media_type,
                media_url=media_url
            )
            
            # Convert to LlamaIndex TextNode with standardized schema
            node = doc.to_llama_index_node()
            
            # Insert into index
            self.index.insert_nodes([node])
            
            # Incrementally update cached chat/sender sets in Redis
            self._update_cached_lists(chat_name=chat_name, sender=sender)
            
            logger.debug(f"Added message to RAG: {doc.get_embedding_text()[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add message to vector store: {e}")
            return False
    
    def add_node(self, node: TextNode) -> bool:
        """Add a pre-constructed TextNode to the vector store.
        
        Args:
            node: LlamaIndex TextNode to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.index.insert_nodes([node])
            logger.debug(f"Added node to RAG: {node.text[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to add node to vector store: {e}")
            return False
    
    def add_nodes(self, nodes: List[TextNode]) -> int:
        """Add multiple nodes to the vector store in batch.
        
        Args:
            nodes: List of TextNode instances
            
        Returns:
            Number of successfully added nodes
        """
        if not nodes:
            return 0
        
        try:
            self.index.insert_nodes(nodes)
            logger.info(f"Added {len(nodes)} nodes to RAG vector store")
            return len(nodes)
        except Exception as e:
            logger.error(f"Failed to add batch nodes to vector store: {e}")
            return 0
    
    def add_document(self, document: "BaseRAGDocument") -> bool:
        """Add a RAG document to the vector store with standardized schema.
        
        This is the preferred method for adding documents as it ensures
        consistent schema across all source types (WhatsApp, files, calls).
        
        Args:
            document: Any BaseRAGDocument subclass instance
                     (WhatsAppMessageDocument, FileDocument, CallRecordingDocument)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from models.base import BaseRAGDocument
            
            if not isinstance(document, BaseRAGDocument):
                raise TypeError(f"Expected BaseRAGDocument, got {type(document)}")
            
            node = document.to_llama_index_node()
            self.index.insert_nodes([node])
            
            logger.debug(f"Added {document.metadata.source_type.value} document to RAG: {document.get_embedding_text()[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document to vector store: {e}")
            return False
    
    def add_documents(self, documents: List["BaseRAGDocument"]) -> int:
        """Add multiple RAG documents to the vector store in batch.
        
        Ensures consistent schema across all source types.
        
        Args:
            documents: List of BaseRAGDocument subclass instances
            
        Returns:
            Number of successfully added documents
        """
        if not documents:
            return 0
        
        try:
            from models.base import BaseRAGDocument
            
            nodes = []
            for doc in documents:
                if not isinstance(doc, BaseRAGDocument):
                    logger.warning(f"Skipping invalid document type: {type(doc)}")
                    continue
                nodes.append(doc.to_llama_index_node())
            
            if nodes:
                self.index.insert_nodes(nodes)
                logger.info(f"Added {len(nodes)} documents to RAG vector store")
            
            return len(nodes)
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")
            return 0
    
    def _fulltext_search(
        self,
        query: str,
        k: int = 10,
        filter_chat_name: Optional[str] = None,
        filter_days: Optional[int] = None
    ) -> List[NodeWithScore]:
        """Perform full-text search on metadata fields (sender, chat_name, message).
        
        Args:
            query: Text to search for
            k: Max results to return
            filter_chat_name: Optional chat filter
            filter_days: Optional time filter
            
        Returns:
            List of matching NodeWithScore objects
        """
        try:
            # Build filter conditions
            must_conditions = []
            
            if filter_chat_name:
                must_conditions.append(
                    FieldCondition(key="chat_name", match=MatchValue(value=filter_chat_name))
                )
            
            if filter_days is not None and filter_days > 0:
                min_timestamp = int(datetime.now().timestamp()) - (filter_days * 24 * 60 * 60)
                must_conditions.append(
                    FieldCondition(key="timestamp", range=Range(gte=min_timestamp))
                )
            
            # Full-text search conditions - at least one must match
            # Use Filter with 'should' to match any of: sender, chat_name, or message
            qdrant_filter = Filter(
                must=must_conditions if must_conditions else None,
                should=[
                    FieldCondition(key="sender", match=MatchText(text=query)),
                    FieldCondition(key="chat_name", match=MatchText(text=query)),
                    FieldCondition(key="message", match=MatchText(text=query))
                ]  # type: ignore[arg-type]
            )
            
            # Scroll through matching documents (no vector search)
            results, _ = self.qdrant_client.scroll(
                collection_name=self.COLLECTION_NAME,
                scroll_filter=qdrant_filter,
                limit=k,
                with_payload=True,
                with_vectors=False
            )
            
            # Convert to NodeWithScore format
            nodes = []
            for record in results:
                payload = record.payload or {}
                
                # Reconstruct text
                chat_name = payload.get("chat_name", "Unknown")
                sender = payload.get("sender", "Unknown")
                message = payload.get("message", "")
                timestamp = payload.get("timestamp", 0)
                
                if message:
                    formatted_time = format_timestamp(str(timestamp))
                    text = f"[{formatted_time}] {sender} in {chat_name}: {message}"
                    
                    node = TextNode(
                        text=text,
                        metadata={k: v for k, v in payload.items() if not k.startswith("_")},
                        id_=str(record.id)
                    )
                    # Full-text matches get a high score to prioritize them
                    nodes.append(NodeWithScore(node=node, score=1.0))
            
            return nodes
            
        except Exception as e:
            logger.debug(f"Full-text search failed (indexes may not exist): {e}")
            return []
    
    @staticmethod
    def _reciprocal_rank_fusion(
        vector_results: List[NodeWithScore],
        fulltext_results: List[NodeWithScore],
        k: int = 10,
        rrf_k: int = 60
    ) -> List[NodeWithScore]:
        """Merge vector and full-text search results using Reciprocal Rank Fusion.
        
        RRF combines results from multiple retrieval methods by scoring each
        result based on its rank position in each list, then sorting by combined
        score. This avoids the need to normalize incompatible score scales
        (cosine similarity vs full-text match).
        
        Formula: RRF_score(d) = sum(1 / (rrf_k + rank_i(d))) for each list i
        
        Args:
            vector_results: Results from vector similarity search
            fulltext_results: Results from full-text metadata search
            k: Maximum number of results to return
            rrf_k: Smoothing constant (default 60, standard in literature)
            
        Returns:
            Merged and re-ranked list of NodeWithScore
        """
        # Map node_id -> (rrf_score, best_node_with_score)
        scores: Dict[str, float] = {}
        node_map: Dict[str, NodeWithScore] = {}
        
        for rank, result in enumerate(vector_results):
            node_id = result.node.id_ if result.node else None
            if not node_id:
                continue
            scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (rrf_k + rank + 1)
            node_map[node_id] = result
        
        for rank, result in enumerate(fulltext_results):
            node_id = result.node.id_ if result.node else None
            if not node_id:
                continue
            scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (rrf_k + rank + 1)
            if node_id not in node_map:
                node_map[node_id] = result
        
        # Sort by RRF score descending, return top-k with RRF score
        sorted_ids = sorted(scores.keys(), key=lambda nid: scores[nid], reverse=True)[:k]
        
        merged = []
        for node_id in sorted_ids:
            original = node_map[node_id]
            merged.append(NodeWithScore(node=original.node, score=scores[node_id]))
        
        return merged
    
    def _metadata_search(
        self,
        k: int = 20,
        filter_chat_name: Optional[str] = None,
        filter_sender: Optional[str] = None,
        filter_days: Optional[int] = None
    ) -> List[NodeWithScore]:
        """Search by metadata filters only, without vector similarity.
        
        Useful for queries like "show me all messages from David in Family Group
        last week" where only metadata filters are needed. Skips the embedding
        API call entirely.
        
        Args:
            k: Max results to return
            filter_chat_name: Filter by chat/group name
            filter_sender: Filter by sender name
            filter_days: Filter by number of days
            
        Returns:
            List of NodeWithScore objects (score=1.0 for all, sorted by timestamp)
        """
        try:
            must_conditions = []
            
            if filter_chat_name:
                must_conditions.append(
                    FieldCondition(key="chat_name", match=MatchValue(value=filter_chat_name))
                )
            
            if filter_sender:
                must_conditions.append(
                    FieldCondition(key="sender", match=MatchValue(value=filter_sender))
                )
            
            if filter_days is not None and filter_days > 0:
                min_timestamp = int(datetime.now().timestamp()) - (filter_days * 24 * 60 * 60)
                must_conditions.append(
                    FieldCondition(key="timestamp", range=Range(gte=min_timestamp))
                )
            
            if not must_conditions:
                logger.debug("Metadata search called with no filters, skipping")
                return []
            
            results, _ = self.qdrant_client.scroll(
                collection_name=self.COLLECTION_NAME,
                scroll_filter=Filter(must=must_conditions),
                limit=k,
                with_payload=True,
                with_vectors=False
            )
            
            nodes = []
            for record in results:
                payload = record.payload or {}
                chat_name = payload.get("chat_name", "Unknown")
                sender = payload.get("sender", "Unknown")
                message = payload.get("message", "")
                timestamp = payload.get("timestamp", 0)
                
                if message:
                    formatted_time = format_timestamp(str(timestamp))
                    text = f"[{formatted_time}] {sender} in {chat_name}: {message}"
                    
                    node = TextNode(
                        text=text,
                        metadata={k: v for k, v in payload.items() if not k.startswith("_")},
                        id_=str(record.id)
                    )
                    nodes.append(NodeWithScore(node=node, score=1.0))
            
            logger.info(f"Metadata search returned {len(nodes)} results")
            return nodes
            
        except Exception as e:
            logger.error(f"Metadata search failed: {e}")
            return []
    
    def search(
        self,
        query: str,
        k: int = 10,
        filter_chat_name: Optional[str] = None,
        filter_sender: Optional[str] = None,
        filter_days: Optional[int] = None,
        include_metadata_search: bool = True,
        metadata_only: bool = False
    ) -> List[NodeWithScore]:
        """Search for relevant messages using hybrid semantic + full-text search.
        
        Performs two searches and merges results:
        1. Vector similarity search (semantic)
        2. Full-text search on metadata (sender, chat_name, message)
        
        If metadata_only=True, skips the vector search entirely and returns
        results based purely on metadata filters. This saves an embedding API
        call for queries that are purely filter-based (e.g., "show me messages
        from David in Family Group last week").
        
        This ensures queries about people find results even when the semantic
        embedding doesn't capture the relationship (e.g., "what is Kobi's last name?").
        
        Args:
            query: The search query
            k: Number of results to return
            filter_chat_name: Optional filter by chat/group name
            filter_sender: Optional filter by sender name
            filter_days: Optional filter by number of days
            include_metadata_search: Include full-text search on metadata fields
            metadata_only: Skip vector search, use only metadata filters
            
        Returns:
            List of NodeWithScore objects with metadata
        """
        try:
            # Metadata-only search: skip vector search entirely
            if metadata_only:
                return self._metadata_search(
                    k=k,
                    filter_chat_name=filter_chat_name,
                    filter_sender=filter_sender,
                    filter_days=filter_days
                )
            
            # Build Qdrant filter conditions
            must_conditions = []
            
            if filter_chat_name:
                must_conditions.append(
                    FieldCondition(
                        key="chat_name",
                        match=MatchValue(value=filter_chat_name)
                    )
                )
            
            if filter_sender:
                must_conditions.append(
                    FieldCondition(
                        key="sender",
                        match=MatchValue(value=filter_sender)
                    )
                )
            
            if filter_days is not None and filter_days > 0:
                min_timestamp = int(datetime.now().timestamp()) - (filter_days * 24 * 60 * 60)
                must_conditions.append(
                    FieldCondition(
                        key="timestamp",
                        range=Range(gte=min_timestamp)
                    )
                )
            
            qdrant_filters = Filter(must=must_conditions) if must_conditions else None
            
            # Use direct Qdrant search to avoid LlamaIndex TextNode validation issues
            # with documents that have None text values
            query_embedding = Settings.embed_model.get_query_embedding(query)
            
            search_results = self.qdrant_client.query_points(
                collection_name=self.COLLECTION_NAME,
                query=query_embedding,
                query_filter=qdrant_filters,
                limit=k,
                with_payload=True
            ).points
            
            # Convert Qdrant results to NodeWithScore, filtering out invalid entries
            valid_results = []
            for result in search_results:
                payload = result.payload or {}
                
                # Try to get text from _node_content first (LlamaIndex storage format),
                # then fall back to reconstructing from payload fields
                text = None
                node_content = payload.get("_node_content")
                if node_content and isinstance(node_content, str):
                    try:
                        import json
                        content_dict = json.loads(node_content)
                        text = content_dict.get("text")
                    except (json.JSONDecodeError, TypeError):
                        pass
                
                # Fallback: reconstruct text from payload metadata
                if not text:
                    chat_name = payload.get("chat_name", "Unknown")
                    sender = payload.get("sender", "Unknown")
                    message = payload.get("message", "")
                    timestamp = payload.get("timestamp", 0)
                    if message:
                        formatted_time = format_timestamp(str(timestamp))
                        text = f"[{formatted_time}] {sender} in {chat_name}: {message}"
                
                # Skip if we still don't have valid text
                if not text:
                    logger.warning(f"Skipping result with no valid text: point_id={result.id}")
                    continue
                
                # Create TextNode with valid text
                node = TextNode(
                    text=text,
                    metadata={k: v for k, v in payload.items() if not k.startswith("_")},
                    id_=str(result.id)
                )
                
                valid_results.append(NodeWithScore(node=node, score=result.score))
            
            # Apply minimum similarity score threshold to vector results
            pre_filter_count = len(valid_results)
            valid_results = [
                r for r in valid_results
                if r.score is not None and r.score >= self.MINIMUM_SIMILARITY_SCORE
            ]
            if pre_filter_count > len(valid_results):
                logger.debug(
                    f"Score threshold filtered {pre_filter_count - len(valid_results)} "
                    f"results below {self.MINIMUM_SIMILARITY_SCORE}"
                )
            
            # Hybrid search: also do full-text search on metadata and merge results
            if include_metadata_search and not filter_sender:
                fulltext_results = self._fulltext_search(
                    query=query,
                    k=k,
                    filter_chat_name=filter_chat_name,
                    filter_days=filter_days
                )
                
                # Merge using Reciprocal Rank Fusion for fair ranking
                if fulltext_results:
                    valid_results = self._reciprocal_rank_fusion(
                        vector_results=valid_results,
                        fulltext_results=fulltext_results,
                        k=k,
                        rrf_k=self.RRF_K
                    )
                    logger.info(
                        f"RRF merged {len(fulltext_results)} fulltext + vector results "
                        f"→ {len(valid_results)} final"
                    )
            
            logger.info(f"RAG search for '{query[:50]}...' returned {len(valid_results)} valid results")
            return valid_results
            
        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            return []
    
    def query(
        self,
        question: str,
        k: int = 10,
        filter_chat_name: Optional[str] = None,
        filter_sender: Optional[str] = None,
        filter_days: Optional[int] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Query the RAG system with a natural language question.
        
        Uses LlamaIndex query engine with retrieved context.
        
        Args:
            question: Natural language question
            k: Number of context documents to retrieve
            filter_chat_name: Optional filter by chat/group name
            filter_sender: Optional filter by sender name
            filter_days: Optional filter by number of days
            conversation_history: Optional previous conversation messages
            
        Returns:
            AI-generated answer based on retrieved context
        """
        try:
            # Ensure LLM is configured (lazy init for faster startup)
            self._ensure_llm_configured()
            
            # Get relevant documents
            results = self.search(
                query=question,
                k=k,
                filter_chat_name=filter_chat_name,
                filter_sender=filter_sender,
                filter_days=filter_days
            )
            
            # Build context from results with token-aware truncation
            context_parts = []
            total_tokens = 0
            
            try:
                import tiktoken
                enc = tiktoken.encoding_for_model("gpt-4o")
            except (ImportError, KeyError):
                enc = None
            
            for result in results:
                # Access text safely - our search method ensures nodes have valid text
                node_text = getattr(result.node, 'text', None) or getattr(result.node, 'get_content', lambda: '')()
                if not node_text:
                    continue
                
                # Estimate token count
                if enc:
                    text_tokens = len(enc.encode(node_text))
                else:
                    text_tokens = len(node_text) // 4  # Rough fallback: ~4 chars/token
                
                if total_tokens + text_tokens > self.MAX_CONTEXT_TOKENS:
                    logger.debug(
                        f"Context token limit reached ({total_tokens}/{self.MAX_CONTEXT_TOKENS}), "
                        f"using {len(context_parts)} of {len(results)} results"
                    )
                    break
                
                context_parts.append(node_text)
                total_tokens += text_tokens
            
            context = "\n".join(context_parts) if context_parts else "[No messages found in the archive]"
            
            # Get current date/time
            tz = ZoneInfo("Asia/Jerusalem")
            now = datetime.now(tz)
            current_datetime = now.strftime("%A, %B %d, %Y at %H:%M")
            hebrew_day = {
                "Monday": "יום שני",
                "Tuesday": "יום שלישי",
                "Wednesday": "יום רביעי",
                "Thursday": "יום חמישי",
                "Friday": "יום שישי",
                "Saturday": "שבת",
                "Sunday": "יום ראשון"
            }.get(now.strftime("%A"), now.strftime("%A"))
            hebrew_date = f"{hebrew_day}, {now.day}/{now.month}/{now.year} בשעה {now.strftime('%H:%M')}"
            
            # Build prompt with conversation history
            history_text = ""
            if conversation_history:
                history_parts = []
                for msg in conversation_history:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    history_parts.append(f"{role.capitalize()}: {content}")
                history_text = "\n\nConversation History:\n" + "\n".join(history_parts)
            
            # Create query prompt with structured instructions
            prompt = f"""You are a helpful AI assistant for a WhatsApp message archive search system.
You have access to retrieved messages from the archive below.

Current Date/Time: {current_datetime}
תאריך ושעה נוכחיים: {hebrew_date}

=== Retrieved Messages ({len(context_parts)} results) ===
{context}
=== End of Retrieved Messages ==={history_text}

User Question: {question}

Instructions:
1. ANALYZE the retrieved messages above to find information relevant to the question.
2. CITE specific messages when possible (mention who said what and when).
3. If multiple messages are relevant, SYNTHESIZE them into a coherent answer.
4. If the retrieved messages don't contain enough information to answer confidently, say so clearly — do NOT fabricate information.
5. If the question is general (like "what day is today?"), answer directly without referencing the archive.
6. Answer in the SAME LANGUAGE as the question.
7. Be concise but thorough. Prefer specific facts over vague summaries."""

            # Use LlamaIndex LLM for response
            response = Settings.llm.complete(prompt)
            answer = str(response)
            
            logger.info(f"RAG query answered: {question[:50]}... (context_docs={len(results)})")
            return answer
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store.
        
        Returns:
            Dictionary with collection stats
        """
        try:
            collection_info = self.qdrant_client.get_collection(self.COLLECTION_NAME)
            return {
                "total_documents": collection_info.points_count,
                "qdrant_server": f"{self.qdrant_host}:{self.qdrant_port}",
                "collection_name": self.COLLECTION_NAME,
                "dashboard_url": f"http://{self.qdrant_host}:{self.qdrant_port}/dashboard"
            }
        except Exception as e:
            logger.error(f"Failed to get RAG stats: {e}")
            return {"error": str(e)}
    
    # =========================================================================
    # Cached chat/sender list methods (Redis-backed)
    # =========================================================================
    
    REDIS_CHAT_SET_KEY = "rag:chat_names"
    REDIS_SENDER_SET_KEY = "rag:sender_names"
    REDIS_LISTS_TTL = 3600  # 1 hour TTL for the cached sets
    
    def _update_cached_lists(self, chat_name: Optional[str] = None, sender: Optional[str] = None) -> None:
        """Incrementally add a chat name and/or sender to the Redis cached sets.
        
        Called on every add_message() so the cache stays up-to-date without
        needing a full Qdrant collection scan.
        
        Args:
            chat_name: Chat/group name to add (if not None)
            sender: Sender name to add (if not None)
        """
        try:
            redis = get_redis_client()
            if chat_name:
                redis.sadd(self.REDIS_CHAT_SET_KEY, chat_name)
                redis.expire(self.REDIS_CHAT_SET_KEY, self.REDIS_LISTS_TTL)
            if sender:
                redis.sadd(self.REDIS_SENDER_SET_KEY, sender)
                redis.expire(self.REDIS_SENDER_SET_KEY, self.REDIS_LISTS_TTL)
        except Exception as e:
            logger.debug(f"Failed to update cached lists in Redis: {e}")
    
    def _rebuild_cached_list(self, field_name: str, redis_key: str) -> List[str]:
        """Rebuild a cached list by scanning the full Qdrant collection.
        
        This is the fallback when the Redis cache is empty (first run or after
        Redis restart). Results are stored in a Redis SET for fast subsequent access.
        
        Args:
            field_name: Qdrant payload field to extract unique values from
            redis_key: Redis SET key to store the results
            
        Returns:
            Sorted list of unique values
        """
        values = set()
        offset = None
        
        while True:
            records, next_offset = self.qdrant_client.scroll(
                collection_name=self.COLLECTION_NAME,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            for record in records:
                payload = record.payload or {}
                value = payload.get(field_name)
                if value:
                    values.add(value)
            
            if next_offset is None:
                break
            offset = next_offset
        
        # Store in Redis SET for fast access
        try:
            redis = get_redis_client()
            if values:
                redis.delete(redis_key)
                redis.sadd(redis_key, *values)
                redis.expire(redis_key, self.REDIS_LISTS_TTL)
            logger.info(f"Rebuilt cached {field_name} list: {len(values)} unique values")
        except Exception as e:
            logger.warning(f"Failed to cache {field_name} list in Redis: {e}")
        
        return sorted(list(values))
    
    def get_chat_list(self) -> List[str]:
        """Get all unique chat names, using Redis cache when available.
        
        First checks Redis SET for cached values. If empty, falls back to
        a full Qdrant collection scan and caches the result.
        
        Returns:
            List of unique chat names sorted alphabetically
        """
        try:
            redis = get_redis_client()
            cached: set = redis.smembers(self.REDIS_CHAT_SET_KEY)  # type: ignore[assignment]
            if cached:
                return sorted(list(cached))
        except Exception as e:
            logger.debug(f"Redis cache miss for chat list: {e}")
        
        # Cache miss — rebuild from Qdrant
        try:
            return self._rebuild_cached_list("chat_name", self.REDIS_CHAT_SET_KEY)
        except Exception as e:
            logger.error(f"Failed to get chat list: {e}")
            return []
    
    def get_sender_list(self) -> List[str]:
        """Get all unique sender names, using Redis cache when available.
        
        First checks Redis SET for cached values. If empty, falls back to
        a full Qdrant collection scan and caches the result.
        
        Returns:
            List of unique sender names sorted alphabetically
        """
        try:
            redis = get_redis_client()
            cached: set = redis.smembers(self.REDIS_SENDER_SET_KEY)  # type: ignore[assignment]
            if cached:
                return sorted(list(cached))
        except Exception as e:
            logger.debug(f"Redis cache miss for sender list: {e}")
        
        # Cache miss — rebuild from Qdrant
        try:
            return self._rebuild_cached_list("sender", self.REDIS_SENDER_SET_KEY)
        except Exception as e:
            logger.error(f"Failed to get sender list: {e}")
            return []
    
    def invalidate_list_caches(self) -> None:
        """Invalidate the Redis-cached chat and sender lists.
        
        Forces a full rebuild on next access. Useful after bulk operations
        or data cleanup.
        """
        try:
            redis = get_redis_client()
            redis.delete(self.REDIS_CHAT_SET_KEY, self.REDIS_SENDER_SET_KEY)
            logger.info("Invalidated cached chat/sender lists")
        except Exception as e:
            logger.warning(f"Failed to invalidate list caches: {e}")


# Create singleton instance getter
_rag_instance: Optional[LlamaIndexRAG] = None


def get_rag() -> LlamaIndexRAG:
    """Get the shared RAG singleton instance.
    
    Returns:
        The LlamaIndexRAG singleton instance
    """
    global _rag_instance
    if _rag_instance is None:
        print("Initializing LlamaIndex RAG instance...", flush=True)
        _rag_instance = LlamaIndexRAG()
        print("✅ LlamaIndex RAG instance initialized", flush=True)
    return _rag_instance
