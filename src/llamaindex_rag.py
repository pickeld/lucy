"""LlamaIndex RAG (Retrieval Augmented Generation) for multi-source knowledge base.

Uses Qdrant as vector store and OpenAI text-embedding-3-large for semantic search.
Configured with 1024 dimensions for optimal Hebrew + English multilingual support.
Uses LlamaIndex CondensePlusContextChatEngine for multi-turn conversations
with automatic query reformulation and Redis-backed chat memory.

Supports data from multiple channel plugins (WhatsApp, Telegram, Email,
Paperless-NG, etc.) via the plugin architecture.

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
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from llama_index.storage.chat_store.redis import RedisChatStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Direction,
    Distance,
    FieldCondition,
    Filter,
    MatchText,
    MatchValue,
    OrderBy,
    PayloadSchemaType,
    Range,
    TextIndexParams,
    TextIndexType,
    TokenizerType,
    VectorParams,
)

from config import settings
from utils.logger import logger
from utils.redis_conn import get_redis_client


def format_timestamp(timestamp: str, timezone: str = "") -> str:
    """Convert Unix timestamp to human-readable format.
    
    Args:
        timestamp: Unix timestamp as string or int
        timezone: Timezone for display (reads from settings if empty)
        
    Returns:
        Formatted datetime string (e.g., "31/12/2024 10:30")
    """
    try:
        if not timezone:
            timezone = settings.get("timezone", "Asia/Jerusalem")
        ts = int(timestamp)
        tz = ZoneInfo(timezone)
        dt = datetime.fromtimestamp(ts, tz=tz)
        return dt.strftime("%d/%m/%Y %H:%M")
    except (ValueError, TypeError, KeyError):
        return str(timestamp)


class ArchiveRetriever(BaseRetriever):
    """Custom retriever that wraps existing hybrid search with metadata filters.
    
    Delegates to LlamaIndexRAG.search() which handles:
    - Vector similarity search (Qdrant)
    - Full-text search on metadata (sender, chat_name, message)
    - Reciprocal Rank Fusion for merging results
    - Minimum similarity score thresholding
    
    Source-agnostic: works with data from any channel plugin
    (WhatsApp, Telegram, Email, Paperless-NG, etc.).
    """
    
    def __init__(
        self,
        rag: "LlamaIndexRAG",
        k: int = 10,
        filter_chat_name: Optional[str] = None,
        filter_sender: Optional[str] = None,
        filter_days: Optional[int] = None,
        **kwargs: Any,
    ):
        """Initialize the archive retriever.
        
        Args:
            rag: The LlamaIndexRAG instance to delegate search to
            k: Number of results to retrieve
            filter_chat_name: Optional filter by chat/group name
            filter_sender: Optional filter by sender name
            filter_days: Optional filter by recency in days
        """
        super().__init__(**kwargs)
        self._rag = rag
        self._k = k
        self._filter_chat_name = filter_chat_name
        self._filter_sender = filter_sender
        self._filter_days = filter_days
    
    # Number of recent messages to always include alongside semantic results.
    # Ensures the LLM has temporal awareness of the latest messages.
    RECENCY_SUPPLEMENT_COUNT = 5
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve relevant messages/documents using hybrid search.
        
        Runs semantic + full-text hybrid search first. Then:
        1. Expands context by fetching surrounding messages from the same chats
        2. Supplements with the most recent messages for temporal awareness
        
        If search returns no results, falls back to timestamp-ordered
        retrieval (language-agnostic recency fallback).
        
        Always returns at least one node so the chat engine's synthesizer
        can generate a proper response (it returns "Empty Response" on empty input).
        
        Args:
            query_bundle: The query bundle from the chat engine
            
        Returns:
            List of NodeWithScore from hybrid search (never empty)
        """
        results = self._rag.search(
            query=query_bundle.query_str,
            k=self._k,
            filter_chat_name=self._filter_chat_name,
            filter_sender=self._filter_sender,
            filter_days=self._filter_days,
        )
        
        # Context expansion: fetch surrounding messages from the same chats
        # so that replies and nearby messages are included as context.
        if results:
            results = self._rag.expand_context(results, max_total=self._k * 2)
        
        # Always supplement with recent messages so the LLM has temporal
        # awareness (knows what the actual latest messages are). This
        # ensures queries like "what's the last message?" get the correct
        # answer even when semantic search returns older matches.
        recent = self._rag.recency_search(
            k=self.RECENCY_SUPPLEMENT_COUNT,
            filter_chat_name=self._filter_chat_name,
            filter_sender=self._filter_sender,
            filter_days=self._filter_days,
        )
        if recent:
            if results:
                # Merge: deduplicate by node ID, keeping originals first
                existing_ids = {nws.node.id_ for nws in results if nws.node}
                for nws in recent:
                    if nws.node and nws.node.id_ not in existing_ids:
                        existing_ids.add(nws.node.id_)
                        results.append(nws)
            else:
                # No semantic results — use recent messages as primary context
                results = recent
                logger.info(
                    f"Semantic search empty, using {len(results)} recent messages"
                )
        
        # Ensure at least one node so the synthesizer doesn't return "Empty Response"
        if not results:
            placeholder = TextNode(
                text="[No relevant messages found in the archive for this query]",
                metadata={"source": "system", "note": "no_results"},
            )
            results = [NodeWithScore(node=placeholder, score=0.0)]
        
        return results

# Backward compat alias
WhatsAppRetriever = ArchiveRetriever


class LlamaIndexRAG:
    """LlamaIndex-based RAG for multi-source knowledge base search and retrieval.
    
    Uses Qdrant server as vector store and OpenAI text-embedding-3-large
    with 1024 dimensions for optimal Hebrew + English multilingual support.
    Uses CondensePlusContextChatEngine for multi-turn conversations
    with automatic query reformulation and Redis-backed chat memory.
    
    Source-agnostic: stores and retrieves data from any channel plugin
    (WhatsApp, Telegram, Email, Paperless-NG, etc.) via BaseRAGDocument.
    
    Connects to Qdrant server at QDRANT_HOST:QDRANT_PORT (default: localhost:6333).
    Dashboard available at: http://localhost:6333/dashboard
    """
    
    _instance = None
    _index = None
    _qdrant_client = None
    _vector_store = None
    _chat_store = None
    
    COLLECTION_NAME = settings.rag_collection_name
    VECTOR_SIZE = int(settings.get("rag_vector_size", "1024"))
    MINIMUM_SIMILARITY_SCORE = float(settings.rag_min_score)
    MAX_CONTEXT_TOKENS = int(settings.rag_max_context_tokens)
    RRF_K = int(settings.get("rag_rrf_k", "60"))
    
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
        self.qdrant_host = settings.qdrant_host
        self.qdrant_port = int(settings.qdrant_port)
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
        """Configure embedding model and cost tracking callback (fast, required at startup).
        
        Uses text-embedding-3-large with dimensions=1024 for best multilingual
        (Hebrew + English) support. The 'large' model significantly outperforms
        'small' on non-English languages. Using 1024 dimensions (reduced from
        native 3072) provides an excellent quality/cost tradeoff — OpenAI's
        Matryoshka representation learning ensures minimal quality loss.
        
        Also sets up the LlamaIndex CallbackManager with a CostTrackingHandler
        to automatically track token usage and costs for all LLM and embedding calls.
        
        Reads model name from settings.embedding_model for configurability.
        """
        model_name = settings.get("embedding_model", "text-embedding-3-large")
        logger.debug(f"Setting up OpenAI embedding model: {model_name}...")
        Settings.embed_model = OpenAIEmbedding(
            api_key=settings.openai_api_key,
            model=model_name,
            dimensions=self.VECTOR_SIZE,
        )
        logger.debug(f"OpenAI embedding model configured ({model_name}, dims={self.VECTOR_SIZE})")
        
        # Set up cost tracking callback manager
        try:
            from cost_callbacks import create_cost_callback_manager
            
            llm_provider = settings.get("llm_provider", "openai").lower()
            llm_model = (
                settings.get("gemini_model", "gemini-pro")
                if llm_provider == "gemini"
                else settings.get("openai_model", "gpt-4o")
            )
            Settings.callback_manager = create_cost_callback_manager(
                llm_provider=llm_provider,
                llm_model=llm_model,
                embed_provider="openai",
                embed_model=model_name,
            )
            logger.info("Cost tracking callback manager configured")
        except Exception as e:
            logger.warning(f"Cost tracking setup failed (non-fatal): {e}")
    
    def _ensure_llm_configured(self):
        """Lazily configure LLM on first use (Gemini import is slow)."""
        if self._llm_configured:
            return
        
        llm_provider = settings.llm_provider.lower()
        logger.info(f"Configuring LLM provider: {llm_provider} (lazy init)...")
        
        actual_model = ""
        if llm_provider == 'gemini':
            try:
                from llama_index.llms.gemini import Gemini
                actual_model = settings.gemini_model
                Settings.llm = Gemini(
                    api_key=settings.google_api_key,
                    model=actual_model,
                    temperature=0.3
                )
                logger.info("Gemini LLM configured")
            except ImportError:
                logger.warning("Gemini LLM not available, falling back to OpenAI")
                llm_provider = "openai"
                actual_model = settings.openai_model
                Settings.llm = LlamaIndexOpenAI(
                    api_key=settings.openai_api_key,
                    model=actual_model,
                    temperature=0.3
                )
        else:
            actual_model = settings.openai_model
            Settings.llm = LlamaIndexOpenAI(
                api_key=settings.openai_api_key,
                model=actual_model,
                temperature=0.3
            )
            logger.info("OpenAI LLM configured")
        
        # Update cost tracking handler with actual provider/model
        try:
            from cost_callbacks import get_cost_handler
            handler = get_cost_handler()
            if handler:
                handler.llm_provider = llm_provider
                handler.llm_model = actual_model
                logger.debug(f"Cost handler updated: LLM={llm_provider}:{actual_model}")
        except Exception:
            pass  # Non-fatal
        
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
        
        Indexes timestamp (integer range queries), source/content_type (keyword filters),
        is_group (boolean filter), and source_id (deduplication lookups).
        """
        index_configs = [
            ("timestamp", PayloadSchemaType.INTEGER, "timestamp integer index"),
            ("source", PayloadSchemaType.KEYWORD, "source keyword index"),
            ("content_type", PayloadSchemaType.KEYWORD, "content_type keyword index"),
            ("source_type", PayloadSchemaType.KEYWORD, "source_type keyword index (legacy)"),
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
    
    # =========================================================================
    # Conversation Chunking (sliding window)
    # =========================================================================
    
    CHUNK_BUFFER_KEY_PREFIX = "rag:chunk_buffer:"
    CHUNK_BUFFER_TTL = 120  # 2 minutes — flush buffer if no new messages
    CHUNK_MAX_MESSAGES = 5  # Flush when buffer reaches this many messages
    
    def _buffer_message_for_chunking(
        self,
        chat_id: str,
        chat_name: str,
        is_group: bool,
        sender: str,
        message: str,
        timestamp: str
    ) -> None:
        """Buffer a message for conversation chunking.
        
        Messages are buffered per chat in a Redis list. When the buffer reaches
        CHUNK_MAX_MESSAGES or the TTL expires, the buffer is flushed as a single
        conversation chunk that gets its own embedding in Qdrant.
        
        This gives isolated messages (like "yes", "me too") conversational context
        in the embedding, dramatically improving retrieval quality.
        
        Args:
            chat_id: WhatsApp chat ID (used as buffer key)
            chat_name: Chat/group display name
            is_group: Whether group chat
            sender: Message sender name
            message: Message text
            timestamp: Unix timestamp as string
        """
        try:
            redis = get_redis_client()
            buffer_key = f"{self.CHUNK_BUFFER_KEY_PREFIX}{chat_id}"
            
            # Store message as JSON in the Redis list
            msg_data = json.dumps({
                "sender": sender,
                "message": message,
                "timestamp": timestamp,
                "chat_name": chat_name,
                "is_group": is_group,
            })
            redis.rpush(buffer_key, msg_data)
            redis.expire(buffer_key, self.CHUNK_BUFFER_TTL)
            
            # Check if buffer is full → flush
            buffer_len = redis.llen(buffer_key)
            if buffer_len >= self.CHUNK_MAX_MESSAGES:
                self._flush_chunk_buffer(chat_id)
                
        except Exception as e:
            logger.debug(f"Chunk buffering failed (non-critical): {e}")
    
    def _flush_chunk_buffer(self, chat_id: str) -> bool:
        """Flush the message buffer for a chat as a conversation chunk.
        
        Reads all buffered messages, concatenates them into a single chunk text,
        and stores it as an additional point in Qdrant with source_type='conversation_chunk'.
        
        Args:
            chat_id: The chat ID whose buffer to flush
            
        Returns:
            True if a chunk was created, False otherwise
        """
        try:
            redis = get_redis_client()
            buffer_key = f"{self.CHUNK_BUFFER_KEY_PREFIX}{chat_id}"
            
            # Atomically get all messages and delete the buffer
            raw_messages = redis.lrange(buffer_key, 0, -1)
            redis.delete(buffer_key)
            
            if not raw_messages or len(raw_messages) < 2:
                return False  # Need at least 2 messages for a meaningful chunk
            
            # Parse buffered messages
            messages = []
            for raw in raw_messages:
                try:
                    messages.append(json.loads(raw))
                except (json.JSONDecodeError, TypeError):
                    continue
            
            if len(messages) < 2:
                return False
            
            # Build chunk text
            chat_name = messages[0].get("chat_name", "Unknown")
            is_group = messages[0].get("is_group", False)
            first_ts = messages[0].get("timestamp", "0")
            last_ts = messages[-1].get("timestamp", "0")
            
            chunk_lines = []
            for msg in messages:
                ts_formatted = format_timestamp(msg.get("timestamp", "0"))
                chunk_lines.append(f"[{ts_formatted}] {msg['sender']}: {msg['message']}")
            
            chunk_text = "\n".join(chunk_lines)
            
            # Create a TextNode for the chunk
            chunk_node = TextNode(
                text=chunk_text,
                metadata={
                    "source_type": "conversation_chunk",
                    "chat_id": chat_id,
                    "chat_name": chat_name,
                    "is_group": is_group,
                    "timestamp": int(last_ts) if last_ts.isdigit() else 0,
                    "first_timestamp": int(first_ts) if first_ts.isdigit() else 0,
                    "message_count": len(messages),
                    "senders": list({m["sender"] for m in messages}),
                    "source_id": f"chunk:{chat_id}:{first_ts}:{last_ts}",
                },
                id_=str(uuid.uuid4()),
            )
            
            self.index.insert_nodes([chunk_node])
            logger.info(
                f"Created conversation chunk: {chat_name} ({len(messages)} msgs, "
                f"{first_ts}→{last_ts})"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to flush chunk buffer for {chat_id}: {e}")
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
        
        Each message is stored as an individual point AND buffered for
        conversation chunking. When the buffer reaches CHUNK_MAX_MESSAGES,
        the buffered messages are also stored as a single conversation chunk
        with its own embedding, giving short messages conversational context.
        
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
            
            # Insert individual message into index
            self.index.insert_nodes([node])
            
            # Buffer for conversation chunking (creates context-rich chunks)
            self._buffer_message_for_chunking(
                chat_id=chat_id,
                chat_name=chat_name,
                is_group=is_group,
                sender=sender,
                message=message,
                timestamp=timestamp,
            )
            
            # Incrementally update cached chat/sender sets in Redis
            self._update_cached_lists(chat_name=chat_name, sender=sender)
            
            logger.debug(f"Added message to RAG: {doc.get_embedding_text()[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add message to vector store: {e}")
            return False
    
    # Safety limit for embedding: truncate text to this many chars before
    # sending to the embedding API.  Covers worst-case tokenisation
    # (base64/HTML ≈ 1 char/token).  8191 token limit → 7000 char safety.
    EMBEDDING_MAX_CHARS = 7_000

    def add_node(self, node: TextNode) -> bool:
        """Add a pre-constructed TextNode to the vector store.
        
        If the node text exceeds EMBEDDING_MAX_CHARS it is truncated to
        avoid hitting the embedding model's token limit (8191 for
        text-embedding-3-large).  A first attempt is made with the full
        text; on a 400 "maximum context length" error the text is
        truncated and retried once.
        
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
            error_str = str(e)
            # Detect embedding token-limit errors and retry with truncated text
            if "maximum context length" in error_str and len(node.text) > self.EMBEDDING_MAX_CHARS:
                logger.warning(
                    f"Node text too long for embedding ({len(node.text)} chars), "
                    f"truncating to {self.EMBEDDING_MAX_CHARS} chars and retrying"
                )
                try:
                    node.text = node.text[:self.EMBEDDING_MAX_CHARS]
                    self.index.insert_nodes([node])
                    logger.debug(f"Added truncated node to RAG: {node.text[:50]}...")
                    return True
                except Exception as retry_err:
                    logger.error(f"Failed to add truncated node: {retry_err}")
                    return False
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
            
            logger.debug(f"Added {document.metadata.source.value}/{document.metadata.content_type.value} document to RAG: {document.get_embedding_text()[:50]}...")
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
    
    @staticmethod
    def _extract_text_from_payload(payload: Dict[str, Any]) -> Optional[str]:
        """Extract display text from a Qdrant point payload.
        
        Handles both WhatsApp messages (which have a 'message' metadata field)
        and Paperless/other documents (which store text only in LlamaIndex's
        internal '_node_content' JSON blob).
        
        Priority:
        1. Reconstruct from 'message' metadata (WhatsApp-style)
        2. Extract from '_node_content' JSON (Paperless/generic documents)
        
        Args:
            payload: Qdrant point payload dict
            
        Returns:
            Extracted text string, or None if no text could be found
        """
        chat_name = payload.get("chat_name", "Unknown")
        sender = payload.get("sender", "Unknown")
        message = payload.get("message", "")
        timestamp = payload.get("timestamp", 0)
        source = payload.get("source", "")
        
        # WhatsApp messages have a 'message' field in metadata
        if message:
            formatted_time = format_timestamp(str(timestamp))
            return f"[{formatted_time}] {sender} in {chat_name}: {message}"
        
        # Paperless/generic documents: extract text from _node_content
        node_content = payload.get("_node_content")
        if node_content and isinstance(node_content, str):
            try:
                content_dict = json.loads(node_content)
                text = content_dict.get("text")
                if text:
                    # For documents, prefix with title/source info for context
                    if source == "paperless":
                        formatted_time = format_timestamp(str(timestamp))
                        # Truncate very long document text for display
                        display_text = text[:2000] if len(text) > 2000 else text
                        if sender:
                            return f"[{formatted_time}] {sender} in {chat_name}:\n{display_text}"
                        return f"[{formatted_time}] Document '{chat_name}':\n{display_text}"
                    return text
            except (json.JSONDecodeError, TypeError):
                pass
        
        return None
    
    # Field-aware full-text search scores: sender matches are most valuable
    # because users often ask "what did X say about Y?"
    FULLTEXT_SCORE_SENDER = float(settings.get("rag_fulltext_score_sender", "0.95"))
    FULLTEXT_SCORE_CHAT_NAME = float(settings.get("rag_fulltext_score_chat_name", "0.85"))
    FULLTEXT_SCORE_MESSAGE = float(settings.get("rag_fulltext_score_message", "0.75"))
    
    # Common Hebrew prefixes (prepositions, conjunctions, articles)
    # that are attached to words: ה (the), ב (in), ל (to), מ (from),
    # ש (that), כ (like), ו (and).  Stripping these helps match
    # inflected forms against stored text.
    _HEBREW_PREFIXES = "הבלמשכו"
    
    @staticmethod
    def _expand_hebrew_tokens(tokens: List[str]) -> List[str]:
        """Expand Hebrew tokens by stripping prefixes and verb patterns.
        
        Hebrew is a morphologically rich language where prefixes (ה, ב, ל, מ, ש, כ, ו)
        attach directly to words, and verb conjugations change the word form significantly.
        For example:
        - "התגרשתי" (I got divorced) → root "גרש" → also matches "גירושין" (divorce)
        - "שהתגרשתי" → strip ש prefix → "התגרשתי" → root "גרש"
        
        This method generates additional search tokens from Hebrew morphological
        variants to improve full-text search recall without requiring a full
        Hebrew NLP library.
        
        Args:
            tokens: Original query tokens
            
        Returns:
            Expanded list of tokens (originals + morphological variants)
        """
        import re as _re
        expanded: List[str] = []
        seen: set = set()
        
        for token in tokens:
            low = token.lower()
            if low in seen:
                continue
            seen.add(low)
            expanded.append(token)
            
            # Only expand Hebrew tokens (contain Hebrew characters)
            if not _re.search(r'[\u0590-\u05FF]', token):
                continue
            
            # Strip common Hebrew prefixes (one or two prefix letters)
            word = token
            for _ in range(2):  # Strip up to 2 prefix letters
                if len(word) > 3 and word[0] in LlamaIndexRAG._HEBREW_PREFIXES:
                    stripped = word[1:]
                    if stripped.lower() not in seen and len(stripped) >= 3:
                        seen.add(stripped.lower())
                        expanded.append(stripped)
                    word = stripped
                else:
                    break
            
            # Hebrew Hitpael verb pattern: הת + root (e.g., התגרשתי → גרש)
            # Strip הת prefix and common suffixes (תי, נו, תם, תן, ת, ה, ו, י)
            if len(token) >= 5 and token[:2] == "הת":
                base = token[2:]
                # Strip verb conjugation suffixes
                for suffix in ["תי", "נו", "תם", "תן", "ת", "ה", "ו", "י"]:
                    if len(base) > 3 and base.endswith(suffix):
                        root = base[:-len(suffix)]
                        if len(root) >= 2 and root.lower() not in seen:
                            seen.add(root.lower())
                            expanded.append(root)
                        break
                # Also add the base without הת prefix
                if base.lower() not in seen and len(base) >= 3:
                    seen.add(base.lower())
                    expanded.append(base)
            
            # Hebrew Piel/Pual patterns with י/ו infix: e.g., גירושין → גרש
            # Try removing common Hebrew noun suffixes (ין, ים, ות, ה)
            for suffix in ["ושין", "ושים", "ין", "ים", "ות", "ה"]:
                if len(token) > len(suffix) + 2 and token.endswith(suffix):
                    stem = token[:-len(suffix)]
                    if len(stem) >= 2 and stem.lower() not in seen:
                        seen.add(stem.lower())
                        expanded.append(stem)
            
            # Strip common verb suffixes from the original token
            for suffix in ["תי", "נו", "תם", "תן", "ת", "ה"]:
                if len(token) > len(suffix) + 2 and token.endswith(suffix):
                    stem = token[:-len(suffix)]
                    if len(stem) >= 3 and stem.lower() not in seen:
                        seen.add(stem.lower())
                        expanded.append(stem)
                    break
        
        return expanded
    
    @staticmethod
    def _tokenize_query(query: str) -> List[str]:
        """Tokenize a query into words for full-text search.
        
        Language-agnostic: splits on word boundaries and keeps tokens
        ≥ 3 characters.  No hardcoded stop-word lists — Qdrant's
        ``should`` (OR) filter handles the matching, so common words
        simply produce more candidates without hurting precision (RRF
        ranking takes care of relevance).
        
        For Hebrew tokens, also generates morphological variants by
        stripping prefixes and verb conjugation patterns to improve
        recall across different word forms.
        
        Args:
            query: The search query string
            
        Returns:
            Deduplicated list of tokens (≥ 3 chars) with Hebrew expansions
        """
        import re as _re
        tokens = _re.findall(r"[\w]{3,}", query, _re.UNICODE)
        # Deduplicate while preserving order
        seen: set = set()
        unique: List[str] = []
        for t in tokens:
            low = t.lower()
            if low not in seen:
                seen.add(low)
                unique.append(t)
        
        # Expand Hebrew tokens with morphological variants
        expanded = LlamaIndexRAG._expand_hebrew_tokens(unique)
        return expanded
    
    def _fulltext_search_by_field(
        self,
        field_name: str,
        tokens: List[str],
        score: float,
        k: int = 10,
        must_conditions: Optional[List] = None,
    ) -> List[NodeWithScore]:
        """Search a single metadata field using OR-matched tokens.
        
        Uses Qdrant's ``should`` filter so that a document matches if
        it contains **any** of the given tokens in the specified field.
        This avoids the AND-logic limitation of ``MatchText`` when the
        full query string contains words absent from the indexed field.
        
        Args:
            field_name: Qdrant payload field to search (sender, chat_name, message)
            tokens: List of keyword tokens to match (OR logic)
            score: Score to assign to matches from this field
            k: Max results
            must_conditions: Additional filter conditions (AND logic)
            
        Returns:
            List of NodeWithScore with the specified score
        """
        if not tokens:
            return []
        
        try:
            # Build OR conditions: match ANY of the tokens in this field
            should_conditions = [
                FieldCondition(key=field_name, match=MatchText(text=token))
                for token in tokens
            ]
            
            qdrant_filter = Filter(
                must=must_conditions or None,
                should=should_conditions,
            )
            
            results, _ = self.qdrant_client.scroll(
                collection_name=self.COLLECTION_NAME,
                scroll_filter=qdrant_filter,
                limit=k,
                with_payload=True,
                with_vectors=False,
            )
            
            nodes = []
            for record in results:
                payload = record.payload or {}
                text = self._extract_text_from_payload(payload)
                
                if text:
                    node = TextNode(
                        text=text,
                        metadata={mk: mv for mk, mv in payload.items() if not mk.startswith("_")},
                        id_=str(record.id),
                    )
                    nodes.append(NodeWithScore(node=node, score=score))
            
            return nodes
        except Exception as e:
            logger.debug(f"Full-text search on '{field_name}' failed: {e}")
            return []
    
    def _fulltext_search(
        self,
        query: str,
        k: int = 10,
        filter_chat_name: Optional[str] = None,
        filter_days: Optional[int] = None
    ) -> List[NodeWithScore]:
        """Perform field-aware full-text search on metadata fields.
        
        Tokenizes the query into words (≥ 3 chars) and searches each
        metadata field using Qdrant ``should`` (OR) conditions — a
        document matches if it contains **any** of the query tokens in
        the searched field.  This is language-agnostic and requires no
        hardcoded stop-word lists.
        
        Runs one query per field (sender, chat_name, message) with
        different scores to prioritize sender matches over message
        content matches.  Results are deduplicated by node ID, keeping
        the highest score.
        
        Args:
            query: Text to search for
            k: Max results to return
            filter_chat_name: Optional chat filter
            filter_days: Optional time filter
            
        Returns:
            List of matching NodeWithScore objects with field-aware scores
        """
        try:
            # Tokenize query into words ≥ 3 chars (language-agnostic)
            tokens = self._tokenize_query(query)
            if not tokens:
                logger.debug("No tokens extracted from query, skipping fulltext search")
                return []
            
            logger.debug(f"Fulltext tokens: {tokens} (from: {query[:60]})")
            
            # Build common filter conditions (AND logic)
            must_conditions: List = []
            
            if filter_chat_name:
                must_conditions.append(
                    FieldCondition(key="chat_name", match=MatchValue(value=filter_chat_name))
                )
            
            if filter_days is not None and filter_days > 0:
                min_timestamp = int(datetime.now().timestamp()) - (filter_days * 24 * 60 * 60)
                must_conditions.append(
                    FieldCondition(key="timestamp", range=Range(gte=min_timestamp))
                )
            
            # Search each field with OR-matched tokens, different scores
            field_searches = [
                ("sender", self.FULLTEXT_SCORE_SENDER),
                ("chat_name", self.FULLTEXT_SCORE_CHAT_NAME),
                ("message", self.FULLTEXT_SCORE_MESSAGE),
            ]
            
            # Collect all results, dedup by node ID keeping highest score
            best_scores: Dict[str, float] = {}
            best_nodes: Dict[str, NodeWithScore] = {}
            
            for field_name, field_score in field_searches:
                results = self._fulltext_search_by_field(
                    field_name=field_name,
                    tokens=tokens,
                    score=field_score,
                    k=k,
                    must_conditions=must_conditions if must_conditions else None,
                )
                for nws in results:
                    node_id = nws.node.id_ if nws.node else None
                    if not node_id:
                        continue
                    nws_score = nws.score or 0.0
                    if node_id not in best_scores or nws_score > best_scores[node_id]:
                        best_scores[node_id] = nws_score
                        best_nodes[node_id] = nws
            
            # Sort by score descending, return top-k
            sorted_nodes = sorted(best_nodes.values(), key=lambda n: n.score or 0.0, reverse=True)[:k]
            return sorted_nodes
            
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
                text = self._extract_text_from_payload(payload)
                
                if text:
                    node = TextNode(
                        text=text,
                        metadata={mk: mv for mk, mv in payload.items() if not mk.startswith("_")},
                        id_=str(record.id)
                    )
                    nodes.append(NodeWithScore(node=node, score=1.0))
            
            logger.info(f"Metadata search returned {len(nodes)} results")
            return nodes
            
        except Exception as e:
            logger.error(f"Metadata search failed: {e}")
            return []
    
    # =========================================================================
    # Recency-aware retrieval (timestamp-ordered fallback)
    # =========================================================================
    
    def recency_search(
        self,
        k: int = 10,
        filter_chat_name: Optional[str] = None,
        filter_sender: Optional[str] = None,
        filter_days: Optional[int] = None,
    ) -> List[NodeWithScore]:
        """Retrieve the most recent messages ordered by timestamp descending.
        
        This method bypasses semantic search entirely and returns messages
        sorted by recency. It's used for temporal queries like "what's the
        last message?" where vector similarity is irrelevant.
        
        Uses Qdrant's ``order_by`` parameter on the ``timestamp`` field
        (which has an integer payload index) for efficient server-side sorting.
        
        Args:
            k: Number of recent messages to return
            filter_chat_name: Optional filter by chat/group name
            filter_sender: Optional filter by sender name
            filter_days: Optional filter by recency in days
            
        Returns:
            List of NodeWithScore ordered by timestamp (most recent first)
        """
        try:
            must_conditions: List = []
            
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
            
            # Exclude conversation chunks — we want individual messages for recency
            must_conditions.append(
                FieldCondition(key="timestamp", range=Range(gt=0))
            )
            
            scroll_filter = Filter(must=must_conditions) if must_conditions else None
            
            # Use order_by to sort by timestamp descending (most recent first)
            records, _ = self.qdrant_client.scroll(
                collection_name=self.COLLECTION_NAME,
                scroll_filter=scroll_filter,
                limit=k,
                with_payload=True,
                with_vectors=False,
                order_by=OrderBy(key="timestamp", direction=Direction.DESC),
            )
            
            nodes = []
            for record in records:
                payload = record.payload or {}
                text = self._extract_text_from_payload(payload)
                
                if text:
                    node = TextNode(
                        text=text,
                        metadata={mk: mv for mk, mv in payload.items() if not mk.startswith("_")},
                        id_=str(record.id),
                    )
                    # Use timestamp as score so most recent messages rank highest
                    ts = payload.get("timestamp", 0)
                    nodes.append(NodeWithScore(node=node, score=float(ts) if ts else 0.0))
            
            logger.info(f"Recency search returned {len(nodes)} messages (most recent first)")
            return nodes
            
        except Exception as e:
            logger.error(f"Recency search failed: {e}")
            return []
    
    # =========================================================================
    # Context expansion (fetch surrounding messages from same chats)
    # =========================================================================
    
    # Time window (seconds) around matched messages to fetch for context.
    # 30 minutes before and after covers typical conversation flow.
    CONTEXT_WINDOW_SECONDS = int(settings.get("rag_context_window_seconds", "1800"))
    
    def expand_context(
        self,
        results: List[NodeWithScore],
        max_total: int = 20,
    ) -> List[NodeWithScore]:
        """Expand search results by fetching surrounding messages from the same chats.
        
        For each unique chat found in the search results, fetches messages
        within a time window around the matched messages. This ensures that
        replies and nearby messages are included as context even if they
        don't match the query semantically.
        
        For example, if the search finds "Are you taking Mario?" sent to Ori,
        this will also fetch Ori's reply "Thanks 🙏 won't happen again" from
        the same chat within the time window.
        
        Args:
            results: Original search results to expand
            max_total: Maximum total nodes to return (original + expanded)
            
        Returns:
            Merged list of original results + surrounding context, deduplicated
        """
        if not results:
            return results
        
        try:
            # Collect unique (chat_name, timestamp) pairs from results
            chat_windows: Dict[str, List[int]] = {}  # chat_name -> [timestamps]
            existing_ids: set = set()
            
            for nws in results:
                node = nws.node
                if not node:
                    continue
                existing_ids.add(node.id_)
                metadata = getattr(node, "metadata", {})
                chat_name = metadata.get("chat_name")
                timestamp = metadata.get("timestamp")
                if chat_name and timestamp and isinstance(timestamp, (int, float)):
                    chat_windows.setdefault(chat_name, []).append(int(timestamp))
            
            if not chat_windows:
                return results
            
            # For each chat, fetch messages in a time window around the matches
            expanded_nodes: List[NodeWithScore] = []
            budget = max_total - len(results)  # How many more nodes we can add
            
            if budget <= 0:
                return results
            
            per_chat_limit = max(3, budget // len(chat_windows))
            
            for chat_name, timestamps in chat_windows.items():
                min_ts = min(timestamps) - self.CONTEXT_WINDOW_SECONDS
                max_ts = max(timestamps) + self.CONTEXT_WINDOW_SECONDS
                
                must_conditions = [
                    FieldCondition(key="chat_name", match=MatchValue(value=chat_name)),
                    FieldCondition(key="timestamp", range=Range(gte=min_ts, lte=max_ts)),
                ]
                
                try:
                    records, _ = self.qdrant_client.scroll(
                        collection_name=self.COLLECTION_NAME,
                        scroll_filter=Filter(must=must_conditions),
                        limit=per_chat_limit,
                        with_payload=True,
                        with_vectors=False,
                        order_by=OrderBy(key="timestamp", direction=Direction.DESC),
                    )
                    
                    for record in records:
                        record_id = str(record.id)
                        if record_id in existing_ids:
                            continue  # Skip duplicates
                        existing_ids.add(record_id)
                        
                        payload = record.payload or {}
                        text = self._extract_text_from_payload(payload)
                        if text:
                            node = TextNode(
                                text=text,
                                metadata={mk: mv for mk, mv in payload.items() if not mk.startswith("_")},
                                id_=record_id,
                            )
                            # Score slightly below original results so they rank after
                            expanded_nodes.append(NodeWithScore(node=node, score=0.5))
                            
                except Exception as e:
                    logger.debug(f"Context expansion for chat '{chat_name}' failed: {e}")
                    continue
            
            if expanded_nodes:
                logger.info(
                    f"Context expansion added {len(expanded_nodes)} surrounding messages "
                    f"from {len(chat_windows)} chat(s)"
                )
                # Merge: original results first, then expanded context
                results = results + expanded_nodes
            
            return results[:max_total]
            
        except Exception as e:
            logger.debug(f"Context expansion failed (non-critical): {e}")
            return results
    
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
            # with documents that have None text values.
            # Fetch more candidates (k * 2) to compensate for score-threshold filtering,
            # especially for morphologically rich languages like Hebrew where semantic
            # similarity may be lower for inflected query forms.
            query_embedding = Settings.embed_model.get_query_embedding(query)
            vector_fetch_limit = k * 2
            
            search_results = self.qdrant_client.query_points(
                collection_name=self.COLLECTION_NAME,
                query=query_embedding,
                query_filter=qdrant_filters,
                limit=vector_fetch_limit,
                with_payload=True
            ).points
            
            # Convert Qdrant results to NodeWithScore, filtering out invalid entries
            valid_results = []
            for result in search_results:
                payload = result.payload or {}
                text = self._extract_text_from_payload(payload)
                
                # Skip if we couldn't extract valid text
                if not text:
                    logger.warning(f"Skipping result with no valid text: point_id={result.id}")
                    continue
                
                # Create TextNode with valid text
                node = TextNode(
                    text=text,
                    metadata={mk: mv for mk, mv in payload.items() if not mk.startswith("_")},
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
    
    # =========================================================================
    # Chat Engine (LlamaIndex built-in conversation management)
    # =========================================================================
    
    @property
    def chat_store(self) -> RedisChatStore:
        """Get or create the Redis-backed chat store (singleton).
        
        Uses RedisChatStore from llama-index-storage-chat-store-redis
        for automatic persistence of conversation history in Redis.
        """
        if LlamaIndexRAG._chat_store is None:
            redis_url = f"redis://{settings.redis_host}:{settings.redis_port}"
            ttl_seconds = int(settings.session_ttl_minutes) * 60
            LlamaIndexRAG._chat_store = RedisChatStore(
                redis_url=redis_url,
                ttl=ttl_seconds,
            )
            logger.info(f"RedisChatStore initialized at {redis_url} (TTL={ttl_seconds}s)")
        return LlamaIndexRAG._chat_store
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt with current date/time.
        
        Returns:
            System prompt string with dynamic date injection
        """
        timezone = settings.get("timezone", "Asia/Jerusalem")
        tz = ZoneInfo(timezone)
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
        
        # Read system prompt template from settings, with runtime placeholder injection
        prompt_template = settings.get("system_prompt", "")
        if not prompt_template:
            prompt_template = (
                "You are a helpful AI assistant for a personal knowledge base "
                "and message archive search system.\n"
                "You have access to retrieved messages and documents from multiple sources "
                "(messaging platforms, documents, emails, etc.) that will be provided as context.\n\n"
                "Current Date/Time: {current_datetime}\n"
                "תאריך ושעה נוכחיים: {hebrew_date}\n\n"
                "Instructions:\n"
                "1. ANALYZE the retrieved messages to find information relevant to the question.\n"
                "2. CITE specific messages when possible — mention who said what and when.\n"
                "3. If multiple messages are relevant, SYNTHESIZE them into a coherent answer.\n"
                "4. For follow-up questions, USE information from earlier in this conversation. "
                "If you already provided an answer about a topic, build on it — do NOT say "
                "\"no information found\" when you discussed it in a previous turn.\n"
                "5. Only say you lack information when BOTH the retrieved context AND the "
                "conversation history don't contain what's needed. Do NOT fabricate information.\n"
                "6. If the question is general (like \"what day is today?\"), answer directly "
                "without referencing the archive.\n"
                "7. Answer in the SAME LANGUAGE as the question.\n"
                "8. Be concise but thorough. Prefer specific facts over vague summaries."
            )
        
        return prompt_template.format(
            current_datetime=current_datetime,
            hebrew_date=hebrew_date,
        )
    
    def create_chat_engine(
        self,
        conversation_id: str,
        filter_chat_name: Optional[str] = None,
        filter_sender: Optional[str] = None,
        filter_days: Optional[int] = None,
        k: int = 10,
    ) -> CondensePlusContextChatEngine:
        """Create a chat engine with memory and filters for a conversation.
        
        The chat engine automatically handles:
        - Condensing follow-up questions into standalone questions
        - Retrieving relevant context using hybrid search
        - Maintaining conversation history in Redis
        - Generating answers with full context
        
        Args:
            conversation_id: Unique conversation identifier (used as Redis key)
            filter_chat_name: Optional filter by chat/group name
            filter_sender: Optional filter by sender name
            filter_days: Optional filter by recency in days
            k: Number of context documents to retrieve
            
        Returns:
            CondensePlusContextChatEngine ready for .chat() calls
        """
        # Ensure LLM is configured (lazy init for faster startup)
        self._ensure_llm_configured()
        
        # Tag cost tracking events with the conversation ID
        try:
            from cost_callbacks import get_cost_handler
            handler = get_cost_handler()
            if handler:
                handler.conversation_id = conversation_id
        except Exception:
            pass  # Non-fatal
        
        # Memory backed by Redis with token limit
        token_limit = int(settings.session_max_history) * 200  # ~200 tokens per turn
        memory = ChatMemoryBuffer.from_defaults(
            chat_store=self.chat_store,
            chat_store_key=conversation_id,
            token_limit=token_limit,
        )
        
        # Custom retriever wrapping existing hybrid search
        retriever = ArchiveRetriever(
            rag=self,
            k=k,
            filter_chat_name=filter_chat_name,
            filter_sender=filter_sender,
            filter_days=filter_days,
        )
        
        # Build system prompt with current datetime
        system_prompt = self._build_system_prompt()
        
        # Context prompt template — handles both populated and empty context.
        # Explicitly instructs the LLM to leverage chat history for follow-up
        # questions, so it doesn't say "no results" when the answer was already
        # provided in a previous turn.
        context_prompt = (
            "Here are the relevant messages from the archive:\n"
            "-----\n"
            "{context_str}\n"
            "-----\n"
            "IMPORTANT: Use BOTH the retrieved messages above AND the chat history "
            "to answer the user's question. If the retrieved messages don't contain "
            "new relevant information but you already discussed the topic in previous "
            "turns, use that prior context to answer — do NOT say 'no results found' "
            "when you already have the information from earlier in the conversation.\n"
            "Only say no relevant messages were found if BOTH the retrieved context "
            "AND the chat history lack the information needed to answer."
        )
        
        engine = CondensePlusContextChatEngine.from_defaults(
            retriever=retriever,
            memory=memory,
            llm=Settings.llm,
            system_prompt=system_prompt,
            context_prompt=context_prompt,
            verbose=(settings.get("log_level", "INFO") == "DEBUG"),
        )
        
        logger.debug(f"Created chat engine for conversation {conversation_id}")
        return engine
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store.
        
        Returns total point count plus per-source breakdowns so the UI
        can show WhatsApp messages vs documents separately.
        
        Returns:
            Dictionary with collection stats including source_counts
        """
        try:
            collection_info = self.qdrant_client.get_collection(self.COLLECTION_NAME)
            total = collection_info.points_count or 0

            # Count points by source type using Qdrant scroll with filters
            source_counts: Dict[str, int] = {}
            for source_value in ("whatsapp", "paperless"):
                try:
                    count_result = self.qdrant_client.count(
                        collection_name=self.COLLECTION_NAME,
                        count_filter=Filter(must=[
                            FieldCondition(key="source", match=MatchValue(value=source_value))
                        ]),
                        exact=True,
                    )
                    source_counts[source_value] = count_result.count
                except Exception:
                    source_counts[source_value] = 0

            return {
                "total_documents": total,
                "whatsapp_messages": source_counts.get("whatsapp", 0),
                "documents": source_counts.get("paperless", 0),
                "source_counts": source_counts,
                "qdrant_server": f"{self.qdrant_host}:{self.qdrant_port}",
                "collection_name": self.COLLECTION_NAME,
                "dashboard_url": f"http://{self.qdrant_host}:{self.qdrant_port}/dashboard"
            }
        except Exception as e:
            logger.error(f"Failed to get RAG stats: {e}")
            return {"error": str(e)}
    
    def delete_by_source(self, source_value: str) -> int:
        """Delete all points matching a specific source value.
        
        Uses Qdrant's delete with filter to remove points where
        source == source_value. This allows selective cleanup
        (e.g., delete only WhatsApp messages or only Paperless documents)
        without dropping the entire collection.
        
        Args:
            source_value: The source field value to match (e.g., "whatsapp", "paperless")
            
        Returns:
            Number of points deleted
        """
        try:
            # Count before delete
            count_before = self.qdrant_client.count(
                collection_name=self.COLLECTION_NAME,
                count_filter=Filter(must=[
                    FieldCondition(key="source", match=MatchValue(value=source_value))
                ]),
                exact=True,
            ).count

            if count_before == 0:
                logger.info(f"No points with source='{source_value}' to delete")
                return 0

            # Delete by filter
            self.qdrant_client.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=Filter(must=[
                    FieldCondition(key="source", match=MatchValue(value=source_value))
                ]),
            )

            # Invalidate caches since data changed
            self.invalidate_list_caches()

            logger.info(f"Deleted {count_before} points with source='{source_value}'")
            return count_before

        except Exception as e:
            logger.error(f"Failed to delete points by source '{source_value}': {e}")
            return 0

    def reset_collection(self) -> bool:
        """Drop and recreate the Qdrant collection with fresh configuration.
        
        This is required when changing embedding models or dimensions, since
        existing vectors are incompatible with the new model. Also invalidates
        Redis-cached chat/sender lists.
        
        WARNING: This permanently deletes ALL stored embeddings. Messages will
        need to be re-ingested from WhatsApp.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.warning(f"Dropping Qdrant collection: {self.COLLECTION_NAME}")
            
            # Delete the collection entirely
            self.qdrant_client.delete_collection(self.COLLECTION_NAME)
            logger.info(f"Deleted collection: {self.COLLECTION_NAME}")
            
            # Clear cached singleton references so they get recreated
            LlamaIndexRAG._vector_store = None
            LlamaIndexRAG._index = None
            
            # Recreate collection with current VECTOR_SIZE
            self.qdrant_client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Recreated collection: {self.COLLECTION_NAME} (dims={self.VECTOR_SIZE})")
            
            # Recreate indexes
            self._ensure_text_indexes()
            self._ensure_payload_indexes()
            
            # Invalidate Redis caches
            self.invalidate_list_caches()
            
            logger.info("Collection reset complete — all embeddings dropped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            return False
    
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
