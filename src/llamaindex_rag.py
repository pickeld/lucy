"""LlamaIndex RAG (Retrieval Augmented Generation) for multi-source knowledge base.

Uses Qdrant as vector store and OpenAI text-embedding-3-large for semantic search.
Configured with 1024 dimensions for optimal Hebrew + English multilingual support.
Uses LlamaIndex CondensePlusContextChatEngine for multi-turn conversations
with automatic query reformulation and Redis-backed chat memory.

Supports data from multiple channel plugins (WhatsApp, Telegram, Email,
Paperless-NG, etc.) via the plugin architecture.

Qdrant Dashboard: http://localhost:6333/dashboard
"""

import hashlib
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
from llama_index.core.chat_engine import CondensePlusContextChatEngine, ContextChatEngine
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.memory import ChatMemoryBuffer
# SimilarityPostprocessor intentionally not imported — see §1.1 in
# plans/rag-hybrid-retrieval-upgrade.md for rationale on why it was
# removed from the main postprocessor chain.
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
    Fusion,
    FusionQuery,
    MatchText,
    MatchValue,
    NamedSparseVector,
    NamedVector,
    OrderBy,
    PayloadSchemaType,
    Prefetch,
    Range,
    SparseVector,
    SparseVectorParams,
    TextIndexParams,
    TextIndexType,
    TokenizerType,
    VectorParams,
)

from config import settings
from utils.logger import logger
from utils.redis_conn import get_redis_client


def deterministic_node_id(source: str, source_id: str, chunk_index: int = 0) -> str:
    """Generate a deterministic UUID-format ID from source metadata.
    
    Using deterministic IDs makes re-ingestion idempotent: re-syncing
    the same content produces the same point ID in Qdrant, turning
    inserts into upserts instead of creating duplicates.
    
    Args:
        source: Data source (e.g., "paperless", "gmail", "whatsapp")
        source_id: Source-specific unique identifier
        chunk_index: Chunk index within the document (0 for single-chunk)
        
    Returns:
        UUID-format string derived from the input fields
    """
    key = f"{source}:{source_id}:{chunk_index}"
    return str(uuid.UUID(hashlib.md5(key.encode("utf-8")).hexdigest()))


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
        filter_sources: Optional[List[str]] = None,
        filter_date_from: Optional[str] = None,
        filter_date_to: Optional[str] = None,
        filter_content_types: Optional[List[str]] = None,
        sort_order: str = "relevance",
        **kwargs: Any,
    ):
        """Initialize the archive retriever.
        
        Args:
            rag: The LlamaIndexRAG instance to delegate search to
            k: Number of results to retrieve
            filter_chat_name: Optional filter by chat/group name
            filter_sender: Optional filter by sender name
            filter_days: Optional filter by recency in days
            filter_sources: Optional list of source values (e.g. ["whatsapp", "gmail"])
            filter_date_from: Optional ISO date string for start of date range
            filter_date_to: Optional ISO date string for end of date range
            filter_content_types: Optional list of content type values (e.g. ["text", "document"])
            sort_order: "relevance" (default) or "newest" for chronological
        """
        super().__init__(**kwargs)
        self._rag = rag
        self._k = k
        self._filter_chat_name = filter_chat_name
        self._filter_sender = filter_sender
        self._filter_days = filter_days
        self._filter_sources = filter_sources
        self._filter_date_from = filter_date_from
        self._filter_date_to = filter_date_to
        self._filter_content_types = filter_content_types
        self._sort_order = sort_order
    
    # Number of recent messages to always include alongside semantic results.
    # Ensures the LLM has temporal awareness of the latest messages.
    RECENCY_SUPPLEMENT_COUNT = 5
    
    def _inject_entity_facts(self, query: str) -> List[NodeWithScore]:
        """Inject known entity facts as high-priority context nodes.
        
        When the query mentions a known person (by name or alias), looks up
        their stored facts in the Entity Store and creates a context node with
        the information. This enables instant answers for factual questions
        like "בת כמה מיה?" (How old is Mia?) using stored birth_date.
        
        Facts are permanent/time-invariant (birth_date, city, ID number, etc.).
        Age is computed from birth_date at query time, not stored.
        
        Args:
            query: The search query string
            
        Returns:
            List of NodeWithScore with entity facts (empty if no matches)
        """
        try:
            import entity_db
            from datetime import datetime
            from zoneinfo import ZoneInfo
            
            # Tokenize query to find person name tokens
            tokens = LlamaIndexRAG._tokenize_query(query)
            if not tokens:
                return []
            
            # Try to resolve each token as a person name
            injected = []
            seen_person_ids: set = set()
            
            for token in tokens:
                matches = entity_db.resolve_name(token)
                for match in matches:
                    pid = match["id"]
                    if pid in seen_person_ids:
                        continue
                    seen_person_ids.add(pid)
                    
                    # Get full person with facts
                    person = entity_db.get_person(pid)
                    if not person:
                        continue
                    
                    facts = person.get("facts", {})
                    if not facts:
                        continue
                    
                    # Build fact text
                    fact_lines = [f"Known facts about {person['canonical_name']}:"]
                    
                    for key, value in facts.items():
                        if key == "birth_date":
                            # Compute age from birth_date
                            try:
                                tz = ZoneInfo(settings.get("timezone", "Asia/Jerusalem"))
                                now = datetime.now(tz)
                                bd = datetime.fromisoformat(value)
                                age = now.year - bd.year - (
                                    (now.month, now.day) < (bd.month, bd.day)
                                )
                                fact_lines.append(
                                    f"- Birth date: {value} (age: {age})"
                                )
                            except (ValueError, TypeError):
                                fact_lines.append(f"- Birth date: {value}")
                        else:
                            label = key.replace("_", " ").title()
                            fact_lines.append(f"- {label}: {value}")
                    
                    # Add aliases
                    aliases = [
                        a["alias"] for a in person.get("aliases", [])
                        if a["alias"] != person["canonical_name"]
                    ]
                    if aliases:
                        fact_lines.append(
                            f"- Also known as: {', '.join(aliases[:5])}"
                        )
                    
                    # Add relationships
                    rels = person.get("relationships", [])
                    for rel in rels[:3]:
                        fact_lines.append(
                            f"- {rel['relationship_type'].title()} of {rel['related_name']}"
                        )
                    
                    fact_text = "\n".join(fact_lines)
                    # Format with consistent source header so the LLM
                    # can cite entity facts like any other source.
                    formatted_fact_text = (
                        f"Entity Store | Person: {person['canonical_name']}:\n"
                        f"{fact_text}"
                    )
                    node = TextNode(
                        text=formatted_fact_text,
                        metadata={
                            "source": "entity_store",
                            "content_type": "entity_facts",
                            "person_name": person["canonical_name"],
                            "person_id": pid,
                        },
                    )
                    # High score so entity facts appear first in context
                    injected.append(NodeWithScore(node=node, score=1.0))
            
            return injected
        except ImportError:
            return []  # entity_db not available
        except Exception:
            return []
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve relevant messages/documents using hybrid search.
        
        Pipeline order (optimized — see plans/rag-hybrid-retrieval-upgrade.md §2.1):
        
        1. Hybrid search (dense + fulltext, fused via QueryFusionRetriever)
        2. **Cohere rerank** — prune to top-N most relevant before expansion
        3. Context expansion (±30min surrounding messages from reranked hits)
        4. Document chunk expansion (Paperless siblings of reranked hits)
        5. Per-chat recency supplement (only from chats in reranked results)
        6. Entity fact injection
        7. Token budget trim
        
        Reranking before expansion means only high-quality seed results
        get expanded, reducing noise and saving Qdrant queries.
        
        Always returns at least one node so the chat engine's synthesizer
        can generate a proper response (it returns "Empty Response" on empty input).
        
        Args:
            query_bundle: The query bundle from the chat engine
            
        Returns:
            List of NodeWithScore from hybrid search (never empty)
        """
        # Common filter kwargs shared across search/recency calls
        _fkw = dict(
            filter_chat_name=self._filter_chat_name,
            filter_sender=self._filter_sender,
            filter_days=self._filter_days,
            filter_sources=self._filter_sources,
            filter_date_from=self._filter_date_from,
            filter_date_to=self._filter_date_to,
            filter_content_types=self._filter_content_types,
        )
        
        # If sort_order is "newest", skip semantic search — just use recency
        if self._sort_order == "newest":
            results = self._rag.recency_search(k=self._k, **_fkw)
            # Still expand document chunks for completeness
            if results:
                results = self._rag.expand_document_chunks(results, max_total=self._k * 3)
            if not results:
                placeholder = TextNode(
                    text="[No relevant messages found in the archive for this query]",
                    metadata={"source": "system", "note": "no_results"},
                )
                results = [NodeWithScore(node=placeholder, score=0.0)]
            return results
        
        results = self._rag.search(
            query=query_bundle.query_str,
            k=self._k,
            **_fkw,
        )
        
        # =====================================================================
        # Step 1: Cohere multilingual reranking (BEFORE expansion)
        # =====================================================================
        # Reranking before expansion ensures that only high-quality seed
        # results get expanded, reducing noise from irrelevant context
        # messages and saving Qdrant queries.  The reranker sees the raw
        # search candidates and selects the most relevant subset.
        try:
            cohere_key = settings.get("cohere_api_key", "")
            if cohere_key and results:
                from llama_index.postprocessor.cohere_rerank import CohereRerank
                
                rerank_top_n = int(settings.get("rag_rerank_top_n", "10"))
                rerank_model = settings.get("rag_rerank_model", "rerank-v3.5")
                reranker = CohereRerank(
                    api_key=cohere_key,
                    top_n=rerank_top_n,
                    model=rerank_model,
                )
                pre_rerank = len(results)
                results = reranker.postprocess_nodes(results, query_bundle)
                logger.info(
                    f"Cohere rerank ({rerank_model}): {pre_rerank} → {len(results)} results"
                )
        except ImportError:
            logger.warning(
                "llama-index-postprocessor-cohere-rerank not installed. "
                "Install with: pip install llama-index-postprocessor-cohere-rerank"
            )
        except Exception as e:
            logger.debug(f"Cohere reranking failed (non-critical): {e}")
        
        # Early budget tracking: compute budget once and skip expansions
        # if we're already close to the limit, avoiding wasteful Qdrant
        # queries for context that will just be trimmed away.
        max_context_chars = self._rag.MAX_CONTEXT_TOKENS * 4  # ~12000 chars default
        
        def _current_chars(nodes: List[NodeWithScore]) -> int:
            return sum(len(getattr(n.node, "text", "") or "") for n in nodes if n.node)
        
        # =====================================================================
        # Step 2: Context expansion (on reranked results only)
        # =====================================================================
        # Fetch surrounding messages from the same chats so that replies
        # and nearby messages are included as context.
        if results and _current_chars(results) < max_context_chars * 0.8:
            results = self._rag.expand_context(results, max_total=self._k * 2)
        
        # =====================================================================
        # Step 3: Document chunk expansion (on reranked results only)
        # =====================================================================
        # When a Paperless document chunk is found, fetch ALL sibling chunks
        # from the same document so the LLM sees the complete content.
        if results and _current_chars(results) < max_context_chars * 0.8:
            results = self._rag.expand_document_chunks(results, max_total=self._k * 3)
        
        # =====================================================================
        # Step 4: Per-chat recency supplement
        # =====================================================================
        # Fetch recent messages only from chats that appear in the
        # (now reranked) search results.  This prevents unrelated recent
        # messages from polluting the context.
        if results:
            chat_names_in_results: set = set()
            for nws in results:
                if nws.node:
                    cn = getattr(nws.node, "metadata", {}).get("chat_name")
                    if cn:
                        chat_names_in_results.add(cn)
            
            existing_ids = {nws.node.id_ for nws in results if nws.node}
            per_chat_limit = max(2, self.RECENCY_SUPPLEMENT_COUNT // max(len(chat_names_in_results), 1))
            
            for chat_name in chat_names_in_results:
                chat_recent = self._rag.recency_search(
                    k=per_chat_limit,
                    filter_chat_name=chat_name,
                    filter_sender=self._filter_sender,
                    filter_days=self._filter_days,
                    filter_sources=self._filter_sources,
                    filter_date_from=self._filter_date_from,
                    filter_date_to=self._filter_date_to,
                    filter_content_types=self._filter_content_types,
                )
                for nws in chat_recent:
                    if nws.node and nws.node.id_ not in existing_ids:
                        existing_ids.add(nws.node.id_)
                        results.append(nws)
        
        # =====================================================================
        # Step 5: Entity fact injection
        # =====================================================================
        # When the query mentions a known person, inject their stored facts
        # as a high-priority context node so the LLM can answer factual
        # questions (age, birth date, etc.) directly.
        try:
            entity_nodes = self._inject_entity_facts(query_bundle.query_str)
            if entity_nodes:
                results = entity_nodes + results
                logger.info(f"Injected {len(entity_nodes)} entity fact node(s)")
        except Exception as e:
            logger.debug(f"Entity fact injection failed (non-critical): {e}")
        
        # Ensure at least one node so the synthesizer doesn't return "Empty Response"
        if not results:
            placeholder = TextNode(
                text="[No relevant messages found in the archive for this query]",
                metadata={"source": "system", "note": "no_results"},
            )
            results = [NodeWithScore(node=placeholder, score=0.0)]
        
        # Context budget: cap total text to avoid exceeding LLM token limits.
        # With full Paperless document chunks (up to 6000 chars each) and sibling
        # expansion, the total context can easily exceed 50K tokens. Use
        # rag_max_context_tokens × 4 chars/token as the character budget.
        # max_context_chars already computed above for early budget checks.
        #
        # Per-result cap: no single result may exceed 40% of the total budget.
        # This prevents a few large Paperless documents from consuming all
        # context space, ensuring shorter results (voice transcriptions, messages)
        # also get included.
        per_result_cap = int(max_context_chars * 0.4)
        total_chars = 0
        trimmed: List[NodeWithScore] = []
        for nws in results:
            node_text = getattr(nws.node, "text", "") if nws.node else ""
            text_len = len(node_text)
            # Cap oversized results (e.g., Paperless documents) so they
            # don't consume the entire budget, leaving room for shorter
            # results like voice transcriptions and messages.
            effective_len = min(text_len, per_result_cap)
            if text_len > per_result_cap and nws.node:
                truncated_text = node_text[:per_result_cap] + "\n[...truncated...]"
                truncated_node = TextNode(
                    text=truncated_text,
                    metadata=getattr(nws.node, "metadata", {}),
                    id_=nws.node.id_,
                )
                nws = NodeWithScore(node=truncated_node, score=nws.score)
                effective_len = len(truncated_text)
            if total_chars + effective_len > max_context_chars and trimmed:
                # Budget exhausted — stop adding more results
                break
            trimmed.append(nws)
            total_chars += effective_len
        
        if len(trimmed) < len(results):
            logger.info(
                f"Context budget trimmed {len(results)} → {len(trimmed)} results "
                f"({total_chars} chars, budget={max_context_chars})"
            )
            results = trimmed
        
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
    _ingestion_pipeline = None
    
    COLLECTION_NAME = settings.rag_collection_name
    VECTOR_SIZE = int(settings.get("rag_vector_size", "1024"))
    MINIMUM_SIMILARITY_SCORE = float(settings.rag_min_score)
    MAX_CONTEXT_TOKENS = int(settings.rag_max_context_tokens)
    
    # Sparse+dense hybrid retrieval (see plans/rag-hybrid-retrieval-upgrade.md §1.3)
    HYBRID_ENABLED = settings.get("rag_hybrid_enabled", "false").lower() == "true"
    # LlamaIndex QdrantVectorStore prepends "text-" to vector_name, so we
    # must use "text-dense" as the actual Qdrant vector name to match.
    DENSE_VECTOR_NAME = "text-dense"  # Named vector for OpenAI embeddings
    SPARSE_VECTOR_NAME = "sparse"     # Named vector for BM25-style sparse vectors
    
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
        """Get or create the Qdrant vector store.
        
        When hybrid mode is enabled, configures the vector store to use
        the 'dense' named vector instead of the default unnamed vector.
        """
        if LlamaIndexRAG._vector_store is None:
            kwargs = {
                "client": self.qdrant_client,
                "collection_name": self.COLLECTION_NAME,
            }
            if self.HYBRID_ENABLED:
                kwargs["vector_name"] = self.DENSE_VECTOR_NAME
            LlamaIndexRAG._vector_store = QdrantVectorStore(**kwargs)
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
    
    @property
    def ingestion_pipeline(self) -> IngestionPipeline:
        """Get or create the LlamaIndex IngestionPipeline with optional embedding cache.
        
        The pipeline provides:
        - Embedding cache via Redis (avoids re-embedding unchanged content)
        - Deduplication via content hash
        - Composable transformation chain
        
        When ``rag_embedding_cache_enabled`` is True (default), embeddings
        are cached in Redis so that re-syncs of unchanged documents skip
        the embedding API call entirely.
        
        Returns:
            IngestionPipeline instance configured with the current vector store
        """
        if LlamaIndexRAG._ingestion_pipeline is None:
            cache = None
            if settings.get("rag_embedding_cache_enabled", "true").lower() == "true":
                try:
                    from llama_index.storage.kvstore.redis import RedisKVStore
                    from llama_index.core.ingestion import IngestionCache
                    
                    redis_url = f"redis://{settings.redis_host}:{settings.redis_port}"
                    redis_kvstore = RedisKVStore(redis_uri=redis_url)
                    cache = IngestionCache(cache=redis_kvstore)
                    logger.info(f"Embedding cache enabled via Redis at {redis_url}")
                except ImportError:
                    logger.warning(
                        "llama-index-storage-kvstore-redis not installed. "
                        "Embedding cache disabled. Install with: "
                        "pip install llama-index-storage-kvstore-redis"
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize embedding cache: {e}")
            
            pipeline_kwargs = {
                "transformations": [Settings.embed_model],
                "vector_store": self.vector_store,
            }
            if cache is not None:
                pipeline_kwargs["cache"] = cache
            
            LlamaIndexRAG._ingestion_pipeline = IngestionPipeline(**pipeline_kwargs)
            logger.info("IngestionPipeline initialized")
        
        return LlamaIndexRAG._ingestion_pipeline
    
    def ingest_nodes(self, nodes: List[TextNode]) -> int:
        """Ingest nodes via the IngestionPipeline (with embedding cache).
        
        Uses the shared IngestionPipeline which provides:
        - Redis-backed embedding cache (skips re-embedding unchanged content)
        - Automatic deduplication by content hash
        
        Falls back to ``add_nodes()`` if the pipeline fails.
        
        Args:
            nodes: List of TextNode instances to ingest
            
        Returns:
            Number of nodes successfully ingested
        """
        if not nodes:
            return 0
        
        try:
            result_nodes = self.ingestion_pipeline.run(nodes=nodes, show_progress=False)
            count = len(result_nodes) if result_nodes else len(nodes)
            logger.info(f"IngestionPipeline ingested {count} nodes")
            
            # After dense vector ingestion, compute and upsert sparse vectors
            if self.HYBRID_ENABLED:
                self._upsert_sparse_vectors(nodes)
            
            return count
        except Exception as e:
            logger.warning(f"IngestionPipeline failed, falling back to add_nodes(): {e}")
            count = self.add_nodes(nodes)
            if self.HYBRID_ENABLED and count > 0:
                self._upsert_sparse_vectors(nodes)
            return count
    
    def _upsert_sparse_vectors(self, nodes: List[TextNode]) -> None:
        """Compute and upsert BM25-style sparse vectors for ingested nodes.
        
        Called after dense vector ingestion.  Uses the Qdrant client directly
        to update each point with a sparse vector in the 'sparse' named vector
        field.  This enables server-side hybrid search (dense + sparse + RRF).
        
        Only runs when ``rag_hybrid_enabled=true`` and the collection has
        a sparse vector configuration.
        
        Args:
            nodes: List of TextNode instances that were just ingested
        """
        try:
            from utils.sparse_vectors import compute_sparse_vector
            from qdrant_client.models import PointVectors
            
            points_to_update = []
            for node in nodes:
                text = getattr(node, "text", "") or ""
                if not text:
                    continue
                
                indices, values = compute_sparse_vector(text)
                if not indices:
                    continue
                
                points_to_update.append(
                    PointVectors(
                        id=node.id_,
                        vector={
                            self.SPARSE_VECTOR_NAME: SparseVector(
                                indices=indices,
                                values=values,
                            )
                        },
                    )
                )
            
            if points_to_update:
                self.qdrant_client.update_vectors(
                    collection_name=self.COLLECTION_NAME,
                    points=points_to_update,
                )
                logger.debug(
                    f"Upserted sparse vectors for {len(points_to_update)} nodes"
                )
        except ImportError:
            logger.debug("sparse_vectors module not available — skipping sparse upsert")
        except Exception as e:
            logger.debug(f"Sparse vector upsert failed (non-critical): {e}")
    
    def _hybrid_search(
        self,
        query: str,
        k: int = 10,
        qdrant_filters: Optional[Filter] = None,
    ) -> List[NodeWithScore]:
        """Perform hybrid search using Qdrant's server-side dense+sparse RRF fusion.
        
        Sends two prefetch queries (dense embedding + sparse BM25 vector) and
        lets Qdrant fuse them via Reciprocal Rank Fusion.  This replaces both
        the vector-only search AND the fulltext search with a single unified
        ranked result set.
        
        Only works when the collection has named vectors (dense + sparse).
        Falls back gracefully if sparse vectors are not available.
        
        Args:
            query: Search query text
            k: Number of results to return
            qdrant_filters: Optional Qdrant filter conditions
            
        Returns:
            List of NodeWithScore from hybrid search
        """
        try:
            from utils.sparse_vectors import compute_query_sparse_vector
            
            # Compute both dense and sparse query vectors
            query_embedding = Settings.embed_model.get_query_embedding(query)
            sparse_indices, sparse_values = compute_query_sparse_vector(query)
            
            if not sparse_indices:
                # No sparse tokens — fall back to dense-only
                logger.debug("No sparse tokens for query, using dense-only")
                return []  # Caller will fall back
            
            # Qdrant hybrid query with prefetch + RRF fusion
            prefetch_limit = k * 3  # Fetch more candidates for fusion
            
            search_results = self.qdrant_client.query_points(
                collection_name=self.COLLECTION_NAME,
                prefetch=[
                    Prefetch(
                        query=query_embedding,
                        using=self.DENSE_VECTOR_NAME,
                        limit=prefetch_limit,
                        filter=qdrant_filters,
                    ),
                    Prefetch(
                        query=SparseVector(
                            indices=sparse_indices,
                            values=sparse_values,
                        ),
                        using=self.SPARSE_VECTOR_NAME,
                        limit=prefetch_limit,
                        filter=qdrant_filters,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=k * 2,
                with_payload=True,
            ).points
            
            # Convert to NodeWithScore
            valid_results = []
            for result in search_results:
                payload = result.payload or {}
                text = self._extract_text_from_payload(payload)
                if not text:
                    continue
                node = TextNode(
                    text=text,
                    metadata={mk: mv for mk, mv in payload.items() if not mk.startswith("_")},
                    id_=str(result.id),
                )
                valid_results.append(NodeWithScore(node=node, score=result.score))
            
            logger.info(
                f"Hybrid search (dense+sparse RRF) for '{query[:50]}...' "
                f"returned {len(valid_results)} results"
            )
            return valid_results
            
        except Exception as e:
            logger.warning(f"Hybrid search failed, will fall back to standard search: {e}")
            return []  # Caller falls back to non-hybrid path
    
    def _ensure_collection(self):
        """Ensure the collection exists in Qdrant with text indexes for metadata search.
        
        When ``rag_hybrid_enabled=true``, creates the collection with named
        vectors (dense + sparse) for server-side hybrid search with RRF fusion.
        Otherwise creates a standard single-vector collection.
        """
        try:
            logger.debug("Fetching existing collections from Qdrant...")
            collections = self.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]
            logger.debug(f"Found collections: {collection_names}")
            
            if self.COLLECTION_NAME not in collection_names:
                if self.HYBRID_ENABLED:
                    # Named vectors: dense (OpenAI embeddings) + sparse (BM25)
                    logger.info(
                        f"Creating HYBRID Qdrant collection: {self.COLLECTION_NAME} "
                        f"(dense={self.VECTOR_SIZE}d + sparse)"
                    )
                    self.qdrant_client.create_collection(
                        collection_name=self.COLLECTION_NAME,
                        vectors_config={
                            self.DENSE_VECTOR_NAME: VectorParams(
                                size=self.VECTOR_SIZE,
                                distance=Distance.COSINE,
                            ),
                        },
                        sparse_vectors_config={
                            self.SPARSE_VECTOR_NAME: SparseVectorParams(),
                        },
                    )
                    logger.info(f"Created hybrid collection: {self.COLLECTION_NAME}")
                else:
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
        
        try:
            # Create keyword index on 'numbers' field for reverse ID/number lookups.
            # Paperless documents store extracted numeric sequences (≥5 digits)
            # as a JSON array of strings, enabling exact-match queries like
            # "למי שייכת תעודה הזהות 227839586?" to find the matching document.
            # Keyword index is faster and more exact than tokenized text matching.
            self.qdrant_client.create_payload_index(
                collection_name=self.COLLECTION_NAME,
                field_name="numbers",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logger.info("Created keyword index on 'numbers' field")
        except Exception as e:
            logger.debug(f"Could not create numbers index (may exist): {e}")
    
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
            ("chat_id", PayloadSchemaType.KEYWORD, "chat_id keyword index"),
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
    CHUNK_BUFFER_TTL = int(settings.get("rag_chunk_buffer_ttl", "120"))
    CHUNK_MAX_MESSAGES = int(settings.get("rag_chunk_max_messages", "5"))
    CHUNK_OVERLAP_MESSAGES = int(settings.get("rag_chunk_overlap_messages", "1"))
    
    # Minimum TTL remaining (seconds) before a buffer is considered
    # "near-expiry" and flushed proactively.  When a buffer's TTL drops
    # below this threshold and it holds ≥ 2 messages, it is flushed to
    # prevent silent data loss from Redis key expiration.
    CHUNK_BUFFER_FLUSH_THRESHOLD = max(
        10,
        int(settings.get("rag_chunk_buffer_ttl", "120")) // 4,
    )
    
    def _flush_expiring_buffers(self) -> int:
        """Scan for chunk buffers near TTL expiry and flush them.
        
        Without this, low-volume chats that never reach CHUNK_MAX_MESSAGES
        have their buffered messages silently deleted when the Redis TTL
        fires.  This method is called on every new message arrival and
        uses SCAN to find active buffers whose TTL is below the flush
        threshold.
        
        Only buffers with ≥ 2 messages are flushed (single messages don't
        benefit from conversation chunking).
        
        Returns:
            Number of buffers flushed
        """
        flushed = 0
        try:
            redis = get_redis_client()
            cursor = 0
            pattern = f"{self.CHUNK_BUFFER_KEY_PREFIX}*"
            
            while True:
                cursor, keys = redis.scan(
                    cursor=cursor,
                    match=pattern,
                    count=50,
                )
                
                for key in keys:
                    # Decode key if bytes
                    key_str = key if isinstance(key, str) else key.decode("utf-8", errors="replace")
                    
                    try:
                        ttl = redis.ttl(key_str)
                        # ttl returns -1 if no expiry, -2 if key doesn't exist
                        if ttl < 0:
                            continue
                        
                        if ttl <= self.CHUNK_BUFFER_FLUSH_THRESHOLD:
                            buf_len = redis.llen(key_str)
                            if buf_len >= 2:
                                # Extract chat_id from the key
                                chat_id = key_str[len(self.CHUNK_BUFFER_KEY_PREFIX):]
                                if self._flush_chunk_buffer(chat_id):
                                    flushed += 1
                                    logger.info(
                                        f"Flushed near-expiry buffer for chat {chat_id} "
                                        f"(TTL={ttl}s, {buf_len} msgs)"
                                    )
                    except Exception as e:
                        logger.debug(f"Error checking buffer TTL for {key_str}: {e}")
                
                if cursor == 0:
                    break
            
        except Exception as e:
            logger.debug(f"Expiring buffer scan failed (non-critical): {e}")
        
        return flushed
    
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
        CHUNK_MAX_MESSAGES, the buffer is flushed as a single conversation chunk
        that gets its own embedding in Qdrant.
        
        Before buffering, scans for other chats' buffers that are near TTL
        expiry and flushes them proactively.  This prevents low-volume chats
        from silently losing buffered messages when the Redis TTL fires.
        
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
            # Proactively flush any buffers about to expire (prevents data loss)
            self._flush_expiring_buffers()
            
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
            
            # Get all messages from the buffer
            raw_messages = redis.lrange(buffer_key, 0, -1)
            
            if not raw_messages:
                redis.delete(buffer_key)
                return False  # Nothing to flush
            
            # Keep the last N messages as overlap for the next chunk.
            # This ensures context continuity between consecutive chunks.
            overlap = self.CHUNK_OVERLAP_MESSAGES
            if overlap > 0 and len(raw_messages) > overlap:
                # Delete all, then re-push the overlap messages
                redis.delete(buffer_key)
                overlap_messages = raw_messages[-overlap:]
                for msg in overlap_messages:
                    redis.rpush(buffer_key, msg)
                redis.expire(buffer_key, self.CHUNK_BUFFER_TTL)
            else:
                redis.delete(buffer_key)
            
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
                    "source": "whatsapp",
                    "content_type": "conversation_chunk",
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
                id_=deterministic_node_id(
                    "whatsapp", f"chunk:{chat_id}:{first_ts}:{last_ts}", 0
                ),
            )
            
            self.ingest_nodes([chunk_node])
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
        media_url: Optional[str] = None,
        media_path: Optional[str] = None,
        message_content_type: Optional[str] = None,
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
            media_path: Local file path to saved media (e.g. data/images/media_xxx.jpg)
            message_content_type: Explicit content type override (e.g. "voice", "image").
                When provided, bypasses auto-detection from has_media/media_type.
                This preserves the correct type even when media download fails.
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from models import WhatsAppMessageDocument
            from models.base import ContentType as ModelContentType
            
            # Deduplication: skip if message already exists
            source_id = f"{chat_id}:{timestamp}"
            if self._message_exists(source_id):
                logger.debug(f"Skipping duplicate message: {source_id}")
                return True  # Not an error, just already stored
            
            # Resolve explicit content_type override to enum
            ct_override = None
            if message_content_type:
                try:
                    ct_override = ModelContentType(message_content_type)
                except ValueError:
                    logger.debug(f"Unknown content_type '{message_content_type}', using auto-detect")
            
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
                media_url=media_url,
                media_path=media_path,
                message_content_type=ct_override,
            )
            
            # Convert to LlamaIndex TextNode with standardized schema
            node = doc.to_llama_index_node()
            
            # Use IngestionPipeline for embedding cache + sparse vectors.
            # Falls back to direct insert if pipeline is unavailable.
            self.ingest_nodes([node])
            
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
        
        Routes through :meth:`ingest_nodes` so the node benefits from the
        Redis-backed embedding cache and (when hybrid mode is enabled)
        automatic sparse-vector upsert.
        
        Proactively truncates node text that exceeds EMBEDDING_MAX_CHARS
        before calling the embedding API, avoiding a wasted API call on
        oversized content.
        
        Args:
            node: LlamaIndex TextNode to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Proactive truncation: avoid a wasted API call on oversized text
            if len(node.text) > self.EMBEDDING_MAX_CHARS:
                logger.info(
                    f"Proactively truncating node text ({len(node.text)} chars "
                    f"→ {self.EMBEDDING_MAX_CHARS} chars)"
                )
                node.text = node.text[:self.EMBEDDING_MAX_CHARS]
            
            self.ingest_nodes([node])
            logger.debug(f"Added node to RAG: {node.text[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to add node to vector store: {e}")
            return False
    
    def add_nodes(self, nodes: List[TextNode]) -> int:
        """Add multiple nodes to the vector store in batch (internal fallback).
        
        .. warning:: This method inserts directly via ``index.insert_nodes()``
           and does **not** use the :class:`IngestionPipeline` (no embedding
           cache, no automatic sparse-vector upsert).  It exists only as
           the last-resort fallback inside :meth:`ingest_nodes`.
        
           External callers should use :meth:`ingest_nodes` instead.
        
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
            self.ingest_nodes([node])
            
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
                self.ingest_nodes(nodes)
                logger.info(f"Added {len(nodes)} documents to RAG vector store")
            
            return len(nodes)
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")
            return 0
    
    # Human-readable labels for source values in retrieved context.
    # Capitalised nicely so the LLM can cite them naturally.
    _SOURCE_LABELS: Dict[str, str] = {
        "whatsapp": "WhatsApp",
        "paperless": "Paperless",
        "gmail": "Gmail",
        "telegram": "Telegram",
        "email": "Email",
        "slack": "Slack",
        "discord": "Discord",
        "sms": "SMS",
        "manual": "Manual",
        "web_scrape": "Web",
        "api_import": "Import",
        "call_recording": "Call Recording",
        "entity_store": "Entity Store",
        "system": "System",
    }

    @staticmethod
    def _extract_text_from_payload(payload: Dict[str, Any]) -> Optional[str]:
        """Extract display text from a Qdrant point payload.
        
        Handles multiple source types with source-specific formatting:
        
        - **WhatsApp messages**: ``WhatsApp | date | sender in chat: message``
        - **Conversation chunks**: ``WhatsApp Conversation | chat | date_range:`` + multi-line
        - **Paperless documents**: ``Paperless | date | Document 'title':`` + full text
        - **Gmail emails**: ``Gmail | date | sender in subject: body``
        - **Entity facts**: Already pre-formatted with ``Entity Store |`` header
        - **Generic documents**: ``Source | date | sender/title:`` + text from _node_content
        
        Each entry starts with a source label so the LLM can identify
        where the information came from and cite it properly.
        
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
        source_type = payload.get("source_type", "")
        content_type = payload.get("content_type", "")
        
        # Resolve a human-readable source label
        source_label = LlamaIndexRAG._SOURCE_LABELS.get(
            source, source.capitalize() if source else "Archive"
        )
        
        # -----------------------------------------------------------------
        # Conversation chunks: pre-formatted multi-message blocks.
        # Format them with a special header instead of the standard
        # single-message format to avoid double-wrapping.
        # -----------------------------------------------------------------
        if source_type == "conversation_chunk" or content_type == "conversation_chunk":
            formatted_time = format_timestamp(str(timestamp))
            first_ts = payload.get("first_timestamp", 0)
            first_formatted = format_timestamp(str(first_ts)) if first_ts else ""
            date_range = f"{first_formatted} → {formatted_time}" if first_formatted else formatted_time
            msg_count = payload.get("message_count", "?")
            
            # The chunk text is already formatted as [timestamp] sender: message lines
            # Use _node_content if available (full text), otherwise try message field
            chunk_text = message
            if not chunk_text:
                node_content = payload.get("_node_content")
                if node_content and isinstance(node_content, str):
                    try:
                        chunk_text = json.loads(node_content).get("text", "")
                    except (json.JSONDecodeError, TypeError):
                        pass
            
            if chunk_text:
                return (
                    f"WhatsApp Conversation | {chat_name} | {date_range} "
                    f"({msg_count} messages):\n{chunk_text}"
                )
        
        # -----------------------------------------------------------------
        # Paperless documents: prefer _node_content (full chunk text, up to
        # 6000 chars) over the 'message' metadata field which may be truncated
        # (old syncs stored only 2000 chars in 'message').  New syncs store
        # full text in 'message' but we keep the _node_content fallback for
        # backward compatibility with existing data.
        # -----------------------------------------------------------------
        if source == "paperless":
            # Try message field first (new syncs store full text here)
            text = message
            # Fall back to _node_content for old syncs with truncated message
            if not text or len(text) < 100:
                node_content = payload.get("_node_content")
                if node_content and isinstance(node_content, str):
                    try:
                        nc_text = json.loads(node_content).get("text", "")
                        if nc_text and len(nc_text) > len(text or ""):
                            text = nc_text
                    except (json.JSONDecodeError, TypeError):
                        pass
            
            if text:
                formatted_time = format_timestamp(str(timestamp))
                if sender and sender != "Unknown":
                    return f"{source_label} | {formatted_time} | {sender} in {chat_name}:\n{text}"
                return f"{source_label} | {formatted_time} | Document '{chat_name}':\n{text}"
        
        # -----------------------------------------------------------------
        # Call recordings: prefer _node_content (full transcript chunk, up to
        # 6000 chars) over the 'message' metadata field which is truncated
        # to 2000 chars.  The full transcript is essential for the LLM to
        # cite the correct passage — e.g. when a user asks "who did princess
        # training?" the relevant sentence may be past the 2000-char mark.
        # -----------------------------------------------------------------
        if source == "call_recording":
            # Try message field first (may be truncated to 2000 chars)
            text = message
            # Prefer _node_content which stores the full embedding text
            # (header + complete transcript chunk)
            node_content = payload.get("_node_content")
            if node_content and isinstance(node_content, str):
                try:
                    nc_text = json.loads(node_content).get("text", "")
                    if nc_text and len(nc_text) > len(text or ""):
                        text = nc_text
                except (json.JSONDecodeError, TypeError):
                    pass
            
            if text:
                formatted_time = format_timestamp(str(timestamp))
                return f"{source_label} | {formatted_time} | {sender} in {chat_name}:\n{text}"
        
        # -----------------------------------------------------------------
        # Standard messages (WhatsApp, Gmail, etc.) use the 'message' field
        # -----------------------------------------------------------------
        if message:
            formatted_time = format_timestamp(str(timestamp))
            return f"{source_label} | {formatted_time} | {sender} in {chat_name}: {message}"
        
        # -----------------------------------------------------------------
        # Generic documents without 'message': extract text from _node_content
        # -----------------------------------------------------------------
        node_content = payload.get("_node_content")
        if node_content and isinstance(node_content, str):
            try:
                content_dict = json.loads(node_content)
                text = content_dict.get("text")
                if text:
                    formatted_time = format_timestamp(str(timestamp))
                    if sender and sender != "Unknown":
                        return f"{source_label} | {formatted_time} | {sender} in {chat_name}:\n{text}"
                    return f"{source_label} | {formatted_time} | Document '{chat_name}':\n{text}"
            except (json.JSONDecodeError, TypeError):
                pass
        
        return None
    
    # =========================================================================
    # Cross-script contact name expansion for fulltext search
    # =========================================================================
    
    def _expand_tokens_with_contact_names(self, tokens: List[str]) -> List[str]:
        """Expand query tokens with cross-script contact name matches.
        
        When a query contains a Hebrew name like 'שירן' that matches a contact
        stored as 'Shiran Waintrob' in English (or vice versa), this method adds
        the alternative-script name tokens so fulltext search on sender/chat_name
        fields can find the match.
        
        Strategy (ordered by richness):
        1. Entity Store aliases (if available): uses person_aliases table which
           links all name variants across scripts for each person
        2. Fallback: sender + chat name lists from Redis cache, with simple
           first-name → full-name-parts mapping
        
        This handles both directions:
        - Hebrew query → English-stored names (שירן → Shiran, Waintrob)
        - English query → Hebrew-stored names (Doron → דורון, עלאני)
        
        Args:
            tokens: Original query tokens (from _tokenize_query)
            
        Returns:
            Expanded token list (originals + cross-script contact name parts)
        """
        try:
            # Strategy 1: Use Entity Store aliases (richer, links all name variants)
            name_map = self._build_entity_name_map()
            
            # Strategy 2: Fallback to sender/chat list if entity store is empty
            if not name_map:
                name_map = self._build_sender_name_map()
            
            if not name_map:
                return tokens
            
            # Check each token against the name map
            MAX_EXPANSION_PER_TOKEN = 10  # Cap per single token match
            MAX_TOTAL_EXPANDED = 50       # Hard cap on total expanded tokens
            
            expanded = list(tokens)
            seen = {t.lower() for t in tokens}
            
            for token in tokens:
                low = token.lower()
                if low in name_map:
                    added_for_token = 0
                    for name_part in name_map[low]:
                        if added_for_token >= MAX_EXPANSION_PER_TOKEN:
                            break
                        if len(expanded) >= MAX_TOTAL_EXPANDED:
                            break
                        if name_part.lower() not in seen:
                            seen.add(name_part.lower())
                            expanded.append(name_part)
                            added_for_token += 1
            
            if len(expanded) > len(tokens):
                logger.debug(
                    f"Cross-script name expansion: {len(tokens)} → {len(expanded)} tokens "
                    f"(added: {[t for t in expanded[len(tokens):]]})"
                )
            
            return expanded
        except Exception as e:
            logger.debug(f"Cross-script name expansion failed (non-critical): {e}")
            return tokens
    
    def _build_entity_name_map(self) -> Dict[str, set]:
        """Build a name→name_parts map from the Entity Store aliases.
        
        For each person, all aliases are cross-linked so that ANY alias
        token maps to ALL other alias parts. This is much richer than the
        sender list approach because it handles multi-script aliases directly.
        
        Returns:
            Dict of lowercased-alias → set of all name parts for that person
        """
        try:
            import entity_db
            persons = entity_db.get_all_persons_summary()
            if not persons:
                return {}
            
            import re
            _NUMERIC_RE = re.compile(r"^[\d+\-#().]+$")
            # Short tokens that are common Hebrew words but also match names
            # (e.g. "בן"=son/Ben, "על"=on, "שם"=name, "גן"=garden, "אור"=light)
            _MIN_NAME_PART_LEN = 3  # Skip 1-2 char parts as map keys (too ambiguous)
            
            name_map: Dict[str, set] = {}
            for person in persons:
                # Collect all name parts from canonical name + all aliases
                all_parts: set = set()
                canonical = person.get("canonical_name", "")
                if canonical:
                    all_parts.update(canonical.split())
                
                for alias in person.get("aliases", []):
                    if isinstance(alias, str):
                        all_parts.update(alias.split())
                    elif isinstance(alias, dict):
                        alias_text = alias.get("alias", "")
                        if alias_text:
                            all_parts.update(alias_text.split())
                
                # Filter out phone numbers, group IDs, and other numeric-only parts
                all_parts = {p for p in all_parts if not _NUMERIC_RE.match(p)}
                
                if not all_parts:
                    continue
                
                # Map each part (lowercased) → all parts for that person,
                # but only use parts with enough length as map keys to avoid
                # common short Hebrew words triggering massive expansions.
                for part in list(all_parts):
                    if len(part) >= _MIN_NAME_PART_LEN:
                        name_map.setdefault(part.lower(), set()).update(all_parts)
            
            return name_map
        except Exception:
            return {}
    
    def _build_sender_name_map(self) -> Dict[str, set]:
        """Build a name→name_parts map from sender/chat lists (fallback).
        
        Uses the Redis-cached sender and chat name lists to build a simple
        first-name → full-name-parts mapping.
        
        Returns:
            Dict of lowercased-first-name → set of full name parts
        """
        try:
            contacts = self.get_sender_list()
            if not contacts:
                return {}
            
            chat_names = self.get_chat_list()
            all_names = set(contacts)
            if chat_names:
                all_names.update(chat_names)
            
            import re
            _NUMERIC_RE = re.compile(r"^[\d+\-#().]+$")
            
            name_map: Dict[str, set] = {}
            for contact in all_names:
                parts = contact.split()
                if not parts:
                    continue
                # Filter out phone numbers / numeric-only parts
                parts = [p for p in parts if not _NUMERIC_RE.match(p)]
                if not parts:
                    continue
                first = parts[0].lower()
                all_parts = set(parts)
                # Only use keys with >= 3 chars to avoid common short-word collisions
                if len(first) >= 3:
                    name_map.setdefault(first, set()).update(all_parts)
            
            return name_map
        except Exception:
            return {}
    
    # Field-aware full-text search scores: sender matches are most valuable
    # because users often ask "what did X say about Y?"
    FULLTEXT_SCORE_SENDER = float(settings.get("rag_fulltext_score_sender", "0.95"))
    FULLTEXT_SCORE_CHAT_NAME = float(settings.get("rag_fulltext_score_chat_name", "0.85"))
    FULLTEXT_SCORE_MESSAGE = float(settings.get("rag_fulltext_score_message", "0.75"))
    FULLTEXT_SCORE_NUMBERS = float(settings.get("rag_fulltext_score_numbers", "0.90"))
    
    # Morphological prefixes to strip during fulltext tokenization.
    # Default covers Hebrew prepositions/conjunctions/articles:
    # ה (the), ב (in), ל (to), מ (from), ש (that), כ (like), ו (and).
    # Configurable via settings key 'rag_morphology_prefixes'.
    _MORPHOLOGY_PREFIXES = settings.get("rag_morphology_prefixes", "")
    
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
            
            # Strip morphological prefixes (one or two prefix letters)
            word = token
            for _ in range(2):  # Strip up to 2 prefix letters
                if len(word) > 3 and LlamaIndexRAG._MORPHOLOGY_PREFIXES and word[0] in LlamaIndexRAG._MORPHOLOGY_PREFIXES:
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
        
        Splits on word boundaries and keeps tokens ≥ 2 characters for
        Hebrew (which has many meaningful 2-char words like בן, בת, אב,
        אם, שם, גן) and ≥ 3 characters for Latin/other scripts.  This
        aligns with Qdrant's text index ``min_token_len=2`` configuration.
        
        No hardcoded stop-word lists — Qdrant's ``should`` (OR) filter
        handles the matching, so common words simply produce more
        candidates without hurting precision (RRF ranking takes care of
        relevance).
        
        For Hebrew tokens, also generates morphological variants by
        stripping prefixes and verb conjugation patterns to improve
        recall across different word forms.
        
        Args:
            query: The search query string
            
        Returns:
            Deduplicated list of tokens with Hebrew expansions
        """
        import re as _re
        import unicodedata as _ud
        # Strip Unicode format characters (category Cf: RTL/LTR marks,
        # zero-width joiners, directional overrides, BOM, soft hyphens)
        # that OCR engines insert — these break tokenization and matching.
        clean_query = "".join(
            ch for ch in query if _ud.category(ch) != "Cf"
        )
        # Capture tokens ≥ 2 chars, then keep 2-char tokens only when they
        # contain Hebrew characters.  Hebrew has many critical 2-char words
        # (בן=son, בת=daughter, אב=father, אם=mother, שם=name, גן=garden)
        # that must reach Qdrant's fulltext index (which already uses
        # min_token_len=2).  Non-Hebrew 2-char tokens (is, to, in, …) are
        # filtered out to avoid noise.
        _HE_RE = _re.compile(r'[\u0590-\u05FF]')
        tokens = [
            t for t in _re.findall(r"[\w]{2,}", clean_query, _re.UNICODE)
            if len(t) >= 3 or _HE_RE.search(t)
        ]
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
    
    # Fields that use keyword indexes (array of strings) instead of text indexes.
    # These fields use MatchValue for exact matching rather than MatchText for
    # tokenized text search.
    _KEYWORD_ARRAY_FIELDS = {"numbers"}
    
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
        
        For text-indexed fields (sender, chat_name, message), uses ``MatchText``
        for tokenized full-text matching.
        
        For keyword-indexed array fields (numbers), uses ``MatchValue``
        for exact element matching, which is faster and more precise.
        
        Args:
            field_name: Qdrant payload field to search
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
            # Build OR conditions: match ANY of the tokens in this field.
            # Keyword array fields (e.g., numbers) use exact MatchValue;
            # text-indexed fields use tokenized MatchText.
            if field_name in self._KEYWORD_ARRAY_FIELDS:
                should_conditions = [
                    FieldCondition(key=field_name, match=MatchValue(value=token))
                    for token in tokens
                ]
            else:
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
    
    def _build_filter_conditions(
        self,
        filter_chat_name: Optional[str] = None,
        filter_sender: Optional[str] = None,
        filter_days: Optional[int] = None,
        filter_sources: Optional[List[str]] = None,
        filter_date_from: Optional[str] = None,
        filter_date_to: Optional[str] = None,
        filter_content_types: Optional[List[str]] = None,
    ) -> List:
        """Build a list of Qdrant ``must`` filter conditions from common params.
        
        Centralises the filter-building logic used by search(), _fulltext_search(),
        recency_search(), and _metadata_search() so new filter types only need
        to be added in one place.
        
        Args:
            filter_chat_name: Filter by exact chat/group name
            filter_sender: Filter by exact sender name
            filter_days: Filter by recency in days (alternative to date range)
            filter_sources: Filter by source values (OR logic — any of the listed sources)
            filter_date_from: ISO date string for start of date range (inclusive)
            filter_date_to: ISO date string for end of date range (inclusive, end-of-day)
            filter_content_types: Filter by content type values (OR logic)
            
        Returns:
            List of Qdrant filter conditions (for ``Filter(must=...)``)
        """
        from datetime import timedelta
        
        must_conditions: List = []
        
        if filter_chat_name:
            must_conditions.append(
                FieldCondition(key="chat_name", match=MatchValue(value=filter_chat_name))
            )
        
        if filter_sender:
            must_conditions.append(
                FieldCondition(key="sender", match=MatchValue(value=filter_sender))
            )
        
        # Date range takes precedence over filter_days when both are provided
        if filter_date_from or filter_date_to:
            if filter_date_from:
                try:
                    ts_from = int(datetime.fromisoformat(filter_date_from).timestamp())
                    must_conditions.append(
                        FieldCondition(key="timestamp", range=Range(gte=ts_from))
                    )
                except (ValueError, TypeError):
                    logger.warning(f"Invalid filter_date_from: {filter_date_from}")
            if filter_date_to:
                try:
                    # Add 1 day to include the full end date
                    dt_to = datetime.fromisoformat(filter_date_to) + timedelta(days=1)
                    ts_to = int(dt_to.timestamp())
                    must_conditions.append(
                        FieldCondition(key="timestamp", range=Range(lte=ts_to))
                    )
                except (ValueError, TypeError):
                    logger.warning(f"Invalid filter_date_to: {filter_date_to}")
        elif filter_days is not None and filter_days > 0:
            min_timestamp = int(datetime.now().timestamp()) - (filter_days * 24 * 60 * 60)
            must_conditions.append(
                FieldCondition(key="timestamp", range=Range(gte=min_timestamp))
            )
        
        # Source filter (OR logic: match ANY of the listed sources)
        if filter_sources:
            source_conditions = [
                FieldCondition(key="source", match=MatchValue(value=s))
                for s in filter_sources
            ]
            if len(source_conditions) == 1:
                must_conditions.append(source_conditions[0])
            else:
                must_conditions.append(Filter(should=source_conditions))
        
        # Content type filter (OR logic: match ANY of the listed types)
        if filter_content_types:
            ct_conditions = [
                FieldCondition(key="content_type", match=MatchValue(value=ct))
                for ct in filter_content_types
            ]
            if len(ct_conditions) == 1:
                must_conditions.append(ct_conditions[0])
            else:
                must_conditions.append(Filter(should=ct_conditions))
        
        return must_conditions
    
    def _fulltext_search(
        self,
        query: str,
        k: int = 10,
        filter_chat_name: Optional[str] = None,
        filter_sender: Optional[str] = None,
        filter_days: Optional[int] = None,
        filter_sources: Optional[List[str]] = None,
        filter_date_from: Optional[str] = None,
        filter_date_to: Optional[str] = None,
        filter_content_types: Optional[List[str]] = None,
    ) -> List[NodeWithScore]:
        """Perform field-aware full-text search on metadata fields.
        
        Tokenizes the query into words (≥ 2 chars for Hebrew, ≥ 3 chars
        otherwise) and searches each metadata field using Qdrant ``should``
        (OR) conditions — a document matches if it contains **any** of the
        query tokens in the searched field.  This is language-agnostic and
        requires no hardcoded stop-word lists.
        
        Runs one query per field (sender, chat_name, message) with
        different scores to prioritize sender matches over message
        content matches.  Results are deduplicated by node ID, keeping
        the highest score.
        
        When ``filter_sender`` is provided, the sender field fulltext search
        is skipped (exact match already applied via filter conditions) but
        chat_name and message searches still run.
        
        Args:
            query: Text to search for
            k: Max results to return
            filter_chat_name: Optional chat filter
            filter_sender: Optional sender filter (exact match)
            filter_days: Optional time filter
            filter_sources: Optional source filter
            filter_date_from: Optional ISO date string for start of range
            filter_date_to: Optional ISO date string for end of range
            filter_content_types: Optional content type filter
            
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
            must_conditions = self._build_filter_conditions(
                filter_chat_name=filter_chat_name,
                filter_sender=filter_sender,
                filter_days=filter_days,
                filter_sources=filter_sources,
                filter_date_from=filter_date_from,
                filter_date_to=filter_date_to,
                filter_content_types=filter_content_types,
            )
            
            # Extract numeric tokens (≥5 digits) for dedicated numbers field search
            import re as _re_local
            numeric_tokens = [t for t in tokens if _re_local.fullmatch(r"\d{5,}", t)]
            
            # Cross-script contact name expansion: when a query token matches
            # a known contact's first name (in any script), add the contact's
            # full name parts as additional tokens.  This ensures that a Hebrew
            # query like "שירן" also finds "Shiran Waintrob" in the sender
            # field, and vice versa.  Only used for sender/chat_name fields.
            contact_tokens = self._expand_tokens_with_contact_names(tokens)
            
            # Search each field with OR-matched tokens, different scores.
            # Skip sender field when filter_sender is set — the exact-match
            # filter already constrains results to that sender, so fulltext
            # search on the sender field would be redundant.
            field_searches = [
                ("chat_name", self.FULLTEXT_SCORE_CHAT_NAME),
                ("message", self.FULLTEXT_SCORE_MESSAGE),
            ]
            if not filter_sender:
                field_searches.insert(0, ("sender", self.FULLTEXT_SCORE_SENDER))
            
            # Add numbers field search when query contains numeric sequences
            if numeric_tokens:
                field_searches.append(("numbers", self.FULLTEXT_SCORE_NUMBERS))
            
            # Collect all results, dedup by node ID keeping highest score
            best_scores: Dict[str, float] = {}
            best_nodes: Dict[str, NodeWithScore] = {}
            
            for field_name, field_score in field_searches:
                # Use contact-expanded tokens for sender/chat_name (cross-script),
                # numeric tokens for numbers field, original tokens for message.
                if field_name == "numbers":
                    search_tokens = numeric_tokens
                elif field_name in ("sender", "chat_name"):
                    search_tokens = contact_tokens
                else:
                    search_tokens = tokens
                results = self._fulltext_search_by_field(
                    field_name=field_name,
                    tokens=search_tokens,
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
    
    def _metadata_search(
        self,
        k: int = 20,
        filter_chat_name: Optional[str] = None,
        filter_sender: Optional[str] = None,
        filter_days: Optional[int] = None,
        filter_sources: Optional[List[str]] = None,
        filter_date_from: Optional[str] = None,
        filter_date_to: Optional[str] = None,
        filter_content_types: Optional[List[str]] = None,
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
            filter_sources: Optional source filter
            filter_date_from: Optional ISO date string for start of range
            filter_date_to: Optional ISO date string for end of range
            filter_content_types: Optional content type filter
            
        Returns:
            List of NodeWithScore objects (score=1.0 for all, sorted by timestamp)
        """
        try:
            must_conditions = self._build_filter_conditions(
                filter_chat_name=filter_chat_name,
                filter_sender=filter_sender,
                filter_days=filter_days,
                filter_sources=filter_sources,
                filter_date_from=filter_date_from,
                filter_date_to=filter_date_to,
                filter_content_types=filter_content_types,
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
        filter_sources: Optional[List[str]] = None,
        filter_date_from: Optional[str] = None,
        filter_date_to: Optional[str] = None,
        filter_content_types: Optional[List[str]] = None,
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
            filter_sources: Optional source filter
            filter_date_from: Optional ISO date string for start of range
            filter_date_to: Optional ISO date string for end of range
            filter_content_types: Optional content type filter
            
        Returns:
            List of NodeWithScore ordered by timestamp (most recent first)
        """
        try:
            must_conditions = self._build_filter_conditions(
                filter_chat_name=filter_chat_name,
                filter_sender=filter_sender,
                filter_days=filter_days,
                filter_sources=filter_sources,
                filter_date_from=filter_date_from,
                filter_date_to=filter_date_to,
                filter_content_types=filter_content_types,
            )
            
            # Require a valid timestamp (>0) and exclude conversation chunks —
            # we want individual messages for recency, not synthetic multi-message
            # chunks which would duplicate content.
            must_conditions.append(
                FieldCondition(key="timestamp", range=Range(gt=0))
            )
            
            must_not_conditions = [
                FieldCondition(key="source_type", match=MatchValue(value="conversation_chunk"))
            ]
            
            scroll_filter = Filter(
                must=must_conditions if must_conditions else None,
                must_not=must_not_conditions,
            )
            
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
                    # Store raw timestamp temporarily; we'll normalize below
                    ts = payload.get("timestamp", 0)
                    nodes.append(NodeWithScore(node=node, score=float(ts) if ts else 0.0))
            
            # ---------------------------------------------------------------
            # Normalize recency scores to the 0.3–0.5 band using linear
            # interpolation.  This replaces the previous raw-timestamp
            # scoring which produced values like 1708012345.0 that broke
            # any downstream score comparison (e.g. SimilarityPostprocessor
            # cutoffs, Cohere rerank score merging).
            #
            # Most recent message → 0.5, oldest in batch → 0.3.
            # These scores sit *below* typical dense-vector cosine
            # similarities (~0.5–0.9) and fulltext heuristic scores
            # (0.75–0.95), so recency supplements don't outrank semantic
            # matches, but above the minimum score threshold (default 0.2)
            # so they aren't filtered out.
            # ---------------------------------------------------------------
            if nodes:
                raw_scores = [n.score for n in nodes if n.score is not None]
                if raw_scores:
                    max_ts = max(raw_scores)
                    min_ts = min(raw_scores)
                    ts_range = max_ts - min_ts if max_ts != min_ts else 1.0
                    for nws in nodes:
                        if nws.score is not None and nws.score > 0:
                            # Linear interpolation: newest=0.5, oldest=0.3
                            nws.score = 0.3 + 0.2 * ((nws.score - min_ts) / ts_range)
                        else:
                            nws.score = 0.3  # Default for missing timestamps
            
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
            # Collect unique chat identifiers and timestamps from results.
            # Prefer chat_id (unique, e.g. '972501234567@c.us') over chat_name
            # (display name, can be duplicated across different chats like "Family").
            # Falls back to chat_name for sources without chat_id (e.g. Paperless).
            chat_windows: Dict[str, Dict] = {}  # key -> {timestamps, filter_field, filter_value}
            existing_ids: set = set()
            
            for nws in results:
                node = nws.node
                if not node:
                    continue
                existing_ids.add(node.id_)
                metadata = getattr(node, "metadata", {})
                chat_id = metadata.get("chat_id")
                chat_name = metadata.get("chat_name")
                timestamp = metadata.get("timestamp")
                if not timestamp or not isinstance(timestamp, (int, float)):
                    continue
                # Use chat_id when available (WhatsApp), fall back to chat_name (Paperless)
                if chat_id:
                    key = f"id:{chat_id}"
                    chat_windows.setdefault(key, {
                        "timestamps": [], "filter_field": "chat_id", "filter_value": chat_id
                    })["timestamps"].append(int(timestamp))
                elif chat_name:
                    key = f"name:{chat_name}"
                    chat_windows.setdefault(key, {
                        "timestamps": [], "filter_field": "chat_name", "filter_value": chat_name
                    })["timestamps"].append(int(timestamp))
            
            if not chat_windows:
                return results
            
            # For each chat, fetch messages in a time window around the matches
            expanded_nodes: List[NodeWithScore] = []
            budget = max_total - len(results)  # How many more nodes we can add
            
            if budget <= 0:
                return results
            
            per_chat_limit = max(3, budget // len(chat_windows))
            
            for chat_key, window_info in chat_windows.items():
                timestamps = window_info["timestamps"]
                filter_field = window_info["filter_field"]
                filter_value = window_info["filter_value"]
                min_ts = min(timestamps) - self.CONTEXT_WINDOW_SECONDS
                max_ts = max(timestamps) + self.CONTEXT_WINDOW_SECONDS
                
                must_conditions = [
                    FieldCondition(key=filter_field, match=MatchValue(value=filter_value)),
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
                    logger.debug(f"Context expansion for chat '{chat_key}' failed: {e}")
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
    
    # Sources whose multi-chunk documents should be expanded to include
    # all sibling chunks when any single chunk matches.  Each entry maps
    # the ``source`` payload value to the expected ``source_id`` prefix.
    _EXPANDABLE_SOURCES: Dict[str, str] = {
        "paperless": "paperless:",
        "call_recording": "call_recording:",
    }
    
    def expand_document_chunks(
        self,
        results: List[NodeWithScore],
        max_total: int = 30,
    ) -> List[NodeWithScore]:
        """Expand results by fetching ALL sibling chunks from matched multi-chunk documents.
        
        When a multi-chunk document (Paperless document or call recording
        transcript) is found in search results, this method fetches all
        other chunks from the same document using the source_id.  This
        ensures the LLM sees the complete content — not just the chunk
        that happened to match the query semantically.
        
        For example:
        - A custody clause chunk → also fetches the children's names chunk
        - A call recording chunk about weather → also fetches the chunk
          that mentions "princess training" later in the same call
        
        Applies to multi-chunk Paperless documents and call recordings.
        WhatsApp messages and single-chunk documents are unaffected.
        
        Args:
            results: Original search results (may include multi-chunk documents)
            max_total: Maximum total nodes to return after expansion
            
        Returns:
            Merged list of original results + sibling document chunks, deduplicated
        """
        if not results:
            return results
        
        try:
            # Collect unique document IDs from results for expandable sources
            # source_id formats:
            #   Paperless:       "paperless:{doc_id}"
            #   Call recordings:  "call_recording:{content_hash}"
            doc_ids: set = set()
            existing_ids: set = set()
            
            for nws in results:
                node = nws.node
                if not node:
                    continue
                existing_ids.add(node.id_)
                metadata = getattr(node, "metadata", {})
                source = metadata.get("source", "")
                source_id = metadata.get("source_id", "")
                # Check if this source type supports chunk expansion
                expected_prefix = self._EXPANDABLE_SOURCES.get(source)
                if expected_prefix and source_id.startswith(expected_prefix):
                    doc_id = source_id
                    # Check if this document has multiple chunks
                    chunk_total = metadata.get("chunk_total")
                    if chunk_total and int(chunk_total) > 1:
                        doc_ids.add(doc_id)
            
            if not doc_ids:
                return results
            
            budget = max_total - len(results)
            if budget <= 0:
                return results
            
            # Fetch sibling chunks for each document.
            # Use chunk_total from the matched chunk's metadata to determine
            # the scroll limit, ensuring we fetch ALL sibling chunks rather
            # than a budget-derived subset.  The downstream context budget
            # trimmer in _retrieve() will handle any overflow.
            sibling_nodes: List[NodeWithScore] = []
            
            # Build a map of doc_source_id -> chunk_total from matched results
            doc_chunk_totals: Dict[str, int] = {}
            for nws in results:
                node = nws.node
                if not node:
                    continue
                metadata = getattr(node, "metadata", {})
                source_id = metadata.get("source_id", "")
                chunk_total = metadata.get("chunk_total")
                if source_id in doc_ids and chunk_total:
                    try:
                        doc_chunk_totals[source_id] = max(
                            doc_chunk_totals.get(source_id, 0),
                            int(chunk_total)
                        )
                    except (ValueError, TypeError):
                        pass
            
            for doc_source_id in doc_ids:
                # Use chunk_total if known, otherwise fall back to budget-based limit
                fallback_limit = max(5, budget // len(doc_ids))
                doc_limit = doc_chunk_totals.get(doc_source_id, fallback_limit)
                
                try:
                    records, _ = self.qdrant_client.scroll(
                        collection_name=self.COLLECTION_NAME,
                        scroll_filter=Filter(must=[
                            FieldCondition(
                                key="source_id",
                                match=MatchValue(value=doc_source_id)
                            )
                        ]),
                        limit=doc_limit,
                        with_payload=True,
                        with_vectors=False,
                    )
                    
                    for record in records:
                        record_id = str(record.id)
                        if record_id in existing_ids:
                            continue  # Skip chunks already in results
                        existing_ids.add(record_id)
                        
                        payload = record.payload or {}
                        text = self._extract_text_from_payload(payload)
                        if text:
                            node = TextNode(
                                text=text,
                                metadata={
                                    mk: mv for mk, mv in payload.items()
                                    if not mk.startswith("_")
                                },
                                id_=record_id,
                            )
                            # Score slightly below original results
                            sibling_nodes.append(
                                NodeWithScore(node=node, score=0.45)
                            )
                            
                except Exception as e:
                    logger.debug(
                        f"Document chunk expansion for '{doc_source_id}' failed: {e}"
                    )
                    continue
            
            if sibling_nodes:
                logger.info(
                    f"Document chunk expansion added {len(sibling_nodes)} sibling "
                    f"chunks from {len(doc_ids)} document(s)"
                )
                results = results + sibling_nodes
            
            return results[:max_total]
            
        except Exception as e:
            logger.debug(f"Document chunk expansion failed (non-critical): {e}")
            return results
    
    def search(
        self,
        query: str,
        k: int = 10,
        filter_chat_name: Optional[str] = None,
        filter_sender: Optional[str] = None,
        filter_days: Optional[int] = None,
        filter_sources: Optional[List[str]] = None,
        filter_date_from: Optional[str] = None,
        filter_date_to: Optional[str] = None,
        filter_content_types: Optional[List[str]] = None,
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
            filter_sources: Optional source filter (list of source values)
            filter_date_from: Optional ISO date string for start of range
            filter_date_to: Optional ISO date string for end of range
            filter_content_types: Optional content type filter
            include_metadata_search: Include full-text search on metadata fields
            metadata_only: Skip vector search, use only metadata filters
            
        Returns:
            List of NodeWithScore objects with metadata
        """
        try:
            _fkw = dict(
                filter_chat_name=filter_chat_name,
                filter_sender=filter_sender,
                filter_days=filter_days,
                filter_sources=filter_sources,
                filter_date_from=filter_date_from,
                filter_date_to=filter_date_to,
                filter_content_types=filter_content_types,
            )
            
            # Metadata-only search: skip vector search entirely
            if metadata_only:
                return self._metadata_search(k=k, **_fkw)
            
            # =================================================================
            # Hybrid search (dense + sparse RRF fusion) — highest priority
            # When rag_hybrid_enabled=true and the collection has sparse
            # vectors, this replaces both vector search and fulltext search
            # with a single Qdrant query that fuses both signal types.
            # =================================================================
            if self.HYBRID_ENABLED:
                must_conditions = self._build_filter_conditions(**_fkw)
                qdrant_filters = Filter(must=must_conditions) if must_conditions else None
                hybrid_results = self._hybrid_search(
                    query=query,
                    k=k,
                    qdrant_filters=qdrant_filters,
                )
                if hybrid_results:
                    logger.info(
                        f"Hybrid search for '{query[:50]}...' returned "
                        f"{len(hybrid_results)} results"
                    )
                    return hybrid_results
                # If hybrid search returns empty, fall through to standard paths
                logger.debug("Hybrid search returned no results, falling back to standard")
            
            # =================================================================
            # Phase 4: QueryFusionRetriever (optional)
            # When enabled, replaces the manual RRF merge below with
            # LlamaIndex's built-in QueryFusionRetriever, which also
            # generates multiple query variants for better recall.
            # =================================================================
            try:
                from llama_index.core.retrievers import QueryFusionRetriever
                
                self._ensure_llm_configured()
                must_conditions = self._build_filter_conditions(**_fkw)
                qdrant_filters = Filter(must=must_conditions) if must_conditions else None
                
                vector_ret = VectorOnlyRetriever(rag=self, k=k, qdrant_filter=qdrant_filters)
                fulltext_ret = FulltextOnlyRetriever(rag=self, k=k, filter_kwargs=_fkw)
                
                num_queries = int(settings.get("rag_query_fusion_num_queries", "3"))
                fusion = QueryFusionRetriever(
                    retrievers=[vector_ret, fulltext_ret],
                    similarity_top_k=k,
                    num_queries=num_queries,
                    mode="reciprocal_rerank",
                    llm=Settings.llm,
                )
                fusion_results = fusion.retrieve(query)
                logger.info(
                    f"QueryFusionRetriever ({num_queries} query variants): "
                    f"{len(fusion_results)} results for '{query[:50]}...'"
                )
                return fusion_results
            except ImportError:
                logger.debug("QueryFusionRetriever not available — using vector-only fallback")
            except Exception as e:
                logger.debug(f"QueryFusionRetriever failed, falling back to vector-only: {e}")
            
            # Build Qdrant filter conditions
            must_conditions = self._build_filter_conditions(**_fkw)
            
            qdrant_filters = Filter(must=must_conditions) if must_conditions else None
            
            # =================================================================
            # Phase 3: HyDE query transform (optional)
            # Generates a hypothetical answer to the query, then uses that
            # answer's embedding for retrieval. Language-agnostic — handles
            # Hebrew morphology naturally because the LLM generates the answer.
            # =================================================================
            hyde_embedding = None
            try:
                self._ensure_llm_configured()
                from llama_index.core.indices.query.query_transform import HyDEQueryTransform
                
                hyde = HyDEQueryTransform(llm=Settings.llm, include_original=True)
                hyde_bundle = hyde.run(query)
                # Use the HyDE-generated embedding if available
                if hasattr(hyde_bundle, 'embedding') and hyde_bundle.embedding:
                    hyde_embedding = hyde_bundle.embedding
                    logger.info(f"HyDE generated hypothetical answer for query: {query[:50]}...")
                elif hasattr(hyde_bundle, 'query_str') and hyde_bundle.query_str:
                    # Embed the hypothetical document text
                    hyde_embedding = Settings.embed_model.get_query_embedding(hyde_bundle.query_str)
                    logger.info(f"HyDE transformed query: {hyde_bundle.query_str[:80]}...")
            except ImportError:
                logger.debug("HyDE query transform not available")
            except Exception as e:
                logger.debug(f"HyDE query transform failed (non-critical): {e}")
            
            # Use direct Qdrant search to avoid LlamaIndex TextNode validation issues
            # with documents that have None text values.
            # Fetch more candidates (k * 2) to compensate for score-threshold filtering,
            # especially for morphologically rich languages like Hebrew where semantic
            # similarity may be lower for inflected query forms.
            query_embedding = hyde_embedding if hyde_embedding is not None else Settings.embed_model.get_query_embedding(query)
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
            
            # Score filtering and hybrid fulltext+vector fusion are now
            # handled by QueryFusionRetriever (primary path above).
            # Minimum similarity filtering is applied inside VectorOnlyRetriever
            # via MINIMUM_SIMILARITY_SCORE.
            # This vector-only path is the fallback when QueryFusion fails.
            
            logger.info(f"RAG search for '{query[:50]}...' returned {len(valid_results)} results (vector-only fallback)")
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
        """Build the system prompt with current date/time and known contacts.
        
        Injects:
        - Current date/time ({current_datetime}, {hebrew_date})
        - Dynamic list of all known sender/contact names from the archive
          so the LLM can disambiguate when a name matches multiple people
        
        Returns:
            System prompt string with dynamic date and contact list injection
        """
        timezone = settings.get("timezone", "Asia/Jerusalem")
        tz = ZoneInfo(timezone)
        now = datetime.now(tz)
        current_datetime = now.strftime("%A, %B %d, %Y at %H:%M")
        # Build a locale-aware local date string.  Uses the OS locale
        # (e.g. ``he_IL``) when available so day names are not hardcoded.
        # Falls back to a numeric-only format that works for any language.
        try:
            import locale as _locale
            saved = _locale.getlocale(_locale.LC_TIME)
            try:
                _locale.setlocale(_locale.LC_TIME, "")  # Use system locale
                local_day = now.strftime("%A")           # Localized day name
            finally:
                _locale.setlocale(_locale.LC_TIME, saved)
        except Exception:
            local_day = now.strftime("%A")  # Fallback to English day name
        hebrew_date = f"{local_day}, {now.day}/{now.month}/{now.year} {now.strftime('%H:%M')}"
        
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
                "8. Be concise but thorough. Prefer specific facts over vague summaries.\n"
                "9. DISAMBIGUATION: When the user mentions a person's name (first name only) "
                "that matches multiple people in the known contacts list below, ASK the user "
                "to clarify which person they mean BEFORE answering. Present the matching "
                "names as numbered options. For example: 'I found multiple people named Doron: "
                "1) Doron Yazkirovich 2) דורון עלאני — which one did you mean?' "
                "Note: names may appear in different languages/scripts (Hebrew and English) "
                "but refer to the same first name (e.g., דורון = Doron, דוד = David). "
                "Only ask if there is genuine ambiguity — if the user provided a full name "
                "or enough context to identify the person, answer directly."
            )
        
        prompt = prompt_template.format(
            current_datetime=current_datetime,
            hebrew_date=hebrew_date,
        )
        
        # Append the dynamic known contacts list and disambiguation instruction.
        # Strategy: use Entity Store (rich aliases) when available, fall back
        # to the flat Redis-cached sender list.
        try:
            contacts_str = ""
            contact_count = 0
            
            # Strategy 1: Entity Store with aliases (richer disambiguation)
            try:
                import entity_db
                persons = entity_db.get_all_persons_summary()
                if persons:
                    contact_entries = []
                    for p in persons:
                        if p.get("is_group"):
                            continue
                        name = p["canonical_name"]
                        aliases = [
                            a for a in p.get("aliases", [])
                            if isinstance(a, str) and a != name
                        ]
                        if aliases:
                            alias_str = "/".join(aliases[:3])
                            contact_entries.append(f"{name} ({alias_str})")
                        else:
                            contact_entries.append(name)
                    contacts_str = ", ".join(contact_entries)
                    contact_count = len(contact_entries)
            except Exception:
                pass  # Fall back to sender list
            
            # Strategy 2: Flat sender list from Redis (fallback)
            if not contacts_str:
                sender_list = self.get_sender_list()
                if sender_list:
                    contacts_str = ", ".join(sender_list)
                    contact_count = len(sender_list)
            
            # Cap contacts string to ~6000 chars (~1500 tokens) to prevent
            # the system prompt from exceeding LLM token limits.
            MAX_CONTACTS_CHARS = 6000
            if len(contacts_str) > MAX_CONTACTS_CHARS:
                # Truncate at the last complete entry before the limit
                truncated = contacts_str[:MAX_CONTACTS_CHARS]
                last_comma = truncated.rfind(", ")
                if last_comma > 0:
                    contacts_str = truncated[:last_comma]
                else:
                    contacts_str = truncated
                contacts_str += f" ... (and more, {contact_count} total)"
            
            if contacts_str:
                prompt += (
                    f"\n\nKnown Contacts ({contact_count} people):\n"
                    f"{contacts_str}\n\n"
                    "CRITICAL DISAMBIGUATION RULE (MUST FOLLOW):\n"
                    "BEFORE answering any question that mentions a person by first name only, "
                    "you MUST scan the Known Contacts list above and check if that first name "
                    "matches MORE THAN ONE person. Names can appear in different scripts — "
                    "Hebrew and English versions of the same name count as matches "
                    "(e.g., דורון = Doron, דוד = David, דנה = Dana, ליאור = Lior).\n\n"
                    "If multiple contacts share the same first name:\n"
                    "- Do NOT answer the question yet\n"
                    "- Do NOT say 'no results found'\n"
                    "- Instead, IMMEDIATELY ask the user to clarify with numbered options\n"
                    "- Include ALL matching contacts from the Known Contacts list, "
                    "even if they don't appear in the retrieved messages\n\n"
                    "Example — user asks 'מה דורון שאל אותי?' and contacts include "
                    "'Doron Yazkirovich' and 'דורון עלאני':\n"
                    "✅ Correct response: 'מצאתי כמה אנשים בשם דורון:\n"
                    "1) Doron Yazkirovich\n"
                    "2) דורון עלאני\n"
                    "לאיזה דורון התכוונת?'\n\n"
                    "❌ Wrong: answering or saying no results without asking first.\n\n"
                    "Only skip disambiguation if the user provided a full name (first + last) "
                    "or enough context to uniquely identify the person."
                )
                logger.debug(f"Injected {contact_count} contacts + disambiguation rule into system prompt")
        except Exception as e:
            logger.debug(f"Failed to inject contacts into system prompt (non-fatal): {e}")
        
        # Append calendar event creation instructions.
        # When the user asks to create a meeting/event/reminder, the LLM outputs
        # a structured [CREATE_EVENT] block that the RichResponseProcessor will
        # parse into a downloadable .ics file.
        prompt += (
            "\n\nCALENDAR EVENT CREATION:\n"
            "When the user asks you to create a calendar event, meeting, appointment, "
            "or reminder, you MUST include a structured event block in your response "
            "using this EXACT format:\n\n"
            "[CREATE_EVENT]\n"
            "title: <event title>\n"
            "start: <YYYY-MM-DDTHH:MM>\n"
            "end: <YYYY-MM-DDTHH:MM>\n"
            "location: <location, optional>\n"
            "description: <description, optional>\n"
            "[/CREATE_EVENT]\n\n"
            "Rules for calendar events:\n"
            "- Use ISO 8601 format for dates (YYYY-MM-DDTHH:MM)\n"
            "- If no end time is specified, omit the 'end' line (default: 1 hour)\n"
            "- If no location is specified, omit the 'location' line\n"
            "- Calculate the actual date from relative expressions like 'tomorrow', "
            "'next Tuesday', 'in 3 days' using the current date/time above\n"
            "- The event block will be automatically converted to an ICS file for download\n"
            "- ALSO write a brief human-readable confirmation message outside the block\n"
            "- Example: User says 'צור לי פגישה עם דוד מחר ב-10 בבוקר'\n"
            "  Response: 'יצרתי לך אירוע ביומן:\n"
            "  📅 פגישה עם דוד — מחר ב-10:00\n\n"
            "  [CREATE_EVENT]\n"
            "  title: פגישה עם דוד\n"
            "  start: 2026-02-16T10:00\n"
            "  end: 2026-02-16T11:00\n"
            "  [/CREATE_EVENT]'"
        )
        
        # Image display instructions — the RichResponseProcessor automatically
        # extracts and displays images from source nodes, so the LLM should
        # never claim it cannot show images.
        prompt += (
            "\n\nIMAGE DISPLAY:\n"
            "When retrieved messages contain images (shown as '[Image: description]' "
            "in the context), the system AUTOMATICALLY displays those images inline "
            "in the chat UI below your response.\n"
            "You do NOT need to display images yourself — they are handled by the system.\n"
            "Your role when images are present:\n"
            "- Describe or discuss the image content as requested by the user\n"
            "- Reference which image you're discussing (sender, chat, date)\n"
            "- NEVER say 'I cannot display images' or 'I can't show images directly' — "
            "the images ARE displayed automatically below your response\n"
            "- If the user asks to 'show' an image, describe it and confirm "
            "it is displayed below your message\n"
        )
        
        return prompt
    
    def create_chat_engine(
        self,
        conversation_id: str,
        filter_chat_name: Optional[str] = None,
        filter_sender: Optional[str] = None,
        filter_days: Optional[int] = None,
        filter_sources: Optional[List[str]] = None,
        filter_date_from: Optional[str] = None,
        filter_date_to: Optional[str] = None,
        filter_content_types: Optional[List[str]] = None,
        sort_order: str = "relevance",
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
            filter_sources: Optional list of source values to filter by
            filter_date_from: Optional ISO date string for start of range
            filter_date_to: Optional ISO date string for end of range
            filter_content_types: Optional list of content type values
            sort_order: "relevance" (default) or "newest"
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
            filter_sources=filter_sources,
            filter_date_from=filter_date_from,
            filter_date_to=filter_date_to,
            filter_content_types=filter_content_types,
            sort_order=sort_order,
        )
        
        # Build system prompt with current datetime
        system_prompt = self._build_system_prompt()
        
        # Context prompt template — handles both populated and empty context.
        # Explicitly instructs the LLM to leverage chat history for follow-up
        # questions, so it doesn't say "no results" when the answer was already
        # provided in a previous turn.
        context_prompt = (
            "Here are the relevant messages and documents from the archive:\n"
            "-----\n"
            "{context_str}\n"
            "-----\n"
            "UNDERSTANDING THE RETRIEVED ITEMS:\n"
            "Each item above is formatted with a source label at the start. "
            "The formats you will see:\n"
            "- **WhatsApp messages**: 'WhatsApp | date | sender in chat: message'\n"
            "- **WhatsApp conversations**: 'WhatsApp Conversation | chat | date_range (N messages):' "
            "followed by multiple [timestamp] sender: message lines — these show a sequence of "
            "messages from the same chat\n"
            "- **Paperless documents**: 'Paperless | date | Document title:' followed by document text\n"
            "- **Gmail emails**: 'Gmail | date | sender in subject: body'\n"
            "- **Entity Store facts**: 'Entity Store | Person: name:' followed by known facts about "
            "a person — these are verified facts from the knowledge base, cite them as 'known facts'\n"
            "- **Other sources**: 'Source | date | sender in context: content'\n\n"
            "CITATION RULES:\n"
            "- When citing information, mention the SOURCE TYPE and approximate date "
            "so the user can verify. Examples:\n"
            "  - 'According to a WhatsApp message from David on 15/01/2024...'\n"
            "  - 'A Paperless document titled \"Divorce Agreement\" from 03/2023 states...'\n"
            "  - 'In an email from Sarah on 10/02/2024...'\n"
            "  - 'Known facts show that David was born on 1990-01-15...'\n"
            "- For conversation chunks with multiple messages, cite the relevant speaker(s)\n"
            "- If multiple sources provide the same information, cite the most "
            "specific or most recent one.\n"
            "- Only cite a source when you are explicitly referencing information "
            "from it — do not add citations to general knowledge.\n\n"
            "IMPORTANT: Use BOTH the retrieved messages above AND the chat history "
            "to answer the user's question. If the retrieved messages don't contain "
            "new relevant information but you already discussed the topic in previous "
            "turns, use that prior context to answer — do NOT say 'no results found' "
            "when you already have the information from earlier in the conversation.\n"
            "When documents mention related concepts (e.g., 'minors'/'קטינים' implies "
            "children exist, a 'divorce agreement'/'הסכם גירושין' implies the parties "
            "were married), EXTRACT and REPORT that implicit information rather than "
            "saying no results were found. Partial answers are better than no answer.\n"
            "When the user asks for suggestions or advice about a person, base your "
            "recommendations on the SPECIFIC situation described in the retrieved messages "
            "and conversation history. Do not give generic advice — tailor it to what the "
            "person is actually going through as evidenced by the messages (e.g., if "
            "messages show anxiety about a medical procedure, give procedure-specific advice, "
            "not generic stress management tips).\n"
            "Only say no relevant messages were found if BOTH the retrieved context "
            "AND the chat history lack the information needed to answer."
        )
        
        # Custom condense prompt for bilingual (Hebrew/English) query rewriting.
        # Handles TWO cases:
        # 1. Follow-up questions: incorporates chat history context
        # 2. First messages (no history): rewrites short/ambiguous queries into
        #    explicit, search-friendly form so the retriever can find relevant
        #    documents.  E.g. "בן כמה בן פיקל?" → more explicit query about
        #    the age/birth date of Pikel's son.
        condense_prompt = (
            "Given the following conversation between a user and an assistant, "
            "and a new message from the user, rewrite it into a standalone "
            "search query that a knowledge-base retriever can use to find the "
            "most relevant documents.\n\n"
            "IMPORTANT RULES:\n"
            "- If the message references something from a previous turn "
            "(like 'and his?', 'what about her?', 'ושל X?', 'מה לגבי Y?'), "
            "you MUST include the full context in the standalone question.\n"
            "- Preserve the language of the message (Hebrew stays Hebrew, "
            "English stays English).\n"
            "- CRITICAL: Always keep person names EXACTLY as the user typed them, "
            "in the SAME SCRIPT (Hebrew/English/Latin). Do NOT transliterate names "
            "into another script. If the user writes 'Doron Yazkirovich' in English, "
            "keep it in English even if the rest of the conversation is in Hebrew. "
            "If the user writes 'דורון עלאני' in Hebrew, keep it in Hebrew.\n"
            "- Include names, IDs, topics, and other specifics from the chat "
            "history that are needed to understand the standalone question.\n"
            "- If the message asks about a different person/entity but the "
            "same topic, include the topic in the rewritten question.\n"
            "- If the user responds with just a person's name (disambiguation), "
            "combine it with the original question from chat history to form a "
            "complete standalone query. Keep the name in its original script.\n"
            "- NUMBERED DISAMBIGUATION: When the user responds with JUST A NUMBER "
            "(like '1', '2', '3') to a disambiguation question in the chat history, "
            "map that number to the corresponding person name from the numbered "
            "options list in the assistant's previous message, then combine it with "
            "the ORIGINAL question to form a complete standalone query. Use the name "
            "EXACTLY as it appeared in the numbered options (same script).\n"
            "- When a follow-up question references someone discussed in previous "
            "turns (e.g., 'How old is she?', 'בת כמה שירן?'), ALWAYS include the "
            "person's FULL NAME (as established in the conversation) in the rewritten "
            "query to maximize search effectiveness.\n"
            "- If there is NO chat history, rewrite the query to be MORE "
            "EXPLICIT and DETAILED for search. Expand shorthand, resolve "
            "ambiguity, and add related terms the user is implicitly asking "
            "about.\n"
            "- NEVER answer the question — only rewrite it.\n\n"
            "Examples:\n"
            "- Chat: 'What is David's ID?' → 'His ID is 038041612'\n"
            "  Follow-up: 'And Mia's?' → 'What is Mia's ID number?'\n"
            "- Chat: 'מה התעודת זהות של דוד?' → 'תעודת הזהות של דוד היא 038041612'\n"
            "  Follow-up: 'ושל בן?' → 'מה מספר תעודת הזהות של בן?'\n"
            "- Chat: 'מה התעודת זהות של דוד?' → '038041612'\n"
            "  Follow-up: 'מה תאריך הלידה שלו?' → 'מה תאריך הלידה של דוד פיקל?'\n"
            "- Chat: 'מה דורון שאל אותי?' → 'Which Doron? 1) Doron Yazkirovich 2) דורון עלאני'\n"
            "  Follow-up: 'Doron Yazkirovich' → 'What did Doron Yazkirovich ask me?'\n"
            "- Chat: 'מה דורון שאל אותי?' → 'לאיזה דורון? 1) Doron Yazkirovich 2) דורון עלאני'\n"
            "  Follow-up: 'דורון עלאני' → 'מה דורון עלאני שאל אותי?'\n"
            "- Chat: 'מה שירן עוברת?' → 'מצאתי: 1) Shiran Waintrob. לאיזה שירן?'\n"
            "  Follow-up: '1' → 'מה Shiran Waintrob עוברת? What is Shiran Waintrob going through?'\n"
            "- Chat: 'What did Doron ask me?' → 'Multiple: 1) Doron Yazkirovich 2) דורון עלאני'\n"
            "  Follow-up: '2' → 'What did דורון עלאני ask me?'\n"
            "- Chat: discussed Shiran Waintrob's stress about an upcoming event\n"
            "  Follow-up: 'בת כמה שירן?' → 'מה הגיל או תאריך הלידה של Shiran Waintrob?'\n"
            "- Chat: discussed Shiran Waintrob's situation\n"
            "  Follow-up: 'מה אתה מציע לה?' → 'What advice for Shiran Waintrob regarding her situation?'\n"
            "- No history. Message: 'בן כמה בן פיקל?'\n"
            "  → 'מה הגיל או תאריך הלידה של הבן של פיקל?'\n"
            "- No history. Message: 'How old is David?'\n"
            "  → 'What is the age or birth date of David?'\n\n"
            "Chat History:\n{chat_history}\n\n"
            "Follow Up Message: {question}\n"
            "Standalone Question:"
        )
        
        verbose = settings.get("log_level", "INFO") == "DEBUG"
        
        # Always use CondensePlusContextChatEngine — the condense step
        # rewrites queries into more explicit, search-friendly form even
        # for first messages (no history), improving retrieval for short
        # or ambiguous Hebrew queries.
        engine = CondensePlusContextChatEngine.from_defaults(
            retriever=retriever,
            memory=memory,
            llm=Settings.llm,
            system_prompt=system_prompt,
            context_prompt=context_prompt,
            condense_prompt=condense_prompt,
            verbose=verbose,
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
            for source_value in ("whatsapp", "paperless", "gmail", "call_recording"):
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
                "gmail_emails": source_counts.get("gmail", 0),
                "call_recordings": source_counts.get("call_recording", 0),
                "source_counts": source_counts,
                "qdrant_server": f"{self.qdrant_host}:{self.qdrant_port}",
                "collection_name": self.COLLECTION_NAME,
                "dashboard_url": f"http://{self.qdrant_host}:{self.qdrant_port}/dashboard"
            }
        except Exception as e:
            logger.error(f"Failed to get RAG stats: {e}")
            return {"error": str(e)}
    
    def create_snapshot(self) -> Optional[Dict[str, Any]]:
        """Create a Qdrant collection snapshot for backup.
        
        Snapshots capture the full collection state (vectors, payloads,
        indexes) and can be used to restore the collection on the same
        or a different Qdrant instance.
        
        Returns:
            Dict with snapshot metadata (name, creation_time, size),
            or None if snapshot creation failed
        """
        try:
            snapshot_info = self.qdrant_client.create_snapshot(
                collection_name=self.COLLECTION_NAME
            )
            
            # Get collection stats for context
            stats = self.get_stats()
            total_points = stats.get("total_documents", "unknown")
            
            result = {
                "snapshot_name": getattr(snapshot_info, "name", str(snapshot_info)),
                "collection": self.COLLECTION_NAME,
                "points_count": total_points,
                "created_at": datetime.now().isoformat(),
            }
            
            logger.info(
                f"Created Qdrant snapshot: {result['snapshot_name']} "
                f"({total_points} points)"
            )
            return result
            
        except Exception as e:
            logger.error(f"Failed to create snapshot: {e}")
            return None
    
    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List all available snapshots for the collection.
        
        Returns:
            List of snapshot info dicts
        """
        try:
            snapshots = self.qdrant_client.list_snapshots(
                collection_name=self.COLLECTION_NAME
            )
            return [
                {
                    "name": getattr(s, "name", str(s)),
                    "creation_time": getattr(s, "creation_time", None),
                    "size": getattr(s, "size", None),
                }
                for s in snapshots
            ]
        except Exception as e:
            logger.error(f"Failed to list snapshots: {e}")
            return []
    
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
            LlamaIndexRAG._ingestion_pipeline = None
            
            # Recreate collection with current VECTOR_SIZE (+ sparse if hybrid enabled)
            if self.HYBRID_ENABLED:
                self.qdrant_client.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config={
                        self.DENSE_VECTOR_NAME: VectorParams(
                            size=self.VECTOR_SIZE,
                            distance=Distance.COSINE,
                        ),
                    },
                    sparse_vectors_config={
                        self.SPARSE_VECTOR_NAME: SparseVectorParams(),
                    },
                )
                logger.info(
                    f"Recreated HYBRID collection: {self.COLLECTION_NAME} "
                    f"(dense={self.VECTOR_SIZE}d + sparse)"
                )
            else:
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


# =========================================================================
# Phase 4: Helper retrievers for QueryFusionRetriever
# =========================================================================

class VectorOnlyRetriever(BaseRetriever):
    """Retriever that performs only vector similarity search via Qdrant.
    
    Used as one of the sub-retrievers for QueryFusionRetriever.
    Delegates to LlamaIndexRAG's direct Qdrant query logic.
    """
    
    def __init__(self, rag: "LlamaIndexRAG", k: int = 10, qdrant_filter=None, **kwargs):
        super().__init__(**kwargs)
        self._rag = rag
        self._k = k
        self._qdrant_filter = qdrant_filter
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Run vector similarity search only."""
        try:
            query_embedding = Settings.embed_model.get_query_embedding(query_bundle.query_str)
            
            query_kwargs = {
                "collection_name": self._rag.COLLECTION_NAME,
                "query": query_embedding,
                "query_filter": self._qdrant_filter,
                "limit": self._k * 2,
                "with_payload": True,
            }
            # When the collection uses named vectors (hybrid mode), we must
            # specify which vector to query against; otherwise Qdrant returns
            # 400 "Not existing vector name".
            if self._rag.HYBRID_ENABLED:
                query_kwargs["using"] = self._rag.DENSE_VECTOR_NAME
            search_results = self._rag.qdrant_client.query_points(**query_kwargs).points
            
            results = []
            for result in search_results:
                payload = result.payload or {}
                text = self._rag._extract_text_from_payload(payload)
                if text:
                    node = TextNode(
                        text=text,
                        metadata={mk: mv for mk, mv in payload.items() if not mk.startswith("_")},
                        id_=str(result.id),
                    )
                    results.append(NodeWithScore(node=node, score=result.score))
            
            # Apply minimum similarity score
            results = [
                r for r in results
                if r.score is not None and r.score >= self._rag.MINIMUM_SIMILARITY_SCORE
            ]
            return results
        except Exception as e:
            logger.error(f"VectorOnlyRetriever failed: {e}")
            return []


class FulltextOnlyRetriever(BaseRetriever):
    """Retriever that performs only full-text search on metadata fields.
    
    Used as one of the sub-retrievers for QueryFusionRetriever.
    Delegates to LlamaIndexRAG._fulltext_search().
    """
    
    def __init__(self, rag: "LlamaIndexRAG", k: int = 10, filter_kwargs: Optional[Dict] = None, **kwargs):
        super().__init__(**kwargs)
        self._rag = rag
        self._k = k
        self._filter_kwargs = filter_kwargs or {}
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Run full-text search only."""
        try:
            return self._rag._fulltext_search(
                query=query_bundle.query_str,
                k=self._k,
                **self._filter_kwargs,
            )
        except Exception as e:
            logger.error(f"FulltextOnlyRetriever failed: {e}")
            return []


# =========================================================================
# Phase 6: EntityExtractionTransform for IngestionPipeline
# =========================================================================

class EntityExtractionTransform:
    """LlamaIndex-compatible transform that runs entity extraction on nodes.
    
    When added to an IngestionPipeline, automatically extracts person facts
    (birth dates, cities, relationships, etc.) from ingested documents and
    stores them in the Entity Store.
    
    This is an alternative to calling entity_extractor directly from the
    sync files. The transform passes nodes through unchanged — it only
    has the side effect of populating the entity store.
    
    Usage:
        pipeline = IngestionPipeline(
            transformations=[
                splitter,
                EntityExtractionTransform(),
                embed_model,
            ],
        )
    """
    
    def __call__(self, nodes: List[TextNode], **kwargs) -> List[TextNode]:
        """Extract entities from nodes and store in entity_db.
        
        Nodes are passed through unchanged. Entity extraction is a
        side effect that populates the entity store.
        
        Args:
            nodes: List of TextNode instances from the pipeline
            
        Returns:
            The same nodes, unchanged
        """
        if settings.get("rag_entity_extraction_in_pipeline", "false").lower() != "true":
            return nodes
        
        if not settings.get("entity_extraction_enabled", "true").lower() == "true":
            return nodes
        
        try:
            from entity_extractor import extract_entities_from_document
            
            for node in nodes:
                metadata = getattr(node, "metadata", {})
                source = metadata.get("source", "")
                title = metadata.get("chat_name", "")
                sender = metadata.get("sender", "")
                text = getattr(node, "text", "")
                source_id = metadata.get("source_id", "")
                
                if not text or len(text) < 20:
                    continue
                
                # Only extract from document-type content (not short messages)
                content_type = metadata.get("content_type", "")
                if content_type in ("document", "text") and source in ("paperless", "gmail"):
                    try:
                        extract_entities_from_document(
                            doc_title=title or "Document",
                            doc_text=text,
                            source_ref=source_id,
                            sender=sender,
                        )
                    except Exception as e:
                        logger.debug(f"Entity extraction failed for node (non-critical): {e}")
        except ImportError:
            logger.debug("entity_extractor not available — skipping pipeline extraction")
        except Exception as e:
            logger.debug(f"EntityExtractionTransform failed (non-critical): {e}")
        
        return nodes


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
