"""LlamaIndex RAG (Retrieval Augmented Generation) for WhatsApp messages.

Uses Qdrant as vector store and OpenAI embeddings for semantic search.
Replaces the previous LangChain-based RAG implementation.

Qdrant Dashboard: http://localhost:6333/dashboard
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores.types import VectorStoreQueryResult
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    Range,
    VectorParams,
)

from config import config
from utils.logger import logger


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
    
    COLLECTION_NAME = "whatsapp_messages"
    VECTOR_SIZE = 1536  # OpenAI embedding dimension
    
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
        
        # Get Qdrant server config from environment
        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        
        # Configure LlamaIndex settings
        self._configure_settings()
        
        self._initialized = True
        self._ensure_collection()
        logger.info(f"LlamaIndex RAG initialized with Qdrant at {self.qdrant_host}:{self.qdrant_port}")
    
    def _configure_settings(self):
        """Configure LlamaIndex global settings."""
        # Set up OpenAI embedding model
        Settings.embed_model = OpenAIEmbedding(
            api_key=config.OPENAI_API_KEY,
            model="text-embedding-ada-002"
        )
        
        # Set up LLM
        llm_provider = os.getenv('LLM_PROVIDER', 'openai').lower()
        if llm_provider == 'gemini':
            try:
                from llama_index.llms.gemini import Gemini
                Settings.llm = Gemini(
                    api_key=config.GOOGLE_API_KEY,
                    model=getattr(config, 'GEMINI_MODEL', 'gemini-pro'),
                    temperature=0.3
                )
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
    
    @property
    def qdrant_client(self) -> QdrantClient:
        """Get or create the Qdrant client."""
        if LlamaIndexRAG._qdrant_client is None:
            LlamaIndexRAG._qdrant_client = QdrantClient(
                host=self.qdrant_host,
                port=self.qdrant_port
            )
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
        """Ensure the collection exists in Qdrant."""
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.COLLECTION_NAME not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=self.VECTOR_SIZE,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
    
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
            # Format timestamp for display
            formatted_time = format_timestamp(timestamp)
            
            # Create text content for embedding
            text_content = f"[{formatted_time}] {sender} in {chat_name}: {message}"
            
            # Create metadata
            metadata = {
                "thread_id": thread_id,
                "chat_id": chat_id,
                "chat_name": chat_name,
                "is_group": is_group,
                "sender": sender,
                "message": message,
                "timestamp": int(timestamp) if timestamp.isdigit() else 0,
                "has_media": has_media,
                "media_type": media_type,
                "source_type": "whatsapp_message",
            }
            
            # Create LlamaIndex TextNode
            node = TextNode(
                text=text_content,
                metadata=metadata,
                id_=f"{chat_id}:{timestamp}"
            )
            
            # Insert into index
            self.index.insert_nodes([node])
            
            logger.debug(f"Added message to RAG: {text_content[:50]}...")
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
    
    def search(
        self,
        query: str,
        k: int = 10,
        filter_chat_name: Optional[str] = None,
        filter_sender: Optional[str] = None,
        filter_days: Optional[int] = None
    ) -> List[NodeWithScore]:
        """Search for relevant messages using semantic similarity.
        
        Args:
            query: The search query
            k: Number of results to return
            filter_chat_name: Optional filter by chat/group name
            filter_sender: Optional filter by sender name
            filter_days: Optional filter by number of days
            
        Returns:
            List of NodeWithScore objects with metadata
        """
        try:
            # Build Qdrant filter conditions
            conditions = []
            
            if filter_chat_name:
                conditions.append(
                    FieldCondition(
                        key="chat_name",
                        match=MatchValue(value=filter_chat_name)
                    )
                )
            
            if filter_sender:
                conditions.append(
                    FieldCondition(
                        key="sender",
                        match=MatchValue(value=filter_sender)
                    )
                )
            
            if filter_days is not None and filter_days > 0:
                min_timestamp = int(datetime.now().timestamp()) - (filter_days * 24 * 60 * 60)
                conditions.append(
                    FieldCondition(
                        key="timestamp",
                        range=Range(gte=min_timestamp)
                    )
                )
            
            # Create retriever with filters
            qdrant_filters = Filter(must=conditions) if conditions else None
            
            retriever = self.index.as_retriever(
                similarity_top_k=k,
                vector_store_kwargs={"qdrant_filters": qdrant_filters} if qdrant_filters else {}
            )
            
            results = retriever.retrieve(query)
            logger.info(f"RAG search for '{query[:50]}...' returned {len(results)} results")
            return results
            
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
            # Get relevant documents
            results = self.search(
                query=question,
                k=k,
                filter_chat_name=filter_chat_name,
                filter_sender=filter_sender,
                filter_days=filter_days
            )
            
            # Build context from results
            context_parts = []
            for result in results:
                context_parts.append(result.node.text)
            
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
            
            # Create query prompt
            prompt = f"""You are a helpful AI assistant for a WhatsApp message archive search system.

Current Date/Time: {current_datetime}
תאריך ושעה נוכחיים: {hebrew_date}

Message Archive Context:
{context}
{history_text}

User Question: {question}

Instructions:
- If the question is about messages/conversations, use the context above to answer
- If the question is general (like "what day is today?"), answer directly
- Answer in the same language as the question
- Be concise and helpful"""

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
    
    def get_chat_list(self) -> List[str]:
        """Get all unique chat names from the vector store.
        
        Returns:
            List of unique chat names sorted alphabetically
        """
        try:
            chat_names = set()
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
                    chat_name = payload.get("chat_name")
                    if chat_name:
                        chat_names.add(chat_name)
                
                if next_offset is None:
                    break
                offset = next_offset
            
            return sorted(list(chat_names))
        except Exception as e:
            logger.error(f"Failed to get chat list: {e}")
            return []
    
    def get_sender_list(self) -> List[str]:
        """Get all unique sender names from the vector store.
        
        Returns:
            List of unique sender names sorted alphabetically
        """
        try:
            senders = set()
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
                    sender = payload.get("sender")
                    if sender:
                        senders.add(sender)
                
                if next_offset is None:
                    break
                offset = next_offset
            
            return sorted(list(senders))
        except Exception as e:
            logger.error(f"Failed to get sender list: {e}")
            return []


# Create singleton instance getter
_rag_instance: Optional[LlamaIndexRAG] = None


def get_rag() -> LlamaIndexRAG:
    """Get the shared RAG singleton instance.
    
    Returns:
        The LlamaIndexRAG singleton instance
    """
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = LlamaIndexRAG()
    return _rag_instance
