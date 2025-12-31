"""RAG (Retrieval Augmented Generation) for querying across all WhatsApp threads.

Uses Qdrant as vector store and OpenAI embeddings for semantic search.
Enables queries like "who said X", "when did Y happen", "what was discussed in group Z".

Qdrant Dashboard: http://localhost:6333/dashboard
"""

from utiles.logger import logger
from config import config
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from typing import List, Optional, Dict, Any
import os


class RAG:
    """RAG (Retrieval Augmented Generation) for querying across all WhatsApp threads.
    
    Uses Qdrant server as vector store and OpenAI embeddings for semantic search.
    Connects to Qdrant server at QDRANT_HOST:QDRANT_PORT (default: localhost:6333).
    
    Dashboard available at: http://localhost:6333/dashboard
    """
    
    _instance = None
    _vectorstore = None
    _embeddings = None
    _qdrant_client = None
    
    COLLECTION_NAME = "whatsapp_messages"
    VECTOR_SIZE = 1536  # OpenAI embedding dimension
    
    def __new__(cls):
        """Singleton pattern to ensure one RAG instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the RAG system."""
        if self._initialized:
            return
        
        # Get Qdrant server config from environment
        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        
        self._initialized = True
        self._llm = None
        self._ensure_collection()
        logger.info(f"RAG system initialized with Qdrant server at {self.qdrant_host}:{self.qdrant_port}")
    
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
    
    @property
    def qdrant_client(self) -> QdrantClient:
        """Get or create the Qdrant client."""
        if RAG._qdrant_client is None:
            RAG._qdrant_client = QdrantClient(
                host=self.qdrant_host,
                port=self.qdrant_port
            )
        return RAG._qdrant_client
    
    @property
    def embeddings(self) -> OpenAIEmbeddings:
        """Get or create the embeddings model."""
        if RAG._embeddings is None:
            RAG._embeddings = OpenAIEmbeddings(
                api_key=config.OPENAI_API_KEY
            )
        return RAG._embeddings
    
    @property
    def vectorstore(self) -> QdrantVectorStore:
        """Get or create the Qdrant vector store connected to the server."""
        if RAG._vectorstore is None:
            RAG._vectorstore = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.COLLECTION_NAME,
                embedding=self.embeddings
            )
        return RAG._vectorstore
    
    @property
    def llm(self):
        """Get or create the LLM for RAG queries."""
        if self._llm is None:
            llm_provider = os.getenv('LLM_PROVIDER', 'openai').lower()
            if llm_provider == 'gemini':
                self._llm = ChatGoogleGenerativeAI(
                    model=getattr(config, 'GEMINI_MODEL', 'gemini-pro'),
                    temperature=0.3,
                    google_api_key=config.GOOGLE_API_KEY
                )
            else:
                self._llm = ChatOpenAI(
                    model=config.OPENAI_MODEL,
                    temperature=0.3,
                    api_key=config.OPENAI_API_KEY
                )
        return self._llm
    
    def add_message(
        self,
        thread_id: str,
        chat_id: str,
        chat_name: str,
        is_group: bool,
        sender: str,
        message: str,
        timestamp: str
    ) -> bool:
        """Add a message to the vector store for RAG queries.
        
        Args:
            thread_id: The LangGraph thread ID
            chat_id: The WhatsApp chat ID
            chat_name: Name of the chat/group
            is_group: Whether this is a group chat
            sender: The message sender
            message: The message content
            timestamp: The message timestamp
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create document with full context
            doc_content = f"[{timestamp}] {sender} in {chat_name}: {message}"
            
            doc = Document(
                page_content=doc_content,
                metadata={
                    "thread_id": thread_id,
                    "chat_id": chat_id,
                    "chat_name": chat_name,
                    "is_group": is_group,
                    "sender": sender,
                    "timestamp": timestamp,
                    "message": message
                }
            )
            
            self.vectorstore.add_documents([doc])
            logger.debug(f"Added message to RAG vector store: {doc_content[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to add message to vector store: {e}")
            return False
    
    def search(
        self,
        query: str,
        k: int = 10,
        filter_chat_name: Optional[str] = None,
        filter_sender: Optional[str] = None
    ) -> List[Document]:
        """Search for relevant messages using semantic similarity.
        
        Args:
            query: The search query
            k: Number of results to return
            filter_chat_name: Optional filter by chat/group name
            filter_sender: Optional filter by sender name
            
        Returns:
            List of relevant Document objects with metadata
        """
        try:
            # Build filter for Qdrant
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            conditions = []
            if filter_chat_name:
                conditions.append(
                    FieldCondition(key="metadata.chat_name", match=MatchValue(value=filter_chat_name))
                )
            if filter_sender:
                conditions.append(
                    FieldCondition(key="metadata.sender", match=MatchValue(value=filter_sender))
                )
            
            qdrant_filter = Filter(must=conditions) if conditions else None
            
            if qdrant_filter:
                results = self.vectorstore.similarity_search(
                    query, k=k, filter=qdrant_filter
                )
            else:
                results = self.vectorstore.similarity_search(query, k=k)
            
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
        filter_sender: Optional[str] = None
    ) -> str:
        """Query the RAG system with a natural language question.
        
        Args:
            question: Natural language question (e.g., "who said they would be late?")
            k: Number of context documents to retrieve
            filter_chat_name: Optional filter by chat/group name
            filter_sender: Optional filter by sender name
            
        Returns:
            AI-generated answer based on retrieved context
        """
        try:
            # Retrieve relevant documents
            docs = self.search(question, k=k, filter_chat_name=filter_chat_name, filter_sender=filter_sender)
            
            if not docs:
                return "I couldn't find any relevant messages to answer your question."
            
            # Build context from retrieved documents
            context_parts = []
            for doc in docs:
                context_parts.append(doc.page_content)
            
            context = "\n".join(context_parts)
            
            # Create prompt for the LLM
            system_prompt = """You are a helpful assistant that answers questions about WhatsApp conversations.
You have access to message history from various chats and groups.

Each message in the context is formatted as: [timestamp] sender in chat_name: message

Based on the context provided, answer the user's question accurately.
If you can't find the answer in the context, say so.
Always mention WHO said something, WHEN they said it, and in WHICH chat/group."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"""Context from message history:
{context}

Question: {question}

Answer:""")
            ]
            
            response = self.llm.invoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)
            # Handle case where content might be a list
            if isinstance(content, list):
                answer = str(content[0]) if content else ""
            else:
                answer = str(content)
            
            logger.info(f"RAG query answered: {question[:50]}...")
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
            
            # Scroll through all points to collect unique chat names
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
                    metadata = payload.get("metadata", {})
                    chat_name = metadata.get("chat_name")
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
            
            # Scroll through all points to collect unique sender names
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
                    metadata = payload.get("metadata", {})
                    sender = metadata.get("sender")
                    if sender:
                        senders.add(sender)
                
                if next_offset is None:
                    break
                offset = next_offset
            
            return sorted(list(senders))
        except Exception as e:
            logger.error(f"Failed to get sender list: {e}")
            return []
    
    def browse(
        self,
        limit: int = 100,
        offset: int = 0,
        filter_chat_name: Optional[str] = None,
        filter_sender: Optional[str] = None
    ) -> Dict[str, Any]:
        """Browse all documents in the vector store.
        
        Note: For better browsing experience, use the Qdrant Dashboard at:
        http://localhost:6333/dashboard
        
        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            filter_chat_name: Optional filter by chat name
            filter_sender: Optional filter by sender
            
        Returns:
            Dictionary with documents and metadata
        """
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            conditions = []
            if filter_chat_name:
                conditions.append(
                    FieldCondition(key="metadata.chat_name", match=MatchValue(value=filter_chat_name))
                )
            if filter_sender:
                conditions.append(
                    FieldCondition(key="metadata.sender", match=MatchValue(value=filter_sender))
                )
            
            qdrant_filter = Filter(must=conditions) if conditions else None
            
            # Scroll through points
            records, next_offset = self.qdrant_client.scroll(
                collection_name=self.COLLECTION_NAME,
                limit=limit,
                offset=offset,
                scroll_filter=qdrant_filter,
                with_payload=True,
                with_vectors=False
            )
            
            # Format results
            documents = []
            for record in records:
                payload = record.payload or {}
                documents.append({
                    "id": str(record.id),
                    "content": payload.get("page_content", ""),
                    "metadata": payload.get("metadata", {})
                })
            
            # Get collection info for total count
            collection_info = self.qdrant_client.get_collection(self.COLLECTION_NAME)
            
            return {
                "documents": documents,
                "total": collection_info.points_count,
                "limit": limit,
                "offset": offset,
                "next_offset": next_offset,
                "dashboard_url": f"http://{self.qdrant_host}:{self.qdrant_port}/dashboard"
            }
        except Exception as e:
            logger.error(f"Failed to browse RAG: {e}")
            return {"error": str(e), "documents": [], "total": 0}
