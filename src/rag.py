"""RAG (Retrieval Augmented Generation) for querying across multiple data sources.

Uses Qdrant as vector store and OpenAI embeddings for semantic search.
Supports multiple data sources:
- WhatsApp messages
- Documents (PDF, Word, text files)
- Call recordings (transcribed audio)

Qdrant Dashboard: http://localhost:6333/dashboard
"""

from utils.logger import logger
from config import config
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue, Range
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from zoneinfo import ZoneInfo
import os

# Import document classes
from classes import (
    BaseRAGDocument,
    WhatsAppMessageDocument,
    FileDocument,
    CallRecordingDocument,
    SourceType
)


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
    except (ValueError, TypeError, KeyError) as e:
        # If conversion fails, return original timestamp
        return str(timestamp)


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
                api_key=config.OPENAI_API_KEY  # type: ignore[arg-type]
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
                    api_key=config.OPENAI_API_KEY  # type: ignore[arg-type]
                )
        return self._llm
    
    def add_document(self, document: BaseRAGDocument) -> bool:
        """Add any document type to the vector store for RAG queries.
        
        This is the primary method for adding documents to the RAG system.
        It accepts any BaseRAGDocument subclass (WhatsApp, File, CallRecording).
        
        Args:
            document: A BaseRAGDocument instance (or subclass)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            langchain_doc = document.to_langchain_document()
            self.vectorstore.add_documents([langchain_doc])
            logger.debug(f"Added {document.metadata.source_type.value} document to RAG: {document.content[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to add document to vector store: {e}")
            return False
    
    def add_documents(self, documents: List[BaseRAGDocument]) -> int:
        """Add multiple documents to the vector store in batch.
        
        Args:
            documents: List of BaseRAGDocument instances
            
        Returns:
            Number of successfully added documents
        """
        if not documents:
            return 0
        
        try:
            langchain_docs = [doc.to_langchain_document() for doc in documents]
            self.vectorstore.add_documents(langchain_docs)
            logger.info(f"Added {len(documents)} documents to RAG vector store")
            return len(documents)
        except Exception as e:
            logger.error(f"Failed to add batch documents to vector store: {e}")
            return 0
    
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
        """Add a WhatsApp message to the vector store for RAG queries.
        
        This is a convenience method that wraps add_document() for backward
        compatibility with existing WhatsApp message handling code.
        
        Args:
            thread_id: The LangGraph thread ID
            chat_id: The WhatsApp chat ID
            chat_name: Name of the chat/group
            is_group: Whether this is a group chat
            sender: The message sender
            message: The message content
            timestamp: The message timestamp (Unix timestamp as string)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create WhatsAppMessageDocument using the new document model
            doc = WhatsAppMessageDocument.from_webhook_payload(
                thread_id=thread_id,
                chat_id=chat_id,
                chat_name=chat_name,
                is_group=is_group,
                sender=sender,
                message=message,
                timestamp=timestamp
            )
            return self.add_document(doc)
        except Exception as e:
            logger.error(f"Failed to add WhatsApp message to vector store: {e}")
            return False
    
    def add_file(
        self,
        file_path: str,
        content: str,
        author: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Add a file document to the vector store for RAG queries.
        
        Convenience method for adding PDF, Word, or text documents.
        
        Args:
            file_path: Path to the file
            content: Extracted text content from the file
            author: Document author (optional)
            title: Document title (optional, defaults to filename)
            description: Document description (optional)
            tags: Optional tags for categorization
            
        Returns:
            True if successful, False otherwise
        """
        try:
            doc = FileDocument.from_file(
                file_path=file_path,
                content=content,
                author=author,
                title=title,
                description=description,
                tags=tags
            )
            return self.add_document(doc)
        except Exception as e:
            logger.error(f"Failed to add file document to vector store: {e}")
            return False
    
    def add_call_recording(
        self,
        recording_id: str,
        transcript: str,
        participants: List[str],
        duration_seconds: int,
        call_type: str = "unknown",
        phone_number: Optional[str] = None,
        confidence_score: float = 1.0,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Add a transcribed call recording to the vector store.
        
        Convenience method for adding transcribed audio recordings.
        
        Args:
            recording_id: Unique recording identifier
            transcript: Transcribed text content
            participants: List of call participants
            duration_seconds: Call duration in seconds
            call_type: Type of call (incoming, outgoing, conference)
            phone_number: Primary phone number
            confidence_score: Transcription confidence (0.0-1.0)
            tags: Optional tags for categorization
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from classes import CallType
            
            # Map string call_type to enum
            call_type_map = {
                "incoming": CallType.INCOMING,
                "outgoing": CallType.OUTGOING,
                "conference": CallType.CONFERENCE,
                "voicemail": CallType.VOICEMAIL,
            }
            call_type_enum = call_type_map.get(call_type.lower(), CallType.UNKNOWN)
            
            doc = CallRecordingDocument.from_transcription(
                recording_id=recording_id,
                transcript=transcript,
                participants=participants,
                duration_seconds=duration_seconds,
                call_type=call_type_enum,
                phone_number=phone_number,
                confidence_score=confidence_score,
                tags=tags
            )
            return self.add_document(doc)
        except Exception as e:
            logger.error(f"Failed to add call recording to vector store: {e}")
            return False
    
    def search(
        self,
        query: str,
        k: int = 10,
        filter_chat_name: Optional[str] = None,
        filter_sender: Optional[str] = None,
        filter_days: Optional[int] = None
    ) -> List[Document]:
        """Search for relevant messages using semantic similarity.
        
        Args:
            query: The search query
            k: Number of results to return
            filter_chat_name: Optional filter by chat/group name
            filter_sender: Optional filter by sender name
            filter_days: Optional filter by number of days (e.g., 1=24h, 3=3 days, 7=week, 30=month, None=all time)
            
        Returns:
            List of relevant Document objects with metadata
        """
        try:
            conditions: List[Any] = []
            if filter_chat_name:
                conditions.append(
                    FieldCondition(key="metadata.chat_name", match=MatchValue(value=filter_chat_name))
                )
            if filter_sender:
                conditions.append(
                    FieldCondition(key="metadata.sender", match=MatchValue(value=filter_sender))
                )
            if filter_days is not None and filter_days > 0:
                # Calculate the timestamp threshold (current time - filter_days in seconds)
                min_timestamp = int(datetime.now().timestamp()) - (filter_days * 24 * 60 * 60)
                conditions.append(
                    FieldCondition(
                        key="metadata.timestamp",
                        range=Range(gte=min_timestamp)
                    )
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
        filter_sender: Optional[str] = None,
        filter_days: Optional[int] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Query the RAG system with a natural language question.
        
        Uses a HYBRID approach:
        - Always retrieves context from the message archive
        - The LLM intelligently decides whether to use the context or answer directly
        - For questions like "what day is today?" - answers directly, ignoring irrelevant context
        - For questions about messages - uses the retrieved context
        - Maintains conversation context across multiple questions
        
        Args:
            question: Natural language question (e.g., "who said they would be late?")
            k: Number of context documents to retrieve
            filter_chat_name: Optional filter by chat/group name
            filter_sender: Optional filter by sender name
            filter_days: Optional filter by number of days (e.g., 1=24h, 3=3 days, 7=week, 30=month, None=all time)
            conversation_history: Optional list of previous conversation messages
                                  [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            
        Returns:
            AI-generated answer based on retrieved context or direct answer
        """
        try:
            # Always retrieve documents for context
            docs = self.search(question, k=k, filter_chat_name=filter_chat_name, filter_sender=filter_sender, filter_days=filter_days)
            
            # Build context from retrieved documents (may be empty)
            context_parts = []
            for doc in docs:
                context_parts.append(doc.page_content)
            
            context = "\n".join(context_parts) if context_parts else "[No messages found in the archive]"
            
            # Get current date/time for context
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
            
            # Create a HYBRID prompt that handles both cases
            system_prompt = """You are a helpful AI assistant for a WhatsApp message archive search system.

You have access to:
1. The user's WhatsApp message history (provided as context below)
2. Current date and time information
3. Our previous conversation (if any)

YOUR TASK: Analyze the user's question and respond appropriately:

IF THE QUESTION IS ABOUT MESSAGES/CONVERSATIONS (e.g., "what did X say?", "show me messages from yesterday", "who mentioned Y?", "what was discussed?"):
- Use the message context provided below to answer
- Mention WHO said something, WHEN they said it, and in WHICH chat/group
- If no relevant messages are found, say "I couldn't find any relevant messages"

IF THE QUESTION IS A GENERAL QUERY NOT REQUIRING MESSAGE HISTORY (e.g., "what day is today?", "hello", "what's 2+2?", "translate X"):
- Answer directly using your knowledge and the current date/time
- IGNORE the message context - it's not relevant to these questions
- Be concise and helpful

IF THE QUESTION REFERENCES PREVIOUS CONVERSATION (e.g., "what was my previous question?", "tell me more", "explain further"):
- Use our conversation history to understand what they're referring to
- Provide contextual follow-up answers

IMPORTANT:
- Answer in the same language as the question
- For date/time questions, use the current date/time provided
- Don't say you don't have access to messages - you DO have access (see context below)
- Remember our conversation context - the user may ask follow-up questions"""

            user_content = f"""Current Date/Time: {current_datetime}
תאריך ושעה נוכחיים: {hebrew_date}

Message Archive Context:
{context}

User Question: {question}"""

            # Build messages list with conversation history
            from langchain_core.messages import AIMessage
            llm_messages: List[Any] = [SystemMessage(content=system_prompt)]
            
            # Add conversation history if provided
            if conversation_history:
                for msg in conversation_history:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "user":
                        llm_messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        llm_messages.append(AIMessage(content=content))
            
            # Add the current question with context
            llm_messages.append(HumanMessage(content=user_content))
            
            response = self.llm.invoke(llm_messages)
            content = response.content if hasattr(response, 'content') else str(response)
            # Handle case where content might be a list
            if isinstance(content, list):
                answer = str(content[0]) if content else ""
            else:
                answer = str(content)
            
            logger.info(f"Hybrid query answered: {question[:50]}... (context_docs={len(docs)})")
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
            conditions: List[Any] = []
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
