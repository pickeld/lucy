"""LlamaIndex-based conversation memory manager for WhatsApp chats.

Replaces LangGraph thread management with LlamaIndex chat memory
backed by a database for persistence.
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI

from config import config
from llamaindex_rag import format_timestamp, get_rag
from utils.logger import logger
from utils.redis_conn import get_redis_client


class ConversationThread:
    """Represents a WhatsApp chat conversation with memory.
    
    Uses LlamaIndex ChatMemoryBuffer for in-context conversation history
    and Redis for persistence across restarts.
    """
    
    def __init__(
        self,
        chat_id: str,
        chat_name: str,
        is_group: bool,
        token_limit: int = 4096
    ):
        """Initialize a conversation thread.
        
        Args:
            chat_id: Normalized WhatsApp chat ID
            chat_name: Display name of the chat
            is_group: Whether this is a group chat
            token_limit: Maximum tokens for memory buffer
        """
        self.chat_id = chat_id
        self.chat_name = chat_name
        self.is_group = is_group
        self._token_limit = token_limit
        
        # Initialize LLM for responses
        llm_provider = os.getenv('LLM_PROVIDER', 'openai').lower()
        if llm_provider == 'gemini':
            try:
                from llama_index.llms.gemini import Gemini
                self._llm = Gemini(
                    api_key=config.GOOGLE_API_KEY,
                    model=getattr(config, 'GEMINI_MODEL', 'gemini-pro'),
                    temperature=float(getattr(config, 'GEMINI_TEMPERATURE', '0.7'))
                )
            except ImportError:
                logger.warning("Gemini LLM not available, falling back to OpenAI")
                self._llm = LlamaIndexOpenAI(
                    api_key=config.OPENAI_API_KEY,
                    model=config.OPENAI_MODEL,
                    temperature=float(getattr(config, 'OPENAI_TEMPERATURE', 0.7))
                )
        else:
            self._llm = LlamaIndexOpenAI(
                api_key=config.OPENAI_API_KEY,
                model=config.OPENAI_MODEL,
                temperature=float(getattr(config, 'OPENAI_TEMPERATURE', 0.7))
            )
        
        # Initialize chat memory
        self._memory = ChatMemoryBuffer.from_defaults(token_limit=token_limit)
        
        # Load existing messages from Redis if available
        self._load_from_redis()
        
        logger.info(f"ConversationThread initialized for {chat_id} ({chat_name})")
    
    def _get_redis_key(self) -> str:
        """Get Redis key for this thread's messages."""
        return f"chat_memory:{self.chat_id}"
    
    def _load_from_redis(self):
        """Load conversation history from Redis."""
        try:
            redis = get_redis_client()
            if redis is None:
                return
            
            key = self._get_redis_key()
            messages_json = redis.lrange(key, 0, -1)
            
            if messages_json:
                import json
                for msg_json in messages_json:
                    msg_data = json.loads(msg_json)
                    role = MessageRole(msg_data.get("role", "user"))
                    content = msg_data.get("content", "")
                    self._memory.put(ChatMessage(role=role, content=content))
                
                logger.debug(f"Loaded {len(messages_json)} messages from Redis for {self.chat_id}")
        except Exception as e:
            logger.warning(f"Failed to load messages from Redis: {e}")
    
    def _save_to_redis(self, message: ChatMessage):
        """Save a message to Redis for persistence."""
        try:
            redis = get_redis_client()
            if redis is None:
                return
            
            import json
            key = self._get_redis_key()
            msg_data = {
                "role": message.role.value,
                "content": message.content,
                "timestamp": datetime.now().isoformat()
            }
            redis.rpush(key, json.dumps(msg_data))
            
            # Keep only last 100 messages in Redis
            redis.ltrim(key, -100, -1)
        except Exception as e:
            logger.warning(f"Failed to save message to Redis: {e}")
    
    def remember(
        self,
        timestamp: str,
        sender: str,
        message: str,
        store_only: bool = True
    ) -> Optional[str]:
        """Store a message in conversation history.
        
        Also adds the message to RAG vector store for cross-thread search.
        
        Args:
            timestamp: Unix timestamp of the message
            sender: Name of the message sender
            message: The message content
            store_only: If True, only store without LLM response
            
        Returns:
            AI response if store_only=False, None otherwise
        """
        try:
            # Format message
            readable_timestamp = format_timestamp(timestamp)
            formatted_message = f"[{readable_timestamp}] {sender}: {message}"
            
            # Add to memory
            user_msg = ChatMessage(role=MessageRole.USER, content=formatted_message)
            self._memory.put(user_msg)
            self._save_to_redis(user_msg)
            
            # Add to RAG vector store
            rag = get_rag()
            rag.add_message(
                thread_id=self.chat_id,
                chat_id=self.chat_id,
                chat_name=self.chat_name,
                is_group=self.is_group,
                sender=sender,
                message=message,
                timestamp=timestamp
            )
            
            logger.debug(f"Stored message: {formatted_message[:50]}...")
            
            if store_only:
                return None
            
            # Generate AI response
            return self.chat(message)
            
        except Exception as e:
            logger.error(f"Error remembering message: {e}")
            return None
    
    def chat(self, query: str) -> str:
        """Generate an AI response based on conversation history.
        
        Args:
            query: The user's query/message
            
        Returns:
            AI-generated response
        """
        try:
            # Build system message with context
            if self.is_group:
                system_content = f"""You are a helpful AI assistant for a WhatsApp group chat.
Chat Name: {self.chat_name}

IMPORTANT: This is a group chat with multiple participants. Each message in the history is formatted as:
[timestamp] sender_name: message_content

The sender_name indicates WHO sent that specific message. Different messages may come from different people.
When asked about what someone said, look at the sender_name prefix to identify their messages.

The LAST message is from the person currently asking you a question."""
            else:
                system_content = f"""You are a helpful AI assistant for WhatsApp.
Chat Type: Personal
Chat Name: {self.chat_name}

Messages are formatted as: [timestamp] sender_name: message_content
The sender_name indicates who sent each message.

Remember conversations and provide contextual responses."""
            
            system_msg = ChatMessage(role=MessageRole.SYSTEM, content=system_content)
            
            # Get conversation history from memory
            history = self._memory.get()
            
            # Build messages for LLM
            messages = [system_msg] + history
            
            # Generate response
            response = self._llm.chat(messages)
            ai_response = str(response.message.content)
            
            # Store AI response in memory
            ai_msg = ChatMessage(role=MessageRole.ASSISTANT, content=ai_response)
            self._memory.put(ai_msg)
            self._save_to_redis(ai_msg)
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history as list of dicts.
        
        Returns:
            List of messages with role and content
        """
        history = self._memory.get()
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in history
        ]
    
    def clear(self):
        """Clear conversation history."""
        self._memory.reset()
        
        try:
            redis = get_redis_client()
            if redis:
                redis.delete(self._get_redis_key())
        except Exception as e:
            logger.warning(f"Failed to clear Redis history: {e}")
        
        logger.info(f"Cleared conversation history for {self.chat_id}")


class ThreadsManager:
    """Manages conversation threads for WhatsApp chats.
    
    Replaces LangGraph ThreadsManager with LlamaIndex-based implementation.
    """
    
    def __init__(self):
        """Initialize the threads manager."""
        self.threads: Dict[str, ConversationThread] = {}
        logger.info("ThreadsManager initialized")
    
    def get_thread(
        self,
        chat_id: str,
        chat_name: str,
        is_group: bool
    ) -> ConversationThread:
        """Get or create a thread for a specific chat.
        
        Args:
            chat_id: The chat ID from WhatsApp
            chat_name: Display name of the chat
            is_group: Whether this is a group chat
            
        Returns:
            ConversationThread instance for the chat
        """
        normalized_id = chat_id.replace("@", "_").replace(".", "_")
        
        if normalized_id not in self.threads:
            logger.debug(f"Creating new thread for chat: {normalized_id}")
            self.threads[normalized_id] = ConversationThread(
                chat_id=normalized_id,
                chat_name=chat_name,
                is_group=is_group
            )
        
        return self.threads[normalized_id]
    
    def clear_thread(self, chat_id: str) -> bool:
        """Clear a specific thread's history.
        
        Args:
            chat_id: The chat ID to clear
            
        Returns:
            True if thread existed and was cleared
        """
        normalized_id = chat_id.replace("@", "_").replace(".", "_")
        
        if normalized_id in self.threads:
            self.threads[normalized_id].clear()
            return True
        return False
    
    def get_all_threads(self) -> List[Dict[str, Any]]:
        """Get info about all active threads.
        
        Returns:
            List of thread info dicts
        """
        return [
            {
                "chat_id": thread.chat_id,
                "chat_name": thread.chat_name,
                "is_group": thread.is_group,
                "message_count": len(thread.get_history())
            }
            for thread in self.threads.values()
        ]


# Module-level singleton
_threads_manager: Optional[ThreadsManager] = None


def get_threads_manager() -> ThreadsManager:
    """Get the shared ThreadsManager singleton.
    
    Returns:
        The ThreadsManager singleton instance
    """
    global _threads_manager
    if _threads_manager is None:
        _threads_manager = ThreadsManager()
    return _threads_manager
