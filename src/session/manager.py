"""Session manager for conversation state persistence.

This module provides the SessionManager class that handles creating,
storing, retrieving, and expiring conversation sessions using Redis.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from utils.logger import logger
from utils.redis_conn import (
    get_redis_client,
    redis_delete,
    redis_delete_pattern,
    redis_get,
    redis_set,
)

from .models import ConversationSession, EntityInfo, EntityType


# Redis key prefix for sessions
SESSION_KEY_PREFIX = "session:"

# Default TTL from environment or 30 minutes
DEFAULT_SESSION_TTL_MINUTES = int(os.getenv("SESSION_TTL_MINUTES", "30"))

# Maximum conversation history turns to keep
MAX_HISTORY_TURNS = int(os.getenv("SESSION_MAX_HISTORY", "20"))


class SessionManager:
    """Manages conversation sessions with Redis persistence.
    
    Handles creation, retrieval, update, and expiration of conversation
    sessions. Uses Redis for storage with configurable TTL.
    
    Usage:
        manager = SessionManager()
        
        # Create new session
        session = manager.create_session()
        
        # Get existing session
        session = manager.get_session(session_id)
        
        # Update session
        session.set_chat_context("Family Group")
        manager.save_session(session)
    """
    
    def __init__(self, ttl_minutes: Optional[int] = None):
        """Initialize the session manager.
        
        Args:
            ttl_minutes: Session timeout in minutes (default from env or 30)
        """
        self.ttl_minutes = ttl_minutes or DEFAULT_SESSION_TTL_MINUTES
        self._redis = get_redis_client()
        logger.info(f"SessionManager initialized with TTL={self.ttl_minutes}min")
    
    def _session_key(self, session_id: str) -> str:
        """Get the Redis key for a session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            Redis key string
        """
        return f"{SESSION_KEY_PREFIX}{session_id}"
    
    def create_session(
        self,
        initial_chat: Optional[str] = None,
        initial_sender: Optional[str] = None
    ) -> ConversationSession:
        """Create a new conversation session.
        
        Args:
            initial_chat: Optional chat/group to focus on initially
            initial_sender: Optional sender to focus on initially
            
        Returns:
            New ConversationSession instance
        """
        session = ConversationSession(ttl_minutes=self.ttl_minutes)
        
        if initial_chat:
            session.set_chat_context(initial_chat)
        
        if initial_sender:
            session.set_sender_context(initial_sender)
        
        # Save to Redis
        self.save_session(session)
        
        logger.info(f"Created new session: {session.session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Retrieve a session by ID.
        
        Args:
            session_id: The session identifier
            
        Returns:
            ConversationSession if found and not expired, None otherwise
        """
        key = self._session_key(session_id)
        data = redis_get(key)
        
        if data is None:
            logger.debug(f"Session not found: {session_id}")
            return None
        
        try:
            session = ConversationSession.from_dict(data)
            
            # Check if expired
            if session.is_expired:
                logger.info(f"Session expired: {session_id}")
                self.delete_session(session_id)
                return None
            
            # Touch the session to update last activity
            session.touch()
            
            return session
            
        except Exception as e:
            logger.error(f"Failed to deserialize session {session_id}: {e}")
            return None
    
    def get_or_create_session(
        self,
        session_id: Optional[str] = None,
        initial_chat: Optional[str] = None,
        initial_sender: Optional[str] = None
    ) -> ConversationSession:
        """Get existing session or create new one.
        
        Args:
            session_id: Optional existing session ID
            initial_chat: Chat filter for new session
            initial_sender: Sender filter for new session
            
        Returns:
            ConversationSession (existing or new)
        """
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
        
        # Create new session
        return self.create_session(
            initial_chat=initial_chat,
            initial_sender=initial_sender
        )
    
    def save_session(self, session: ConversationSession) -> bool:
        """Save a session to Redis.
        
        Updates the last_activity timestamp before saving.
        
        Args:
            session: The session to save
            
        Returns:
            True if saved successfully
        """
        try:
            session.touch()
            
            # Trim history if too long
            if len(session.turns) > MAX_HISTORY_TURNS:
                session.turns = session.turns[-MAX_HISTORY_TURNS:]
            
            key = self._session_key(session.session_id)
            ttl_seconds = session.ttl_minutes * 60
            
            redis_set(key, session.to_dict(), expire=ttl_seconds)
            
            logger.debug(f"Saved session: {session.session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save session {session.session_id}: {e}")
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            True if session was deleted
        """
        key = self._session_key(session_id)
        deleted = redis_delete(key)
        
        if deleted:
            logger.info(f"Deleted session: {session_id}")
        
        return deleted
    
    def clear_all_sessions(self) -> int:
        """Delete all sessions.
        
        Returns:
            Number of sessions deleted
        """
        pattern = f"{SESSION_KEY_PREFIX}*"
        count = redis_delete_pattern(pattern)
        logger.info(f"Cleared {count} sessions")
        return count
    
    def add_turn_to_session(
        self,
        session: ConversationSession,
        user_query: str,
        assistant_response: str,
        reformulated_query: Optional[str] = None,
        retrieved_ids: Optional[List[str]] = None,
        filters: Optional[Dict] = None,
        entities: Optional[List[str]] = None,
        auto_save: bool = True
    ) -> ConversationSession:
        """Add a conversation turn and optionally save.
        
        Convenience method that adds a turn and saves in one operation.
        
        Args:
            session: The session to update
            user_query: User's query
            assistant_response: Assistant's response
            reformulated_query: Query after reformulation
            retrieved_ids: IDs of retrieved messages
            filters: Filters used
            entities: Entities mentioned
            auto_save: Whether to save after adding turn
            
        Returns:
            Updated session
        """
        session.add_turn(
            user_query=user_query,
            assistant_response=assistant_response,
            reformulated_query=reformulated_query,
            retrieved_ids=retrieved_ids,
            filters=filters,
            entities=entities
        )
        
        if auto_save:
            self.save_session(session)
        
        return session
    
    def update_session_context(
        self,
        session: ConversationSession,
        chat_name: Optional[str] = None,
        sender_name: Optional[str] = None,
        time_range: Optional[tuple] = None,
        auto_save: bool = True
    ) -> ConversationSession:
        """Update session context filters.
        
        Args:
            session: The session to update
            chat_name: Chat filter (None to keep, empty string to clear)
            sender_name: Sender filter (None to keep, empty string to clear)
            time_range: Time range tuple (start, end) or None
            auto_save: Whether to save after updating
            
        Returns:
            Updated session
        """
        if chat_name is not None:
            session.set_chat_context(chat_name if chat_name else None)
        
        if sender_name is not None:
            session.set_sender_context(sender_name if sender_name else None)
        
        if time_range is not None:
            session.active_time_range = time_range if time_range else None
        
        if auto_save:
            self.save_session(session)
        
        return session
    
    def extract_and_track_entities(
        self,
        session: ConversationSession,
        text: str,
        known_chats: Optional[List[str]] = None,
        known_senders: Optional[List[str]] = None
    ) -> List[EntityInfo]:
        """Extract and track entities from text.
        
        Simple entity extraction that matches against known chats and senders.
        For more sophisticated extraction, integrate with NER.
        
        Args:
            session: The session to update
            text: Text to extract entities from
            known_chats: List of known chat/group names
            known_senders: List of known sender names
            
        Returns:
            List of extracted entities
        """
        extracted = []
        text_lower = text.lower()
        
        # Check for known chats
        if known_chats:
            for chat in known_chats:
                if chat.lower() in text_lower:
                    entity = session.add_entity(chat, EntityType.GROUP)
                    extracted.append(entity)
                    # Auto-set as active chat if mentioned
                    session.set_chat_context(chat)
        
        # Check for known senders
        if known_senders:
            for sender in known_senders:
                if sender.lower() in text_lower:
                    entity = session.add_entity(sender, EntityType.PERSON)
                    extracted.append(entity)
        
        return extracted
    
    def get_session_stats(self) -> Dict:
        """Get statistics about active sessions.
        
        Returns:
            Dictionary with session statistics
        """
        try:
            pattern = f"{SESSION_KEY_PREFIX}*"
            keys = self._redis.keys(pattern)  # type: ignore[union-attr]
            key_count = len(list(keys)) if keys else 0  # type: ignore[arg-type]
            
            return {
                "active_sessions": key_count,
                "ttl_minutes": self.ttl_minutes,
                "max_history": MAX_HISTORY_TURNS
            }
        except Exception as e:
            logger.error(f"Failed to get session stats: {e}")
            return {"error": str(e)}


# Singleton instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get the shared SessionManager singleton.
    
    Returns:
        The SessionManager singleton instance
    """
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
