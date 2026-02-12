"""Session data models for conversational context management.

This module defines the data models used to maintain conversational context
across multi-turn interactions with the RAG system.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """Types of entities that can be tracked in conversation context."""
    PERSON = "person"
    GROUP = "group"
    TOPIC = "topic"
    DATE = "date"
    LOCATION = "location"
    EVENT = "event"
    OTHER = "other"


class EntityInfo(BaseModel):
    """Information about an entity mentioned in conversation.
    
    Entities are people, groups, topics, or other nouns that may be
    referenced by pronouns or demonstratives in follow-up questions.
    
    Attributes:
        name: The entity's name or identifier
        entity_type: Type of entity (person, group, topic, etc.)
        first_mentioned_turn: Turn number when entity was first mentioned
        last_mentioned_turn: Turn number when entity was last mentioned
        mentions_count: Number of times this entity has been mentioned
        attributes: Additional attributes known about the entity
    """
    name: str
    entity_type: EntityType
    first_mentioned_turn: int = 0
    last_mentioned_turn: int = 0
    mentions_count: int = 1
    attributes: Dict[str, Any] = Field(default_factory=dict)
    
    def update_mention(self, turn: int) -> None:
        """Update entity with a new mention.
        
        Args:
            turn: The turn number where entity was mentioned
        """
        self.last_mentioned_turn = turn
        self.mentions_count += 1


class Reference(BaseModel):
    """A resolved reference from conversation context.
    
    References are pronouns (he, she, they, it) or demonstratives
    (this, that, these) that have been resolved to specific entities.
    
    Attributes:
        reference_text: The original reference text (e.g., "she", "it")
        resolved_to: The entity name it resolves to
        entity_type: Type of the resolved entity
        turn_number: Turn where this resolution occurred
        confidence: Confidence score of the resolution (0-1)
    """
    reference_text: str
    resolved_to: str
    entity_type: EntityType
    turn_number: int
    confidence: float = 1.0


class Fact(BaseModel):
    """An established fact from conversation/reasoning.
    
    Facts are conclusions or information extracted during conversation
    that can be used for multi-hop reasoning.
    
    Attributes:
        statement: The fact statement
        source_turn: Turn number where this fact was established
        source_messages: IDs of messages that support this fact
        confidence: Confidence score (0-1)
    """
    statement: str
    source_turn: int
    source_messages: List[str] = Field(default_factory=list)
    confidence: float = 1.0


class ConversationTurn(BaseModel):
    """A single turn in the conversation.
    
    Represents a question-answer pair with associated metadata.
    
    Attributes:
        turn_number: Sequential turn number in the conversation
        timestamp: When this turn occurred
        user_query: The original user query
        reformulated_query: The query after reformulation (if different)
        assistant_response: The assistant's response
        retrieved_message_ids: IDs of messages retrieved for this turn
        filters_applied: Filters that were applied during retrieval
        entities_mentioned: Entities mentioned in this turn
    """
    turn_number: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(ZoneInfo("UTC")))
    user_query: str
    reformulated_query: Optional[str] = None
    assistant_response: str = ""
    retrieved_message_ids: List[str] = Field(default_factory=list)
    filters_applied: Dict[str, Any] = Field(default_factory=dict)
    entities_mentioned: List[str] = Field(default_factory=list)


class ConversationSession(BaseModel):
    """Complete conversation session state.
    
    Maintains all context needed for multi-turn conversations with the
    RAG system, including active filters, entity tracking, and history.
    
    Attributes:
        session_id: Unique session identifier
        created_at: When the session was created
        last_activity: Last interaction timestamp
        ttl_minutes: Session timeout in minutes
        
        # Context State
        active_chat_filter: Currently focused chat/group name
        active_sender_filter: Currently focused sender name
        active_time_range: Active time range filter (start, end timestamps)
        
        # Entity Memory
        mentioned_entities: Entities mentioned in this session
        resolved_references: History of resolved references
        
        # Conversation History
        turns: List of conversation turns
        retrieved_context: IDs of all retrieved messages
        
        # Reasoning State
        pending_queries: Sub-queries waiting to be executed
        established_facts: Facts established through conversation
    """
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(ZoneInfo("UTC")))
    last_activity: datetime = Field(default_factory=lambda: datetime.now(ZoneInfo("UTC")))
    ttl_minutes: int = 30
    
    # Context State
    active_chat_filter: Optional[str] = None
    active_sender_filter: Optional[str] = None
    active_time_range: Optional[Tuple[int, int]] = None  # (start_timestamp, end_timestamp)
    
    # Entity Memory
    mentioned_entities: Dict[str, EntityInfo] = Field(default_factory=dict)
    resolved_references: List[Reference] = Field(default_factory=list)
    
    # Conversation History
    turns: List[ConversationTurn] = Field(default_factory=list)
    retrieved_context: List[str] = Field(default_factory=list)
    
    # Reasoning State
    pending_queries: List[str] = Field(default_factory=list)
    established_facts: List[Fact] = Field(default_factory=list)
    
    @property
    def current_turn_number(self) -> int:
        """Get the current turn number."""
        return len(self.turns)
    
    @property
    def is_expired(self) -> bool:
        """Check if the session has expired."""
        now = datetime.now(ZoneInfo("UTC"))
        delta = now - self.last_activity
        return delta.total_seconds() > (self.ttl_minutes * 60)
    
    def touch(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now(ZoneInfo("UTC"))
    
    def add_entity(
        self,
        name: str,
        entity_type: EntityType,
        attributes: Optional[Dict[str, Any]] = None
    ) -> EntityInfo:
        """Add or update an entity in the session.
        
        Args:
            name: Entity name
            entity_type: Type of entity
            attributes: Optional additional attributes
            
        Returns:
            The EntityInfo for this entity
        """
        name_lower = name.lower()
        
        if name_lower in self.mentioned_entities:
            entity = self.mentioned_entities[name_lower]
            entity.update_mention(self.current_turn_number)
            if attributes:
                entity.attributes.update(attributes)
        else:
            entity = EntityInfo(
                name=name,
                entity_type=entity_type,
                first_mentioned_turn=self.current_turn_number,
                last_mentioned_turn=self.current_turn_number,
                attributes=attributes or {}
            )
            self.mentioned_entities[name_lower] = entity
        
        return entity
    
    def get_recent_entities(
        self,
        entity_type: Optional[EntityType] = None,
        limit: int = 5
    ) -> List[EntityInfo]:
        """Get recently mentioned entities.
        
        Args:
            entity_type: Optional filter by entity type
            limit: Maximum number of entities to return
            
        Returns:
            List of EntityInfo sorted by last mention (most recent first)
        """
        entities = list(self.mentioned_entities.values())
        
        if entity_type:
            entities = [e for e in entities if e.entity_type == entity_type]
        
        # Sort by last mentioned turn, descending
        entities.sort(key=lambda e: e.last_mentioned_turn, reverse=True)
        
        return entities[:limit]
    
    def resolve_reference(
        self,
        reference_text: str,
        entity_type: Optional[EntityType] = None
    ) -> Optional[str]:
        """Attempt to resolve a pronoun or demonstrative reference.
        
        Args:
            reference_text: The reference to resolve (e.g., "she", "it")
            entity_type: Optional hint about expected entity type
            
        Returns:
            Resolved entity name or None if cannot resolve
        """
        reference_lower = reference_text.lower()
        
        # Map common pronouns to entity types
        pronoun_type_hints = {
            "he": EntityType.PERSON,
            "him": EntityType.PERSON,
            "his": EntityType.PERSON,
            "she": EntityType.PERSON,
            "her": EntityType.PERSON,
            "hers": EntityType.PERSON,
            "they": EntityType.PERSON,  # Could be group too
            "them": EntityType.PERSON,
            "their": EntityType.PERSON,
            "it": None,  # Could be topic, event, etc.
            "this": None,
            "that": None,
            "these": None,
            "those": None,
        }
        
        # Determine entity type to look for
        target_type = entity_type or pronoun_type_hints.get(reference_lower)
        
        # Get recent entities of the appropriate type
        recent = self.get_recent_entities(entity_type=target_type, limit=3)
        
        if not recent:
            # Fall back to any recent entity
            recent = self.get_recent_entities(limit=3)
        
        if recent:
            # Return the most recently mentioned entity
            resolved = recent[0].name
            
            # Record this resolution
            self.resolved_references.append(Reference(
                reference_text=reference_text,
                resolved_to=resolved,
                entity_type=recent[0].entity_type,
                turn_number=self.current_turn_number,
                confidence=0.8 if target_type else 0.5
            ))
            
            return resolved
        
        return None
    
    def add_turn(
        self,
        user_query: str,
        assistant_response: str = "",
        reformulated_query: Optional[str] = None,
        retrieved_ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        entities: Optional[List[str]] = None
    ) -> ConversationTurn:
        """Add a new conversation turn.
        
        Args:
            user_query: The user's query
            assistant_response: The assistant's response
            reformulated_query: Query after reformulation
            retrieved_ids: IDs of retrieved messages
            filters: Filters applied during retrieval
            entities: Entities mentioned in this turn
            
        Returns:
            The new ConversationTurn
        """
        turn = ConversationTurn(
            turn_number=self.current_turn_number,
            user_query=user_query,
            reformulated_query=reformulated_query,
            assistant_response=assistant_response,
            retrieved_message_ids=retrieved_ids or [],
            filters_applied=filters or {},
            entities_mentioned=entities or []
        )
        
        self.turns.append(turn)
        self.touch()
        
        # Add retrieved IDs to overall context
        if retrieved_ids:
            for msg_id in retrieved_ids:
                if msg_id not in self.retrieved_context:
                    self.retrieved_context.append(msg_id)
        
        return turn
    
    def get_conversation_history(
        self,
        max_turns: int = 10
    ) -> List[Dict[str, str]]:
        """Get conversation history in chat format.
        
        Args:
            max_turns: Maximum number of turns to include
            
        Returns:
            List of {"role": "user"|"assistant", "content": "..."} dicts
        """
        history = []
        recent_turns = self.turns[-max_turns:] if max_turns else self.turns
        
        for turn in recent_turns:
            history.append({"role": "user", "content": turn.user_query})
            if turn.assistant_response:
                history.append({"role": "assistant", "content": turn.assistant_response})
        
        return history
    
    def get_active_filters(self) -> Dict[str, Any]:
        """Get currently active filters as a dictionary.
        
        Returns:
            Dictionary of filter name to value
        """
        filters = {}
        
        if self.active_chat_filter:
            filters["filter_chat_name"] = self.active_chat_filter
        
        if self.active_sender_filter:
            filters["filter_sender"] = self.active_sender_filter
        
        if self.active_time_range:
            # Convert to filter_days for backward compatibility
            # or use start/end timestamps directly
            filters["time_range"] = self.active_time_range
        
        return filters
    
    def set_chat_context(self, chat_name: Optional[str]) -> None:
        """Set or clear the active chat filter.
        
        Args:
            chat_name: Chat/group name to focus on, or None to clear
        """
        self.active_chat_filter = chat_name
        if chat_name:
            self.add_entity(chat_name, EntityType.GROUP)
        self.touch()
    
    def set_sender_context(self, sender_name: Optional[str]) -> None:
        """Set or clear the active sender filter.
        
        Args:
            sender_name: Sender name to focus on, or None to clear
        """
        self.active_sender_filter = sender_name
        if sender_name:
            self.add_entity(sender_name, EntityType.PERSON)
        self.touch()
    
    def add_fact(
        self,
        statement: str,
        source_messages: Optional[List[str]] = None,
        confidence: float = 1.0
    ) -> Fact:
        """Add an established fact to the session.
        
        Args:
            statement: The fact statement
            source_messages: IDs of messages supporting this fact
            confidence: Confidence score (0-1)
            
        Returns:
            The created Fact
        """
        fact = Fact(
            statement=statement,
            source_turn=self.current_turn_number,
            source_messages=source_messages or [],
            confidence=confidence
        )
        self.established_facts.append(fact)
        return fact
    
    def get_context_summary(self) -> str:
        """Get a human-readable summary of the current context.
        
        Returns:
            Summary string describing active context
        """
        parts = []
        
        if self.active_chat_filter:
            parts.append(f"Chat: {self.active_chat_filter}")
        
        if self.active_sender_filter:
            parts.append(f"Sender: {self.active_sender_filter}")
        
        if self.active_time_range:
            start, end = self.active_time_range
            start_dt = datetime.fromtimestamp(start, ZoneInfo("UTC"))
            end_dt = datetime.fromtimestamp(end, ZoneInfo("UTC"))
            parts.append(f"Time: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
        
        recent_people = self.get_recent_entities(EntityType.PERSON, limit=3)
        if recent_people:
            names = [e.name for e in recent_people]
            parts.append(f"People: {', '.join(names)}")
        
        recent_topics = self.get_recent_entities(EntityType.TOPIC, limit=3)
        if recent_topics:
            topics = [e.name for e in recent_topics]
            parts.append(f"Topics: {', '.join(topics)}")
        
        if not parts:
            return "No active context"
        
        return " | ".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize session to dictionary for storage.
        
        Returns:
            Dictionary representation of the session
        """
        return self.model_dump(mode="json")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationSession":
        """Deserialize session from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            ConversationSession instance
        """
        return cls.model_validate(data)
