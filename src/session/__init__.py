"""Session management for conversational context.

This module provides session management for maintaining context across
multi-turn conversations with the RAG system.
"""

from .models import (
    ConversationSession,
    ConversationTurn,
    EntityInfo,
    EntityType,
    Reference,
    Fact,
)
from .manager import SessionManager, get_session_manager

__all__ = [
    "ConversationSession",
    "ConversationTurn",
    "EntityInfo",
    "EntityType",
    "Reference",
    "Fact",
    "SessionManager",
    "get_session_manager",
]
