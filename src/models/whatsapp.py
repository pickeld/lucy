"""WhatsApp message document for RAG system.

This module provides the WhatsAppMessageDocument class for handling
WhatsApp messages in the RAG vector store using LlamaIndex.
"""

from datetime import datetime
from typing import Any, Dict, Optional, TYPE_CHECKING
from zoneinfo import ZoneInfo

from pydantic import Field

from .base import BaseRAGDocument, DocumentMetadata, SourceType

if TYPE_CHECKING:
    from llama_index.core.schema import TextNode


class WhatsAppMessageDocument(BaseRAGDocument):
    """Document class for WhatsApp messages.
    
    Extends BaseRAGDocument with WhatsApp-specific fields for handling
    messages from individual chats and group conversations.
    
    Attributes:
        thread_id: Conversation thread ID for context
        chat_id: WhatsApp chat ID (e.g., '972501234567@c.us' or '120363...@g.us')
        chat_name: Display name of the chat or group
        is_group: Whether this message is from a group chat
        sender: Name of the message sender
        message: Original message body text
        has_media: Whether message has media attachment
        media_type: MIME type of media if present
        media_url: URL to media file if present
    """
    
    thread_id: str = Field(..., description="Thread ID for conversation context")
    chat_id: str = Field(..., description="WhatsApp chat ID")
    chat_name: str = Field(..., description="Chat or group display name")
    is_group: bool = Field(default=False, description="Whether group chat")
    sender: str = Field(..., description="Message sender name")
    message: str = Field(..., description="Original message body")
    has_media: bool = Field(default=False, description="Has media attachment")
    media_type: Optional[str] = Field(default=None, description="MIME type of media")
    media_url: Optional[str] = Field(default=None, description="URL to media file")
    
    @classmethod
    def get_source_type(cls) -> SourceType:
        """Get the source type for WhatsApp messages."""
        return SourceType.WHATSAPP
    
    @classmethod
    def from_webhook_payload(
        cls,
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
    ) -> "WhatsAppMessageDocument":
        """Create a WhatsAppMessageDocument from webhook payload data.
        
        This factory method provides a convenient way to create documents
        from the existing WhatsApp webhook handler format.
        
        Args:
            thread_id: Thread ID for conversation context
            chat_id: WhatsApp chat ID
            chat_name: Display name of chat/group
            is_group: Whether group chat
            sender: Message sender name
            message: Message body text
            timestamp: Unix timestamp as string
            has_media: Whether has media attachment
            media_type: MIME type if media
            media_url: Media URL if media
            
        Returns:
            WhatsAppMessageDocument instance
        """
        # Parse Unix timestamp
        try:
            ts = int(timestamp)
            dt = datetime.fromtimestamp(ts, tz=ZoneInfo("UTC"))
        except (ValueError, TypeError):
            dt = datetime.now(ZoneInfo("UTC"))
        
        # Create metadata
        metadata = DocumentMetadata(
            source_id=f"{chat_id}:{timestamp}",
            source_type=SourceType.WHATSAPP,
            created_at=dt,
            custom_fields={
                "thread_id": thread_id,
                "chat_id": chat_id,
                "chat_name": chat_name,
                "is_group": is_group
            }
        )
        
        return cls(
            content=message,
            author=sender,
            timestamp=dt,
            metadata=metadata,
            thread_id=thread_id,
            chat_id=chat_id,
            chat_name=chat_name,
            is_group=is_group,
            sender=sender,
            message=message,
            has_media=has_media,
            media_type=media_type,
            media_url=media_url
        )
    
    def to_searchable_content(self) -> str:
        """Format message for display in search results.
        
        Returns human-readable format: [timestamp] sender in chat: message
        
        Returns:
            Formatted string for search result display
        """
        formatted_time = self.format_timestamp()
        return f"[{formatted_time}] {self.sender} in {self.chat_name}: {self.message}"
    
    def get_embedding_text(self) -> str:
        """Get optimized text for embedding generation.
        
        Creates a semantically rich text that helps embeddings understand:
        - Who sent the message (full name as sender)
        - Where it was sent (chat/group name)
        - The message content
        
        The format is designed to answer questions like:
        - "What did X say?"
        - "What is X's last name?"
        - "Messages in Y group"
        
        Returns:
            Text optimized for semantic embedding and retrieval
        """
        formatted_time = self.format_timestamp()
        
        # Parse sender name parts for better semantic understanding
        sender_parts = self.sender.split() if self.sender else []
        
        # Build semantically explicit text
        parts = [f"Date: {formatted_time}"]
        
        if self.sender:
            parts.append(f"Sender name: {self.sender}")
            # If sender has multiple parts (first + last name), make it explicit
            if len(sender_parts) >= 2:
                parts.append(f"First name: {sender_parts[0]}")
                parts.append(f"Last name: {' '.join(sender_parts[1:])}")
        
        if self.chat_name:
            if self.is_group:
                parts.append(f"Group: {self.chat_name}")
            else:
                parts.append(f"Chat with: {self.chat_name}")
        
        parts.append(f"Message: {self.message}")
        
        return " | ".join(parts)
    
    def to_llama_index_node(self) -> "TextNode":
        """Convert to LlamaIndex TextNode with WhatsApp-specific metadata.
        
        Adds WhatsApp-specific fields to the standard metadata.
        
        Returns:
            LlamaIndex TextNode with full metadata
        """
        from llama_index.core.schema import TextNode
        
        # Get base metadata
        node_metadata = self.metadata.to_qdrant_payload()
        
        # Add WhatsApp-specific fields
        node_metadata.update({
            "document_id": self.id,
            "author": self.author,
            "timestamp": int(self.timestamp.timestamp()),
            "thread_id": self.thread_id,
            "chat_id": self.chat_id,
            "chat_name": self.chat_name,
            "is_group": self.is_group,
            "sender": self.sender,
            "message": self.message,
            "has_media": self.has_media,
            "media_type": self.media_type
        })
        
        return TextNode(
            text=self.get_embedding_text(),
            metadata=node_metadata,
            id_=self.id
        )
