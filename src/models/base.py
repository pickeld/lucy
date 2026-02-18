"""Base classes for RAG document standardization.

This module provides the foundational classes for handling multiple data sources
in the RAG system, including messages, documents, and call recordings from any
channel plugin (WhatsApp, Telegram, Email, Paperless-NG, etc.).

All document types inherit from BaseRAGDocument to ensure consistent interface
for vector store integration with LlamaIndex.

Taxonomy:
    - Source: Where data came from (whatsapp, telegram, email, paperless, manual, ...)
    - ContentType: What the data is (text, image, voice, document, call_recording, ...)
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from uuid import uuid4
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from llama_index.core.schema import TextNode


# =============================================================================
# Source — where data came from (maps 1:1 to plugins)
# =============================================================================

class Source(str, Enum):
    """The platform or plugin that originated the data.
    
    Each value maps to a channel plugin. Used for filtering by origin
    and for routing data to the correct plugin for display/actions.
    """
    
    # Messaging platforms
    WHATSAPP = "whatsapp"
    TELEGRAM = "telegram"
    SLACK = "slack"
    DISCORD = "discord"
    SMS = "sms"
    
    # Email
    EMAIL = "email"
    
    # Document management
    PAPERLESS = "paperless"
    
    # Web/Social
    WEB_SCRAPE = "web_scrape"
    SOCIAL_MEDIA = "social_media"
    
    # Manual / API
    MANUAL = "manual"
    API_IMPORT = "api_import"


# =============================================================================
# ContentType — what the data is (source-agnostic)
# =============================================================================

class ContentType(str, Enum):
    """What the data actually is, independent of where it came from.
    
    A document (PDF) can come from Paperless, WhatsApp, Email, or manual upload.
    A voice note can come from WhatsApp or Telegram. These are independent axes.
    """
    
    # Messages
    TEXT = "text"
    IMAGE = "image"
    VOICE = "voice"
    VIDEO = "video"
    STICKER = "sticker"
    LOCATION = "location"
    CONTACT_CARD = "contact_card"
    
    # Documents
    DOCUMENT = "document"
    SPREADSHEET = "spreadsheet"
    
    # Audio
    CALL_RECORDING = "call_recording"
    
    # Conversation chunks (synthetic, created by RAG chunking)
    CONVERSATION_CHUNK = "conversation_chunk"
    
    # Other
    UNKNOWN = "unknown"


# =============================================================================
# Backward compatibility — SourceType is deprecated, use Source instead
# =============================================================================

class SourceType(str, Enum):
    """DEPRECATED: Use Source and ContentType instead.
    
    This enum conflated sources (platforms) with content types.
    Kept for backward compatibility with existing Qdrant payloads
    that have a 'source_type' field.
    
    Migration: existing payloads with source_type='whatsapp' map to
    Source.WHATSAPP. Payloads with source_type='document' map to
    ContentType.DOCUMENT with Source.MANUAL (or whatever plugin created them).
    """
    
    # These are actually sources (platforms)
    WHATSAPP = "whatsapp"
    TELEGRAM = "telegram"
    SLACK = "slack"
    DISCORD = "discord"
    SMS = "sms"
    EMAIL = "email"
    
    # These are actually content types (kept for legacy payload compat)
    DOCUMENT = "document"
    CALL_RECORDING = "call_recording"
    VOICE_NOTE = "voice_note"
    
    # These are sources
    WEB_SCRAPE = "web_scrape"
    SOCIAL_MEDIA = "social_media"
    MANUAL_ENTRY = "manual_entry"
    API_IMPORT = "api_import"


def source_type_to_source_and_content(
    source_type: SourceType,
) -> tuple:
    """Convert a legacy SourceType to (Source, ContentType) pair.
    
    Used during migration to map old payloads to the new taxonomy.
    
    Args:
        source_type: Legacy SourceType value
        
    Returns:
        Tuple of (Source, ContentType)
    """
    # Map legacy SourceType values that are actually content types
    content_type_map = {
        SourceType.DOCUMENT: (Source.MANUAL, ContentType.DOCUMENT),
        SourceType.CALL_RECORDING: (Source.MANUAL, ContentType.CALL_RECORDING),
        SourceType.VOICE_NOTE: (Source.MANUAL, ContentType.VOICE),
    }
    
    if source_type in content_type_map:
        return content_type_map[source_type]
    
    # Map legacy SourceType values that are actually sources
    source_map = {
        SourceType.WHATSAPP: Source.WHATSAPP,
        SourceType.TELEGRAM: Source.TELEGRAM,
        SourceType.SLACK: Source.SLACK,
        SourceType.DISCORD: Source.DISCORD,
        SourceType.SMS: Source.SMS,
        SourceType.EMAIL: Source.EMAIL,
        SourceType.WEB_SCRAPE: Source.WEB_SCRAPE,
        SourceType.SOCIAL_MEDIA: Source.SOCIAL_MEDIA,
        SourceType.MANUAL_ENTRY: Source.MANUAL,
        SourceType.API_IMPORT: Source.API_IMPORT,
    }
    
    source = source_map.get(source_type, Source.MANUAL)
    return (source, ContentType.TEXT)


# =============================================================================
# DocumentMetadata
# =============================================================================

class DocumentMetadata(BaseModel):
    """Common metadata structure for all RAG document types.
    
    Provides standardized metadata fields that apply across all data sources,
    enabling consistent filtering, searching, and display in the RAG system.
    
    Attributes:
        source_id: Unique identifier from the source system
        source: Where the data came from (plugin/platform)
        content_type: What the data is (text, image, document, etc.)
        source_type: DEPRECATED — legacy field for backward compat with existing payloads
        created_at: When the original content was created
        indexed_at: When the document was added to the RAG vector store
        tags: Optional list of tags for filtering and categorization
        language: Detected or specified language of the content
        person_ids: Entity IDs of persons structurally linked to this asset (sender, author, participants)
        mentioned_person_ids: Entity IDs of persons mentioned in the content
        custom_fields: Additional source-specific metadata
    """
    
    source_id: str = Field(..., description="Unique identifier from source system")
    source: Source = Field(..., description="Where the data came from (plugin/platform)")
    content_type: ContentType = Field(
        default=ContentType.TEXT,
        description="What the data is (text, image, document, etc.)"
    )
    # Legacy field — kept for backward compat with existing Qdrant payloads
    source_type: Optional[SourceType] = Field(
        default=None,
        description="DEPRECATED: Use source and content_type instead"
    )
    created_at: datetime = Field(..., description="When original content was created")
    indexed_at: datetime = Field(
        default_factory=lambda: datetime.now(ZoneInfo("UTC")),
        description="When document was added to RAG"
    )
    tags: List[str] = Field(default_factory=list, description="Tags for filtering")
    language: Optional[str] = Field(default=None, description="Content language (e.g., 'en', 'he')")
    # Person-asset graph fields: entity IDs linking this asset to persons
    person_ids: List[int] = Field(
        default_factory=list,
        description="Entity IDs of persons structurally linked (sender, author, participants)"
    )
    mentioned_person_ids: List[int] = Field(
        default_factory=list,
        description="Entity IDs of persons mentioned in the content"
    )
    custom_fields: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional source-specific metadata"
    )
    
    def to_qdrant_payload(self) -> Dict[str, Any]:
        """Convert metadata to Qdrant-compatible payload format.
        
        Writes both new fields (source, content_type) and legacy field
        (source_type) for backward compatibility during transition.
        Includes person_ids and mentioned_person_ids for the person-asset graph.
        
        Returns:
            Dictionary suitable for Qdrant vector store metadata
        """
        payload = {
            "source_id": self.source_id,
            "source": self.source.value,
            "content_type": self.content_type.value,
            # Legacy field — write for backward compat with existing search code
            "source_type": self.source_type.value if self.source_type else self.source.value,
            "created_at": int(self.created_at.timestamp()),
            "indexed_at": int(self.indexed_at.timestamp()),
            "tags": self.tags,
            "language": self.language,
            # Person-asset graph: entity IDs for Qdrant keyword filtering
            "person_ids": self.person_ids,
            "mentioned_person_ids": self.mentioned_person_ids,
            **self.custom_fields
        }
        return payload
    
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            Source: lambda v: v.value,
            ContentType: lambda v: v.value,
            SourceType: lambda v: v.value,
        }
    }


# =============================================================================
# BaseRAGDocument
# =============================================================================

class BaseRAGDocument(BaseModel, ABC):
    """Abstract base class for all RAG document types.
    
    Provides a common interface for documents from different sources
    (WhatsApp messages, files, call recordings, Telegram messages, emails)
    to be processed uniformly by the RAG system.
    
    All subclasses must implement:
        - to_searchable_content(): Format content for display in search results
        - get_embedding_text(): Get optimized text for embedding generation
    
    Attributes:
        id: Unique document identifier (auto-generated UUID if not provided)
        content: Main textual content for embedding and retrieval
        author: Who created or sent the content
        timestamp: When the content was created
        metadata: Structured metadata for filtering and context
    """
    
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique document identifier"
    )
    content: str = Field(..., description="Main textual content for embedding")
    author: str = Field(..., description="Creator/sender of the content")
    timestamp: datetime = Field(..., description="When content was created")
    metadata: DocumentMetadata = Field(..., description="Structured metadata")
    
    @field_validator("content")
    @classmethod
    def content_must_not_be_empty(cls, v: str) -> str:
        """Validate that content is not empty or whitespace only."""
        if not v or not v.strip():
            raise ValueError("Content cannot be empty")
        return v.strip()
    
    @abstractmethod
    def to_searchable_content(self) -> str:
        """Format content for display in search results.
        
        Should return a human-readable format including relevant context
        like sender, timestamp, and source information.
        
        Returns:
            Formatted string suitable for displaying in search results
        """
        pass
    
    @abstractmethod
    def get_embedding_text(self) -> str:
        """Get optimized text for embedding generation.
        
        May include additional context to improve semantic search quality.
        
        Returns:
            Text optimized for embedding generation
        """
        pass
    
    def to_llama_index_node(self) -> "TextNode":
        """Convert to LlamaIndex TextNode for vector store integration.
        
        Creates a LlamaIndex TextNode with the embedding text as content
        and all relevant metadata for filtering and retrieval.
        
        Returns:
            LlamaIndex TextNode object ready for vector store indexing
        """
        from llama_index.core.schema import TextNode
        
        # Combine base metadata with source-specific fields
        node_metadata = self.metadata.to_qdrant_payload()
        node_metadata.update({
            "document_id": self.id,
            "author": self.author,
            "timestamp": int(self.timestamp.timestamp()),
        })
        
        return TextNode(
            text=self.get_embedding_text(),
            metadata=node_metadata,
            id_=self.id
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary representation.
        
        Returns:
            Dictionary with all document fields
        """
        return self.model_dump(mode="json")
    
    @classmethod
    def get_source(cls) -> Source:
        """Get the source for this document class.
        
        Subclasses should override this to return their specific source.
        
        Returns:
            Source enum value
        """
        raise NotImplementedError("Subclasses must implement get_source()")
    
    @classmethod
    def get_content_type(cls) -> ContentType:
        """Get the default content type for this document class.
        
        Subclasses should override this to return their specific content type.
        
        Returns:
            ContentType enum value
        """
        raise NotImplementedError("Subclasses must implement get_content_type()")
    
    # Backward compat — delegates to get_source()
    @classmethod
    def get_source_type(cls) -> SourceType:
        """DEPRECATED: Use get_source() and get_content_type() instead."""
        source = cls.get_source()
        # Map Source back to SourceType for legacy callers
        try:
            return SourceType(source.value)
        except ValueError:
            return SourceType.MANUAL_ENTRY
    
    def format_timestamp(self, timezone: str = "") -> str:
        """Format timestamp for human-readable display.
        
        Args:
            timezone: Timezone for display (reads from settings if empty)
            
        Returns:
            Formatted datetime string (e.g., "31/12/2024 10:30")
        """
        try:
            if not timezone:
                from config import settings as _settings
                timezone = _settings.get("timezone", "Asia/Jerusalem")
            tz = ZoneInfo(timezone)
            dt = self.timestamp.astimezone(tz)
            return dt.strftime("%d/%m/%Y %H:%M")
        except Exception:
            return str(self.timestamp)
    
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            Source: lambda v: v.value,
            ContentType: lambda v: v.value,
            SourceType: lambda v: v.value,
        },
        "arbitrary_types_allowed": True
    }
