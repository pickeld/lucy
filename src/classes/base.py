"""Base classes for RAG document standardization.

This module provides the foundational classes for handling multiple data sources
in the RAG system, including WhatsApp messages, documents, and call recordings.

All document types inherit from BaseRAGDocument to ensure consistent interface
for vector store integration.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, field_validator


class SourceType(str, Enum):
    """Enumeration of supported data source types for RAG."""
    
    WHATSAPP_MESSAGE = "whatsapp_message"
    DOCUMENT = "document"
    CALL_RECORDING = "call_recording"


class DocumentMetadata(BaseModel):
    """Common metadata structure for all RAG document types.
    
    Provides standardized metadata fields that apply across all data sources,
    enabling consistent filtering, searching, and display in the RAG system.
    
    Attributes:
        source_id: Unique identifier from the source system
        source_type: Type of data source (WhatsApp, document, call, etc.)
        created_at: When the original content was created
        indexed_at: When the document was added to the RAG vector store
        tags: Optional list of tags for filtering and categorization
        language: Detected or specified language of the content
        custom_fields: Additional source-specific metadata
    """
    
    source_id: str = Field(..., description="Unique identifier from source system")
    source_type: SourceType = Field(..., description="Type of data source")
    created_at: datetime = Field(..., description="When original content was created")
    indexed_at: datetime = Field(
        default_factory=lambda: datetime.now(ZoneInfo("UTC")),
        description="When document was added to RAG"
    )
    tags: List[str] = Field(default_factory=list, description="Tags for filtering")
    language: Optional[str] = Field(default=None, description="Content language (e.g., 'en', 'he')")
    custom_fields: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional source-specific metadata"
    )
    
    def to_qdrant_payload(self) -> Dict[str, Any]:
        """Convert metadata to Qdrant-compatible payload format.
        
        Returns:
            Dictionary suitable for Qdrant vector store metadata
        """
        return {
            "source_id": self.source_id,
            "source_type": self.source_type.value,
            "created_at": int(self.created_at.timestamp()),
            "indexed_at": int(self.indexed_at.timestamp()),
            "tags": self.tags,
            "language": self.language,
            **self.custom_fields
        }
    
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            SourceType: lambda v: v.value
        }
    }


class BaseRAGDocument(BaseModel, ABC):
    """Abstract base class for all RAG document types.
    
    Provides a common interface for documents from different sources
    (WhatsApp messages, files, call recordings) to be processed uniformly
    by the RAG system.
    
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
    
    def to_langchain_document(self) -> "Document":
        """Convert to LangChain Document for vector store integration.
        
        Creates a LangChain Document with the embedding text as page_content
        and all relevant metadata for filtering and retrieval.
        
        Returns:
            LangChain Document object ready for vector store indexing
        """
        from langchain_core.documents import Document
        
        # Combine base metadata with source-specific fields
        langchain_metadata = self.metadata.to_qdrant_payload()
        langchain_metadata.update({
            "document_id": self.id,
            "author": self.author,
            "timestamp": int(self.timestamp.timestamp()),
        })
        
        return Document(
            page_content=self.get_embedding_text(),
            metadata=langchain_metadata
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary representation.
        
        Returns:
            Dictionary with all document fields
        """
        return self.model_dump(mode="json")
    
    @classmethod
    def get_source_type(cls) -> SourceType:
        """Get the source type for this document class.
        
        Subclasses should override this to return their specific source type.
        
        Returns:
            SourceType enum value
        """
        raise NotImplementedError("Subclasses must implement get_source_type()")
    
    def format_timestamp(self, timezone: str = "Asia/Jerusalem") -> str:
        """Format timestamp for human-readable display.
        
        Args:
            timezone: Timezone for display (default: Asia/Jerusalem)
            
        Returns:
            Formatted datetime string (e.g., "31/12/2024 10:30")
        """
        try:
            tz = ZoneInfo(timezone)
            dt = self.timestamp.astimezone(tz)
            return dt.strftime("%d/%m/%Y %H:%M")
        except Exception:
            return str(self.timestamp)
    
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            SourceType: lambda v: v.value
        },
        "arbitrary_types_allowed": True
    }
