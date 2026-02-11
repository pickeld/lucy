"""File document class for RAG system.

This module provides the FileDocument class for handling
documents like PDFs, Word files, and text files in the RAG vector store using LlamaIndex.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from zoneinfo import ZoneInfo

from pydantic import Field, field_validator

from .base import BaseRAGDocument, DocumentMetadata, SourceType

if TYPE_CHECKING:
    from llama_index.core.schema import TextNode


class FileType(str, Enum):
    """Supported file types for document ingestion."""
    
    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    TXT = "txt"
    MD = "md"
    RTF = "rtf"
    HTML = "html"
    CSV = "csv"
    XLSX = "xlsx"
    XLS = "xls"
    UNKNOWN = "unknown"
    
    @classmethod
    def from_extension(cls, extension: str) -> "FileType":
        """Get FileType from file extension.
        
        Args:
            extension: File extension (with or without dot)
            
        Returns:
            Corresponding FileType enum value
        """
        ext = extension.lower().lstrip(".")
        try:
            return cls(ext)
        except ValueError:
            return cls.UNKNOWN


class FileDocument(BaseRAGDocument):
    """Document class for file-based documents.
    
    Extends BaseRAGDocument with file-specific fields for handling
    PDFs, Word documents, text files, and other document formats.
    
    Attributes:
        file_path: Original file path or storage location
        file_name: File name without path
        file_type: Type of file (PDF, DOCX, TXT, etc.)
        file_size: Size in bytes
        mime_type: MIME type of the file
        page_count: Number of pages (if applicable)
        chunk_index: Index of this chunk if document was split
        total_chunks: Total number of chunks if document was split
        title: Document title if available
        description: Document description or summary
    """
    
    file_path: str = Field(..., description="Original file path or storage location")
    file_name: str = Field(..., description="File name without path")
    file_type: FileType = Field(..., description="Type of file")
    file_size: int = Field(default=0, description="Size in bytes")
    mime_type: Optional[str] = Field(default=None, description="MIME type")
    page_count: Optional[int] = Field(default=None, description="Number of pages")
    chunk_index: int = Field(default=0, description="Index of this chunk")
    total_chunks: int = Field(default=1, description="Total chunks in document")
    title: Optional[str] = Field(default=None, description="Document title")
    description: Optional[str] = Field(default=None, description="Document description")
    
    @field_validator("file_size")
    @classmethod
    def file_size_must_be_non_negative(cls, v: int) -> int:
        """Validate that file size is non-negative."""
        if v < 0:
            raise ValueError("File size cannot be negative")
        return v
    
    @classmethod
    def get_source_type(cls) -> SourceType:
        """Get the source type for file documents."""
        return SourceType.DOCUMENT
    
    @classmethod
    def from_file(
        cls,
        file_path: str,
        content: str,
        author: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        chunk_index: int = 0,
        total_chunks: int = 1,
        created_at: Optional[datetime] = None,
        tags: Optional[List[str]] = None
    ) -> "FileDocument":
        """Create a FileDocument from a file path and extracted content.
        
        This factory method provides a convenient way to create documents
        from file processing pipelines.
        
        Args:
            file_path: Path to the file
            content: Extracted text content
            author: Document author (defaults to "Unknown")
            title: Document title (defaults to filename)
            description: Document description
            chunk_index: Index if content was chunked
            total_chunks: Total chunks in document
            created_at: Original creation date (defaults to file mtime)
            tags: Optional tags for categorization
            
        Returns:
            FileDocument instance
        """
        path = Path(file_path)
        
        # Determine file properties
        file_name = path.name
        file_type = FileType.from_extension(path.suffix)
        
        # Get file size if file exists
        try:
            file_size = path.stat().st_size if path.exists() else 0
        except Exception:
            file_size = 0
        
        # Get file modification time as creation date if not provided
        if created_at is None:
            try:
                mtime = path.stat().st_mtime if path.exists() else datetime.now().timestamp()
                created_at = datetime.fromtimestamp(mtime, tz=ZoneInfo("UTC"))
            except Exception:
                created_at = datetime.now(ZoneInfo("UTC"))
        
        # Determine MIME type
        mime_types = {
            FileType.PDF: "application/pdf",
            FileType.DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            FileType.DOC: "application/msword",
            FileType.TXT: "text/plain",
            FileType.MD: "text/markdown",
            FileType.RTF: "application/rtf",
            FileType.HTML: "text/html",
            FileType.CSV: "text/csv",
            FileType.XLSX: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            FileType.XLS: "application/vnd.ms-excel",
        }
        mime_type = mime_types.get(file_type, "application/octet-stream")
        
        # Create metadata
        metadata = DocumentMetadata(
            source_id=f"file:{file_path}:{chunk_index}",
            source_type=SourceType.DOCUMENT,
            created_at=created_at,
            tags=tags or [],
            custom_fields={
                "file_path": file_path,
                "file_name": file_name,
                "file_type": file_type.value,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks
            }
        )
        
        return cls(
            content=content,
            author=author or "Unknown",
            timestamp=created_at,
            metadata=metadata,
            file_path=file_path,
            file_name=file_name,
            file_type=file_type,
            file_size=file_size,
            mime_type=mime_type,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            title=title or file_name,
            description=description
        )
    
    def to_searchable_content(self) -> str:
        """Format document for display in search results.
        
        Returns human-readable format with title and content preview.
        
        Returns:
            Formatted string for search result display
        """
        formatted_time = self.format_timestamp()
        chunk_info = f" (chunk {self.chunk_index + 1}/{self.total_chunks})" if self.total_chunks > 1 else ""
        
        title = self.title or self.file_name
        return f"[{formatted_time}] {title}{chunk_info} by {self.author}: {self.content[:200]}..."
    
    def get_embedding_text(self) -> str:
        """Get optimized text for embedding generation.
        
        Includes title and description for better semantic context.
        
        Returns:
            Text optimized for embedding
        """
        parts = []
        
        if self.title:
            parts.append(f"Title: {self.title}")
        
        if self.description:
            parts.append(f"Description: {self.description}")
        
        parts.append(self.content)
        
        return "\n\n".join(parts)
    
    def to_llama_index_node(self) -> "TextNode":
        """Convert to LlamaIndex TextNode with file-specific metadata.
        
        Adds file-specific fields to the standard metadata.
        
        Returns:
            LlamaIndex TextNode with full metadata
        """
        from llama_index.core.schema import TextNode
        
        # Get base metadata
        node_metadata = self.metadata.to_qdrant_payload()
        
        # Add file-specific fields
        node_metadata.update({
            "document_id": self.id,
            "author": self.author,
            "timestamp": int(self.timestamp.timestamp()),
            "file_path": self.file_path,
            "file_name": self.file_name,
            "file_type": self.file_type.value,
            "file_size": self.file_size,
            "mime_type": self.mime_type,
            "page_count": self.page_count,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "title": self.title,
            "description": self.description
        })
        
        return TextNode(
            text=self.get_embedding_text(),
            metadata=node_metadata,
            id_=self.id
        )
