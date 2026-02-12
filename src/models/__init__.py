"""RAG Document Classes for multi-source data standardization.

This package provides Pydantic v2 models for handling multiple data sources
in the RAG system, ensuring consistent interface for vector store integration
with LlamaIndex.

Taxonomy:
    - Source: Where data came from (whatsapp, telegram, email, paperless, manual, ...)
    - ContentType: What the data is (text, image, voice, document, call_recording, ...)

Classes:
    - BaseRAGDocument: Abstract base class for all document types
    - DocumentMetadata: Common metadata structure
    - Source: Enum for data origin platform/channel
    - ContentType: Enum for data content classification
    - SourceType: DEPRECATED — use Source + ContentType instead
    - WhatsAppMessageDocument: WhatsApp message documents
    - FileDocument: PDF, Word, and text file documents
    - CallRecordingDocument: Transcribed call recordings

Usage:
    from models import WhatsAppMessageDocument, FileDocument, CallRecordingDocument
    from models import Source, ContentType
    
    # Create a WhatsApp message document
    doc = WhatsAppMessageDocument.from_webhook_payload(
        thread_id="thread-123",
        chat_id="972501234567@c.us",
        chat_name="John Doe",
        is_group=False,
        sender="John",
        message="Hello!",
        timestamp="1704067200"
    )
    
    # Convert to LlamaIndex TextNode for vector store
    text_node = doc.to_llama_index_node()
"""

from .base import (
    BaseRAGDocument,
    ContentType,
    DocumentMetadata,
    Source,
    SourceType,  # Deprecated — backward compat
    source_type_to_source_and_content,
)
from .whatsapp import WhatsAppMessageDocument
from .document import FileDocument, FileType
from .call_recording import CallRecordingDocument, CallType

__all__ = [
    # Taxonomy enums (new)
    "Source",
    "ContentType",
    # Base classes
    "BaseRAGDocument",
    "DocumentMetadata",
    # Deprecated (backward compat)
    "SourceType",
    "source_type_to_source_and_content",
    # WhatsApp
    "WhatsAppMessageDocument",
    # Documents
    "FileDocument",
    "FileType",
    # Call Recordings
    "CallRecordingDocument",
    "CallType",
]
