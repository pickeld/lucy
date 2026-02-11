"""RAG Document Classes for multi-source data standardization.

This package provides Pydantic v2 models for handling multiple data sources
in the RAG system, ensuring consistent interface for vector store integration
with LlamaIndex.

Classes:
    - BaseRAGDocument: Abstract base class for all document types
    - DocumentMetadata: Common metadata structure
    - SourceType: Enum for data source classification
    - WhatsAppMessageDocument: WhatsApp message documents
    - FileDocument: PDF, Word, and text file documents
    - CallRecordingDocument: Transcribed call recordings

Usage:
    from classes import WhatsAppMessageDocument, FileDocument, CallRecordingDocument
    
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

from .base import BaseRAGDocument, DocumentMetadata, SourceType
from .whatsapp import WhatsAppMessageDocument
from .document import FileDocument, FileType
from .call_recording import CallRecordingDocument, CallType

__all__ = [
    # Base classes
    "BaseRAGDocument",
    "DocumentMetadata",
    "SourceType",
    # WhatsApp
    "WhatsAppMessageDocument",
    # Documents
    "FileDocument",
    "FileType",
    # Call Recordings
    "CallRecordingDocument",
    "CallType",
]
