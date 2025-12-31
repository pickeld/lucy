"""Custom exception classes for WhatsApp-GPT application.

This module provides a hierarchy of exceptions for better error handling
and debugging throughout the application.
"""

from typing import Optional, Any, Dict


class WhatsAppGPTError(Exception):
    """Base exception for all WhatsApp-GPT errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# =============================================================================
# Configuration Errors
# =============================================================================

class ConfigurationError(WhatsAppGPTError):
    """Raised when there's a configuration issue."""
    pass


class MissingConfigError(ConfigurationError):
    """Raised when a required configuration value is missing."""
    
    def __init__(self, config_key: str):
        super().__init__(
            f"Missing required configuration: {config_key}",
            {"config_key": config_key}
        )


# =============================================================================
# External API Errors
# =============================================================================

class ExternalAPIError(WhatsAppGPTError):
    """Base exception for external API errors."""
    
    def __init__(
        self, 
        service: str, 
        message: str, 
        status_code: Optional[int] = None,
        response_body: Optional[str] = None
    ):
        self.service = service
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(
            f"[{service}] {message}",
            {
                "service": service,
                "status_code": status_code,
                "response_body": response_body
            }
        )


class WAHAAPIError(ExternalAPIError):
    """Raised when WAHA API requests fail."""
    
    def __init__(
        self, 
        message: str, 
        status_code: Optional[int] = None,
        response_body: Optional[str] = None
    ):
        super().__init__("WAHA", message, status_code, response_body)


class OpenAIAPIError(ExternalAPIError):
    """Raised when OpenAI API requests fail."""
    
    def __init__(
        self, 
        message: str, 
        status_code: Optional[int] = None,
        response_body: Optional[str] = None
    ):
        super().__init__("OpenAI", message, status_code, response_body)


class QdrantAPIError(ExternalAPIError):
    """Raised when Qdrant API requests fail."""
    
    def __init__(
        self, 
        message: str, 
        status_code: Optional[int] = None,
        response_body: Optional[str] = None
    ):
        super().__init__("Qdrant", message, status_code, response_body)


class LangGraphAPIError(ExternalAPIError):
    """Raised when LangGraph API requests fail."""
    
    def __init__(
        self, 
        message: str, 
        status_code: Optional[int] = None,
        response_body: Optional[str] = None
    ):
        super().__init__("LangGraph", message, status_code, response_body)


# =============================================================================
# Storage Errors
# =============================================================================

class StorageError(WhatsAppGPTError):
    """Base exception for storage-related errors."""
    pass


class RedisError(StorageError):
    """Raised when Redis operations fail."""
    
    def __init__(self, operation: str, message: str, key: Optional[str] = None):
        super().__init__(
            f"Redis {operation} failed: {message}",
            {"operation": operation, "key": key}
        )


class VectorStoreError(StorageError):
    """Raised when vector store operations fail."""
    
    def __init__(self, operation: str, message: str):
        super().__init__(
            f"Vector store {operation} failed: {message}",
            {"operation": operation}
        )


# =============================================================================
# Message Processing Errors
# =============================================================================

class MessageProcessingError(WhatsAppGPTError):
    """Raised when message processing fails."""
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None, 
        chat_id: Optional[str] = None
    ):
        final_details = details or {}
        if chat_id:
            final_details["chat_id"] = chat_id
        super().__init__(message, final_details)


class InvalidPayloadError(MessageProcessingError):
    """Raised when webhook payload is invalid or malformed."""
    
    def __init__(self, reason: str, payload: Optional[Dict] = None):
        super().__init__(
            f"Invalid webhook payload: {reason}",
            {"reason": reason, "payload_keys": list(payload.keys()) if payload else None}
        )


class MediaProcessingError(MessageProcessingError):
    """Raised when media processing fails."""
    
    def __init__(self, media_type: str, reason: str):
        super().__init__(
            f"Failed to process {media_type}: {reason}",
            {"media_type": media_type}
        )


# =============================================================================
# RAG Errors
# =============================================================================

class RAGError(WhatsAppGPTError):
    """Base exception for RAG-related errors."""
    pass


class EmbeddingError(RAGError):
    """Raised when embedding generation fails."""
    
    def __init__(self, message: str):
        super().__init__(f"Embedding generation failed: {message}")


class SearchError(RAGError):
    """Raised when RAG search fails."""
    
    def __init__(self, query: str, reason: str):
        super().__init__(
            f"Search failed: {reason}",
            {"query": query[:100] if query else None}
        )


class QueryError(RAGError):
    """Raised when RAG query processing fails."""
    
    def __init__(self, question: str, reason: str):
        super().__init__(
            f"Query processing failed: {reason}",
            {"question": question[:100] if question else None}
        )


# =============================================================================
# Contact/Group Errors
# =============================================================================

class ContactError(WhatsAppGPTError):
    """Raised when contact operations fail."""
    
    def __init__(self, contact_id: str, operation: str, reason: str):
        super().__init__(
            f"Contact {operation} failed for {contact_id}: {reason}",
            {"contact_id": contact_id, "operation": operation}
        )


class GroupError(WhatsAppGPTError):
    """Raised when group operations fail."""
    
    def __init__(self, group_id: str, operation: str, reason: str):
        super().__init__(
            f"Group {operation} failed for {group_id}: {reason}",
            {"group_id": group_id, "operation": operation}
        )


# =============================================================================
# Retry-related
# =============================================================================

class RetryExhaustedError(WhatsAppGPTError):
    """Raised when all retry attempts have been exhausted."""
    
    def __init__(self, operation: str, attempts: int, last_error: Optional[Exception] = None):
        self.last_error = last_error
        super().__init__(
            f"All {attempts} retry attempts exhausted for {operation}",
            {
                "operation": operation,
                "attempts": attempts,
                "last_error": str(last_error) if last_error else None
            }
        )
