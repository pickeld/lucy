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


# =============================================================================
# RAG Errors
# =============================================================================

class RAGError(WhatsAppGPTError):
    """Base exception for RAG-related errors."""
    pass
