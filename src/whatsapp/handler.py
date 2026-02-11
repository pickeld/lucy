"""WhatsApp message handling with type-specific classes.

This module provides a class hierarchy for handling different types of WhatsApp
messages (text, image, voice, etc.) with a common base class and type-specific
subclasses for specialized processing.

Usage:
    >>> from whatsapp.handler import create_whatsapp_message
    >>> msg = create_whatsapp_message(payload)
    >>> json_output = msg.to_json()
"""

import base64
import os
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

import httpx

from config import config
from whatsapp.contact import Contact, ContactManager
from whatsapp.group import Group, GroupManager
from utils.logger import logger
from models import WhatsAppMessageDocument


# =============================================================================
# Message Content Types
# =============================================================================


class ContentType(str, Enum):
    """Enumeration of supported message content types."""
    TEXT = "text"
    IMAGE = "image"
    VOICE = "voice"
    VIDEO = "video"
    DOCUMENT = "document"
    STICKER = "sticker"
    LOCATION = "location"
    CONTACT = "contact"
    UNKNOWN = "unknown"


# Global managers
contact_manager = ContactManager()
group_manager = GroupManager()


# make dir for media storage
if not os.path.exists("tmp/images"):
    os.makedirs("tmp/images")


class WhatsappMSG(ABC):
    """Base class for all WhatsApp message types.
    
    This abstract base class provides common functionality for all message types
    and defines the interface that subclasses must implement.
    
    Attributes:
        contact: Contact information of the message sender
        group: Group information if this is a group message
        is_group: Whether this message is from a group chat
        timestamp: Unix timestamp of the message
        message: Text body of the message (may be caption for media)
        to: Recipient of the message
        _payload: Original webhook payload (stored for subclass access)
    """
    
    # Class-level content type - subclasses override this
    content_type: ContentType = ContentType.UNKNOWN
    
    def __init__(self, payload: Dict[str, Any]):
        """Initialize base message attributes from webhook payload.
        
        Args:
            payload: The webhook payload dictionary from WAHA
        """
        self._payload = payload
        self.contact: Contact = contact_manager.get_contact(payload)
        self.group: Group = group_manager.get_group(payload)
        self.is_group: bool = True if self.group.id else False
        self.timestamp: Optional[int] = payload.get("timestamp")
        self.message: Optional[str] = payload.get("body", None)
        self.to: Optional[str] = payload.get("to", None)

    def __str__(self) -> str:
        return f"[{self.content_type.value}] {self.group.name}/{self.contact.name}: {self.message}"

    @abstractmethod
    def to_json(self) -> Dict[str, Any]:
        """Convert message to JSON format optimized for AI models.
        
        This method must be implemented by all subclasses to provide
        type-specific JSON serialization.
        
        Returns:
            Dictionary with standardized message structure
        """
        pass
    
    def _base_json(self) -> Dict[str, Any]:
        """Generate the base JSON structure common to all message types.
        
        Returns:
            Dictionary with common message fields
        """
        # Format timestamp
        timestamp_data = self._format_timestamp()
        
        # Determine chat info
        chat_id = self.group.id if self.is_group else self.contact.id
        chat_name = self.group.name if self.is_group else self.contact.name
        chat_type = "group" if self.is_group else "direct"
        
        # Format message for AI models (same format used in Thread.remember())
        formatted_time = timestamp_data.get("formatted", "")
        sender_name = self.contact.name or "Unknown"
        formatted_message = f"[{formatted_time}] {sender_name}: {self.message or ''}"
        
        return {
            "type": "whatsapp_message",
            "version": "1.0",
            "content_type": self.content_type.value,
            "timestamp": timestamp_data,
            "chat": {
                "id": chat_id or "unknown",
                "name": chat_name or "Unknown",
                "type": chat_type
            },
            "sender": {
                "id": self.contact.id,
                "name": self.contact.name,
                "number": self.contact.number,
                "is_business": self.contact.is_business
            },
            "context": {
                "is_group": self.is_group,
                "formatted_message": formatted_message
            }
        }
    
    def _format_timestamp(self) -> Dict[str, Any]:
        """Format timestamp into multiple representations.
        
        Returns:
            Dictionary with unix, iso, and formatted timestamp versions
        """
        if not self.timestamp:
            return {"unix": None, "iso": None, "formatted": "Unknown"}
        
        try:
            ts = int(self.timestamp)
            dt = datetime.fromtimestamp(ts, tz=ZoneInfo("Asia/Jerusalem"))
            return {
                "unix": ts,
                "iso": dt.isoformat(),
                "formatted": dt.strftime("%d/%m/%Y %H:%M")
            }
        except (ValueError, TypeError):
            return {"unix": self.timestamp, "iso": None, "formatted": "Unknown"}

    def to_rag_document(self, thread_id: str) -> Optional[WhatsAppMessageDocument]:
        """Convert WhatsApp message to a RAG document for vector store indexing.
        
        Creates a WhatsAppMessageDocument from the parsed webhook payload,
        suitable for storing in the RAG vector store for semantic search.
        
        Args:
            thread_id: The conversation thread ID
            
        Returns:
            WhatsAppMessageDocument instance, or None if message has no content
        """
        if not self.message:
            return None
        
        # Determine chat_id and chat_name based on group/direct message
        chat_id = self.group.id if self.is_group else self.contact.id
        chat_name = self.group.name if self.is_group else self.contact.name
        
        return WhatsAppMessageDocument.from_webhook_payload(
            thread_id=thread_id,
            chat_id=chat_id or "unknown",
            chat_name=chat_name or "Unknown",
            is_group=self.is_group,
            sender=self.contact.name or "Unknown",
            message=self.message,
            timestamp=str(self.timestamp) if self.timestamp else "0",
            has_media=False,
            media_type=None,
            media_url=None
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to a dictionary representation.
        
        Returns:
            Dictionary with all message attributes
        """
        def serialize(value: Any) -> Any:
            if hasattr(value, "to_dict"):
                return value.to_dict()
            elif isinstance(value, dict):
                return {k: serialize(v) for k, v in value.items()}
            elif isinstance(value, (list, tuple, set)):
                return [serialize(v) for v in value]
            elif isinstance(value, (str, int, float, bool, type(None))):
                return value
            else:
                return str(value)

        return {
            k: serialize(v)
            for k, v in self.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }


class TextMessage(WhatsappMSG):
    """Text-only WhatsApp message.
    
    Handles plain text messages without any media attachments.
    """
    
    content_type = ContentType.TEXT
    
    def __init__(self, payload: Dict[str, Any]):
        super().__init__(payload)
    
    def to_json(self) -> Dict[str, Any]:
        """Convert text message to JSON format.
        
        Returns:
            Dictionary with text message structure
        """
        result = self._base_json()
        result["content"] = {
            "type": self.content_type.value,
            "text": self.message,
            "media": None
        }
        return result


class MediaMessageBase(WhatsappMSG):
    """Base class for messages with media attachments.
    
    Provides common functionality for handling media (download, encoding, etc.)
    that is shared by image, voice, video, and document messages.
    
    Attributes:
        has_media: Whether media was successfully loaded
        media_base64: Base64-encoded media content
        media_url: URL to the media file
        media_type: MIME type of the media
        saved_path: Local path if media was saved (debug mode)
    """
    
    def __init__(self, payload: Dict[str, Any]):
        super().__init__(payload)
        self.has_media: bool = payload.get("hasMedia", False)
        self.media_base64: Optional[str] = None
        self.media_url: Optional[str] = None
        self.media_type: Optional[str] = None  # MIME type
        self.saved_path: Optional[str] = None
        
        self._load_media(payload)
    
    def _load_media(self, payload: Dict[str, Any]) -> None:
        """Load and process media from payload.
        
        Args:
            payload: The webhook payload containing media information
        """
        if not self.has_media:
            return
        
        media = payload.get("media", {})
        self.media_url = media.get('url')
        self.media_type = media.get('mimetype')
        
        # Only fetch media if URL is present
        if not self.media_url:
            logger.warning(f"Media message has no URL, skipping media download.")
            self.has_media = False
            return
        
        try:
            response = httpx.get(
                self.media_url, 
                headers={"X-Api-Key": config.waha_api_key}
            )
            self.media_base64 = base64.standard_b64encode(response.content).decode("utf-8")
            
            if config.log_level == "DEBUG" and self.media_type:
                # save media to file
                extension = self.media_type.split("/")[-1]
                filename = f"tmp/images/media_{payload.get('id')}.{extension}"
                with open(filename, "wb") as f:
                    f.write(response.content)
                logger.debug(f"Saved media to {filename}")
                self.saved_path = filename
        except Exception as e:
            logger.error(f"Failed to download media: {e}")
            self.has_media = False
    
    def _media_json(self) -> Optional[Dict[str, Any]]:
        """Generate media-specific JSON structure.
        
        Returns:
            Dictionary with media information, or None if no media
        """
        if not self.has_media:
            return None
        
        return {
            "mime_type": self.media_type,
            "url": self.media_url,
            "base64": self.media_base64
        }
    
    def to_rag_document(self, thread_id: str) -> Optional[WhatsAppMessageDocument]:
        """Convert media message to RAG document with media info.
        
        Args:
            thread_id: The conversation thread ID
            
        Returns:
            WhatsAppMessageDocument with media metadata
        """
        # Include caption or description in the RAG document
        message_content = self.message or f"[{self.content_type.value} attachment]"
        
        chat_id = self.group.id if self.is_group else self.contact.id
        chat_name = self.group.name if self.is_group else self.contact.name
        
        return WhatsAppMessageDocument.from_webhook_payload(
            thread_id=thread_id,
            chat_id=chat_id or "unknown",
            chat_name=chat_name or "Unknown",
            is_group=self.is_group,
            sender=self.contact.name or "Unknown",
            message=message_content,
            timestamp=str(self.timestamp) if self.timestamp else "0",
            has_media=self.has_media,
            media_type=self.media_type,
            media_url=self.media_url
        )


class ImageMessage(MediaMessageBase):
    """WhatsApp image message.
    
    Handles image attachments with optional caption and image description.
    
    Attributes:
        description: AI-generated description of the image (optional)
    """
    
    content_type = ContentType.IMAGE
    
    def __init__(self, payload: Dict[str, Any]):
        super().__init__(payload)
        self.description: Optional[str] = None  # Can be set by vision API
    
    def to_json(self) -> Dict[str, Any]:
        """Convert image message to JSON format.
        
        Returns:
            Dictionary with image message structure
        """
        result = self._base_json()
        media_data = self._media_json()
        if media_data:
            media_data["description"] = self.description
        
        result["content"] = {
            "type": self.content_type.value,
            "text": self.message,  # Caption
            "media": media_data
        }
        return result


class VoiceMessage(MediaMessageBase):
    """WhatsApp voice message (push-to-talk).
    
    Handles voice recordings with optional transcription.
    
    Attributes:
        transcription: Text transcription of the voice message (optional)
    """
    
    content_type = ContentType.VOICE
    
    def __init__(self, payload: Dict[str, Any]):
        super().__init__(payload)
        self.transcription: Optional[str] = None  # Can be set by transcription API
    
    def to_json(self) -> Dict[str, Any]:
        """Convert voice message to JSON format.
        
        Returns:
            Dictionary with voice message structure
        """
        result = self._base_json()
        media_data = self._media_json()
        if media_data:
            media_data["transcription"] = self.transcription
        
        result["content"] = {
            "type": self.content_type.value,
            "text": self.transcription,  # Use transcription as text if available
            "media": media_data
        }
        return result


class VideoMessage(MediaMessageBase):
    """WhatsApp video message.
    
    Handles video attachments with optional caption.
    """
    
    content_type = ContentType.VIDEO
    
    def __init__(self, payload: Dict[str, Any]):
        super().__init__(payload)
    
    def to_json(self) -> Dict[str, Any]:
        """Convert video message to JSON format.
        
        Returns:
            Dictionary with video message structure
        """
        result = self._base_json()
        result["content"] = {
            "type": self.content_type.value,
            "text": self.message,  # Caption
            "media": self._media_json()
        }
        return result


class DocumentMessage(MediaMessageBase):
    """WhatsApp document message.
    
    Handles document attachments (PDF, Word, etc.) with optional caption.
    
    Attributes:
        filename: Original filename of the document
    """
    
    content_type = ContentType.DOCUMENT
    
    def __init__(self, payload: Dict[str, Any]):
        super().__init__(payload)
        self.filename: Optional[str] = payload.get("_data", {}).get("filename")
    
    def to_json(self) -> Dict[str, Any]:
        """Convert document message to JSON format.
        
        Returns:
            Dictionary with document message structure
        """
        result = self._base_json()
        media_data = self._media_json()
        if media_data:
            media_data["filename"] = self.filename
        
        result["content"] = {
            "type": self.content_type.value,
            "text": self.message,  # Caption
            "media": media_data
        }
        return result


class StickerMessage(MediaMessageBase):
    """WhatsApp sticker message.
    
    Handles sticker attachments.
    """
    
    content_type = ContentType.STICKER
    
    def __init__(self, payload: Dict[str, Any]):
        super().__init__(payload)
    
    def to_json(self) -> Dict[str, Any]:
        """Convert sticker message to JSON format.
        
        Returns:
            Dictionary with sticker message structure
        """
        result = self._base_json()
        result["content"] = {
            "type": self.content_type.value,
            "text": "[sticker]",
            "media": self._media_json()
        }
        return result


class LocationMessage(WhatsappMSG):
    """WhatsApp location message.
    
    Handles location shares with coordinates.
    
    Attributes:
        latitude: Location latitude
        longitude: Location longitude
        location_name: Optional name/label for the location
    """
    
    content_type = ContentType.LOCATION
    
    def __init__(self, payload: Dict[str, Any]):
        super().__init__(payload)
        location_data = payload.get("_data", {})
        self.latitude: Optional[float] = location_data.get("lat")
        self.longitude: Optional[float] = location_data.get("lng")
        self.location_name: Optional[str] = location_data.get("loc")
    
    def to_json(self) -> Dict[str, Any]:
        """Convert location message to JSON format.
        
        Returns:
            Dictionary with location message structure
        """
        result = self._base_json()
        result["content"] = {
            "type": self.content_type.value,
            "text": self.location_name or f"Location: {self.latitude}, {self.longitude}",
            "media": None,
            "location": {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "name": self.location_name
            }
        }
        return result


class ContactMessage(WhatsappMSG):
    """WhatsApp contact/vCard message.
    
    Handles shared contact information.
    
    Attributes:
        vcard: vCard data string
        shared_contact_name: Name from the shared contact
    """
    
    content_type = ContentType.CONTACT
    
    def __init__(self, payload: Dict[str, Any]):
        super().__init__(payload)
        self.vcard: Optional[str] = payload.get("_data", {}).get("body")
        self.shared_contact_name: Optional[str] = self._extract_contact_name()
    
    def _extract_contact_name(self) -> Optional[str]:
        """Extract contact name from vCard data.
        
        Returns:
            Contact name or None
        """
        if not self.vcard:
            return None
        
        # Simple extraction of FN (formatted name) from vCard
        for line in self.vcard.split('\n'):
            if line.startswith('FN:'):
                return line[3:].strip()
        return None
    
    def to_json(self) -> Dict[str, Any]:
        """Convert contact message to JSON format.
        
        Returns:
            Dictionary with contact message structure
        """
        result = self._base_json()
        result["content"] = {
            "type": self.content_type.value,
            "text": f"Shared contact: {self.shared_contact_name}",
            "media": None,
            "contact": {
                "name": self.shared_contact_name,
                "vcard": self.vcard
            }
        }
        return result


class UnknownMessage(WhatsappMSG):
    """Fallback for unrecognized message types.
    
    Used when the message type cannot be determined.
    """
    
    content_type = ContentType.UNKNOWN
    
    def __init__(self, payload: Dict[str, Any]):
        super().__init__(payload)
        self.raw_type: Optional[str] = payload.get("_data", {}).get("type")
    
    def to_json(self) -> Dict[str, Any]:
        """Convert unknown message to JSON format.
        
        Returns:
            Dictionary with unknown message structure
        """
        result = self._base_json()
        result["content"] = {
            "type": self.content_type.value,
            "text": self.message,
            "media": None,
            "raw_type": self.raw_type
        }
        return result


def create_whatsapp_message(payload: Dict[str, Any]) -> WhatsappMSG:
    """Factory function to create the appropriate message type from payload.
    
    Analyzes the webhook payload and returns an instance of the appropriate
    message subclass based on the content type.
    
    Args:
        payload: The webhook payload dictionary from WAHA
        
    Returns:
        An instance of the appropriate WhatsappMSG subclass
    
    Example:
        >>> msg = create_whatsapp_message(payload)
        >>> json_output = msg.to_json()
        >>> print(json.dumps(json_output, indent=2))
    """
    has_media = payload.get("hasMedia", False)
    data_type = payload.get("_data", {}).get("type", "").lower()
    
    # Check for specific message types
    if data_type == "location":
        return LocationMessage(payload)
    
    if data_type in ["vcard", "contact"]:
        return ContactMessage(payload)
    
    if has_media:
        # Determine media type
        if data_type == "image":
            return ImageMessage(payload)
        elif data_type in ["ptt", "audio"]:  # ptt = push-to-talk (voice memo)
            return VoiceMessage(payload)
        elif data_type == "video":
            return VideoMessage(payload)
        elif data_type == "document":
            return DocumentMessage(payload)
        elif data_type == "sticker":
            return StickerMessage(payload)
        else:
            # Check MIME type as fallback
            mime_type = payload.get("media", {}).get("mimetype", "").lower()
            if mime_type.startswith("image/"):
                return ImageMessage(payload)
            elif mime_type.startswith("audio/"):
                return VoiceMessage(payload)
            elif mime_type.startswith("video/"):
                return VideoMessage(payload)
            elif mime_type.startswith("application/"):
                return DocumentMessage(payload)
    
    # Default to text message if no media and has body
    if payload.get("body"):
        return TextMessage(payload)
    
    # Fallback to unknown
    return UnknownMessage(payload)
