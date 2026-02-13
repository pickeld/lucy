"""WhatsApp integration module.

This module provides classes and functions for handling WhatsApp messages,
contacts, and groups through the WAHA API.

Main exports:
    - create_whatsapp_message: Factory function to create message objects
    - WhatsappMSG: Base class for all message types
    - Contact, ContactManager: Contact handling
    - Group, GroupManager: Group handling
    - ContentType: Enum of message content types
"""

from whatsapp.contact import Contact, ContactManager
from whatsapp.group import Group, GroupManager
from whatsapp.handler import (
    ContentType,
    WhatsappMSG,
    TextMessage,
    ImageMessage,
    VoiceMessage,
    VideoMessage,
    DocumentMessage,
    StickerMessage,
    LocationMessage,
    ContactMessage,
    UnknownMessage,
    MediaMessageBase,
    create_whatsapp_message,
    contact_manager,
    group_manager,
)

__all__ = [
    # Contact management
    "Contact",
    "ContactManager",
    "contact_manager",
    # Group management
    "Group",
    "GroupManager",
    "group_manager",
    # Message types
    "ContentType",
    "WhatsappMSG",
    "TextMessage",
    "ImageMessage",
    "VoiceMessage",
    "VideoMessage",
    "DocumentMessage",
    "StickerMessage",
    "LocationMessage",
    "ContactMessage",
    "UnknownMessage",
    "MediaMessageBase",
    # Factory function
    "create_whatsapp_message",
]
