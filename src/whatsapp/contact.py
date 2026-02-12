"""WhatsApp contact management with Redis caching.

This module provides classes for managing WhatsApp contacts, including
fetching contact information from WAHA and caching in Redis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from config import settings
from utils.globals import send_request
from utils.logger import logger
from utils.redis_conn import redis_get, redis_set


class ContactManager:
    """Manager for WhatsApp contact operations with Redis caching."""
    
    def __init__(self) -> None:
        pass

    def get_contact(self, payload) -> Contact:
        """Get contact information from payload, using cache when available.
        
        Args:
            payload: The webhook payload containing contact information
            
        Returns:
            Contact object with extracted information
        """
        _from = payload.get("from", None)
        _participant = payload.get("participant", None)
        
        # Direct message: from ends with @c.us
        if _from and _from.endswith("@c.us"):
            contact_data = redis_get(f"contact:{_from}")
            if not contact_data:
                contact_data = self.fetch_contact(_from)
                redis_set(f"contact:{_from}", contact_data)
            contact = Contact()
            contact.extract(contact_data)
            return contact
        
        # Group message: participant ends with @c.us
        elif _participant and _participant.endswith("@c.us"):
            contact_data = redis_get(f"contact:{_participant}")
            if not contact_data:
                contact_data = self.fetch_contact(_participant)
                redis_set(f"contact:{_participant}", contact_data)
            contact = Contact()
            contact.extract(contact_data)
            return contact
        
        # Linked ID case: participant ends with @lid
        elif _participant and _participant.endswith("@lid"):
            contact_id = redis_get(f"contact_alias:{_participant}")
            if contact_id:
                contact_data = redis_get(f"contact:{contact_id}")
                if contact_data:
                    contact = Contact()
                    contact.extract(contact_data)
                    return contact
                else:
                    contact_data = self.fetch_contact(contact_id)
                    redis_set(f"contact:{contact_id}",
                              contact_data)
            else:
                contact_data = self.fetch_contact(_participant)

            redis_set(f"contact_alias:{_participant}",
                      contact_data.get("id"))
            redis_set(f"contact:{contact_data.get('id')}",
                      contact_data)
            contact = Contact()
            contact.extract(contact_data)
            return contact
        
        # Fallback: return empty Contact with data from payload if available
        logger.warning(f"Could not resolve contact from payload: from={_from}, participant={_participant}")
        contact = Contact()
        # Try to extract name from _data.notifyName if available
        notify_name = payload.get("_data", {}).get("notifyName")
        if notify_name:
            contact.name = notify_name
        contact.id = _participant or _from
        return contact

    def fetch_contact(self, contact_id: str) -> Dict[str, Any]:
        """Fetch contact information from WAHA API.
        
        Args:
            contact_id: The WhatsApp contact ID
            
        Returns:
            Dictionary with contact information
        """
        params = {"contactId": contact_id, "session": settings.waha_session_name}
        try:
            response = send_request(
                method="GET", endpoint="/api/contacts", params=params)
            return response
        except Exception as e:
            logger.error(f"WAHA contact fetch failed for {contact_id}: {e}")
            return {}


@dataclass
class Contact:
    """Represents a WhatsApp contact.
    
    Attributes:
        id: WhatsApp contact ID (e.g., '1234567890@c.us')
        number: Phone number
        name: Display name (from contact list or pushname)
        pushname: Name set by the user themselves
        short_name: Shortened version of name
        status_muted: Whether status updates are muted
        is_business: Whether this is a business account
        is_enterprise: Whether this is an enterprise account
        type: Contact type
        is_me: Whether this is the current user
        is_user: Whether this is a user contact
        is_group: Whether this is a group
        is_wa_contact: Whether this is a WhatsApp contact
        is_my_contact: Whether this is in the address book
        is_blocked: Whether this contact is blocked
    """
    id: Optional[str] = None
    number: Optional[str] = None
    name: Optional[str] = None
    pushname: Optional[str] = None
    short_name: Optional[str] = None
    status_muted: bool = False
    is_business: bool = False
    is_enterprise: bool = False
    type: Optional[str] = None
    is_me: bool = False
    is_user: bool = False
    is_group: bool = False
    is_wa_contact: bool = False
    is_my_contact: bool = False
    is_blocked: bool = False

    def __str__(self) -> str:
        return f"Name: {self.name}, Number: {self.number}"

    def extract(self, data: Dict[str, Any]) -> "Contact":
        """Extract contact information from API response data.
        
        Args:
            data: Dictionary with contact data from WAHA API
            
        Returns:
            Self with extracted data
        """
        self.id = data.get("id")
        self.number = data.get("number")
        self.name = data.get("name", data.get("pushname"))
        self.pushname = data.get("pushname")
        self.short_name = data.get("shortName")
        self.status_muted = bool(data.get("statusMuted", False))
        self.is_business = bool(data.get("isBusiness", False))
        self.is_enterprise = bool(data.get("isEnterprise", False))
        self.type = data.get("type")
        self.is_me = bool(data.get("isMe", False))
        self.is_user = bool(data.get("isUser", False))
        self.is_group = bool(data.get("isGroup", False))
        self.is_wa_contact = bool(data.get("isWAContact", False))
        self.is_my_contact = bool(data.get("isMyContact", False))
        self.is_blocked = bool(data.get("isBlocked", False))
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert contact to dictionary representation.
        
        Returns:
            Dictionary with all contact attributes
        """
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }
