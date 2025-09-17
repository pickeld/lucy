from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from config import config
from utiles.globals import send_request
from utiles.logger import logger


class ContactManager:
    def __init__(self) -> None:
        self.contacts = {}

    def get_contact(self, payload) -> "Contact":
        contact_id = payload.get("from")
        if not contact_id:
            raise ValueError("Payload missing 'from' field")
        if contact_id not in self.contacts:
            self.contacts[contact_id] = self.fetch_contact(contact_id)
        return self.contacts[contact_id]

    def fetch_contact(self, contact_id: str) -> "Contact":
        params = {"contactId": contact_id, "session": config.waha_session_name}
        try:
            resp = send_request(
                method="GET", endpoint="/api/contacts", params=params)
            if isinstance(resp, dict):
                return Contact().extract(resp)
            logger.error(
                f"Unexpected WAHA response for {contact_id}: {type(resp)}")
        except Exception as e:
            logger.error(f"WAHA contact fetch failed for {contact_id}: {e}")
        # Graceful fallback so callers always get a Contact
        return Contact(id=contact_id, number=contact_id)


@dataclass
class Contact:
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
        return f"Contact Name({self.name}, number={self.number}, name={self.name})"

    def extract(self, data: Dict[str, Any]) -> "Contact":
        # populate fields in-place and return self
        self.id = data.get("id")
        self.number = data.get("number")
        self.name = data.get("name")
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
