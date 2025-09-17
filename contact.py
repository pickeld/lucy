from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from config import config
from utiles.globals import send_request
from utiles.logger import logger
from utiles.redis_conn import redis_get, redis_set


class ContactManager:
    def __init__(self) -> None:
        pass

    def get_contact(self, payload) -> Contact:
        _from = payload.get("from", None)
        _participant = payload.get("participant", None)
        if _from and _from.endswith("@c.us"):
            logger.debug(f"Redis lookup for contact: contact:{_from}")
            contact_data = redis_get(f"contact:{_from}")
            if not contact_data:
                contact_data = self.fetch_contact(_from)
                redis_set(f"contact:{_from}", contact_data)
            contact = Contact()
            contact.extract(contact_data)
            return contact
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

    def fetch_contact(self, contact_id: str):
        params = {"contactId": contact_id, "session": config.waha_session_name}
        try:
            response = send_request(
                method="GET", endpoint="/api/contacts", params=params)
            logger.debug(f"Fetched contact data: {response}")
            return response
        except Exception as e:
            logger.error(f"WAHA contact fetch failed for {contact_id}: {e}")
            return {}


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
        return f"Name: {self.name}, Number: {self.number}"

    def extract(self, data) -> "Contact":
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

    def to_dict(self):
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }
