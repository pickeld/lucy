"""Person resolver for linking assets to entity IDs at ingestion time.

Resolves sender names, WhatsApp IDs, phone numbers, and email addresses
to person entity IDs using the Entity Store.  This module is the bridge
between the ingestion pipeline and the person-asset graph.

Used by:
- WhatsApp message handler (resolve sender → person_id)
- Paperless sync (resolve document author/correspondents)
- Call recording sync (resolve participants + phone_number)
- Gmail sync (resolve sender/recipients)
- Entity extractor (resolve mentioned names → mentioned_person_ids)

The resolver is designed to be:
- **Fast**: In-memory LRU cache per chat session avoids repeated DB lookups
- **Fuzzy**: Falls back to name matching when phone/email are unavailable
- **Non-blocking**: Resolution failure never blocks asset ingestion
"""

import functools
from typing import Dict, List, Optional, Tuple

from utils.logger import logger


# ---------------------------------------------------------------------------
# LRU caches — avoid repeated DB lookups within the same session
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=512)
def _resolve_by_whatsapp_id(whatsapp_id: str) -> Optional[int]:
    """Resolve a WhatsApp ID to a person_id (cached)."""
    try:
        import identity_db
        person = identity_db.get_person_by_whatsapp_id(whatsapp_id)
        return person["id"] if person else None
    except Exception:
        return None


@functools.lru_cache(maxsize=512)
def _resolve_by_phone(phone: str) -> Optional[int]:
    """Resolve a phone number to a person_id (cached)."""
    try:
        import identity_db
        return identity_db.find_person_by_phone(phone)
    except Exception:
        return None


@functools.lru_cache(maxsize=512)
def _resolve_by_email(email: str) -> Optional[int]:
    """Resolve an email address to a person_id (cached)."""
    try:
        import identity_db
        return identity_db.find_person_by_email(email)
    except Exception:
        return None


@functools.lru_cache(maxsize=1024)
def _resolve_by_name(name: str) -> Optional[int]:
    """Resolve a display name or alias to a person_id (cached).

    Uses exact match on canonical_name or aliases. Returns the first
    match — may be ambiguous for common first names, so callers should
    prefer phone/email/whatsapp_id when available.
    """
    try:
        import identity_db
        person = identity_db.get_person_by_name(name)
        return person["id"] if person else None
    except Exception:
        return None


def clear_caches() -> None:
    """Clear all resolution caches.

    Call this after entity merges, bulk imports, or when cache
    staleness is a concern.
    """
    _resolve_by_whatsapp_id.cache_clear()
    _resolve_by_phone.cache_clear()
    _resolve_by_email.cache_clear()
    _resolve_by_name.cache_clear()


# ---------------------------------------------------------------------------
# Main resolution API
# ---------------------------------------------------------------------------

def resolve_person(
    name: Optional[str] = None,
    whatsapp_id: Optional[str] = None,
    phone: Optional[str] = None,
    email: Optional[str] = None,
) -> Optional[int]:
    """Resolve a person to their entity ID using all available identifiers.

    Tries identifiers in priority order (most specific → least):
    1. WhatsApp ID (e.g., '972501234567@c.us')
    2. Phone number (e.g., '+972501234567')
    3. Email address
    4. Display name / alias

    Args:
        name: Display name or alias (e.g., 'Shiran Waintrob', 'שירן')
        whatsapp_id: WhatsApp contact ID
        phone: Phone number
        email: Email address

    Returns:
        Person entity ID (integer), or None if not found
    """
    # 1. WhatsApp ID — most specific for WhatsApp messages
    if whatsapp_id:
        pid = _resolve_by_whatsapp_id(whatsapp_id)
        if pid is not None:
            return pid

    # 2. Phone number — works across sources
    if phone:
        pid = _resolve_by_phone(phone)
        if pid is not None:
            return pid

    # 3. Email — works for Gmail, Paperless
    if email:
        pid = _resolve_by_email(email)
        if pid is not None:
            return pid

    # 4. Name — least specific, may be ambiguous
    if name:
        pid = _resolve_by_name(name)
        if pid is not None:
            return pid

    return None


def resolve_persons_from_names(names: List[str]) -> List[int]:
    """Resolve a list of names to person IDs.

    Used for resolving participants, recipients, and mentioned names.
    Skips names that can't be resolved.

    Args:
        names: List of display names to resolve

    Returns:
        List of resolved person IDs (may be shorter than input)
    """
    person_ids = []
    seen: set = set()
    for name in names:
        if not name or not name.strip():
            continue
        pid = resolve_person(name=name.strip())
        if pid is not None and pid not in seen:
            seen.add(pid)
            person_ids.append(pid)
    return person_ids


def resolve_whatsapp_sender(
    sender_name: str,
    chat_id: str,
    is_group: bool,
) -> Optional[int]:
    """Resolve a WhatsApp message sender to a person_id.

    For 1:1 chats, the chat_id IS the sender's WhatsApp ID.
    For group chats, we can only resolve by sender name.

    Args:
        sender_name: The sender's display name
        chat_id: WhatsApp chat ID (e.g., '972501234567@c.us' or group ID)
        is_group: Whether this is a group chat

    Returns:
        Person entity ID, or None
    """
    # For 1:1 chats, the chat_id is the contact's WhatsApp ID
    if not is_group and chat_id and (
        chat_id.endswith("@c.us") or chat_id.endswith("@lid")
    ):
        pid = _resolve_by_whatsapp_id(chat_id)
        if pid is not None:
            return pid

        # Extract phone number from chat_id (strip @c.us suffix)
        phone_part = chat_id.split("@")[0]
        if phone_part.isdigit():
            pid = _resolve_by_phone(phone_part)
            if pid is not None:
                return pid

    # Fall back to name resolution
    return resolve_person(name=sender_name)


def resolve_and_link(
    asset_type: str,
    asset_ref: str,
    sender_name: Optional[str] = None,
    sender_whatsapp_id: Optional[str] = None,
    sender_phone: Optional[str] = None,
    sender_email: Optional[str] = None,
    participant_names: Optional[List[str]] = None,
    mentioned_names: Optional[List[str]] = None,
) -> Tuple[List[int], List[int]]:
    """Resolve all persons associated with an asset and create links.

    This is the all-in-one function for ingestion pipelines.
    It resolves sender/participants → person_ids and mentioned names →
    mentioned_person_ids, then creates person_assets links in the
    entity store.

    Args:
        asset_type: Asset kind ('whatsapp_msg', 'document', 'call_recording', 'gmail')
        asset_ref: Qdrant point source_id
        sender_name: Primary sender/author name
        sender_whatsapp_id: Sender's WhatsApp ID (for WhatsApp messages)
        sender_phone: Sender's phone number
        sender_email: Sender's email address
        participant_names: Additional participant names (for calls, group chats)
        mentioned_names: Names mentioned in the content

    Returns:
        Tuple of (person_ids, mentioned_person_ids) — the resolved entity IDs
    """
    import identity_db

    person_ids: List[int] = []
    mentioned_person_ids: List[int] = []
    seen_ids: set = set()

    # Resolve primary sender/author
    sender_pid = resolve_person(
        name=sender_name,
        whatsapp_id=sender_whatsapp_id,
        phone=sender_phone,
        email=sender_email,
    )
    if sender_pid is not None:
        person_ids.append(sender_pid)
        seen_ids.add(sender_pid)
        try:
            identity_db.link_person_asset(
                person_id=sender_pid,
                asset_type=asset_type,
                asset_ref=asset_ref,
                role="sender",
            )
        except Exception as e:
            logger.debug(f"Failed to link sender asset: {e}")

    # Resolve additional participants
    if participant_names:
        for pname in participant_names:
            pid = resolve_person(name=pname)
            if pid is not None and pid not in seen_ids:
                person_ids.append(pid)
                seen_ids.add(pid)
                try:
                    identity_db.link_person_asset(
                        person_id=pid,
                        asset_type=asset_type,
                        asset_ref=asset_ref,
                        role="participant",
                    )
                except Exception as e:
                    logger.debug(f"Failed to link participant asset: {e}")

    # Resolve mentioned names
    if mentioned_names:
        for mname in mentioned_names:
            pid = resolve_person(name=mname)
            if pid is not None and pid not in seen_ids:
                mentioned_person_ids.append(pid)
                seen_ids.add(pid)
                try:
                    identity_db.link_person_asset(
                        person_id=pid,
                        asset_type=asset_type,
                        asset_ref=asset_ref,
                        role="mentioned",
                    )
                except Exception as e:
                    logger.debug(f"Failed to link mentioned asset: {e}")

    return person_ids, mentioned_person_ids
