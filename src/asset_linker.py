"""Asset linker for creating asset↔asset edges at ingestion time.

Provides helper functions that ingestion pipelines call to:
- Generate canonical asset_id values (shared across chunks)
- Create structural edges in the entity_db asset_asset_edges table
- Populate asset graph metadata on Qdrant payloads

Relation types:
    thread_member  — email thread membership, chat thread grouping
    attachment_of  — email↔attachment, document↔embedded file
    chunk_of       — parent document↔chunk (Paperless, call recordings)
    reply_to       — WhatsApp quoted messages, email In-Reply-To
    references     — shared ID number, same calendar UID, etc.
    transcript_of  — call recording↔transcript chunks
"""

import hashlib
import uuid
from typing import Dict, List, Optional

from utils.logger import logger


# ---------------------------------------------------------------------------
# Asset ID generation
# ---------------------------------------------------------------------------

def generate_asset_id(source: str, source_native_id: str) -> str:
    """Generate a canonical asset_id from source + native ID.

    All chunks of the same document/email/call share the same asset_id.
    Individual chunks get unique Qdrant point IDs but share this asset_id.

    Examples:
        generate_asset_id("whatsapp", "972501234567@c.us:1708012345")
          → "wa:972501234567@c.us:1708012345"
        generate_asset_id("gmail", "msg_abc123")
          → "gm:msg_abc123"
        generate_asset_id("paperless", "42")
          → "pl:42"
        generate_asset_id("call_recording", "a1b2c3d4e5f6...")
          → "cr:a1b2c3d4e5f6..."

    Args:
        source: Data source (whatsapp, gmail, paperless, call_recording)
        source_native_id: Source-specific unique identifier

    Returns:
        Canonical asset_id string
    """
    prefix_map = {
        "whatsapp": "wa",
        "gmail": "gm",
        "paperless": "pl",
        "call_recording": "cr",
        "email": "em",
        "telegram": "tg",
        "manual": "mn",
    }
    prefix = prefix_map.get(source, source[:2])
    return f"{prefix}:{source_native_id}"


# ---------------------------------------------------------------------------
# Edge creation helpers (non-blocking, never fail the caller)
# ---------------------------------------------------------------------------

def link_attachment(
    parent_ref: str,
    child_ref: str,
    provenance: str = "ingestion",
) -> None:
    """Create an attachment_of edge: child is an attachment of parent.

    Args:
        parent_ref: Parent asset's source_id (e.g. the email)
        child_ref: Child asset's source_id (e.g. the attachment)
        provenance: Where this edge was derived from
    """
    try:
        import identity_db
        identity_db.link_assets(
            src_asset_ref=child_ref,
            dst_asset_ref=parent_ref,
            relation_type="attachment_of",
            provenance=provenance,
        )
    except Exception as e:
        logger.debug(f"Failed to link attachment edge (non-critical): {e}")


def link_thread_member(
    thread_id: str,
    asset_ref: str,
    provenance: str = "ingestion",
) -> None:
    """Create a thread_member edge: asset belongs to thread.

    The thread_id acts as a virtual "thread node" — edges go from
    thread_id → asset_ref.

    Args:
        thread_id: Thread identifier (email thread ID, chat_id, etc.)
        asset_ref: Asset's source_id
        provenance: Where this edge was derived from
    """
    try:
        import identity_db
        identity_db.link_assets(
            src_asset_ref=f"thread:{thread_id}",
            dst_asset_ref=asset_ref,
            relation_type="thread_member",
            provenance=provenance,
        )
    except Exception as e:
        logger.debug(f"Failed to link thread member edge (non-critical): {e}")


def link_chunk(
    parent_ref: str,
    chunk_ref: str,
    provenance: str = "ingestion",
) -> None:
    """Create a chunk_of edge: chunk belongs to parent document.

    Args:
        parent_ref: Parent asset's source_id (the whole document/recording)
        chunk_ref: Chunk's source_id (individual chunk)
        provenance: Where this edge was derived from
    """
    try:
        import identity_db
        identity_db.link_assets(
            src_asset_ref=chunk_ref,
            dst_asset_ref=parent_ref,
            relation_type="chunk_of",
            provenance=provenance,
        )
    except Exception as e:
        logger.debug(f"Failed to link chunk edge (non-critical): {e}")


def link_reply(
    reply_ref: str,
    original_ref: str,
    provenance: str = "ingestion",
) -> None:
    """Create a reply_to edge: reply is a response to original.

    Args:
        reply_ref: The reply asset's source_id
        original_ref: The original asset's source_id
        provenance: Where this edge was derived from
    """
    try:
        import identity_db
        identity_db.link_assets(
            src_asset_ref=reply_ref,
            dst_asset_ref=original_ref,
            relation_type="reply_to",
            provenance=provenance,
        )
    except Exception as e:
        logger.debug(f"Failed to link reply edge (non-critical): {e}")


def link_transcript(
    transcript_ref: str,
    recording_ref: str,
    provenance: str = "ingestion",
) -> None:
    """Create a transcript_of edge: transcript chunk belongs to recording.

    Args:
        transcript_ref: Transcript chunk's source_id
        recording_ref: Audio recording's asset_id
        provenance: Where this edge was derived from
    """
    try:
        import identity_db
        identity_db.link_assets(
            src_asset_ref=transcript_ref,
            dst_asset_ref=recording_ref,
            relation_type="transcript_of",
            provenance=provenance,
        )
    except Exception as e:
        logger.debug(f"Failed to link transcript edge (non-critical): {e}")


def link_reference(
    asset_ref_a: str,
    asset_ref_b: str,
    provenance: str = "ingestion",
) -> None:
    """Create a references edge: two assets share a common reference.

    Used for shared ID numbers, calendar UIDs, file hashes, etc.

    Args:
        asset_ref_a: First asset's source_id
        asset_ref_b: Second asset's source_id
        provenance: What the shared reference is
    """
    try:
        import identity_db
        identity_db.link_assets(
            src_asset_ref=asset_ref_a,
            dst_asset_ref=asset_ref_b,
            relation_type="references",
            provenance=provenance,
        )
    except Exception as e:
        logger.debug(f"Failed to link reference edge (non-critical): {e}")
