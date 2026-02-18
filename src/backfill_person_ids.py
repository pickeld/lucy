#!/usr/bin/env python3
"""Backfill person_ids on existing Qdrant points.

Scans all points in the Qdrant collection, resolves their sender /
chat_name / participants to entity IDs via the person_resolver module,
and updates the Qdrant payload with ``person_ids`` and
``mentioned_person_ids`` fields.  Also populates the ``person_assets``
junction table in the entity store.

Usage:
    # From the src/ directory (so imports resolve):
    cd src && python backfill_person_ids.py

    # Dry-run (no writes):
    cd src && python backfill_person_ids.py --dry-run

    # Limit to N points:
    cd src && python backfill_person_ids.py --limit 500

This is a one-time migration script.  It is idempotent — re-running
it won't create duplicate links because both the Qdrant payload
update and the person_assets INSERT use upsert semantics.
"""

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure src/ is on the import path
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings
from utils.logger import logger


def _resolve_point(payload: Dict[str, Any]) -> tuple:
    """Resolve a Qdrant point's payload to (person_ids, mentioned_person_ids, asset_type, asset_ref).

    Returns:
        Tuple of (person_ids, mentioned_person_ids, asset_type, asset_ref)
    """
    from person_resolver import resolve_person, resolve_persons_from_names

    source = payload.get("source", "")
    source_id = payload.get("source_id", "")
    sender = payload.get("sender", "")
    chat_name = payload.get("chat_name", "")
    content_type = payload.get("content_type", "")

    person_ids: List[int] = []
    mentioned_ids: List[int] = []
    asset_type = "unknown"
    asset_ref = source_id

    # Determine asset type from source
    if source == "whatsapp":
        asset_type = "whatsapp_msg"
    elif source == "paperless":
        asset_type = "document"
    elif source == "call_recording":
        asset_type = "call_recording"
    elif source == "gmail":
        asset_type = "gmail"
    elif content_type == "conversation_chunk":
        asset_type = "whatsapp_msg"  # Conversation chunks are WhatsApp
    else:
        asset_type = source or "unknown"

    # Skip system/entity_store nodes
    if source in ("system", "entity_store"):
        return [], [], asset_type, asset_ref

    # Resolve sender
    if sender and sender != "Unknown":
        # For WhatsApp 1:1 chats, try to use chat_id as WhatsApp ID
        chat_id = payload.get("chat_id", "")
        is_group = payload.get("is_group", False)

        if source == "whatsapp" and not is_group and chat_id:
            pid = resolve_person(name=sender, whatsapp_id=chat_id)
        elif source == "gmail":
            # Try to extract email from sender field
            pid = resolve_person(name=sender, email=sender if "@" in sender else None)
        elif source == "call_recording":
            phone = payload.get("phone_number", "")
            pid = resolve_person(name=sender, phone=phone or None)
        else:
            pid = resolve_person(name=sender)

        if pid is not None:
            person_ids.append(pid)

    # Resolve participants (call recordings)
    participants = payload.get("participants", [])
    if isinstance(participants, list) and len(participants) > 1:
        for pname in participants[1:]:  # Skip first (already resolved as sender)
            if pname and pname != "Unknown":
                pid = resolve_person(name=pname)
                if pid is not None and pid not in person_ids:
                    person_ids.append(pid)

    return person_ids, mentioned_ids, asset_type, asset_ref


def backfill(
    dry_run: bool = False,
    limit: Optional[int] = None,
    batch_size: int = 100,
) -> Dict[str, int]:
    """Scan all Qdrant points and backfill person_ids.

    Args:
        dry_run: If True, don't write anything — just count what would change
        limit: Maximum points to process (None = all)
        batch_size: Points per scroll batch

    Returns:
        Dict with counts: scanned, updated, links_created, skipped, errors
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointVectors
    import entity_db

    qdrant_host = settings.qdrant_host
    qdrant_port = int(settings.qdrant_port)
    collection_name = settings.rag_collection_name

    logger.info(
        f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}, "
        f"collection: {collection_name}"
    )
    client = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=30)

    # Verify collection exists
    try:
        info = client.get_collection(collection_name)
        total_points = info.points_count or 0
        logger.info(f"Collection has {total_points} points")
    except Exception as e:
        logger.error(f"Cannot access collection {collection_name}: {e}")
        return {"error": str(e)}

    scanned = 0
    updated = 0
    links_created = 0
    skipped = 0
    errors = 0

    # Scroll through all points
    offset = None
    start_time = time.time()

    while True:
        if limit and scanned >= limit:
            break

        try:
            points, next_offset = client.scroll(
                collection_name=collection_name,
                offset=offset,
                limit=min(batch_size, (limit - scanned) if limit else batch_size),
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:
            logger.error(f"Scroll failed at offset {offset}: {e}")
            errors += 1
            break

        if not points:
            break

        batch_updates = []

        for point in points:
            scanned += 1
            payload = point.payload or {}

            # Skip points that already have person_ids populated
            existing_pids = payload.get("person_ids", [])
            if existing_pids and isinstance(existing_pids, list) and len(existing_pids) > 0:
                skipped += 1
                continue

            try:
                person_ids, mentioned_ids, asset_type, asset_ref = _resolve_point(payload)
            except Exception as e:
                logger.debug(f"Resolution failed for point {point.id}: {e}")
                errors += 1
                continue

            if not person_ids and not mentioned_ids:
                skipped += 1
                continue

            if not dry_run:
                # Update Qdrant payload
                try:
                    client.set_payload(
                        collection_name=collection_name,
                        payload={
                            "person_ids": person_ids,
                            "mentioned_person_ids": mentioned_ids,
                        },
                        points=[point.id],
                    )
                    updated += 1
                except Exception as e:
                    logger.debug(f"Payload update failed for point {point.id}: {e}")
                    errors += 1
                    continue

                # Create person_assets links
                for pid in person_ids:
                    try:
                        entity_db.link_person_asset(
                            person_id=pid,
                            asset_type=asset_type,
                            asset_ref=asset_ref,
                            role="sender",
                        )
                        links_created += 1
                    except Exception:
                        pass

                for pid in mentioned_ids:
                    try:
                        entity_db.link_person_asset(
                            person_id=pid,
                            asset_type=asset_type,
                            asset_ref=asset_ref,
                            role="mentioned",
                        )
                        links_created += 1
                    except Exception:
                        pass
            else:
                updated += 1  # Would update
                links_created += len(person_ids) + len(mentioned_ids)

            # Progress logging every 200 points
            if scanned % 200 == 0:
                elapsed = time.time() - start_time
                rate = scanned / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Progress: {scanned}/{total_points} scanned, "
                    f"{updated} updated, {skipped} skipped, "
                    f"{errors} errors ({rate:.0f} pts/sec)"
                )

        offset = next_offset
        if offset is None:
            break

    elapsed = time.time() - start_time
    mode = "DRY RUN" if dry_run else "COMPLETE"

    result = {
        "status": mode,
        "scanned": scanned,
        "updated": updated,
        "links_created": links_created,
        "skipped": skipped,
        "errors": errors,
        "elapsed_seconds": round(elapsed, 1),
    }

    logger.info(
        f"Backfill {mode}: {scanned} scanned, {updated} updated, "
        f"{links_created} links, {skipped} skipped, {errors} errors "
        f"({elapsed:.1f}s)"
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Backfill person_ids on existing Qdrant points"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't write anything — just count what would change",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum points to process (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Points per scroll batch (default: 100)",
    )

    args = parser.parse_args()

    result = backfill(
        dry_run=args.dry_run,
        limit=args.limit,
        batch_size=args.batch_size,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
