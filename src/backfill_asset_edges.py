#!/usr/bin/env python3
"""Backfill asset↔asset edges and payload fields on existing Qdrant points.

Scans all points in the Qdrant collection, derives structural relationships
(thread membership, chunk grouping, attachment edges), and:
1. Updates Qdrant payloads with asset_id, parent_asset_id, thread_id, chunk_group_id
2. Creates asset_asset_edges in the entity store

Usage:
    # From the src/ directory:
    cd src && python backfill_asset_edges.py

    # Dry-run (no writes):
    cd src && python backfill_asset_edges.py --dry-run

    # Limit to N points:
    cd src && python backfill_asset_edges.py --limit 500

This is a one-time migration script.  It is idempotent — re-running
it won't create duplicate edges because both the Qdrant payload
update and the asset_asset_edges INSERT use upsert semantics.
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


def _derive_asset_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Derive asset graph fields from existing payload data.

    Returns a dict with keys: asset_id, parent_asset_id, thread_id,
    chunk_group_id, edges (list of edge dicts to create).
    """
    from asset_linker import generate_asset_id

    source = payload.get("source", "")
    source_id = payload.get("source_id", "")
    content_type = payload.get("content_type", "")
    chat_id = payload.get("chat_id", "")

    result = {
        "asset_id": "",
        "parent_asset_id": "",
        "thread_id": "",
        "chunk_group_id": "",
        "edges": [],
    }

    if not source_id:
        return result

    # --- WhatsApp messages ---
    if source == "whatsapp":
        if content_type == "conversation_chunk":
            result["asset_id"] = generate_asset_id("whatsapp", source_id)
            result["thread_id"] = chat_id
            result["chunk_group_id"] = f"wa_chunk:{chat_id}"
        else:
            result["asset_id"] = generate_asset_id("whatsapp", source_id)
            result["thread_id"] = chat_id
            if chat_id:
                result["edges"].append({
                    "src_asset_ref": f"thread:{chat_id}",
                    "dst_asset_ref": source_id,
                    "relation_type": "thread_member",
                    "provenance": "backfill",
                })

    # --- Gmail emails ---
    elif source == "gmail":
        # Extract base message ID (strip chunk suffix if present)
        # source_id format: "gmail:msg_id" or "gmail:msg_id:att:filename"
        msg_id = source_id.replace("gmail:", "")
        is_attachment = ":att:" in source_id

        if is_attachment:
            parts = msg_id.split(":att:")
            parent_msg_id = parts[0] if parts else msg_id
            result["asset_id"] = generate_asset_id("gmail", msg_id)
            result["parent_asset_id"] = generate_asset_id("gmail", parent_msg_id)
            result["chunk_group_id"] = f"gm:{msg_id}"

            parent_source_id = f"gmail:{parent_msg_id}"
            result["edges"].append({
                "src_asset_ref": source_id,
                "dst_asset_ref": parent_source_id,
                "relation_type": "attachment_of",
                "provenance": "backfill",
            })
        else:
            result["asset_id"] = generate_asset_id("gmail", msg_id)
            result["chunk_group_id"] = f"gm:{msg_id}"

        # Thread membership
        thread_id = payload.get("thread_id", "")
        if thread_id:
            result["thread_id"] = thread_id
            result["edges"].append({
                "src_asset_ref": f"thread:{thread_id}",
                "dst_asset_ref": source_id,
                "relation_type": "thread_member",
                "provenance": "backfill",
            })

    # --- Paperless documents ---
    elif source == "paperless":
        # source_id format: "paperless:{doc_id}"
        doc_id = source_id.replace("paperless:", "")
        result["asset_id"] = generate_asset_id("paperless", doc_id)
        result["chunk_group_id"] = f"pl:{doc_id}"

        # Multi-chunk documents: create chunk_of edges
        chunk_index = payload.get("chunk_index")
        chunk_total = payload.get("chunk_total")
        if chunk_total and int(chunk_total) > 1 and chunk_index is not None:
            result["edges"].append({
                "src_asset_ref": f"{source_id}:{chunk_index}",
                "dst_asset_ref": source_id,
                "relation_type": "chunk_of",
                "provenance": "backfill",
            })

    # --- Call recordings ---
    elif source == "call_recording":
        # source_id format: "call_recording:{content_hash}"
        content_hash = source_id.replace("call_recording:", "")
        result["asset_id"] = generate_asset_id("call_recording", content_hash)
        result["chunk_group_id"] = f"cr:{content_hash}"

        # Multi-chunk transcripts: create chunk_of + transcript_of edges
        chunk_index = payload.get("chunk_index")
        chunk_total = payload.get("chunk_total")
        if chunk_total and int(chunk_total) > 1 and chunk_index is not None:
            result["edges"].append({
                "src_asset_ref": f"{source_id}:{chunk_index}",
                "dst_asset_ref": source_id,
                "relation_type": "chunk_of",
                "provenance": "backfill",
            })
            result["edges"].append({
                "src_asset_ref": f"{source_id}:{chunk_index}",
                "dst_asset_ref": source_id,
                "relation_type": "transcript_of",
                "provenance": "backfill",
            })

    return result


def backfill(
    dry_run: bool = False,
    limit: Optional[int] = None,
    batch_size: int = 100,
) -> Dict[str, int]:
    """Scan all Qdrant points and backfill asset graph fields + edges.

    Args:
        dry_run: If True, don't write anything — just count what would change
        limit: Maximum points to process (None = all)
        batch_size: Points per scroll batch

    Returns:
        Dict with counts: scanned, updated, edges_created, skipped, errors
    """
    from qdrant_client import QdrantClient
    import identity_db

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
    edges_created = 0
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

        for point in points:
            scanned += 1
            payload = point.payload or {}

            # Skip points that already have asset_id populated
            existing_aid = payload.get("asset_id", "")
            if existing_aid and isinstance(existing_aid, str) and len(existing_aid) > 2:
                skipped += 1
                continue

            # Skip system/entity_store nodes
            source = payload.get("source", "")
            if source in ("system", "entity_store"):
                skipped += 1
                continue

            try:
                fields = _derive_asset_fields(payload)
            except Exception as e:
                logger.debug(f"Field derivation failed for point {point.id}: {e}")
                errors += 1
                continue

            if not fields["asset_id"]:
                skipped += 1
                continue

            if not dry_run:
                # Update Qdrant payload with asset graph fields
                try:
                    client.set_payload(
                        collection_name=collection_name,
                        payload={
                            "asset_id": fields["asset_id"],
                            "parent_asset_id": fields["parent_asset_id"],
                            "thread_id": fields["thread_id"],
                            "chunk_group_id": fields["chunk_group_id"],
                        },
                        points=[point.id],
                    )
                    updated += 1
                except Exception as e:
                    logger.debug(f"Payload update failed for point {point.id}: {e}")
                    errors += 1
                    continue

                # Create asset↔asset edges
                for edge in fields["edges"]:
                    try:
                        identity_db.link_assets(
                            src_asset_ref=edge["src_asset_ref"],
                            dst_asset_ref=edge["dst_asset_ref"],
                            relation_type=edge["relation_type"],
                            provenance=edge.get("provenance", "backfill"),
                        )
                        edges_created += 1
                    except Exception:
                        pass  # Duplicate edges silently ignored
            else:
                updated += 1  # Would update
                edges_created += len(fields["edges"])

            # Progress logging every 200 points
            if scanned % 200 == 0:
                elapsed = time.time() - start_time
                rate = scanned / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Progress: {scanned}/{total_points} scanned, "
                    f"{updated} updated, {edges_created} edges, "
                    f"{skipped} skipped, {errors} errors ({rate:.0f} pts/sec)"
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
        "edges_created": edges_created,
        "skipped": skipped,
        "errors": errors,
        "elapsed_seconds": round(elapsed, 1),
    }

    logger.info(
        f"Asset edge backfill {mode}: {scanned} scanned, {updated} updated, "
        f"{edges_created} edges, {skipped} skipped, {errors} errors "
        f"({elapsed:.1f}s)"
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Backfill asset↔asset edges on existing Qdrant points"
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
