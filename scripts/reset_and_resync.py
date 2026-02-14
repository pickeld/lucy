#!/usr/bin/env python3
"""Selective RAG reset — choose which data to delete and optionally re-sync.

Usage (from project root):
    python3 scripts/reset_and_resync.py

Options:
    1. Delete WhatsApp messages only
    2. Delete documents (Paperless) only
    3. Delete ALL (full collection reset + Paperless re-sync)
    4. Delete all previous chat conversations

Requires the app to be running on localhost:8765.
"""

import sys
import os

# Add src/ to path so we can import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import requests

APP_BASE = "http://localhost:8765"


def get_current_stats() -> dict:
    """Fetch current RAG stats from the API."""
    try:
        resp = requests.get(f"{APP_BASE}/rag/stats", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except requests.ConnectionError:
        print(f"  ❌ Cannot connect to app at {APP_BASE}")
        print("     Make sure the app is running (python3 src/app.py)")
        sys.exit(1)
    return {}


def delete_by_source(source: str) -> int:
    """Delete all RAG vectors for a specific source type."""
    resp = requests.post(
        f"{APP_BASE}/rag/delete-by-source",
        json={"source": source, "confirm": True},
        timeout=30,
    )
    if resp.status_code == 200:
        data = resp.json()
        return data.get("deleted", 0)
    else:
        print(f"  ❌ Delete failed ({resp.status_code}): {resp.text}")
        return 0


def reset_full_collection() -> bool:
    """Drop and recreate the entire Qdrant collection."""
    resp = requests.post(
        f"{APP_BASE}/rag/reset",
        json={"confirm": True},
        timeout=30,
    )
    if resp.status_code == 200:
        data = resp.json()
        print(f"  ✅ {data.get('message', 'Collection reset')}")
        return True
    else:
        print(f"  ❌ Reset failed ({resp.status_code}): {resp.text}")
        return False


def remove_paperless_tags():
    """Remove rag-indexed tag from all Paperless documents."""
    try:
        from config import settings

        paperless_url = settings.get("paperless_url", "")
        paperless_token = settings.get("paperless_token", "")
        processed_tag_name = settings.get("paperless_processed_tag", "rag-indexed")

        if not paperless_url or not paperless_token:
            print("  ⚠️  Paperless URL or token not configured, skipping tag removal")
            return

        headers = {"Authorization": f"Token {paperless_token}"}
        base = paperless_url.rstrip("/")

        # Find the tag ID
        tag_resp = requests.get(
            f"{base}/api/tags/",
            params={"name__iexact": processed_tag_name},
            headers=headers,
            timeout=10,
        )
        tag_resp.raise_for_status()
        tags = tag_resp.json().get("results", [])

        if not tags:
            print(f"  ℹ️  Tag '{processed_tag_name}' not found in Paperless")
            return

        tag_id = tags[0]["id"]
        print(f"  Found tag '{processed_tag_name}' (id={tag_id})")

        # Find all documents with this tag
        doc_ids = []
        page = 1
        while True:
            docs_resp = requests.get(
                f"{base}/api/documents/",
                params={
                    "tags__id__all": str(tag_id),
                    "page": page,
                    "page_size": 100,
                },
                headers=headers,
                timeout=10,
            )
            docs_resp.raise_for_status()
            data = docs_resp.json()
            doc_ids.extend(d["id"] for d in data.get("results", []))
            if not data.get("next"):
                break
            page += 1

        if doc_ids:
            bulk_resp = requests.post(
                f"{base}/api/documents/bulk_edit/",
                json={
                    "documents": doc_ids,
                    "method": "modify_tags",
                    "parameters": {
                        "add_tags": [],
                        "remove_tags": [tag_id],
                    },
                },
                headers=headers,
                timeout=30,
            )
            bulk_resp.raise_for_status()
            print(f"  ✅ Removed '{processed_tag_name}' tag from {len(doc_ids)} documents")
        else:
            print(f"  ℹ️  No documents found with tag '{processed_tag_name}'")

    except Exception as e:
        print(f"  ⚠️  Tag removal failed (non-fatal): {e}")


def sync_paperless():
    """Trigger Paperless document sync to RAG."""
    print("\n  Triggering Paperless document sync...")
    try:
        resp = requests.post(
            f"{APP_BASE}/plugins/paperless/sync",
            timeout=300,
        )
        if resp.status_code == 200:
            data = resp.json()
            print(f"  ✅ Sync complete:")
            print(f"     Synced:  {data.get('synced', 0)}")
            print(f"     Tagged:  {data.get('tagged', 0)}")
            print(f"     Skipped: {data.get('skipped', 0)}")
            print(f"     Errors:  {data.get('errors', 0)}")
        else:
            print(f"  ❌ Sync failed ({resp.status_code}): {resp.text}")
    except Exception as e:
        print(f"  ❌ Sync request failed: {e}")


def delete_all_conversations():
    """Delete all chat conversations from SQLite."""
    try:
        resp = requests.get(f"{APP_BASE}/conversations", params={"limit": 200}, timeout=10)
        if resp.status_code != 200:
            print(f"  ❌ Failed to fetch conversations ({resp.status_code})")
            return

        convos = resp.json().get("conversations", [])
        if not convos:
            print("  ℹ️  No conversations found")
            return

        print(f"  Found {len(convos)} conversations")
        deleted = 0
        for c in convos:
            cid = c.get("id", "")
            if cid:
                del_resp = requests.delete(f"{APP_BASE}/conversations/{cid}", timeout=10)
                if del_resp.status_code == 200:
                    deleted += 1
        print(f"  ✅ Deleted {deleted} conversations")

    except Exception as e:
        print(f"  ❌ Failed to delete conversations: {e}")


def main():
    print("=" * 60)
    print("RAG RESET TOOL — Selective Data Management")
    print("=" * 60)

    # Show current stats
    stats = get_current_stats()
    total = stats.get("total_documents", "?")
    wa_count = stats.get("whatsapp_messages", "?")
    doc_count = stats.get("documents", "?")
    collection = stats.get("collection_name", "?")

    print(f"\n  Collection: {collection}")
    print(f"  Total vectors:      {total}")
    print(f"  WhatsApp messages:  {wa_count}")
    print(f"  Documents:          {doc_count}")

    # Show menu
    print("\n  What would you like to do?\n")
    print("  1) Delete WhatsApp messages only")
    print("  2) Delete documents (Paperless) only + re-sync")
    print("  3) Delete ALL (full collection reset + Paperless re-sync)")
    print("  4) Delete all chat conversations (sidebar history)")
    print("  5) Exit")

    choice = input("\n  Enter choice [1-5]: ").strip()

    if choice == "1":
        print(f"\n[1] Deleting WhatsApp messages ({wa_count} vectors)...")
        confirm = input("  Are you sure? (y/N): ").strip().lower()
        if confirm != "y":
            print("  Cancelled.")
            return
        deleted = delete_by_source("whatsapp")
        print(f"  ✅ Deleted {deleted} WhatsApp message vectors")

    elif choice == "2":
        print(f"\n[2] Deleting documents ({doc_count} vectors) + re-sync...")
        confirm = input("  Are you sure? (y/N): ").strip().lower()
        if confirm != "y":
            print("  Cancelled.")
            return

        print("\n  Step 1: Deleting document vectors...")
        deleted = delete_by_source("paperless")
        print(f"  ✅ Deleted {deleted} document vectors")

        print("\n  Step 2: Removing Paperless tags...")
        remove_paperless_tags()

        print("\n  Step 3: Re-syncing Paperless documents...")
        sync_paperless()

    elif choice == "3":
        print(f"\n[3] FULL RESET — deleting ALL {total} vectors + re-sync...")
        confirm = input("  ⚠️  This will delete EVERYTHING. Are you sure? (y/N): ").strip().lower()
        if confirm != "y":
            print("  Cancelled.")
            return

        print("\n  Step 1: Resetting Qdrant collection...")
        if not reset_full_collection():
            return

        print("\n  Step 2: Removing Paperless tags...")
        remove_paperless_tags()

        print("\n  Step 3: Re-syncing Paperless documents...")
        sync_paperless()

    elif choice == "4":
        print("\n[4] Deleting all chat conversations...")
        confirm = input("  Are you sure? This removes all sidebar chat history. (y/N): ").strip().lower()
        if confirm != "y":
            print("  Cancelled.")
            return
        delete_all_conversations()

    elif choice == "5":
        print("  Bye!")
        return

    else:
        print(f"  ❌ Invalid choice: {choice}")
        return

    # Show final stats
    print("\n" + "-" * 40)
    final_stats = get_current_stats()
    print(f"  Final stats:")
    print(f"    Total vectors:      {final_stats.get('total_documents', '?')}")
    print(f"    WhatsApp messages:  {final_stats.get('whatsapp_messages', '?')}")
    print(f"    Documents:          {final_stats.get('documents', '?')}")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
