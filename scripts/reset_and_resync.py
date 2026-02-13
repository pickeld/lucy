#!/usr/bin/env python3
"""Reset Qdrant collection, remove rag-indexed tag from Paperless, and re-sync.

Usage:
    python3 scripts/reset_and_resync.py

Requires the app to be running on localhost:8765 and Paperless to be accessible.
Reads Paperless credentials from the app's settings DB.
"""

import json
import sys
import time

import requests

APP_BASE = "http://localhost:8765"


def main():
    print("=" * 60)
    print("FULL RAG RE-INGEST: Reset Qdrant + Remove tags + Re-sync")
    print("=" * 60)

    # Step 1: Reset Qdrant collection
    print("\n[1/3] Resetting Qdrant collection...")
    try:
        resp = requests.post(
            f"{APP_BASE}/rag/reset",
            json={"confirm": True},
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            print(f"  ✅ Collection reset: {data.get('message', 'OK')}")
        else:
            print(f"  ❌ Reset failed ({resp.status_code}): {resp.text}")
            sys.exit(1)
    except requests.ConnectionError:
        print(f"  ❌ Cannot connect to app at {APP_BASE}")
        print("     Make sure the app is running (python3 src/app.py)")
        sys.exit(1)

    # Step 2: Remove rag-indexed tag from all Paperless documents
    print("\n[2/3] Removing 'rag-indexed' tag from Paperless documents...")
    try:
        # Get Paperless settings from the app
        resp = requests.get(f"{APP_BASE}/settings", timeout=10)
        if resp.status_code != 200:
            print(f"  ⚠️  Could not fetch settings ({resp.status_code}), skipping tag removal")
        else:
            settings_data = resp.json()
            settings_list = settings_data if isinstance(settings_data, list) else settings_data.get("settings", [])
            
            paperless_url = ""
            paperless_token = ""
            processed_tag_name = "rag-indexed"
            
            for s in settings_list:
                key = s.get("key", "")
                val = s.get("value", "")
                if key == "paperless_url":
                    paperless_url = val
                elif key == "paperless_token":
                    paperless_token = val
                elif key == "paperless_processed_tag":
                    processed_tag_name = val or "rag-indexed"
            
            if paperless_url and paperless_token:
                headers = {"Authorization": f"Token {paperless_token}"}
                
                # Find the tag ID
                tag_resp = requests.get(
                    f"{paperless_url.rstrip('/')}/api/tags/",
                    params={"name__iexact": processed_tag_name},
                    headers=headers,
                    timeout=10,
                )
                tag_resp.raise_for_status()
                tags = tag_resp.json().get("results", [])
                
                if tags:
                    tag_id = tags[0]["id"]
                    print(f"  Found tag '{processed_tag_name}' (id={tag_id})")
                    
                    # Find all documents with this tag
                    doc_ids = []
                    page = 1
                    while True:
                        docs_resp = requests.get(
                            f"{paperless_url.rstrip('/')}/api/documents/",
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
                        # Remove tag from all documents
                        bulk_resp = requests.post(
                            f"{paperless_url.rstrip('/')}/api/documents/bulk_edit/",
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
                else:
                    print(f"  ℹ️  Tag '{processed_tag_name}' not found in Paperless (nothing to remove)")
            else:
                print("  ⚠️  Paperless URL or token not configured, skipping tag removal")
    except Exception as e:
        print(f"  ⚠️  Tag removal failed (non-fatal): {e}")

    # Step 3: Trigger re-sync
    print("\n[3/3] Triggering Paperless document sync...")
    try:
        resp = requests.post(
            f"{APP_BASE}/plugins/paperless/sync",
            timeout=300,  # Long timeout for large syncs
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

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
