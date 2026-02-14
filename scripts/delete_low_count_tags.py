#!/usr/bin/env python3
"""Temporary script: delete all Paperless-NGX tags with <= X documents.

Uses the /api/bulk_edit_objects/ endpoint (same as the Paperless-NGX UI)
with configurable batch sizes for reliable batch deletion.

Usage:
    python scripts/delete_low_count_tags.py [--max-docs X] [--dry-run]

Requires PAPERLESS_URL and PAPERLESS_TOKEN env vars (or in .env / settings DB).

Examples:
    # Dry-run: preview which tags would be deleted (default threshold: 0)
    python scripts/delete_low_count_tags.py --dry-run

    # Delete all tags with 0 documents
    python scripts/delete_low_count_tags.py --max-docs 0

    # Delete all tags with 2 or fewer documents, batch size 20
    python scripts/delete_low_count_tags.py --max-docs 2 --batch-size 20
"""

import argparse
import math
import os
import sys
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Resolve Paperless credentials
# ---------------------------------------------------------------------------

def _load_env_file():
    """Try loading a .env file if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def _load_from_settings_db() -> tuple[str | None, str | None]:
    """Try reading PAPERLESS_URL / PAPERLESS_TOKEN from the settings DB."""
    db_path = os.path.join(os.path.dirname(__file__), "..", "data", "settings.db")
    if not os.path.exists(db_path):
        return None, None
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT key, value FROM settings WHERE key IN ('paperless_url', 'paperless_token')")
        rows = {k: v for k, v in cur.fetchall()}
        conn.close()
        return rows.get("paperless_url"), rows.get("paperless_token")
    except Exception:
        return None, None


def get_credentials() -> tuple[str, str]:
    _load_env_file()

    url = os.environ.get("PAPERLESS_URL")
    token = os.environ.get("PAPERLESS_TOKEN")

    if not url or not token:
        db_url, db_token = _load_from_settings_db()
        url = url or db_url
        token = token or db_token

    if not url or not token:
        print("ERROR: PAPERLESS_URL and PAPERLESS_TOKEN must be set "
              "(env vars, .env file, or data/settings.db)")
        sys.exit(1)

    return url.rstrip("/"), token


def build_session(token: str) -> requests.Session:
    """Build a requests session with retry logic."""
    session = requests.Session()
    session.headers.update({"Authorization": f"Token {token}"})

    retry = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 502, 503, 504],
        allowed_methods=["DELETE", "GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ---------------------------------------------------------------------------
# Paperless API helpers
# ---------------------------------------------------------------------------

def fetch_all_tags(base_url: str, session: requests.Session) -> list[dict]:
    """Fetch every tag (handles pagination)."""
    tags: list[dict] = []
    page = 1
    while True:
        resp = session.get(
            f"{base_url}/api/tags/",
            params={"page": page, "page_size": 100},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        tags.extend(data.get("results", []))
        if not data.get("next"):
            break
        page += 1
    return tags


def batch_delete_tags(
    base_url: str,
    session: requests.Session,
    tag_ids: list[int],
    batch_size: int = 25,
    delay: float = 1.0,
) -> tuple[int, int]:
    """Delete tags using POST /api/bulk_edit_objects/ (Paperless-NGX native batch).
    
    Sends tag IDs in chunks to avoid server overload.
    Returns (success_count, fail_count).
    """
    total_batches = math.ceil(len(tag_ids) / batch_size)
    ok = 0
    fail = 0

    for i in range(0, len(tag_ids), batch_size):
        batch = tag_ids[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        print(f"  Batch {batch_num}/{total_batches}: deleting {len(batch)} tags …", end=" ", flush=True)

        try:
            resp = session.post(
                f"{base_url}/api/bulk_edit_objects/",
                json={
                    "objects": batch,
                    "object_type": "tags",
                    "operation": "delete",
                },
                timeout=120,
            )
            resp.raise_for_status()
            ok += len(batch)
            print("✓")
        except requests.exceptions.HTTPError as exc:
            # Log the response body for debugging
            body = ""
            try:
                body = exc.response.text[:200]
            except Exception:
                pass
            print(f"✗ {exc.response.status_code}: {body}")
            fail += len(batch)
        except Exception as exc:
            print(f"✗ {exc}")
            fail += len(batch)

        # Small delay between batches to let the server breathe
        if i + batch_size < len(tag_ids):
            time.sleep(delay)

    return ok, fail


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Delete Paperless-NGX tags that have <= X documents."
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="Delete tags whose document_count is <= this value (default: 0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Number of tags per bulk_edit_objects call (default: 25)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to wait between batches (default: 1.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list tags that would be deleted; don't actually delete.",
    )
    args = parser.parse_args()

    base_url, token = get_credentials()
    session = build_session(token)

    print(f"Connecting to {base_url} …")
    tags = fetch_all_tags(base_url, session)
    print(f"Fetched {len(tags)} tags total.\n")

    candidates = [
        t for t in tags
        if t.get("document_count", 0) <= args.max_docs
    ]
    candidates.sort(key=lambda t: t.get("document_count", 0))

    if not candidates:
        print(f"No tags found with document_count <= {args.max_docs}. Nothing to do.")
        return

    # Preview
    print(f"Tags with <= {args.max_docs} document(s):  ({len(candidates)} tags)")
    print("-" * 60)
    for t in candidates:
        print(f"  id={t['id']:>5}  docs={t.get('document_count', 0):>3}  name={t['name']}")
    print("-" * 60)

    if args.dry_run:
        print("\n[DRY RUN] No tags were deleted.")
        return

    # Confirmation
    answer = input(
        f"\nDelete these {len(candidates)} tag(s) in batches of {args.batch_size}? [y/N] "
    ).strip().lower()
    if answer != "y":
        print("Aborted.")
        return

    print(f"\nDeleting {len(candidates)} tags …\n")
    t0 = time.time()
    ok, fail = batch_delete_tags(
        base_url, session,
        [t["id"] for t in candidates],
        batch_size=args.batch_size,
        delay=args.delay,
    )
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Deleted: {ok}  Failed: {fail}")


if __name__ == "__main__":
    main()
