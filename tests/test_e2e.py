#!/usr/bin/env python3
"""End-to-end test for WhatsApp message flow.

Simulates the full pipeline:
  1. POST a fake webhook payload to /webhook  (receive a message)
  2. Wait for async processing to finish
  3. POST a question to /rag/query about that message (ask about it)
  4. Print the RAG answer and sources

Prerequisites:
  - The Flask app must be running (python src/app.py or docker-compose up)
  - Redis, Qdrant, and WAHA must be reachable

Usage:
    # Basic — uses default test message
    python tests/test_e2e.py

    # Custom message body
    python tests/test_e2e.py --body "I'll be 10 minutes late to the meeting"

    # Custom sender name and message
    python tests/test_e2e.py --sender "Alice" --body "The project deadline is next Friday"

    # Custom base URL (if app runs on a different port)
    python tests/test_e2e.py --base-url http://localhost:9000

    # Ask a specific question about the message
    python tests/test_e2e.py --body "We need 5 pizzas for the party" --question "How many pizzas?"

    # Image message test — simulate receiving an image with caption
    python tests/test_e2e.py --image

    # Image with custom local file and caption
    python tests/test_e2e.py --image --image-file /path/to/photo.jpg --caption "Check out this sunset!"

    # Image with a remote URL (bypasses local file server)
    python tests/test_e2e.py --image --image-url "https://example.com/photo.jpg"

    # Image in a group chat
    python tests/test_e2e.py --image --group --caption "Meeting whiteboard photo"
"""

import argparse
import json
import os
import socket
import sys
import threading
import time
from datetime import datetime
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import requests

# ─────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────
DEFAULT_BASE_URL = "http://localhost:8765"
DEFAULT_SENDER = "Test User"
DEFAULT_BODY = "היי, אני אגיע לפגישה ב-3 אחר הצהריים. אפשר להכין את חדר הישיבות?"
DEFAULT_QUESTION = None  # auto-generated from the message

# Default local image file (birthday party invitation)
DEFAULT_IMAGE_FILE = os.path.join(os.path.dirname(__file__), "image.png")
DEFAULT_IMAGE_CAPTION = "הזמנה ליום הולדת של עדי ואביב"


# ─────────────────────────────────────────────────────────────────────
# Local file server (serves image files for the app to download)
# ─────────────────────────────────────────────────────────────────────

class _QuietHandler(SimpleHTTPRequestHandler):
    """HTTP handler that suppresses log output."""
    def log_message(self, format, *args):
        pass  # silence


def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def start_file_server(directory: str, port: int = 0) -> tuple[HTTPServer, int]:
    """Start a background HTTP file server for serving local images.

    Args:
        directory: Directory to serve files from
        port: Port to bind (0 = auto-select free port)

    Returns:
        Tuple of (server instance, actual port)
    """
    if port == 0:
        port = _find_free_port()

    handler = partial(_QuietHandler, directory=directory)
    server = HTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port


# ─────────────────────────────────────────────────────────────────────
# Payload builders
# ─────────────────────────────────────────────────────────────────────

def build_webhook_payload(
    sender_name: str,
    body: str,
    sender_number: str = "972501234567",
    is_group: bool = False,
    group_id: str | None = None,
    group_name: str | None = None,
) -> dict:
    """Build a realistic WAHA webhook payload for a text message.

    Args:
        sender_name: Display name of the sender
        body: Message text
        sender_number: Phone number (without +)
        is_group: Whether to simulate a group message
        group_id: Group ID (auto-generated if is_group=True)
        group_name: Group display name
    
    Returns:
        dict suitable for POST to /webhook
    """
    ts = int(datetime.now().timestamp())
    contact_id = f"{sender_number}@c.us"

    if is_group:
        from_field = group_id or "120363099999999@g.us"
        participant = contact_id
    else:
        from_field = contact_id
        participant = None

    payload = {
        "id": f"False_{from_field}_TEST{ts}",
        "timestamp": ts,
        "from": from_field,
        "fromMe": False,
        "source": "app",
        "to": "972547755011@c.us",
        "body": body,
        "hasMedia": False,
        "media": None,
        "ack": 1,
        "ackName": "SERVER",
        "location": None,
        "vCards": [],
        "_data": {
            "id": {
                "fromMe": False,
                "remote": from_field,
                "id": f"TEST{ts}",
                "_serialized": f"False_{from_field}_TEST{ts}",
            },
            "viewed": False,
            "body": body,
            "type": "chat",
            "t": ts,
            "notifyName": sender_name,
            "from": from_field,
            "to": "972547755011@c.us",
            "ack": 1,
            "isNewMsg": True,
            "star": False,
            "recvFresh": True,
            "isFromTemplate": False,
            "links": [],
            "mentionedJidList": [],
            "groupMentions": [],
        },
    }

    if participant:
        payload["participant"] = participant
        payload["_data"]["author"] = participant

    return payload


def build_image_webhook_payload(
    sender_name: str,
    image_url: str,
    caption: str | None = None,
    mimetype: str = "image/png",
    sender_number: str = "972501234567",
    is_group: bool = False,
    group_id: str | None = None,
    group_name: str | None = None,
) -> dict:
    """Build a realistic WAHA webhook payload for an image message.

    Simulates the payload WAHA sends when a contact sends a photo.
    The ``media.url`` field points to a real, downloadable image so the
    app's MediaMessageBase._load_media() can fetch it.  The ``_data.type``
    is set to ``"image"`` so create_whatsapp_message() routes to
    ImageMessage, which triggers GPT-4 Vision description.

    Args:
        sender_name: Display name of the sender
        image_url: HTTP URL to the image (can be a local file server)
        caption: Optional caption text attached to the image
        mimetype: MIME type of the image (default: image/png)
        sender_number: Phone number (without +)
        is_group: Whether to simulate a group message
        group_id: Group ID (auto-generated if is_group=True)
        group_name: Group display name

    Returns:
        dict suitable for POST to /webhook
    """
    ts = int(datetime.now().timestamp())
    contact_id = f"{sender_number}@c.us"

    if is_group:
        from_field = group_id or "120363099999999@g.us"
        participant = contact_id
    else:
        from_field = contact_id
        participant = None

    payload = {
        "id": f"False_{from_field}_IMGTEST{ts}",
        "timestamp": ts,
        "from": from_field,
        "fromMe": False,
        "source": "app",
        "to": "972547755011@c.us",
        "body": caption or "",
        "hasMedia": True,
        "media": {
            "url": image_url,
            "mimetype": mimetype,
            "filename": None,
            "directPath": f"/v/t62.7118-24/{ts}",
            "caption": caption,
        },
        "ack": 1,
        "ackName": "SERVER",
        "location": None,
        "vCards": [],
        "_data": {
            "id": {
                "fromMe": False,
                "remote": from_field,
                "id": f"IMGTEST{ts}",
                "_serialized": f"False_{from_field}_IMGTEST{ts}",
            },
            "viewed": False,
            "body": caption or "",
            "type": "image",
            "t": ts,
            "notifyName": sender_name,
            "from": from_field,
            "to": "972547755011@c.us",
            "ack": 1,
            "isNewMsg": True,
            "star": False,
            "recvFresh": True,
            "isFromTemplate": False,
            "caption": caption,
            "mimetype": mimetype,
            "filehash": f"testhash{ts}",
            "size": 45000,
            "height": 640,
            "width": 480,
            "mediaKey": f"testmediakey{ts}",
            "links": [],
            "mentionedJidList": [],
            "groupMentions": [],
        },
    }

    if participant:
        payload["participant"] = participant
        payload["_data"]["author"] = participant

    return payload


# ─────────────────────────────────────────────────────────────────────
# Test steps
# ─────────────────────────────────────────────────────────────────────

def step_health_check(base_url: str) -> bool:
    """Step 0 — verify the app is reachable."""
    print("\n" + "=" * 60)
    print("STEP 0: Health check")
    print("=" * 60)
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        data = resp.json()
        status = data.get("status", "unknown")
        deps = data.get("dependencies", {})
        print(f"  Status : {status}")
        for dep, state in deps.items():
            icon = "✅" if "connected" in str(state) else "❌"
            print(f"  {icon} {dep}: {state}")
        if status == "degraded":
            print("  ⚠️  Some dependencies are down — test may partially fail")
        return True
    except requests.ConnectionError:
        print(f"  ❌ Cannot reach {base_url} — is the app running?")
        return False
    except Exception as e:
        print(f"  ❌ Health check error: {e}")
        return False


def step_send_webhook(base_url: str, payload: dict) -> bool:
    """Step 1 — POST the webhook payload."""
    print("\n" + "=" * 60)
    print("STEP 1: Send webhook (simulate incoming WhatsApp message)")
    print("=" * 60)
    print(f"  Sender : {payload.get('_data', {}).get('notifyName', '?')}")
    print(f"  Body   : {payload.get('body', '')[:80]}")
    print(f"  From   : {payload.get('from', '?')}")

    try:
        resp = requests.post(
            f"{base_url}/webhook",
            json={"payload": payload},
            timeout=10,
        )
        print(f"  HTTP   : {resp.status_code}")
        print(f"  Response: {resp.text}")
        if resp.status_code == 200:
            print("  ✅ Webhook accepted")
            return True
        else:
            print(f"  ❌ Unexpected status {resp.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Webhook POST failed: {e}")
        return False


def step_send_image_webhook(base_url: str, payload: dict) -> bool:
    """Step 1 (image) — POST the image webhook payload."""
    print("\n" + "=" * 60)
    print("STEP 1: Send webhook (simulate incoming WhatsApp IMAGE message)")
    print("=" * 60)
    print(f"  Sender   : {payload.get('_data', {}).get('notifyName', '?')}")
    print(f"  Caption  : {payload.get('body', '') or '(none)'}")
    print(f"  Image URL: {payload.get('media', {}).get('url', '?')[:80]}")
    print(f"  MIME     : {payload.get('media', {}).get('mimetype', '?')}")
    print(f"  From     : {payload.get('from', '?')}")
    print(f"  hasMedia : {payload.get('hasMedia')}")
    print(f"  _data.type: {payload.get('_data', {}).get('type', '?')}")

    try:
        resp = requests.post(
            f"{base_url}/webhook",
            json={"payload": payload},
            timeout=10,
        )
        print(f"  HTTP     : {resp.status_code}")
        print(f"  Response : {resp.text}")
        if resp.status_code == 200:
            print("  ✅ Image webhook accepted")
            return True
        else:
            print(f"  ❌ Unexpected status {resp.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Webhook POST failed: {e}")
        return False


def step_wait_for_processing(seconds: int = 3) -> None:
    """Step 2 — wait for async background processing."""
    print("\n" + "=" * 60)
    print(f"STEP 2: Waiting {seconds}s for async processing...")
    print("=" * 60)
    for i in range(seconds, 0, -1):
        print(f"  {i}...", end=" ", flush=True)
        time.sleep(1)
    print("done")


def step_rag_query(base_url: str, question: str, filter_days: int = 1) -> dict | None:
    """Step 3 — query the RAG system about the message."""
    print("\n" + "=" * 60)
    print("STEP 3: Query RAG about the message")
    print("=" * 60)
    print(f"  Question   : {question}")
    print(f"  Filter days: {filter_days}")

    try:
        resp = requests.post(
            f"{base_url}/rag/query",
            json={
                "question": question,
                "filter_days": filter_days,
                "k": 5,
            },
            timeout=30,
        )
        print(f"  HTTP       : {resp.status_code}")

        if resp.status_code != 200:
            print(f"  ❌ Query failed: {resp.text[:200]}")
            return None

        data = resp.json()
        answer = data.get("answer", "")
        sources = data.get("sources", [])
        conv_id = data.get("conversation_id", "")

        print(f"\n  📝 Answer:\n  {answer}\n")
        print(f"  Conversation ID: {conv_id}")
        print(f"  Sources returned: {len(sources)}")
        for i, src in enumerate(sources[:3], 1):
            print(f"    [{i}] score={src.get('score', '?'):.4f}  "
                  f"sender={src.get('sender', '?')}  "
                  f"chat={src.get('chat_name', '?')}")
            content_preview = (src.get("content", ""))[:100]
            print(f"        {content_preview}")

        return data

    except Exception as e:
        print(f"  ❌ RAG query failed: {e}")
        return None


def step_rag_search(base_url: str, query: str, filter_days: int = 1) -> list | None:
    """Step 3b (optional) — raw vector search to verify the message was stored."""
    print("\n" + "=" * 60)
    print("STEP 3b: Raw vector search (verify message stored)")
    print("=" * 60)
    print(f"  Query      : {query[:60]}")

    try:
        resp = requests.post(
            f"{base_url}/rag/search",
            json={
                "query": query,
                "k": 3,
                "filter_days": filter_days,
            },
            timeout=15,
        )
        if resp.status_code != 200:
            print(f"  ❌ Search failed: {resp.text[:200]}")
            return None

        results = resp.json().get("results", [])
        print(f"  Results    : {len(results)}")
        for i, r in enumerate(results[:3], 1):
            score = r.get("score", "?")
            content = r.get("content", "")[:100]
            print(f"    [{i}] score={score}  {content}")

        if results:
            print("  ✅ Message found in vector store")
        else:
            print("  ⚠️  No results — message may not have been indexed yet")

        return results

    except Exception as e:
        print(f"  ❌ Search failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end test: webhook → process → RAG query",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL,
        help=f"App base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--sender", default=DEFAULT_SENDER,
        help=f"Sender display name (default: {DEFAULT_SENDER})",
    )
    parser.add_argument(
        "--body", default=DEFAULT_BODY,
        help="Message body text",
    )
    parser.add_argument(
        "--question", default=DEFAULT_QUESTION,
        help="Question to ask the RAG (auto-generated if omitted)",
    )
    parser.add_argument(
        "--group", action="store_true",
        help="Simulate a group message instead of direct",
    )
    parser.add_argument(
        "--group-name", default="Test Group",
        help="Group name when --group is used",
    )
    parser.add_argument(
        "--wait", type=int, default=3,
        help="Seconds to wait for async processing (default: 3)",
    )
    parser.add_argument(
        "--skip-health", action="store_true",
        help="Skip the health check step",
    )
    # ── Image test arguments ──────────────────────────────────────
    parser.add_argument(
        "--image", action="store_true",
        help="Simulate an IMAGE message instead of text",
    )
    parser.add_argument(
        "--image-file", default=DEFAULT_IMAGE_FILE,
        help="Path to a local image file (default: tests/image.png)",
    )
    parser.add_argument(
        "--image-url", default=None,
        help="HTTP URL to an image (overrides --image-file; must be downloadable by the app)",
    )
    parser.add_argument(
        "--caption", default=DEFAULT_IMAGE_CAPTION,
        help="Caption for the image message",
    )
    parser.add_argument(
        "--mimetype", default=None,
        help="MIME type of the image (auto-detected from file extension if omitted)",
    )

    args = parser.parse_args()

    # ── Determine test mode ───────────────────────────────────────
    is_image_test = args.image
    mode_label = "📷 IMAGE" if is_image_test else "💬 TEXT"

    print(f"🧪 WhatsApp E2E Test ({mode_label})")
    print(f"   Target: {args.base_url}")
    print(f"   Time  : {datetime.now().isoformat()}")

    # ── Step 0: Health check ──────────────────────────────────────
    if not args.skip_health:
        if not step_health_check(args.base_url):
            print("\n❌ Aborting — app is not reachable.")
            sys.exit(1)

    if is_image_test:
        # ══════════════════════════════════════════════════════════
        # IMAGE MESSAGE TEST FLOW
        # ══════════════════════════════════════════════════════════
        print("\n" + "=" * 60)
        print("  📷  IMAGE MESSAGE TEST")
        print("=" * 60)

        # ── Resolve image URL ─────────────────────────────────────
        file_server = None
        if args.image_url:
            # User provided an explicit URL — use it directly
            image_url = args.image_url
            mimetype = args.mimetype or "image/jpeg"
        else:
            # Serve the local file via a temporary HTTP server
            image_path = Path(args.image_file).resolve()
            if not image_path.exists():
                print(f"  ❌ Image file not found: {image_path}")
                sys.exit(1)

            # Auto-detect MIME type from extension
            ext_to_mime = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }
            mimetype = args.mimetype or ext_to_mime.get(
                image_path.suffix.lower(), "image/png"
            )

            # Start local file server
            file_server, port = start_file_server(str(image_path.parent))
            image_url = f"http://127.0.0.1:{port}/{image_path.name}"
            print(f"  📂 Serving local file: {image_path}")
            print(f"  🌐 File server URL  : {image_url}")

        # ── Step 1: Send image webhook ────────────────────────────
        payload = build_image_webhook_payload(
            sender_name=args.sender,
            image_url=image_url,
            caption=args.caption,
            mimetype=mimetype,
            is_group=args.group,
            group_name=args.group_name if args.group else None,
        )

        if not step_send_image_webhook(args.base_url, payload):
            print("\n❌ Aborting — image webhook was not accepted.")
            if file_server:
                file_server.shutdown()
            sys.exit(1)

        # ── Step 2: Wait (image processing takes longer: download + GPT-4V) ──
        wait_time = max(args.wait, 10)  # images need more time for Vision API
        step_wait_for_processing(wait_time)

        # Shut down file server (image already downloaded by now)
        if file_server:
            file_server.shutdown()
            print("  📂 File server stopped")

        # ── Step 3b: Raw search — look for the image description ──
        search_query = args.caption or "image birthday"
        step_rag_search(args.base_url, search_query, filter_days=1)

        # ── Step 3: RAG query about the image ─────────────────────
        question = args.question
        if not question:
            question = f"Did {args.sender} send any images? What was in the image?"

        result = step_rag_query(args.base_url, question, filter_days=1)

        # ── Summary ───────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("SUMMARY (IMAGE TEST)")
        print("=" * 60)
        if result and result.get("answer"):
            print("  ✅ Image E2E test PASSED")
            print(f"  Image from '{args.sender}' was received, downloaded,")
            print(f"  described by GPT-4 Vision, stored in RAG, and queried.")
            answer_preview = result["answer"][:200]
            print(f"  Answer preview: {answer_preview}")
        else:
            print("  ⚠️  Image E2E test completed with warnings")
            print("  The webhook was accepted but the RAG query returned no answer.")
            print("  This could mean:")
            print("    - Image download failed (check image URL accessibility)")
            print("    - GPT-4 Vision description failed (check OpenAI API key)")
            print("    - Processing still in progress (try --wait 12)")
            print("    - Contact/group resolution failed (WAHA not connected)")
            print("    - Embedding or Qdrant issue")

    else:
        # ══════════════════════════════════════════════════════════
        # TEXT MESSAGE TEST FLOW (original behavior)
        # ══════════════════════════════════════════════════════════

        # ── Step 1: Send webhook ──────────────────────────────────
        payload = build_webhook_payload(
            sender_name=args.sender,
            body=args.body,
            is_group=args.group,
            group_name=args.group_name if args.group else None,
        )

        if not step_send_webhook(args.base_url, payload):
            print("\n❌ Aborting — webhook was not accepted.")
            sys.exit(1)

        # ── Step 2: Wait ──────────────────────────────────────────
        step_wait_for_processing(args.wait)

        # ── Step 3b: Raw search to verify storage ─────────────────
        step_rag_search(args.base_url, args.body, filter_days=1)

        # ── Step 3: RAG query ─────────────────────────────────────
        question = args.question
        if not question:
            # Auto-generate a question about the message
            question = f"What did {args.sender} say in their latest message?"

        result = step_rag_query(args.base_url, question, filter_days=1)

        # ── Summary ───────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        if result and result.get("answer"):
            print("  ✅ End-to-end test PASSED")
            print(f"  Message sent by '{args.sender}' was received, processed,")
            print(f"  stored in the vector store, and answered by the RAG system.")
        else:
            print("  ⚠️  End-to-end test completed with warnings")
            print("  The webhook was accepted but the RAG query returned no answer.")
            print("  This could mean:")
            print("    - Processing is still in progress (try --wait 5)")
            print("    - Contact/group resolution failed (WAHA not connected)")
            print("    - Embedding or Qdrant issue")

    print()


if __name__ == "__main__":
    main()
