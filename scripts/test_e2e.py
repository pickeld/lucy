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
    python scripts/test_e2e.py

    # Custom message body
    python scripts/test_e2e.py --body "I'll be 10 minutes late to the meeting"

    # Custom sender name and message
    python scripts/test_e2e.py --sender "Alice" --body "The project deadline is next Friday"

    # Custom base URL (if app runs on a different port)
    python scripts/test_e2e.py --base-url http://localhost:9000

    # Ask a specific question about the message
    python scripts/test_e2e.py --body "We need 5 pizzas for the party" --question "How many pizzas?"
"""

import argparse
import json
import sys
import time
from datetime import datetime

import requests

# ─────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────
DEFAULT_BASE_URL = "http://localhost:8765"
DEFAULT_SENDER = "Test User"
DEFAULT_BODY = "היי, אני אגיע לפגישה ב-3 אחר הצהריים. אפשר להכין את חדר הישיבות?"
DEFAULT_QUESTION = None  # auto-generated from the message


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

    args = parser.parse_args()

    print("🧪 WhatsApp E2E Test")
    print(f"   Target: {args.base_url}")
    print(f"   Time  : {datetime.now().isoformat()}")

    # ── Step 0: Health check ──────────────────────────────────────
    if not args.skip_health:
        if not step_health_check(args.base_url):
            print("\n❌ Aborting — app is not reachable.")
            sys.exit(1)

    # ── Step 1: Send webhook ──────────────────────────────────────
    payload = build_webhook_payload(
        sender_name=args.sender,
        body=args.body,
        is_group=args.group,
        group_name=args.group_name if args.group else None,
    )

    if not step_send_webhook(args.base_url, payload):
        print("\n❌ Aborting — webhook was not accepted.")
        sys.exit(1)

    # ── Step 2: Wait ──────────────────────────────────────────────
    step_wait_for_processing(args.wait)

    # ── Step 3b: Raw search to verify storage ─────────────────────
    step_rag_search(args.base_url, args.body, filter_days=1)

    # ── Step 3: RAG query ─────────────────────────────────────────
    question = args.question
    if not question:
        # Auto-generate a question about the message
        question = f"What did {args.sender} say in their latest message?"

    result = step_rag_query(args.base_url, question, filter_days=1)

    # ── Summary ───────────────────────────────────────────────────
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
