#!/usr/bin/env python3
"""Script to retrieve and display info and metadata about a LangGraph thread."""

import asyncio
import os
import sys
import json
from datetime import datetime
from typing import Optional

from langgraph_sdk import get_client


async def get_thread_info(thread_id: str, api_url: Optional[str] = None, show_messages: bool = False):
    """Get and display info and metadata about a specific thread.
    
    Args:
        thread_id: The ID of the thread to inspect.
        api_url: The LangGraph API URL. Defaults to env var or localhost.
        show_messages: Whether to also fetch and display message history.
    """
    url = api_url or os.getenv("LANGGRAPH_API_URL", "http://127.0.0.1:2024")
    
    print(f"Connecting to LangGraph server at: {url}")
    client = get_client(url=url)
    
    print(f"\nFetching thread: {thread_id}")
    print("=" * 60)
    
    try:
        # Get thread info
        thread = await client.threads.get(thread_id)
        
        print("\n📋 THREAD INFO")
        print("-" * 40)
        print(f"Thread ID:    {thread.get('thread_id', 'N/A')}")
        print(f"Created At:   {format_timestamp(thread.get('created_at'))}")
        print(f"Updated At:   {format_timestamp(thread.get('updated_at'))}")
        
        # Display metadata
        metadata = thread.get("metadata", {})
        print("\n📌 METADATA")
        print("-" * 40)
        if metadata:
            for key, value in metadata.items():
                print(f"  {key}: {value}")
        else:
            print("  (No metadata)")
        
        # Display values/state if available
        values = thread.get("values", {})
        if values:
            print("\n📊 STATE VALUES")
            print("-" * 40)
            for key, value in values.items():
                if key == "messages":
                    print(f"  messages: [{len(value)} message(s)]")
                else:
                    print(f"  {key}: {value}")
        
        # Get thread state for more details
        try:
            state = await client.threads.get_state(thread_id)
            if state:
                print("\n🔧 STATE DETAILS")
                print("-" * 40)
                # Handle both dict and list responses
                if isinstance(state, dict):
                    print(f"  Checkpoint ID: {state.get('checkpoint_id', 'N/A')}")
                    print(f"  Parent Config: {state.get('parent_config', 'N/A')}")
                    
                    state_values = state.get("values", {})
                    if state_values and isinstance(state_values, dict):
                        print("\n  State Values:")
                        for key, value in state_values.items():
                            if key == "messages":
                                print(f"    messages: [{len(value)} message(s)]")
                            else:
                                print(f"    {key}: {value}")
                else:
                    print(f"  State type: {type(state).__name__}")
        except Exception as e:
            print(f"\n  (Could not fetch state details: {e})")
        
        # Optionally show messages
        if show_messages:
            print("\n💬 MESSAGE HISTORY")
            print("-" * 40)
            try:
                state = await client.threads.get_state(thread_id)
                messages = []
                if isinstance(state, dict):
                    values = state.get("values", {})
                    if isinstance(values, dict):
                        messages = values.get("messages", [])
                
                if messages:
                    for i, msg in enumerate(messages):
                        msg_type = msg.get("type", "unknown")
                        content = msg.get("content", "")
                        
                        # Truncate long messages
                        if len(content) > 200:
                            content = content[:200] + "..."
                        
                        # Format message type with emoji
                        type_emoji = {
                            "human": "👤",
                            "ai": "🤖",
                            "system": "⚙️"
                        }.get(msg_type, "📝")
                        
                        print(f"\n  [{i+1}] {type_emoji} {msg_type.upper()}")
                        print(f"      {content}")
                else:
                    print("  (No messages in thread)")
            except Exception as e:
                print(f"  Error fetching messages: {e}")
        
        # Get run history
        print("\n🏃 RUN HISTORY (last 5)")
        print("-" * 40)
        try:
            runs = await client.runs.list(thread_id=thread_id, limit=5)
            if runs:
                for run in runs:
                    run_id = run.get("run_id", "N/A")
                    status = run.get("status", "N/A")
                    created = format_timestamp(run.get("created_at"))
                    
                    status_emoji = {
                        "success": "✅",
                        "error": "❌",
                        "pending": "⏳",
                        "running": "🔄"
                    }.get(status, "❓")
                    
                    print(f"  {status_emoji} {run_id[:8]}... | {status} | {created}")
            else:
                print("  (No runs found)")
        except Exception as e:
            print(f"  (Could not fetch run history: {e})")
        
        print("\n" + "=" * 60)
        print("✅ Thread info retrieved successfully")
        
        # Return the full thread data
        return thread
        
    except Exception as e:
        print(f"\n❌ Error fetching thread: {e}")
        print("\nPossible causes:")
        print("  - Thread ID does not exist")
        print("  - LangGraph server is not running")
        print("  - Invalid thread ID format")
        return None


def format_timestamp(ts) -> str:
    """Format a timestamp for display."""
    if not ts:
        return "N/A"
    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
        except Exception:
            return ts
    return str(ts)


async def search_threads_by_metadata(key: str, value: str, api_url: Optional[str] = None):
    """Search for threads by metadata key-value pair.
    
    Args:
        key: Metadata key to search for.
        value: Metadata value to match.
        api_url: The LangGraph API URL.
    """
    url = api_url or os.getenv("LANGGRAPH_API_URL", "http://127.0.0.1:2024")
    
    print(f"Searching for threads with {key}={value}")
    print("=" * 60)
    
    client = get_client(url=url)
    
    try:
        threads = await client.threads.search(
            metadata={key: value},
            limit=100
        )
        
        if threads:
            print(f"\nFound {len(threads)} thread(s):\n")
            for thread in threads:
                thread_id = thread.get("thread_id", "N/A")
                metadata = thread.get("metadata") or {}
                name = metadata.get("name") or metadata.get("chat_name") or "Unnamed"
                print(f"  - {thread_id} | {name}")
        else:
            print("\nNo threads found matching criteria.")
            
        return threads
        
    except Exception as e:
        print(f"\n❌ Error searching threads: {e}")
        return []


async def list_all_threads(api_url: Optional[str] = None, limit: int = 20):
    """List all threads with summary info.
    
    Args:
        api_url: The LangGraph API URL.
        limit: Maximum number of threads to list.
    """
    url = api_url or os.getenv("LANGGRAPH_API_URL", "http://127.0.0.1:2024")
    
    print(f"Listing threads (limit: {limit})")
    print("=" * 60)
    
    client = get_client(url=url)
    
    try:
        threads = await client.threads.search(limit=limit)
        
        if threads:
            print(f"\nFound {len(threads)} thread(s):\n")
            print(f"{'Thread ID':<40} | {'Name':<25} | {'Updated'}")
            print("-" * 85)
            
            for thread in threads:
                thread_id = thread.get("thread_id", "N/A")
                metadata = thread.get("metadata") or {}
                name = (metadata.get("name") or metadata.get("chat_name") or "Unnamed")[:25]
                updated = format_timestamp(thread.get("updated_at"))[:19]
                
                print(f"{thread_id:<40} | {name:<25} | {updated}")
        else:
            print("\nNo threads found.")
            
        return threads
        
    except Exception as e:
        print(f"\n❌ Error listing threads: {e}")
        return []


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Get info and metadata about a LangGraph thread",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get info for a specific thread
  python get_thread_info.py abc123-def456-789

  # Get info with message history
  python get_thread_info.py abc123-def456-789 --messages

  # List all threads
  python get_thread_info.py --list

  # Search by metadata
  python get_thread_info.py --search chat_id=12345_c_us

  # Export to JSON
  python get_thread_info.py abc123-def456-789 --json > thread.json
        """
    )
    
    parser.add_argument(
        "thread_id",
        nargs="?",
        help="The thread ID to inspect"
    )
    parser.add_argument(
        "--messages", "-m",
        action="store_true",
        help="Include message history in output"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all threads"
    )
    parser.add_argument(
        "--search", "-s",
        help="Search by metadata (format: key=value)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Limit for list/search results (default: 20)"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output as JSON (for thread_id mode)"
    )
    parser.add_argument(
        "--url",
        help="LangGraph API URL (default: from LANGGRAPH_API_URL env or http://127.0.0.1:2024)"
    )
    
    args = parser.parse_args()
    
    api_url = args.url or os.getenv("LANGGRAPH_API_URL", "http://127.0.0.1:2024")
    
    print("=" * 60)
    print("LangGraph Thread Info Tool")
    print("=" * 60)
    print(f"\nTarget server: {api_url}\n")
    
    if args.list:
        await list_all_threads(api_url, args.limit)
    elif args.search:
        if "=" not in args.search:
            print("Error: --search format must be key=value")
            sys.exit(1)
        key, value = args.search.split("=", 1)
        await search_threads_by_metadata(key, value, api_url)
    elif args.thread_id:
        thread = await get_thread_info(args.thread_id, api_url, args.messages)
        if args.json and thread:
            print("\n📄 JSON OUTPUT")
            print("-" * 40)
            print(json.dumps(thread, indent=2, default=str))
    else:
        parser.print_help()
        print("\n💡 Tip: Use --list to see all available threads")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
