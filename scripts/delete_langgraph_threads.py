import asyncio
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from langgraph_sdk import get_client


from typing import Optional


async def delete_all_threads(api_url: Optional[str] = None):
    """Delete all threads from LangGraph server.
    
    Args:
        api_url: The LangGraph API URL. Defaults to env var or localhost.
    """
    url = api_url or os.getenv("LANGGRAPH_API_URL", "http://127.0.0.1:2024")
    
    print(f"Connecting to LangGraph server at: {url}")
    client = get_client(url=url)
    
    # Get all threads
    print("Fetching all threads...")
    threads = await client.threads.search(limit=1000)
    
    if not threads:
        print("No threads found. Nothing to delete.")
        return
    
    print(f"Found {len(threads)} threads to delete.")
    
    # Confirm deletion
    confirm = input(f"\nAre you sure you want to delete ALL {len(threads)} threads? This cannot be undone. (yes/no): ")
    if confirm.lower() != 'yes':
        print("Aborted. No threads were deleted.")
        return
    
    # Delete each thread
    deleted_count = 0
    failed_count = 0
    
    for i, thread in enumerate(threads):
        thread_id = thread.get("thread_id") or thread.get("id")
        metadata = thread.get("metadata") or {}
        thread_name = metadata.get("name", "Unknown")
        
        if not thread_id:
            print(f"  [{i+1}/{len(threads)}] Skipped: No thread ID found")
            failed_count += 1
            continue
        
        try:
            await client.threads.delete(thread_id)
            deleted_count += 1
            print(f"  [{i+1}/{len(threads)}] Deleted: {thread_name} ({thread_id})")
        except Exception as e:
            failed_count += 1
            print(f"  [{i+1}/{len(threads)}] Failed to delete {thread_id}: {e}")
    
    print(f"\n--- Summary ---")
    print(f"Deleted: {deleted_count} threads")
    print(f"Failed: {failed_count} threads")


async def delete_all_runs(api_url: Optional[str] = None):
    """Delete all runs from LangGraph server (if supported).
    
    Args:
        api_url: The LangGraph API URL. Defaults to env var or localhost.
    """
    api_url = api_url or os.getenv("LANGGRAPH_API_URL", "http://127.0.0.1:2024")
    
    print(f"\nNote: Run history is typically tied to threads.")
    print("Deleting threads should also remove associated runs.")


async def main():
    """Main entry point."""
    print("=" * 60)
    print("LangGraph Thread Deletion Script")
    print("=" * 60)
    print()
    
    api_url = os.getenv("LANGGRAPH_API_URL", "http://127.0.0.1:2024")
    
    print(f"Target server: {api_url}")
    print()
    
    try:
        await delete_all_threads(api_url)
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure the LangGraph dev server is running:")
        print("  langgraph dev")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
