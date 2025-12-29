import asyncio
import os
import sys
from typing import Optional

from langgraph_sdk import get_client
from langsmith import Client as LangSmithClient


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


def delete_langsmith_traces(project_name: Optional[str] = None, recreate: bool = True):
    """Delete all traces from LangSmith by deleting and recreating the project.
    
    The LangSmith SDK doesn't support deleting individual runs/traces directly.
    The only way to delete all traces is to delete the entire project.
    
    Args:
        project_name: The LangSmith project name. Defaults to env var LANGCHAIN_PROJECT.
        recreate: Whether to recreate the project after deletion. Defaults to True.
    """
    project = project_name or os.getenv("LANGCHAIN_PROJECT", "default")
    api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
    
    if not api_key:
        print("\nNo LangSmith API key found. Set LANGCHAIN_API_KEY or LANGSMITH_API_KEY.")
        print("Skipping LangSmith trace deletion.")
        return
    
    print(f"\n{'=' * 60}")
    print("LangSmith Traces Deletion")
    print("=" * 60)
    print(f"\nConnecting to LangSmith...")
    print(f"Project: {project}")
    
    try:
        client = LangSmithClient()
        
        # Check if project exists and get trace count
        try:
            runs = list(client.list_runs(
                project_name=project,
                is_root=True,
                limit=1000
            ))
            trace_count = len(runs)
            print(f"Found {trace_count} traces in project '{project}'.")
        except Exception:
            print(f"Project '{project}' not found or no traces.")
            return
        
        if trace_count == 0:
            print("No traces found. Nothing to delete.")
            return
        
        # Confirm deletion
        print(f"\nWARNING: This will DELETE the entire project '{project}' and all its traces.")
        if recreate:
            print("The project will be recreated as an empty project after deletion.")
        confirm = input(f"\nAre you sure you want to delete ALL {trace_count} traces? This cannot be undone. (yes/no): ")
        if confirm.lower() != 'yes':
            print("Aborted. No traces were deleted.")
            return
        
        # Delete the project (this deletes all traces)
        print(f"\nDeleting project '{project}'...")
        try:
            client.delete_project(project_name=project)
            print(f"Project '{project}' and all {trace_count} traces deleted successfully.")
        except Exception as e:
            print(f"Failed to delete project: {e}")
            return
        
        # Optionally recreate the project
        if recreate:
            print(f"\nRecreating project '{project}'...")
            try:
                client.create_project(project_name=project)
                print(f"Project '{project}' recreated successfully.")
            except Exception as e:
                print(f"Failed to recreate project: {e}")
                print("You may need to create it manually or it will be created on first trace.")
        
        print(f"\n--- LangSmith Traces Summary ---")
        print(f"Deleted: {trace_count} traces (by deleting project)")
        
    except Exception as e:
        print(f"\nError connecting to LangSmith: {e}")
        print("Make sure LANGCHAIN_API_KEY is set correctly.")


async def main():
    """Main entry point."""
    print("=" * 60)
    print("LangGraph Thread & Traces Deletion Script")
    print("=" * 60)
    print()
    
    api_url = os.getenv("LANGGRAPH_API_URL", "http://127.0.0.1:2024")
    
    print(f"LangGraph Target server: {api_url}")
    print()
    
    # First delete LangSmith traces (synchronous)
    delete_langsmith_traces()
    
    # Then delete LangGraph threads (async)
    try:
        await delete_all_threads(api_url)
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure the LangGraph dev server is running:")
        print("  langgraph dev")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
