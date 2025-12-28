#!/usr/bin/env python3
"""
Script to send a test message using the same logic as app.py.

This simulates a WhatsApp message from a group chat for testing purposes.

Usage:
    python scripts/send_test_message.py

Environment Variables:
    LANGGRAPH_API_URL: The LangGraph API URL (default: http://127.0.0.1:2024)
"""

import os
import sys
from datetime import datetime

# Change to project root directory so .env file can be found
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

# Add src to path for imports
sys.path.insert(0, os.path.join(project_root, 'src'))

from langgraph_client import ThreadsManager, Thread


def send_test_message(
    chat_name: str = "testing group",
    sender_name: str = "sam",
    message: str = "im 32 years old",
    is_group: bool = True
):
    """Send a test message using the same logic as app.py.
    
    Args:
        chat_name: Name of the group/chat
        sender_name: Name of the message sender
        message: The message content
        is_group: Whether this is a group chat
    """
    # Use the same memory manager as app.py
    memory_manager = ThreadsManager()
    
    # Create chat_id from chat_name (simulating what WhatsappMSG does)
    chat_id = chat_name.replace(" ", "_").lower()
    timestamp = datetime.now().isoformat()
    
    print(f"Creating thread for chat: {chat_name}")
    thread: Thread = memory_manager.get_thread(
        is_group=is_group,
        chat_name=chat_name,
        chat_id=chat_id
    )
    
    print(f"\nSending message:")
    print(f"  {'Group' if is_group else 'Chat'}: {chat_name}")
    print(f"  Sender: {sender_name}")
    print(f"  Message: {message}")
    print(f"  Timestamp: {timestamp}")
    
    # Use remember() just like in app.py webhook handler
    success = thread.remember(
        timestamp=timestamp,
        sender=sender_name,
        message=message
    )
    
    if success:
        print(f"\n✓ Message stored successfully!")
        print(f"  Thread: {thread.to_string()}")
    else:
        print(f"\n✗ Failed to store message")


def main():
    """Main entry point."""
    print("=" * 60)
    print("LangGraph Test Message Sender")
    print("=" * 60)
    print()
    
    # Test message configuration - same as requested
    send_test_message(
        chat_name="testing group",
        sender_name="sam",
        message="im 32 years old",
        is_group=True
    )


if __name__ == "__main__":
    main()
