#!/usr/bin/env python3
"""
Live demo of LangGraph with cross-agent memory access.
This shows the supervisor agent reading all conversation threads.
"""

import sys
import os
sys.path.insert(0, 'src')

from memory_agent import LangGraphMemoryManager
from datetime import datetime

print("="*60)
print("🚀 LangGraph Cross-Agent Memory Demo")
print("="*60)

# Initialize manager
print("\n📦 Initializing LangGraphMemoryManager...")
manager = LangGraphMemoryManager()
print("✅ Manager initialized with MemorySaver checkpointer")

# Create agents for different chats
print("\n👥 Creating agents for different chats...")

# Agent 1: Alice
print("\n1️⃣  Creating agent for Alice...")
alice_agent = manager.get_agent("alice_chat", "Alice", False)
print("✅ Alice's agent created")

# Agent 2: Bob
print("\n2️⃣  Creating agent for Bob...")
bob_agent = manager.get_agent("bob_chat", "Bob", False)
print("✅ Bob's agent created")

# Agent 3: Project Group
print("\n3️⃣  Creating agent for Project Group...")
group_agent = manager.get_agent("project_group@g.us", "Project Team", True)
print("✅ Group agent created")

# Simulate conversations
print("\n" + "="*60)
print("💬 Simulating Conversations")
print("="*60)

print("\n📤 Alice: 'I'm working on the machine learning module'")
response1 = alice_agent.send_message(
    sender="Alice",
    message="I'm working on the machine learning module",
    timestamp=datetime.now().isoformat()
)
print(f"🤖 AI Response: {response1[:100]}...")

print("\n📤 Bob: 'I'm building the frontend dashboard'")
response2 = bob_agent.send_message(
    sender="Bob",
    message="I'm building the frontend dashboard",
    timestamp=datetime.now().isoformat()
)
print(f"🤖 AI Response: {response2[:100]}...")

print("\n📤 Group (Charlie): 'When is our next standup meeting?'")
response3 = group_agent.send_message(
    sender="Charlie",
    message="When is our next standup meeting?",
    timestamp=datetime.now().isoformat()
)
print(f"🤖 AI Response: {response3[:100]}...")

# Now use the supervisor to read ALL conversations
print("\n" + "="*60)
print("🔍 SUPERVISOR AGENT - Cross-Agent Memory Access")
print("="*60)

print("\n📊 Getting supervisor agent...")
supervisor = manager.get_supervisor()
print("✅ Supervisor initialized")

print("\n🔎 Supervisor querying: 'What are people working on across all chats?'")
supervisor_response = supervisor.query("What are people working on across all the different chats?")
print(f"\n🎯 Supervisor Response:\n{supervisor_response}")

print("\n🔎 Supervisor searching for 'meeting'...")
search_result = supervisor.search_conversations("meeting")
print(f"\n🎯 Search Result:\n{search_result}")

print("\n🔎 Getting overall summary...")
summary = supervisor.get_all_conversations_summary()
print(f"\n🎯 Summary:\n{summary}")

print("\n" + "="*60)
print("✅ DEMO COMPLETE")
print("="*60)
print("\nKey Takeaway:")
print("- ✅ Multiple agents with separate conversations")
print("- ✅ Supervisor can read ALL threads")
print("- ✅ Cross-agent search and summarization")
print("- ✅ No database setup needed (MemorySaver)")
print("\nThis solves the limitation you had with Letta!")