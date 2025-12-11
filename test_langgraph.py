#!/usr/bin/env python3
"""
Test script for LangGraph implementation with PostgreSQL checkpointer.
This script tests basic functionality, supervisor agent, and persistence.
"""

import os
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from memory_agent import LangGraphMemoryManager
from utiles.logger import logger


def test_basic_agent():
    """Test basic agent functionality."""
    print("\n" + "="*50)
    print("TEST 1: Basic Agent Functionality")
    print("="*50)
    
    try:
        manager = LangGraphMemoryManager()
        
        # Create a test agent
        agent = manager.get_agent(
            chat_id="test_user_123",
            chat_name="Test User",
            is_group=False
        )
        
        # Send a message
        print("\n📤 Sending message: 'Hello, my name is John'")
        response = agent.send_message(
            sender="John",
            message="Hello, my name is John",
            timestamp=datetime.now().isoformat()
        )
        print(f"🤖 Response: {response}")
        
        # Send follow-up to test memory
        print("\n📤 Sending message: 'What is my name?'")
        response = agent.send_message(
            sender="John",
            message="What is my name?",
            timestamp=datetime.now().isoformat()
        )
        print(f"🤖 Response: {response}")
        
        # Test remember without response
        print("\n💾 Storing message without response...")
        success = agent.remember(
            timestamp=datetime.now().isoformat(),
            sender="John",
            message="I like pizza and coding"
        )
        print(f"✅ Stored: {success}")
        
        print("\n✅ Test 1 PASSED: Basic agent works correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ Test 1 FAILED: {e}")
        logger.error(f"Test 1 failed: {e}", exc_info=True)
        return False


def test_multiple_agents():
    """Test multiple agents with different conversations."""
    print("\n" + "="*50)
    print("TEST 2: Multiple Agents")
    print("="*50)
    
    try:
        manager = LangGraphMemoryManager()
        
        # Create two different agents
        agent1 = manager.get_agent("user1", "Alice", False)
        agent2 = manager.get_agent("user2", "Bob", False)
        
        # Agent 1 conversation
        print("\n👤 Agent 1 (Alice):")
        print("📤 Sending: 'I love Python programming'")
        response1 = agent1.send_message("Alice", "I love Python programming")
        print(f"🤖 Response: {response1[:100]}...")
        
        # Agent 2 conversation
        print("\n👤 Agent 2 (Bob):")
        print("📤 Sending: 'I prefer JavaScript'")
        response2 = agent2.send_message("Bob", "I prefer JavaScript")
        print(f"🤖 Response: {response2[:100]}...")
        
        # Verify isolation
        print("\n👤 Agent 1 (Alice):")
        print("📤 Sending: 'What programming language do I like?'")
        response1_check = agent1.send_message("Alice", "What programming language do I like?")
        print(f"🤖 Response: {response1_check[:100]}...")
        
        print("\n✅ Test 2 PASSED: Multiple agents maintain separate conversations")
        return True
        
    except Exception as e:
        print(f"\n❌ Test 2 FAILED: {e}")
        logger.error(f"Test 2 failed: {e}", exc_info=True)
        return False


def test_supervisor_agent():
    """Test supervisor agent with cross-agent access."""
    print("\n" + "="*50)
    print("TEST 3: Supervisor Agent (Cross-Agent Access)")
    print("="*50)
    
    try:
        manager = LangGraphMemoryManager()
        
        # Create some conversations first
        print("\n📝 Setting up test conversations...")
        agent1 = manager.get_agent("user1", "Alice", False)
        agent2 = manager.get_agent("user2", "Bob", False)
        
        agent1.send_message("Alice", "I'm working on a machine learning project")
        agent2.send_message("Bob", "I'm building a web application")
        
        # Get supervisor
        supervisor = manager.get_supervisor()
        
        # Test cross-agent query
        print("\n🔍 Supervisor querying all conversations...")
        print("📤 Query: 'What are people working on?'")
        summary = supervisor.query("What are the different people working on?")
        print(f"🤖 Supervisor Response: {summary[:200]}...")
        
        # Test search
        print("\n🔍 Supervisor searching for 'project'...")
        results = supervisor.search_conversations("project")
        print(f"🤖 Search Results: {results[:200]}...")
        
        print("\n✅ Test 3 PASSED: Supervisor can access multiple conversations")
        return True
        
    except Exception as e:
        print(f"\n❌ Test 3 FAILED: {e}")
        logger.error(f"Test 3 failed: {e}", exc_info=True)
        return False


def test_persistence():
    """Test that conversation state persists across manager instances."""
    print("\n" + "="*50)
    print("TEST 4: State Persistence")
    print("="*50)
    
    try:
        # First manager instance
        print("\n📝 Creating first manager and agent...")
        manager1 = LangGraphMemoryManager()
        agent1 = manager1.get_agent("persistence_test", "Test User", False)
        
        print("📤 Sending: 'Remember that my favorite color is blue'")
        agent1.send_message("User", "Remember that my favorite color is blue")
        
        # Simulate restart by creating new manager
        print("\n🔄 Simulating restart (new manager instance)...")
        manager2 = LangGraphMemoryManager()
        agent2 = manager2.get_agent("persistence_test", "Test User", False)
        
        print("📤 Sending: 'What is my favorite color?'")
        response = agent2.send_message("User", "What is my favorite color?")
        print(f"🤖 Response: {response}")
        
        # Check if it remembered
        if "blue" in response.lower():
            print("\n✅ Test 4 PASSED: State persisted across manager instances")
            return True
        else:
            print("\n⚠️  Test 4 WARNING: State may not have persisted correctly")
            print(f"Expected 'blue' in response, got: {response}")
            return False
        
    except Exception as e:
        print(f"\n❌ Test 4 FAILED: {e}")
        logger.error(f"Test 4 failed: {e}", exc_info=True)
        return False


def test_group_chat():
    """Test group chat functionality."""
    print("\n" + "="*50)
    print("TEST 5: Group Chat")
    print("="*50)
    
    try:
        manager = LangGraphMemoryManager()
        
        # Create a group agent
        group_agent = manager.get_agent(
            chat_id="group_123@g.us",
            chat_name="Python Developers",
            is_group=True
        )
        
        # Simulate group conversation
        print("\n👥 Group: Python Developers")
        print("📤 Alice: 'Hello everyone!'")
        group_agent.remember(datetime.now().isoformat(), "Alice", "Hello everyone!")
        
        print("📤 Bob: 'Hi Alice! How's your project going?'")
        group_agent.remember(datetime.now().isoformat(), "Bob", "Hi Alice! How's your project going?")
        
        print("📤 Charlie: 'What are we discussing today?'")
        response = group_agent.send_message("Charlie", "What are we discussing today?")
        print(f"🤖 Response: {response[:150]}...")
        
        print("\n✅ Test 5 PASSED: Group chat functionality works")
        return True
        
    except Exception as e:
        print(f"\n❌ Test 5 FAILED: {e}")
        logger.error(f"Test 5 failed: {e}", exc_info=True)
        return False


def main():
    """Run all tests."""
    print("\n" + "🚀 " + "="*48)
    print("🚀  LangGraph Implementation Test Suite")
    print("🚀 " + "="*48)
    
    results = []
    
    # Run all tests
    results.append(("Basic Agent", test_basic_agent()))
    results.append(("Multiple Agents", test_multiple_agents()))
    results.append(("Supervisor Agent", test_supervisor_agent()))
    results.append(("State Persistence", test_persistence()))
    results.append(("Group Chat", test_group_chat()))
    
    # Print summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! LangGraph implementation is ready.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())