"""
LangGraph SDK Client for connecting to LangGraph Studio/Dev server.

This module provides a client that communicates with the LangGraph dev server
(http://127.0.0.1:2024) allowing conversations to be visible in LangGraph Studio.
"""

import os
from typing import List, Optional, Dict, Any
from datetime import datetime

from langgraph_sdk import get_client
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from utiles.logger import logger


class StudioMemoryManager:
    """Manages LangGraph threads via the LangGraph Studio API."""

    def __init__(self, api_url: Optional[str] = None):
        """Initialize the LangGraph Studio client.
        
        Args:
            api_url: The LangGraph dev server URL. Defaults to http://127.0.0.1:2024
        """
        self.api_url = api_url or os.getenv("LANGGRAPH_API_URL", "http://127.0.0.1:2024")
        self.client = get_client(url=self.api_url)
        self.graph_name = "memory_agent"  # Must match the graph name in langgraph.json
        
        # Cache for thread instances
        self.threads: Dict[str, 'StudioThread'] = {}
        
        logger.info(f"StudioMemoryManager initialized with API URL: {self.api_url}")

    def get_thread(self, chat_id: str, chat_name: str, is_group: bool) -> 'StudioThread':
        """Get or create a thread for a specific chat.
        
        Args:
            chat_id: The chat identifier
            chat_name: Display name for the chat
            is_group: Whether this is a group chat
            
        Returns:
            StudioThread instance for this chat
        """
        # Normalize chat_id for use as thread_id
        normalized_id = chat_id.replace("@", "_").replace(".", "_")
        
        if normalized_id not in self.threads:
            logger.debug(f"Creating new studio thread for chat: {normalized_id}")
            self.threads[normalized_id] = StudioThread(
                client=self.client,
                graph_name=self.graph_name,
                chat_id=normalized_id,
                chat_name=chat_name,
                is_group=is_group
            )
        
        return self.threads[normalized_id]


class StudioThread:
    """Individual chat thread that communicates with LangGraph Studio."""

    def __init__(
        self,
        client,
        graph_name: str,
        chat_id: str,
        chat_name: str,
        is_group: bool
    ):
        """Initialize a studio thread.
        
        Args:
            client: The LangGraph SDK client
            graph_name: Name of the graph in langgraph.json
            chat_id: Normalized chat identifier (used as thread_id)
            chat_name: Display name for the chat
            is_group: Whether this is a group chat
        """
        self.client = client
        self.graph_name = graph_name
        self.chat_id = chat_id
        self.chat_name = chat_name
        self.is_group = is_group
        self._thread_id = None
        
        logger.info(f"StudioThread initialized for {chat_id} ({chat_name})")

    async def _ensure_thread(self) -> str:
        """Ensure a thread exists for this chat, create if needed.
        
        Returns:
            The thread_id
        """
        if self._thread_id:
            return self._thread_id
        
        # Search for existing thread with matching metadata
        threads = await self.client.threads.search(
            metadata={"chat_id": self.chat_id},
            limit=1
        )
        
        if threads:
            self._thread_id = threads[0]["thread_id"]
            logger.debug(f"Found existing thread: {self._thread_id}")
        else:
            # Use chat_name as the thread name (group name for groups, person name for DMs)
            thread_name = self.chat_name
            
            # Create new thread with metadata
            # Note: LangGraph Studio uses thread_id or a specific field for display
            thread = await self.client.threads.create(
                thread_id=None,  # Let it auto-generate
                metadata={
                    "chat_id": self.chat_id,
                    "chat_name": self.chat_name,
                    "is_group": self.is_group,
                    "thread_name": thread_name,  # Alternative field name
                    "name": thread_name,  # Display name in Studio
                    "display_name": thread_name  # Try another field name
                }
            )
            self._thread_id = thread["thread_id"]
            logger.debug(f"Created new thread: {self._thread_id} ({thread_name})")
        
        return self._thread_id

    async def send_message(
        self,
        sender: str,
        message: str,
        timestamp: Optional[str] = None
    ) -> str:
        """Send a message and get a response.
        
        Args:
            sender: Name of the message sender
            message: The message content
            timestamp: Optional timestamp (defaults to now)
            
        Returns:
            The AI response text
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        thread_id = await self._ensure_thread()
        
        # Format message with metadata
        formatted_message = f"[{timestamp}] {sender}: {message}"
        
        # Create the input state
        input_state = {
            "messages": [{"role": "user", "content": formatted_message}],
            "chat_id": self.chat_id,
            "chat_name": self.chat_name,
            "is_group": self.is_group
        }
        
        try:
            # Invoke the graph via the API
            result = await self.client.runs.wait(
                thread_id=thread_id,
                assistant_id=self.graph_name,
                input=input_state,
                metadata={
                    "sender": sender,
                    "chat_name": self.chat_name
                }
            )
            
            # Extract the AI response
            if result and "messages" in result:
                messages = result["messages"]
                if messages:
                    last_message = messages[-1]
                    if isinstance(last_message, dict):
                        return last_message.get("content", "No response")
                    return str(last_message)
            
            return "No response generated"
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            raise

    async def remember(
        self,
        timestamp: str,
        sender: str,
        message: str
    ) -> bool:
        """Store a message in the conversation history without triggering AI response.
        
        Sets action="store" in the input state so the graph routes to the store node
        instead of the chat node (which invokes the LLM).
        
        Args:
            timestamp: When the message was sent
            sender: Who sent the message
            message: The message content
            
        Returns:
            True if successful, False otherwise
        """
        try:
            thread_id = await self._ensure_thread()
            
            formatted_message = f"[{timestamp}] {sender}: {message}"
            
            # Set action="store" to route to store node (no LLM invocation)
            input_state = {
                "messages": [{"role": "user", "content": formatted_message}],
                "chat_id": self.chat_id,
                "chat_name": self.chat_name,
                "is_group": self.is_group,
                "action": "store"  # This routes to store node, skipping LLM
            }
            
            # Run the graph with store action
            await self.client.runs.wait(
                thread_id=thread_id,
                assistant_id=self.graph_name,
                input=input_state,
                metadata={"sender": sender, "action": "store"}
            )
            
            logger.debug(f"Stored message in thread {thread_id}: {formatted_message[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error remembering message: {e}")
            return False

    async def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve conversation history.
        
        Args:
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of message dictionaries
        """
        try:
            thread_id = await self._ensure_thread()
            
            # Get thread state
            state = await self.client.threads.get_state(thread_id)
            
            if state and "values" in state and "messages" in state["values"]:
                messages = state["values"]["messages"]
                return messages[-limit:]
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting history: {e}")
            return []

    def to_string(self) -> str:
        """Return string representation of this thread."""
        return f"StudioThread(chat_id={self.chat_id}, chat_name={self.chat_name}, is_group={self.is_group})"


# Synchronous wrapper for use in Flask
class SyncStudioMemoryManager:
    """Synchronous wrapper for StudioMemoryManager for use in Flask."""
    
    def __init__(self, api_url: Optional[str] = None):
        """Initialize the sync wrapper.
        
        Args:
            api_url: The LangGraph dev server URL
        """
        import asyncio
        self.api_url = api_url or os.getenv("LANGGRAPH_API_URL", "http://127.0.0.1:2024")
        self._async_manager = None
        self._loop = None
        self.threads: Dict[str, 'SyncStudioThread'] = {}
        
        logger.info(f"SyncStudioMemoryManager initialized with API URL: {self.api_url}")
    
    def _get_loop(self):
        """Get or create an event loop."""
        import asyncio
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
            return self._loop
    
    def _run_async(self, coro):
        """Run an async coroutine synchronously."""
        import asyncio
        loop = self._get_loop()
        if loop.is_running():
            # We're in an async context, create a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    
    def get_thread(self, chat_id: str, chat_name: str, is_group: bool) -> 'SyncStudioThread':
        """Get or create a thread for a specific chat."""
        normalized_id = chat_id.replace("@", "_").replace(".", "_")
        
        if normalized_id not in self.threads:
            logger.debug(f"Creating new sync studio thread for chat: {normalized_id}")
            self.threads[normalized_id] = SyncStudioThread(
                api_url=self.api_url,
                graph_name="memory_agent",
                chat_id=normalized_id,
                chat_name=chat_name,
                is_group=is_group
            )
        
        return self.threads[normalized_id]


class SyncStudioThread:
    """Synchronous wrapper for StudioThread."""
    
    def __init__(
        self,
        api_url: str,
        graph_name: str,
        chat_id: str,
        chat_name: str,
        is_group: bool
    ):
        """Initialize a sync studio thread."""
        self.api_url = api_url
        self.graph_name = graph_name
        self.chat_id = chat_id
        self.chat_name = chat_name
        self.is_group = is_group
        self._thread_id = None
        
        logger.info(f"SyncStudioThread initialized for {chat_id} ({chat_name})")
    
    def _run_async(self, coro):
        """Run an async coroutine synchronously."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        except RuntimeError:
            return asyncio.run(coro)
    
    async def _async_ensure_thread(self, client) -> str:
        """Ensure a thread exists for this chat."""
        if self._thread_id:
            return self._thread_id
        
        threads = await client.threads.search(
            metadata={"chat_id": self.chat_id},
            limit=1
        )
        
        if threads:
            self._thread_id = threads[0]["thread_id"]
            logger.debug(f"Found existing thread: {self._thread_id}")
        else:
            # Use chat_name as the thread name (group name for groups, person name for DMs)
            thread_name = self.chat_name
            
            # Create new thread with metadata
            # Note: LangGraph Studio uses thread_id or a specific field for display
            thread = await client.threads.create(
                thread_id=None,  # Let it auto-generate
                metadata={
                    "chat_id": self.chat_id,
                    "chat_name": self.chat_name,
                    "is_group": self.is_group,
                    "thread_name": thread_name,  # Alternative field name
                    "name": thread_name,  # Display name in Studio
                    "display_name": thread_name  # Try another field name
                }
            )
            self._thread_id = thread["thread_id"]
            logger.debug(f"Created new thread: {self._thread_id} ({thread_name})")
        
        return self._thread_id
    
    def send_message(self, sender: str, message: str, timestamp: Optional[str] = None) -> str:
        """Send a message and get a response (sync)."""
        async def _send():
            client = get_client(url=self.api_url)
            
            if timestamp is None:
                ts = datetime.now().isoformat()
            else:
                ts = timestamp
            
            thread_id = await self._async_ensure_thread(client)
            formatted_message = f"[{ts}] {sender}: {message}"
            
            # Set action="chat" to invoke LLM and get a response
            input_state = {
                "messages": [{"role": "user", "content": formatted_message}],
                "chat_id": self.chat_id,
                "chat_name": self.chat_name,
                "is_group": self.is_group,
                "action": "chat"  # This routes to chat node, invoking LLM
            }
            
            result = await client.runs.wait(
                thread_id=thread_id,
                assistant_id=self.graph_name,
                input=input_state,
                metadata={"sender": sender, "chat_name": self.chat_name, "action": "chat"}
            )
            
            if result and "messages" in result:
                messages = result["messages"]
                if messages:
                    last_message = messages[-1]
                    if isinstance(last_message, dict):
                        return last_message.get("content", "No response")
                    return str(last_message)
            
            return "No response generated"
        
        return self._run_async(_send())
    
    def remember(self, timestamp: str, sender: str, message: str) -> bool:
        """Store a message in the conversation history without triggering AI response.
        
        Sets action="store" in the input state so the graph routes to the store node
        instead of the chat node (which invokes the LLM).
        """
        async def _remember():
            client = get_client(url=self.api_url)
            thread_id = await self._async_ensure_thread(client)
            
            formatted_message = f"[{timestamp}] {sender}: {message}"
            
            # Set action="store" to route to store node (no LLM invocation)
            input_state = {
                "messages": [{"role": "user", "content": formatted_message}],
                "chat_id": self.chat_id,
                "chat_name": self.chat_name,
                "is_group": self.is_group,
                "action": "store"  # This routes to store node, skipping LLM
            }
            
            # Wait for the run to complete to ensure message is stored
            await client.runs.wait(
                thread_id=thread_id,
                assistant_id=self.graph_name,
                input=input_state,
                metadata={"sender": sender, "action": "store"}
            )
            
            logger.debug(f"Stored message in thread {thread_id}: {formatted_message[:50]}...")
            return True
        
        try:
            return self._run_async(_remember())
        except Exception as e:
            logger.error(f"Error remembering message: {e}")
            return False
    
    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve conversation history (sync)."""
        async def _get_history():
            client = get_client(url=self.api_url)
            thread_id = await self._async_ensure_thread(client)
            
            state = await client.threads.get_state(thread_id)
            
            if state and "values" in state and "messages" in state["values"]:
                return state["values"]["messages"][-limit:]
            
            return []
        
        try:
            return self._run_async(_get_history())
        except Exception as e:
            logger.error(f"Error getting history: {e}")
            return []
    
    def to_string(self) -> str:
        """Return string representation of this thread."""
        return f"SyncStudioThread(chat_id={self.chat_id}, chat_name={self.chat_name}, is_group={self.is_group})"
