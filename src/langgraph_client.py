"""LangGraph client for managing WhatsApp conversation threads.

Provides thread management, message storage, and LLM interaction
for WhatsApp conversations using LangGraph.
"""

import asyncio
import concurrent.futures
import os
import traceback
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph_sdk import get_client

from config import config
from rag import RAG, format_timestamp
from utils.logger import logger

# Module-level RAG singleton instance for efficient reuse
_rag_instance: Optional[RAG] = None


def get_rag() -> RAG:
    """Get the shared RAG singleton instance.
    
    Returns:
        The RAG singleton instance
    """
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAG()
    return _rag_instance


class ThreadState(TypedDict, total=False):
    """State structure for LangGraph thread conversations."""
    messages: Annotated[List[BaseMessage], add_messages]
    chat_id: str
    chat_name: str
    is_group: bool
    action: str  # "store" or "chat" - controls routing to skip or invoke LLM


def create_graph():
    """Create a standalone LangGraph graph for development and testing.

    The graph supports two modes:
    - action="store": Just stores the message without invoking LLM
    - action="chat": Invokes LLM to generate a response
    """
    # Initialize LLM based on configured provider
    llm_provider = os.getenv('LLM_PROVIDER', 'openai').lower()

    if llm_provider == 'gemini':
        llm = ChatGoogleGenerativeAI(
            model=getattr(config, 'GEMINI_MODEL', 'gemini-pro'),
            temperature=float(getattr(config, 'GEMINI_TEMPERATURE', '0.7')),
            google_api_key=config.GOOGLE_API_KEY
        )
    else:
        llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=float(getattr(config, 'OPENAI_TEMPERATURE', 0.7)),
            api_key=config.OPENAI_API_KEY  # type: ignore[arg-type]
        )

    def store_node(state: ThreadState) -> Dict[str, Any]:
        """Store-only node - just passes through without LLM invocation."""
        logger.debug("Store node: storing message without LLM invocation")
        # Messages are already added to state via add_messages reducer
        # Just return empty dict to preserve state without adding new messages
        return {}

    def chat_node(state: ThreadState) -> Dict[str, List[AIMessage]]:
        """Chat node - invokes LLM to generate a response."""
        logger.info("Chat node: invoking LLM for response")
        messages = state.get("messages", [])
        chat_name = state.get("chat_name", "Unknown")
        is_group = state.get("is_group", False)

        # Validate that we have messages to process
        if not messages:
            logger.warning(
                "Chat node: No messages to process, returning empty response")
            return {"messages": [AIMessage(content="I don't see any messages to respond to.")]}

        # Filter out any messages with empty content
        valid_messages: List[BaseMessage] = []
        for msg in messages:
            content = msg.content if hasattr(msg, 'content') else str(msg)
            # Handle case where content might be a list
            if isinstance(content, list):
                content = str(content[0]) if content else ""
            if content and content.strip():
                valid_messages.append(msg)

        if not valid_messages:
            logger.warning("Chat node: All messages have empty content")
            return {"messages": [AIMessage(content="I don't see any messages to respond to.")]}

        messages = valid_messages

        # Log the messages for debugging
        for i, msg in enumerate(messages[-5:]):  # Log last 5 messages
            content = msg.content if hasattr(msg, 'content') else str(msg)
            logger.debug(f"  Message {i}: {content[:100]}...")

        # Add system message with context
        # IMPORTANT: Messages are formatted as "[timestamp] sender_name: message_content"
        # Each message includes the actual sender's name, which may be different from the user asking
        if is_group:
            system_msg = SystemMessage(content=f"""You are a helpful AI assistant for a WhatsApp group chat.
Chat Name: {chat_name}

IMPORTANT: This is a group chat with multiple participants. Each message in the history is formatted as:
[timestamp] sender_name: message_content

The sender_name indicates WHO sent that specific message. Different messages may come from different people.
When a user asks about what someone said (e.g., "what did Adi say?"), look at the sender_name prefix of each message to identify messages from that specific person.

The LAST message in the conversation is from the person currently asking you a question. Use the sender_name from that message to identify who is asking.

Remember conversations and provide contextual responses based on the chat history.""")
        else:
            system_msg = SystemMessage(content=f"""You are a helpful AI assistant for WhatsApp.
Chat Type: Personal
Chat Name: {chat_name}

Messages are formatted as: [timestamp] sender_name: message_content
The sender_name indicates who sent each message.

Remember conversations and provide contextual responses based on the chat history.""")

        # Invoke the LLM with full message history
        try:
            response = llm.invoke([system_msg] + messages)
            # Handle case where content might be a list (Gemini)
            response_content = response.content if hasattr(response, 'content') else str(response)
            if isinstance(response_content, list):
                response_content = str(response_content[0]) if response_content else ""
            # Ensure we return a proper AIMessage
            if not isinstance(response, AIMessage):
                response = AIMessage(content=str(response_content))

            return {"messages": [response]}
        except Exception as e:
            logger.error(f"Chat node: LLM invocation failed: {e}")
            logger.error(traceback.format_exc())
            # Return an error message instead of crashing
            error_response = AIMessage(
                content=f"Sorry, I encountered an error processing your request: {str(e)}")
            return {"messages": [error_response]}

    def route_by_action(state: ThreadState) -> str:
        """Route messages to store or chat node based on action."""
        action = state.get("action", None)  # Don't default to anything yet
        messages = state.get("messages", [])
        logger.info(f"[ROUTE] action={action}, num_messages={len(messages)}")

        # App messages are formatted as "[timestamp] sender: message"
        if messages:
            last_msg = messages[-1]
            last_content = last_msg.content if hasattr(
                last_msg, 'content') else str(last_msg)
            # Handle case where content might be a list
            if isinstance(last_content, list):
                last_content = str(last_content[0]) if last_content else ""

            is_formatted_msg = isinstance(last_content, str) and last_content.startswith(
                '[') and '] ' in last_content and ': ' in last_content

            # If message is NOT in app format and action was store, override to chat
            if not is_formatted_msg and action == "store":
                return "chat"

        # Use explicit action if provided, otherwise default to chat
        if action == "store":
            logger.info("[ROUTE] -> store node (action=store)")
            return "store"

        logger.info(f"[ROUTE] -> chat node (action={action})")
        return "chat"

    # Build the graph with conditional routing
    workflow = StateGraph(ThreadState)
    workflow.add_node("store", store_node)
    workflow.add_node("chat", chat_node)

    # Add conditional edge from START based on action
    workflow.add_conditional_edges(START, route_by_action, {
                                   "store": "store", "chat": "chat"})
    workflow.add_edge("store", END)
    workflow.add_edge("chat", END)

    # Compile without checkpointer for dev mode (LangGraph dev provides its own)
    return workflow.compile()


class ThreadsManager:
    """Manages LangGraph threads for WhatsApp conversations."""

    def __init__(self, api_url: Optional[str] = None):
        """Initialize the threads manager.

        Args:
            api_url: The LangGraph dev server URL
        """
        self.api_url = api_url or os.getenv(
            "LANGGRAPH_API_URL", "http://127.0.0.1:2024")
        self._async_manager = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.threads: Dict[str, 'Thread'] = {}

        logger.info(f"ThreadsManager initialized with API URL: {self.api_url}")

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create an event loop."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
            return self._loop

    def _run_async(self, coro: Any) -> Any:
        """Run an async coroutine synchronously."""
        loop = self._get_loop()
        if loop.is_running():
            # We're in an async context, create a task
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)

    def get_thread(self, chat_id: str, chat_name: str, is_group: bool) -> 'Thread':
        """Get or create a thread for a specific chat.
        
        Args:
            chat_id: The chat ID from WhatsApp
            chat_name: Display name of the chat
            is_group: Whether this is a group chat
            
        Returns:
            Thread instance for the chat
        """
        normalized_id = chat_id.replace("@", "_").replace(".", "_")

        if normalized_id not in self.threads:
            logger.debug(f"Creating new thread for chat: {normalized_id}")
            self.threads[normalized_id] = Thread(
                api_url=self.api_url,
                graph_name="memory_agent",
                chat_id=normalized_id,
                chat_name=chat_name,
                is_group=is_group
            )

        return self.threads[normalized_id]


class Thread:
    """Represents a chat conversation thread (DM or group) in LangGraph."""

    def __init__(
        self, 
        api_url: str, 
        graph_name: str, 
        chat_id: str, 
        chat_name: str, 
        is_group: bool
    ):
        """Initialize a thread.
        
        Args:
            api_url: The LangGraph API URL
            graph_name: Name of the graph to use
            chat_id: Normalized chat ID
            chat_name: Display name of the chat
            is_group: Whether this is a group chat
        """
        self.api_url = api_url
        self.graph_name = graph_name
        self.chat_id = chat_id
        self.chat_name = chat_name
        self.is_group = is_group
        self._thread_id: Optional[str] = None

        logger.info(f"Thread initialized for {chat_id} ({chat_name})")

    def _run_async(self, coro: Any) -> Any:
        """Run an async coroutine synchronously."""
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        except RuntimeError:
            return asyncio.run(coro)

    async def _async_ensure_thread(self, client: Any) -> str:
        """Ensure a thread exists for this chat.
        
        Args:
            client: The LangGraph client
            
        Returns:
            The thread ID
        """
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
            thread = await client.threads.create(
                thread_id=None,  # Let it auto-generate
                metadata={
                    "chat_id": self.chat_id,
                    "chat_name": self.chat_name,
                    "is_group": self.is_group,
                    "thread_name": thread_name,
                    "name": thread_name,
                    "display_name": thread_name
                }
            )
            self._thread_id = thread["thread_id"]
            logger.debug(
                f"Created new thread: {self._thread_id} ({thread_name})")

        # At this point _thread_id is guaranteed to be set
        assert self._thread_id is not None
        return self._thread_id

    def remember(
        self, 
        timestamp: str, 
        sender: str, 
        message: str, 
        store: bool = True
    ) -> bool:
        """Store a message in the conversation history without triggering AI response.

        Sets action="store" in the input state so the graph routes to the store node
        instead of the chat node (which invokes the LLM).

        Also adds the message to RAG vector store for cross-thread semantic search.
        
        Args:
            timestamp: Unix timestamp of the message
            sender: Name of the message sender
            message: The message content
            store: If True, store only without LLM. If False, invoke LLM.
            
        Returns:
            True if successful, False otherwise
        """
        async def _remember() -> bool:
            client = get_client(url=self.api_url)
            thread_id = await self._async_ensure_thread(client)

            # Convert Unix timestamp to human-readable format
            readable_timestamp = format_timestamp(timestamp)
            formatted_message = f"[{readable_timestamp}] {sender}: {message}"

            input_state = {
                "messages": [{"role": "user", "content": formatted_message}],
                "chat_id": self.chat_id,
                "chat_name": self.chat_name,
                "is_group": self.is_group,
                # This routes to store node, skipping LLM
                "action": "store" if store else "chat"
            }

            # Wait for the run to complete to ensure message is stored
            await client.runs.wait(
                thread_id=thread_id,
                assistant_id=self.graph_name,
                input=input_state,
                metadata={"sender": sender, "action": "store"}
            )

            logger.debug(
                f"Stored message in thread {thread_id}: {formatted_message[:50]}...")

            # Add message to RAG vector store for cross-thread search
            # Use the shared RAG singleton instance
            rag = get_rag()
            rag.add_message(
                thread_id=thread_id,
                chat_id=self.chat_id,
                chat_name=self.chat_name,
                is_group=self.is_group,
                sender=sender,
                message=message,
                timestamp=timestamp
            )

            return True

        try:
            return self._run_async(_remember())
        except Exception as e:
            logger.error(f"Error remembering message: {type(e).__name__}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def to_string(self) -> str:
        """Return string representation of this thread."""
        return f"Thread(chat_id={self.chat_id}, chat_name={self.chat_name}, is_group={self.is_group})"

    def __repr__(self) -> str:
        """Return detailed string representation."""
        return self.to_string()


# Create the graph at module load time
graph = create_graph()
