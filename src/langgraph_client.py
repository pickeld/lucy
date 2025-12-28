import os
import sys

# Add src directory to Python path for LangGraph dev compatibility
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from typing import Annotated, List, Optional, Dict, Any, TypedDict
from datetime import datetime

from langgraph_sdk import get_client
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from config import config
from utiles.logger import logger


class ThreadState(TypedDict, total=False):
    """State structure for LangGraph thread conversations."""
    messages: Annotated[List[BaseMessage], add_messages]
    chat_id: str
    chat_name: str
    is_group: bool
    # Note: 'action' is intentionally NOT in the state to avoid persistence issues
    # It's passed via input but not stored in checkpoints


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
            api_key=config.OPENAI_API_KEY
        )

    def store_node(state: ThreadState):
        """Store-only node - just passes through without LLM invocation."""
        print(f"[STORE NODE] Storing message without LLM invocation")
        logger.debug(f"Store node: storing message without LLM invocation")
        # Messages are already added to state via add_messages reducer
        # Just return empty dict to preserve state without adding new messages
        return {}

    def chat_node(state: ThreadState):
        """Chat node - invokes LLM to generate a response."""
        print(f"[CHAT NODE] Invoking LLM for response")
        logger.info(f"Chat node: invoking LLM for response")
        messages = state.get("messages", [])
        chat_name = state.get("chat_name", "Unknown")
        is_group = state.get("is_group", False)

        print(f"[CHAT NODE] {len(messages)} messages in history, chat_name={chat_name}")
        logger.info(f"Chat node: {len(messages)} messages in history, chat_name={chat_name}, is_group={is_group}")
        
        # Log the messages for debugging
        for i, msg in enumerate(messages[-5:]):  # Log last 5 messages
            content = msg.content if hasattr(msg, 'content') else str(msg)
            print(f"[CHAT NODE]   Message {i}: {content[:100]}...")
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
            logger.info(f"Chat node: Sending {len(messages) + 1} messages to LLM (including system)")
            response = llm.invoke([system_msg] + messages)
            response_content = response.content if hasattr(response, 'content') else str(response)
            logger.info(f"Chat node: LLM response received ({len(response_content)} chars): {response_content[:100]}...")
            
            # Ensure we return a proper AIMessage
            if not isinstance(response, AIMessage):
                response = AIMessage(content=response_content)
            
            return {"messages": [response]}
        except Exception as e:
            logger.error(f"Chat node: LLM invocation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return an error message instead of crashing
            error_response = AIMessage(content=f"Sorry, I encountered an error processing your request: {str(e)}")
            return {"messages": [error_response]}

    def route_by_action(state: ThreadState) -> str:
        """Route to appropriate node based on action field.
        
        IMPORTANT: The 'action' field can get persisted in thread state.
        To handle Studio UI properly (which doesn't send action), we check
        the last message - if it's a simple user message without the
        timestamp/sender format, it's likely from the Studio UI and should
        route to chat.
        """
        action = state.get("action", None)  # Don't default to anything yet
        messages = state.get("messages", [])
        
        print(f"[ROUTE] action={action}, num_messages={len(messages)}")
        logger.info(f"[ROUTE] action={action}, num_messages={len(messages)}")
        
        # Check if this is likely a Studio UI message (no timestamp/sender format)
        # App messages are formatted as "[timestamp] sender: message"
        if messages:
            last_msg = messages[-1]
            last_content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
            # Handle case where content might be a list
            if isinstance(last_content, list):
                last_content = str(last_content[0]) if last_content else ""
            # Studio UI sends plain messages, app sends "[timestamp] sender: message"
            is_formatted_msg = isinstance(last_content, str) and last_content.startswith('[') and '] ' in last_content and ': ' in last_content
            print(f"[ROUTE] Last message formatted={is_formatted_msg}: {str(last_content)[:50]}...")
            
            # If message is NOT in app format and action was store, override to chat
            if not is_formatted_msg and action == "store":
                print(f"[ROUTE] -> chat node (Studio UI message detected, overriding store)")
                logger.info(f"[ROUTE] -> chat node (Studio UI detected)")
                return "chat"
        
        # Use explicit action if provided, otherwise default to chat
        if action == "store":
            print(f"[ROUTE] -> store node")
            logger.info(f"[ROUTE] -> store node (action=store)")
            return "store"
        
        print(f"[ROUTE] -> chat node")
        logger.info(f"[ROUTE] -> chat node (action={action})")
        return "chat"

    # Build the graph with conditional routing
    workflow = StateGraph(ThreadState)
    workflow.add_node("store", store_node)
    workflow.add_node("chat", chat_node)
    
    # Add conditional edge from START based on action
    workflow.add_conditional_edges(START, route_by_action, {"store": "store", "chat": "chat"})
    workflow.add_edge("store", END)
    workflow.add_edge("chat", END)

    # Compile without checkpointer for dev mode (LangGraph dev provides its own)
    return workflow.compile()


# Export the compiled graph for LangGraph dev
graph = create_graph()


# =============================================================================
# LangGraph SDK Client Classes (for app.py to communicate with LangGraph Studio)
# =============================================================================


class ThreadsManager:
    """Manages LangGraph threads for WhatsApp conversations."""
    
    def __init__(self, api_url: Optional[str] = None):
        """Initialize the threads manager.
        
        Args:
            api_url: The LangGraph dev server URL
        """
        import asyncio
        self.api_url = api_url or os.getenv("LANGGRAPH_API_URL", "http://127.0.0.1:2024")
        self._async_manager = None
        self._loop = None
        self.threads: Dict[str, 'Thread'] = {}
        
        logger.info(f"ThreadsManager initialized with API URL: {self.api_url}")
    
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
    
    def get_thread(self, chat_id: str, chat_name: str, is_group: bool) -> 'Thread':
        """Get or create a thread for a specific chat."""
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
        """Initialize a thread."""
        self.api_url = api_url
        self.graph_name = graph_name
        self.chat_id = chat_id
        self.chat_name = chat_name
        self.is_group = is_group
        self._thread_id = None
        
        logger.info(f"Thread initialized for {chat_id} ({chat_name})")
    
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
        return f"Thread(chat_id={self.chat_id}, chat_name={self.chat_name}, is_group={self.is_group})"
