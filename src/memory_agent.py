from typing import Annotated, TypedDict, List, Optional, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver, PostgresSaver
from config import config
from utiles.logger import logger
from datetime import datetime


# Define the state structure for our agent
class AgentState(TypedDict):
    """State structure for LangGraph agent conversations."""
    messages: Annotated[List[BaseMessage], add_messages]
    chat_id: str
    chat_name: str
    is_group: bool


class LangGraphMemoryManager:
    """Manages LangGraph agents with in-memory checkpointer for persistent memory."""
    
    def __init__(self):
        """Initialize the LangGraph memory manager with MemorySaver backend."""
        logger.debug("Initializing LangGraphMemoryManager")
        
        # Initialize MemorySaver checkpointer (in-memory, cross-agent accessible)
        # For PostgreSQL persistence, install langgraph-checkpoint-postgres
        self.checkpointer = MemorySaver()
        
        # Initialize OpenAI LLM
        self.llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=float(getattr(config, 'OPENAI_TEMPERATURE', 0.7)),
            api_key=config.OPENAI_API_KEY
        )
        
        # Cache for agent graphs
        self.agents = {}
        
        # Create supervisor agent with access to all conversations
        self.supervisor_agent = self._create_supervisor_agent()
        
        logger.info("LangGraphMemoryManager initialized successfully with MemorySaver")

    def _create_agent_graph(self, chat_id: str, chat_name: str, is_group: bool):
        """Create a LangGraph agent for a specific chat."""
        
        def chat_node(state: AgentState):
            """Main chat processing node."""
            messages = state["messages"]
            
            # Add system message with context
            system_msg = SystemMessage(content=f"""You are a helpful AI assistant for WhatsApp.
Chat Type: {'Group' if is_group else 'Personal'}
Chat Name: {chat_name}

Remember conversations and provide contextual responses based on the chat history.""")
            
            # Invoke the LLM with full message history
            response = self.llm.invoke([system_msg] + messages)
            
            return {"messages": [response]}
        
        # Build the graph
        workflow = StateGraph(AgentState)
        workflow.add_node("chat", chat_node)
        workflow.add_edge(START, "chat")
        workflow.add_edge("chat", END)
        
        # Compile with checkpointer for persistence
        return workflow.compile(checkpointer=self.checkpointer)

    def _create_supervisor_agent(self):
        """Create a supervisor agent that can read all conversation threads."""
        
        def supervisor_node(state: AgentState):
            """Supervisor node with cross-agent memory access."""
            messages = state["messages"]
            
            # Get recent conversations from all threads
            all_conversations = self._get_all_recent_conversations(limit=50)
            
            context = f"""You are a supervisor AI with access to all WhatsApp conversations.
            
Recent conversations across all chats:
{all_conversations}

Use this context to provide insights, summaries, or analysis across multiple conversations."""
            
            system_msg = SystemMessage(content=context)
            response = self.llm.invoke([system_msg] + messages)
            
            return {"messages": [response]}
        
        workflow = StateGraph(AgentState)
        workflow.add_node("supervisor", supervisor_node)
        workflow.add_edge(START, "supervisor")
        workflow.add_edge("supervisor", END)
        
        return workflow.compile(checkpointer=self.checkpointer)

    def _get_all_recent_conversations(self, limit: int = 50) -> str:
        """Retrieve recent messages from all conversation threads."""
        try:
            # Access all checkpoints from MemorySaver
            conversations = []
            checkpoint_data = self.checkpointer.storage
            
            # Iterate through all thread checkpoints
            for thread_id, checkpoint_dict in list(checkpoint_data.items())[:limit]:
                if thread_id == "supervisor":
                    continue
                    
                # Get the most recent checkpoint
                if checkpoint_dict:
                    latest = max(checkpoint_dict.keys())
                    checkpoint = checkpoint_dict[latest]
                    
                    # Extract messages
                    if 'channel_values' in checkpoint:
                        messages = checkpoint['channel_values'].get('messages', [])
                        if messages:
                            msg_text = "\n".join([
                                f"  {msg.content if hasattr(msg, 'content') else str(msg)}"
                                for msg in messages[-5:]  # Last 5 messages
                            ])
                            conversations.append(f"Thread {thread_id}:\n{msg_text}\n")
            
            return "\n".join(conversations) if conversations else "No recent conversations found."
        except Exception as e:
            logger.error(f"Error retrieving conversations: {e}")
            return "Error accessing conversation history."

    def get_agent(self, chat_id: str, chat_name: str, is_group: bool):
        """Get or create an agent for a specific chat."""
        # Normalize chat_id
        normalized_id = chat_id.replace("@", "_").replace(".", "_")
        
        if normalized_id not in self.agents:
            logger.debug(f"Creating new agent for chat: {normalized_id}")
            self.agents[normalized_id] = LangGraphAgent(
                chat_id=normalized_id,
                chat_name=chat_name,
                is_group=is_group,
                graph=self._create_agent_graph(normalized_id, chat_name, is_group)
            )
        
        return self.agents[normalized_id]

    def get_supervisor(self):
        """Get the supervisor agent with cross-chat access."""
        return LangGraphSupervisor(
            graph=self.supervisor_agent,
            manager=self
        )


class LangGraphAgent:
    """Individual chat agent with persistent memory."""
    
    def __init__(self, chat_id: str, chat_name: str, is_group: bool, graph):
        self.chat_id = chat_id
        self.chat_name = chat_name
        self.is_group = is_group
        self.graph = graph
        self.config = {"configurable": {"thread_id": chat_id}}
        
        logger.info(f"LangGraphAgent initialized for {chat_id}")

    def send_message(self, sender: str, message: str, timestamp: Optional[str] = None) -> str:
        """Send a message and get a response."""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Format message with metadata
        formatted_message = f"[{timestamp}] {sender}: {message}"
        
        # Create state with message
        state = {
            "messages": [HumanMessage(content=formatted_message)],
            "chat_id": self.chat_id,
            "chat_name": self.chat_name,
            "is_group": self.is_group
        }
        
        # Add metadata for LangSmith UI display
        config_with_metadata = {
            **self.config,
            "metadata": {
                "chat_name": self.chat_name,
                "chat_id": self.chat_id,
                "sender": sender,
                "is_group": self.is_group
            },
            "run_name": f"{self.chat_name} - {sender}"
        }
        
        # Invoke graph with persistent state and metadata
        result = self.graph.invoke(state, config_with_metadata)
        
        # Extract AI response
        if result and "messages" in result and result["messages"]:
            last_message = result["messages"][-1]
            return last_message.content if hasattr(last_message, 'content') else str(last_message)
        
        return "No response generated"

    def remember(self, timestamp: str, sender: str, message: str) -> bool:
        """Store a message in the conversation history without generating a response."""
        try:
            formatted_message = f"[{timestamp}] {sender}: {message}"
            
            state = {
                "messages": [HumanMessage(content=formatted_message)],
                "chat_id": self.chat_id,
                "chat_name": self.chat_name,
                "is_group": self.is_group
            }
            
            # Just invoke to store in checkpoint, don't return response
            self.graph.invoke(state, self.config)
            return True
        except Exception as e:
            logger.error(f"Error storing message: {e}")
            return False

    def get_history(self, limit: int = 10) -> List[BaseMessage]:
        """Retrieve conversation history."""
        try:
            # Get state from checkpointer
            state = self.graph.get_state(self.config)
            if state and "messages" in state.values:
                return state.values["messages"][-limit:]
            return []
        except Exception as e:
            logger.error(f"Error retrieving history: {e}")
            return []


class LangGraphSupervisor:
    """Supervisor agent with access to all conversation threads."""
    
    def __init__(self, graph, manager):
        self.graph = graph
        self.manager = manager
        self.config = {"configurable": {"thread_id": "supervisor"}}
        logger.info("LangGraphSupervisor initialized")

    def query(self, question: str) -> str:
        """Query across all conversations."""
        state = {
            "messages": [HumanMessage(content=question)],
            "chat_id": "supervisor",
            "chat_name": "Supervisor",
            "is_group": False
        }
        
        result = self.graph.invoke(state, self.config)
        
        if result and "messages" in result and result["messages"]:
            last_message = result["messages"][-1]
            return last_message.content if hasattr(last_message, 'content') else str(last_message)
        
        return "No response generated"

    def get_all_conversations_summary(self) -> str:
        """Get a summary of all recent conversations."""
        return self.query("Provide a summary of the most recent conversations across all chats.")

    def search_conversations(self, keyword: str) -> str:
        """Search for a keyword across all conversations."""
        return self.query(f"Search all conversations for mentions of: {keyword}")