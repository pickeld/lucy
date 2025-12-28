import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from typing import Annotated, TypedDict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Define the state structure for our agent
class AgentState(TypedDict):
    """State structure for LangGraph agent conversations."""
    messages: Annotated[List[BaseMessage], add_messages]
    chat_id: str
    chat_name: str
    is_group: bool


def create_chat_graph():
    """Create a LangGraph agent for chat conversations."""
    
    # Initialize LLM based on configured provider
    llm_provider = os.getenv('LLM_PROVIDER', 'openai').lower()
    
    if llm_provider == 'gemini':
        llm = ChatGoogleGenerativeAI(
            model=os.getenv('GEMINI_MODEL', 'gemini-pro'),
            temperature=float(os.getenv('GEMINI_TEMPERATURE', '0.7')),
            google_api_key=os.getenv('GOOGLE_API_KEY')
        )
    else:
        llm = ChatOpenAI(
            model=os.getenv('OPENAI_MODEL', 'gpt-4o'),
            temperature=float(os.getenv('OPENAI_TEMPERATURE', '0.7')),
            api_key=os.getenv('OPENAI_API_KEY')
        )

    def chat_node(state: AgentState):
        """Main chat processing node."""
        messages = state["messages"]
        chat_name = state.get("chat_name", "User")
        is_group = state.get("is_group", False)

        # Add system message with context
        system_msg = SystemMessage(content=f"""You are a helpful AI assistant for WhatsApp.
Chat Type: {'Group' if is_group else 'Personal'}
Chat Name: {chat_name}
Remember conversations and provide contextual responses based on the chat history.""")

        # Invoke the LLM with full message history
        response = llm.invoke([system_msg] + messages)

        return {"messages": [response]}

    # Build the graph
    workflow = StateGraph(AgentState)
    workflow.add_node("chat", chat_node)
    workflow.add_edge(START, "chat")
    workflow.add_edge("chat", END)

    # Compile with in-memory checkpointer for dev mode
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# Create the graph instance for LangGraph to discover
graph = create_chat_graph()
