import streamlit as st
import requests
import ast
import json
import logging
import sys
import os
import uuid
from typing import Optional, Dict, Any

# Configure logging to stderr (which Streamlit doesn't capture)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

# Also optionally log to file if LOG_FILE env var is set
log_file = os.environ.get("LOG_FILE")
if log_file:
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

logger.info("🚀 Starting Streamlit app...")

# Configuration
API_BASE_URL = "http://localhost:8765"

st.set_page_config(
    page_title="WhatsApp RAG Assistant",
    page_icon="💬",
    layout="wide"
)

# Custom CSS — WhatsApp-inspired theme
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    /* WhatsApp-style chat bubbles */
    [data-testid="stChatMessage"][data-testid-type="user"] .stMarkdown {
        background-color: #DCF8C6;
        border-radius: 10px;
        padding: 8px 12px;
    }
    [data-testid="stChatMessage"][data-testid-type="assistant"] .stMarkdown {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 8px 12px;
        border: 1px solid #e0e0e0;
    }
    .search-result {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .metadata {
        font-size: 0.8rem;
        color: #666;
    }
    .context-box {
        background-color: #e8f4ea;
        border: 1px solid #4caf50;
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin-bottom: 1rem;
    }
    .context-label {
        font-weight: bold;
        color: #2e7d32;
    }
    /* Connection status indicator */
    .status-connected {
        color: #25D366;
        font-weight: bold;
    }
    .status-disconnected {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# API HELPER FUNCTIONS
# =============================================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_chat_list(api_url: str) -> list:
    """Fetch all unique chat names from the RAG API (cached 5 min)."""
    logger.debug(f"Fetching chat list from {api_url}/rag/chats")
    try:
        response = requests.get(f"{api_url}/rag/chats", timeout=10)
        logger.debug(f"Chat list response: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            return data.get("chats", [])
    except requests.exceptions.Timeout:
        logger.warning("Timeout fetching chat list")
    except requests.exceptions.ConnectionError:
        logger.warning("Connection error fetching chat list - API may not be running")
    except Exception as e:
        logger.error(f"Error fetching chat list: {e}")
    return []


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_sender_list(api_url: str) -> list:
    """Fetch all unique sender names from the RAG API (cached 5 min)."""
    logger.debug(f"Fetching sender list from {api_url}/rag/senders")
    try:
        response = requests.get(f"{api_url}/rag/senders", timeout=10)
        logger.debug(f"Sender list response: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            return data.get("senders", [])
    except requests.exceptions.Timeout:
        logger.warning("Timeout fetching sender list")
    except requests.exceptions.ConnectionError:
        logger.warning("Connection error fetching sender list - API may not be running")
    except Exception as e:
        logger.error(f"Error fetching sender list: {e}")
    return []


@st.cache_data(ttl=60)  # Cache for 1 minute
def get_rag_stats(api_url: str) -> dict:
    """Fetch RAG statistics (cached 1 min)."""
    logger.debug(f"Fetching RAG stats from {api_url}/rag/stats")
    try:
        response = requests.get(f"{api_url}/rag/stats", timeout=10)
        logger.debug(f"RAG stats response: {response.status_code}")
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.Timeout:
        logger.warning("Timeout fetching RAG stats")
    except requests.exceptions.ConnectionError:
        logger.warning("Connection error fetching RAG stats - API may not be running")
    except Exception as e:
        logger.error(f"Error fetching RAG stats: {e}")
    return {}


def check_api_health(api_url: str) -> dict:
    """Check API health status (cached 30 seconds)."""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code in (200, 503):
            return response.json()
    except Exception:
        pass
    return {"status": "unreachable", "dependencies": {}}


def export_chat_history(messages: list) -> str:
    """Export chat history as a formatted text string."""
    lines = []
    for msg in messages:
        role = "You" if msg["role"] == "user" else "Assistant"
        lines.append(f"[{role}]\n{msg['content']}\n")
    return "\n".join(lines)


def delete_conversation(api_url: str, conversation_id: str) -> bool:
    """Delete a conversation's data (filters + chat history)."""
    try:
        response = requests.delete(
            f"{api_url}/conversation/{conversation_id}",
            timeout=10
        )
        return response.status_code == 200
    except Exception:
        pass
    return False


logger.info("✅ Imports and functions loaded successfully")
print("✅ Imports and functions loaded successfully", flush=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

logger.info("Initializing session state...")
print("Initializing session state...", flush=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

if "active_filters" not in st.session_state:
    st.session_state.active_filters = {}

if "api_url" not in st.session_state:
    st.session_state.api_url = API_BASE_URL


# =============================================================================
# SIDEBAR
# =============================================================================

logger.info("Building sidebar...")
print("Building sidebar...", flush=True)

with st.sidebar:
    logger.info("Rendering sidebar header...")
    print("Rendering sidebar header...", flush=True)
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/6b/WhatsApp.svg", width=50)
    st.header("⚙️ Settings")
    
    api_url = st.text_input("API URL", value=st.session_state.api_url)
    st.session_state.api_url = api_url
    
    # Connection status indicator
    health = check_api_health(api_url)
    api_status = health.get("status", "unreachable")
    if api_status == "up":
        st.markdown('<span class="status-connected">🟢 API Connected</span>', unsafe_allow_html=True)
    elif api_status == "degraded":
        st.markdown('<span class="status-disconnected">🟡 API Degraded</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-disconnected">🔴 API Unreachable</span>', unsafe_allow_html=True)
    k_results = st.slider("Context documents (k)", min_value=1, max_value=50, value=10)
    
    st.markdown("---")
    st.subheader("🔍 Filters")
    
    # Get chat list for dropdown
    logger.info("Fetching chat list...")
    print("Fetching chat list...", flush=True)
    chat_list = get_chat_list(api_url)
    logger.info(f"Got {len(chat_list)} chats")
    print(f"Got {len(chat_list)} chats", flush=True)
    chat_options = [""] + chat_list
    filter_chat = st.selectbox(
        "Chat/Group",
        options=chat_options,
        index=0,
        format_func=lambda x: "All chats" if x == "" else x
    )
    
    # Get sender list for dropdown
    logger.info("Fetching sender list...")
    print("Fetching sender list...", flush=True)
    sender_list = get_sender_list(api_url)
    logger.info(f"Got {len(sender_list)} senders")
    print(f"Got {len(sender_list)} senders", flush=True)
    sender_options = [""] + sender_list
    filter_sender = st.selectbox(
        "Sender",
        options=sender_options,
        index=0,
        format_func=lambda x: "All senders" if x == "" else x
    )
    
    # Date range filter
    DATE_RANGE_OPTIONS = {
        "All time": None,
        "Last 24 hours": 1,
        "Last 3 days": 3,
        "Last week": 7,
        "Last month": 30
    }
    filter_date_range = st.selectbox(
        "Time range",
        options=list(DATE_RANGE_OPTIONS.keys()),
        index=0
    )
    filter_days = DATE_RANGE_OPTIONS[filter_date_range]

    st.markdown("---")
    
    # Conversation info
    st.subheader("🧠 Conversation")
    if st.session_state.conversation_id:
        st.caption(f"ID: `{st.session_state.conversation_id[:8]}...`")
        st.caption(f"Messages: {len(st.session_state.messages)}")
        
        # Show active filters
        filters = st.session_state.active_filters
        if filters:
            if filters.get("chat_name"):
                st.caption(f"💬 {filters['chat_name']}")
            if filters.get("sender"):
                st.caption(f"👤 {filters['sender']}")
        
        col_new, col_export = st.columns(2)
        with col_new:
            if st.button("🔄 New", key="new_conversation"):
                st.session_state.conversation_id = None
                st.session_state.messages = []
                st.session_state.active_filters = {}
                st.rerun()
        with col_export:
            if st.session_state.messages:
                export_text = export_chat_history(st.session_state.messages)
                st.download_button(
                    "📥 Export",
                    data=export_text,
                    file_name="chat_history.txt",
                    mime="text/plain",
                    key="export_chat",
                )
    else:
        st.caption("No active conversation")
    
    st.markdown("---")
    
    # Show RAG stats
    st.subheader("📊 Statistics")
    stats = get_rag_stats(api_url)
    if stats:
        st.metric("Total Documents", stats.get("total_documents", 0))
        st.caption(f"Collection: {stats.get('collection_name', 'N/A')}")
        if stats.get("dashboard_url"):
            st.markdown(f"[🔗 Qdrant Dashboard]({stats['dashboard_url']})")


# =============================================================================
# MAIN CONTENT
# =============================================================================

st.title("💬 WhatsApp RAG Assistant")
st.caption("Powered by LlamaIndex CondensePlusContextChatEngine + Qdrant")

# Create tabs
tab_chat, tab_search = st.tabs(["💬 Chat", "🔍 Search"])

# === CHAT TAB ===
with tab_chat:
    # Context indicator box
    filters = st.session_state.active_filters
    has_filters = filters.get("chat_name") or filters.get("sender")
    
    if has_filters:
        context_parts = []
        if filters.get("chat_name"):
            context_parts.append(f"💬 **Chat:** {filters['chat_name']}")
        if filters.get("sender"):
            context_parts.append(f"👤 **Sender:** {filters['sender']}")
        if filters.get("days"):
            context_parts.append(f"📅 **Days:** {filters['days']}")
        
        st.markdown(
            f"""<div class="context-box">
            <span class="context-label">🧠 Active Filters:</span> {' | '.join(context_parts)}
            </div>""",
            unsafe_allow_html=True
        )
    
    # Control buttons row
    col1, col2, col3 = st.columns([5, 1, 1])
    with col2:
        if st.button("🧹 Clear Filters", key="clear_filters", disabled=not has_filters):
            st.session_state.active_filters = {}
            st.rerun()
    with col3:
        if st.button("🗑️ Clear Chat", key="clear_chat"):
            if st.session_state.conversation_id:
                delete_conversation(api_url, st.session_state.conversation_id)
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.session_state.active_filters = {}
            st.rerun()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Question input
    question = st.chat_input("Ask a question about your WhatsApp messages...")
    
    if question:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(question)
        
        # Query the RAG endpoint
        answer = ""
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching and generating answer..."):
                try:
                    # Build request payload
                    payload = {
                        "question": question,
                        "k": k_results
                    }
                    
                    # Include conversation ID if we have one
                    if st.session_state.conversation_id:
                        payload["conversation_id"] = st.session_state.conversation_id
                    
                    # Include explicit filters from sidebar
                    if filter_chat.strip():
                        payload["filter_chat_name"] = filter_chat.strip()
                    if filter_sender.strip():
                        payload["filter_sender"] = filter_sender.strip()
                    if filter_days is not None:
                        payload["filter_days"] = filter_days
                    
                    # Call the RAG query endpoint
                    response = requests.post(
                        f"{api_url}/rag/query",
                        json=payload,
                        timeout=300
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        raw_answer = data.get("answer", "No answer received")
                        
                        # Update conversation state from response
                        if data.get("conversation_id"):
                            st.session_state.conversation_id = data["conversation_id"]
                        
                        if data.get("filters"):
                            st.session_state.active_filters = data["filters"]
                        
                        # Parse response
                        if isinstance(raw_answer, str):
                            stripped = raw_answer.strip()
                            if (stripped.startswith('{') and stripped.endswith('}')) or \
                               (stripped.startswith('[') and stripped.endswith(']')):
                                try:
                                    raw_answer = ast.literal_eval(stripped)
                                except (ValueError, SyntaxError):
                                    try:
                                        raw_answer = json.loads(stripped)
                                    except json.JSONDecodeError:
                                        pass
                        
                        if isinstance(raw_answer, dict):
                            answer = raw_answer.get("text", str(raw_answer))
                        elif isinstance(raw_answer, list):
                            texts = []
                            for item in raw_answer:
                                if isinstance(item, dict) and "text" in item:
                                    texts.append(item["text"])
                                else:
                                    texts.append(str(item))
                            answer = "\n".join(texts)
                        else:
                            answer = str(raw_answer)
                        
                        st.markdown(answer)
                        
                        # Show source citations as expandable section
                        sources = data.get("sources", [])
                        if sources:
                            with st.expander(f"📎 Sources ({len(sources)} messages)", expanded=False):
                                for i, src in enumerate(sources):
                                    score = src.get("score")
                                    score_str = f" — {score:.2%} relevant" if score else ""
                                    sender = src.get("sender", "Unknown")
                                    chat = src.get("chat_name", "Unknown")
                                    content = src.get("content", "")
                                    
                                    st.markdown(
                                        f"**{i+1}. {sender}** in _{chat}_{score_str}\n\n"
                                        f"> {content[:200]}{'...' if len(content) > 200 else ''}"
                                    )
                                    if i < len(sources) - 1:
                                        st.divider()
                        
                        # Show filter context if active
                        active_filters = data.get("filters", {})
                        if active_filters.get("chat_name"):
                            st.caption(f"📍 Chat: {active_filters['chat_name']}")
                    else:
                        error_data = response.json()
                        answer = f"❌ Error ({response.status_code}): {error_data.get('error', 'Unknown error')}"
                        st.error(answer)
                
                except requests.exceptions.ConnectionError:
                    answer = "❌ Connection error: Could not connect to the API. Make sure the server is running."
                    st.error(answer)
                except requests.exceptions.Timeout:
                    answer = "❌ Timeout: The request took too long to complete."
                    st.error(answer)
                except Exception as e:
                    answer = f"❌ Unexpected error: {str(e)}"
                    st.error(answer)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})


# === SEARCH TAB ===
with tab_search:
    st.subheader("🔍 Semantic Search")
    st.caption("Search for specific messages using natural language")
    
    search_query = st.text_input("Search query", placeholder="meeting tomorrow...")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        search_k = st.number_input("Results", min_value=1, max_value=100, value=20)
    
    if st.button("🔍 Search", type="primary"):
        if search_query:
            with st.spinner("Searching..."):
                try:
                    payload = {
                        "query": search_query,
                        "k": search_k
                    }
                    if filter_chat.strip():
                        payload["filter_chat_name"] = filter_chat.strip()
                    if filter_sender.strip():
                        payload["filter_sender"] = filter_sender.strip()
                    if filter_days is not None:
                        payload["filter_days"] = filter_days
                    
                    response = requests.post(
                        f"{api_url}/rag/search",
                        json=payload,
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        results = data.get("results", [])
                        
                        st.success(f"Found {len(results)} results")
                        
                        for i, result in enumerate(results):
                            with st.expander(f"📝 Result {i+1} (Score: {result.get('score', 'N/A'):.4f})", expanded=i < 3):
                                st.markdown(result.get("content", "No content"))
                                
                                # Show metadata
                                metadata = result.get("metadata", {})
                                if metadata:
                                    cols = st.columns(3)
                                    with cols[0]:
                                        st.caption(f"👤 {metadata.get('sender', 'Unknown')}")
                                    with cols[1]:
                                        st.caption(f"💬 {metadata.get('chat_name', 'Unknown')}")
                                    with cols[2]:
                                        st.caption(f"📅 {metadata.get('timestamp', 'N/A')}")
                    else:
                        st.error(f"Search failed: {response.text}")
                
                except Exception as e:
                    st.error(f"Search error: {str(e)}")
        else:
            st.warning("Please enter a search query")


# Footer
st.markdown("---")
st.caption("WhatsApp RAG Assistant • Powered by LlamaIndex CondensePlusContextChatEngine + Qdrant")
