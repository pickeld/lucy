import streamlit as st
import requests
import ast
import json

# Configuration
API_BASE_URL = "http://localhost:8765"

st.set_page_config(
    page_title="WhatsApp RAG Assistant",
    page_icon="💬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
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
</style>
""", unsafe_allow_html=True)


def get_chat_list(api_url: str) -> list:
    """Fetch all unique chat names from the RAG API."""
    try:
        response = requests.get(f"{api_url}/rag/chats", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("chats", [])
    except Exception:
        pass
    return []


def get_sender_list(api_url: str) -> list:
    """Fetch all unique sender names from the RAG API."""
    try:
        response = requests.get(f"{api_url}/rag/senders", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("senders", [])
    except Exception:
        pass
    return []


def get_rag_stats(api_url: str) -> dict:
    """Fetch RAG statistics."""
    try:
        response = requests.get(f"{api_url}/rag/stats", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return {}


def get_threads(api_url: str) -> list:
    """Fetch active conversation threads."""
    try:
        response = requests.get(f"{api_url}/threads", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("threads", [])
    except Exception:
        pass
    return []


# Sidebar configuration
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/6b/WhatsApp.svg", width=50)
    st.header("⚙️ Settings")
    
    api_url = st.text_input("API URL", value=API_BASE_URL)
    k_results = st.slider("Context documents (k)", min_value=1, max_value=50, value=10)
    
    st.markdown("---")
    st.subheader("🔍 Filters")
    
    # Get chat list for dropdown
    chat_list = get_chat_list(api_url)
    chat_options = [""] + chat_list
    filter_chat = st.selectbox(
        "Chat/Group",
        options=chat_options,
        index=0,
        format_func=lambda x: "All chats" if x == "" else x
    )
    
    # Get sender list for dropdown
    sender_list = get_sender_list(api_url)
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
    
    # Show RAG stats
    st.subheader("📊 Statistics")
    stats = get_rag_stats(api_url)
    if stats:
        st.metric("Total Documents", stats.get("total_documents", 0))
        st.caption(f"Collection: {stats.get('collection_name', 'N/A')}")
        if stats.get("dashboard_url"):
            st.markdown(f"[🔗 Qdrant Dashboard]({stats['dashboard_url']})")


# Main content
st.title("💬 WhatsApp RAG Assistant")
st.caption("Powered by LlamaIndex + Qdrant")

# Create tabs
tab_chat, tab_search, tab_threads = st.tabs(["💬 Chat", "🔍 Search", "📋 Threads"])

# === CHAT TAB ===
with tab_chat:
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Clear button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🗑️ Clear", key="clear_chat"):
            st.session_state.messages = []
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
                    # Build conversation history
                    conversation_history = []
                    if len(st.session_state.messages) > 1:
                        for msg in st.session_state.messages[:-1]:
                            conversation_history.append({
                                "role": msg["role"],
                                "content": msg["content"]
                            })
                    
                    # Build request payload
                    payload = {
                        "question": question,
                        "k": k_results
                    }
                    
                    if conversation_history:
                        payload["conversation_history"] = conversation_history
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


# === THREADS TAB ===
with tab_threads:
    st.subheader("📋 Active Conversation Threads")
    st.caption("View and manage conversation memory")
    
    if st.button("🔄 Refresh Threads"):
        st.rerun()
    
    threads = get_threads(api_url)
    
    if threads:
        for thread in threads:
            chat_type = "👥" if thread.get("is_group") else "👤"
            with st.expander(f"{chat_type} {thread.get('chat_name', 'Unknown')} ({thread.get('message_count', 0)} messages)"):
                st.json(thread)
                
                if st.button(f"🗑️ Clear Thread", key=f"clear_{thread.get('chat_id')}"):
                    try:
                        response = requests.post(
                            f"{api_url}/threads/{thread.get('chat_id')}/clear",
                            timeout=10
                        )
                        if response.status_code == 200:
                            st.success("Thread cleared!")
                            st.rerun()
                        else:
                            st.error("Failed to clear thread")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    else:
        st.info("No active conversation threads found")


# Footer
st.markdown("---")
st.caption("WhatsApp RAG Assistant • Powered by LlamaIndex + Qdrant")
