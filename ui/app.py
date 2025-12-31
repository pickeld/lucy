import streamlit as st
import requests
import ast
import json

# Configuration
API_BASE_URL = "http://localhost:8765"

st.set_page_config(
    page_title="RAG Q&A Assistant",
    page_icon="💬",
    layout="centered"
)


def get_chat_list(api_url: str) -> list:
    """Fetch all unique chat names from the RAG API.
    
    Args:
        api_url: The base URL of the API
        
    Returns:
        List of chat names sorted alphabetically
    """
    try:
        response = requests.get(f"{api_url}/rag/chats", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("chats", [])
    except Exception:
        pass
    return []


def get_sender_list(api_url: str) -> list:
    """Fetch all unique sender names from the RAG API.
    
    Args:
        api_url: The base URL of the API
        
    Returns:
        List of sender names sorted alphabetically
    """
    try:
        response = requests.get(f"{api_url}/rag/senders", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("senders", [])
    except Exception:
        pass
    return []


st.title("💬 RAG Q&A Assistant")
st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", value=API_BASE_URL)
    k_results = st.slider("Number of context documents (k)",
                          min_value=1, max_value=50, value=10)
    
    # Get chat list for dropdown
    chat_list = get_chat_list(api_url)
    chat_options = [""] + chat_list  # Add empty option for "no filter"
    filter_chat = st.selectbox(
        "Filter by chat name (optional)",
        options=chat_options,
        index=0,
        format_func=lambda x: "All chats" if x == "" else x
    )
    
    # Get sender list for dropdown
    sender_list = get_sender_list(api_url)
    sender_options = [""] + sender_list  # Add empty option for "no filter"
    filter_sender = st.selectbox(
        "Filter by sender (optional)",
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
        "Filter by time range (optional)",
        options=list(DATE_RANGE_OPTIONS.keys()),
        index=0
    )
    filter_days = DATE_RANGE_OPTIONS[filter_date_range]

    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    # Show RAG stats
    st.markdown("---")
    st.header("RAG Stats")
    if st.button("Refresh Stats"):
        try:
            stats_response = requests.get(f"{api_url}/rag/stats", timeout=10)
            if stats_response.status_code == 200:
                st.session_state.rag_stats = stats_response.json()
        except Exception as e:
            st.error(f"Failed to fetch stats: {e}")

    if "rag_stats" in st.session_state:
        st.json(st.session_state.rag_stats)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Question input
question = st.chat_input("Ask a question...")

if question:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})

    # Display user message
    with st.chat_message("user"):
        st.markdown(question)

    # Query the RAG endpoint
    answer = ""  # Initialize answer to avoid "possibly unbound" error
    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer..."):
            try:
                # Build conversation history from session state (exclude current question)
                # Only include previous messages, not the one we just added
                conversation_history = []
                if len(st.session_state.messages) > 1:
                    # Get all messages except the last one (which is the current question)
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

                # Add conversation history for context
                if conversation_history:
                    payload["conversation_history"] = conversation_history

                # Add optional filters
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

                    # Try to parse string representations of dict/list
                    if isinstance(raw_answer, str):
                        # Check if it looks like a dict or list string
                        stripped = raw_answer.strip()
                        if (stripped.startswith('{') and stripped.endswith('}')) or \
                           (stripped.startswith('[') and stripped.endswith(']')):
                            try:
                                # Try ast.literal_eval for Python dict format
                                raw_answer = ast.literal_eval(stripped)
                            except (ValueError, SyntaxError):
                                try:
                                    # Try JSON parsing as fallback
                                    raw_answer = json.loads(stripped)
                                except json.JSONDecodeError:
                                    pass  # Keep as string

                    # Handle different response formats
                    if isinstance(raw_answer, dict):
                        # Format: {'type': 'text', 'text': '...', 'extras': {...}}
                        answer = raw_answer.get("text", str(raw_answer))
                    elif isinstance(raw_answer, list):
                        # Format: [{'type': 'text', 'text': '...'}]
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
