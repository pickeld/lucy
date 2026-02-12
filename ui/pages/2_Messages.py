"""Message Browser page for WhatsApp-GPT.

Browse, search, and filter WhatsApp messages stored in the RAG vector store.
"""

import streamlit as st
import requests
from datetime import datetime
from zoneinfo import ZoneInfo

# Configuration
API_BASE_URL = "http://localhost:8765"

st.set_page_config(
    page_title="Messages — WhatsApp RAG Assistant",
    page_icon="📱",
    layout="wide",
)

st.title("📱 Message Browser")
st.caption("Browse WhatsApp messages stored in the RAG vector store")

api_url = st.session_state.get("api_url", API_BASE_URL)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@st.cache_data(ttl=300)
def get_chat_list(url: str) -> list:
    """Fetch chat list for filter dropdown."""
    try:
        resp = requests.get(f"{url}/rag/chats", timeout=10)
        if resp.status_code == 200:
            return resp.json().get("chats", [])
    except Exception:
        pass
    return []


@st.cache_data(ttl=300)
def get_sender_list(url: str) -> list:
    """Fetch sender list for filter dropdown."""
    try:
        resp = requests.get(f"{url}/rag/senders", timeout=10)
        if resp.status_code == 200:
            return resp.json().get("senders", [])
    except Exception:
        pass
    return []


def fetch_messages(url: str, chat_name: str = "", sender: str = "",
                   days: int = 0, limit: int = 50, offset: str = "") -> dict:
    """Fetch messages from the API."""
    try:
        params: dict = {"limit": limit}
        if chat_name:
            params["chat_name"] = chat_name
        if sender:
            params["sender"] = sender
        if days and days > 0:
            params["days"] = days
        if offset:
            params["offset"] = offset

        resp = requests.get(f"{url}/rag/messages", params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        st.error(f"Failed to fetch messages: {e}")
    return {"messages": [], "count": 0, "has_more": False, "next_offset": None}


def format_timestamp(ts: int) -> str:
    """Format Unix timestamp for display."""
    try:
        dt = datetime.fromtimestamp(ts, tz=ZoneInfo("Asia/Jerusalem"))
        return dt.strftime("%d/%m/%Y %H:%M")
    except Exception:
        return str(ts)


# =============================================================================
# SIDEBAR FILTERS
# =============================================================================

with st.sidebar:
    st.subheader("🔍 Filters")

    chat_list = get_chat_list(api_url)
    chat_options = [""] + chat_list
    filter_chat = st.selectbox(
        "Chat/Group",
        options=chat_options,
        index=0,
        format_func=lambda x: "All chats" if x == "" else x,
        key="msg_filter_chat",
    )

    sender_list = get_sender_list(api_url)
    sender_options = [""] + sender_list
    filter_sender = st.selectbox(
        "Sender",
        options=sender_options,
        index=0,
        format_func=lambda x: "All senders" if x == "" else x,
        key="msg_filter_sender",
    )

    DATE_RANGE_OPTIONS = {
        "All time": 0,
        "Last 24 hours": 1,
        "Last 3 days": 3,
        "Last week": 7,
        "Last month": 30,
    }
    filter_date_range = st.selectbox(
        "Time range",
        options=list(DATE_RANGE_OPTIONS.keys()),
        index=0,
        key="msg_filter_date",
    )
    filter_days = DATE_RANGE_OPTIONS[filter_date_range]

    page_size = st.slider("Messages per page", min_value=10, max_value=200, value=50, step=10)

    if st.button("🔄 Refresh", key="msg_refresh"):
        st.cache_data.clear()
        st.rerun()


# =============================================================================
# PAGINATION STATE
# =============================================================================

if "msg_offset" not in st.session_state:
    st.session_state.msg_offset = ""
if "msg_page" not in st.session_state:
    st.session_state.msg_page = 1


# =============================================================================
# FETCH AND DISPLAY MESSAGES
# =============================================================================

data = fetch_messages(
    url=api_url,
    chat_name=filter_chat.strip(),
    sender=filter_sender.strip(),
    days=filter_days,
    limit=page_size,
    offset=st.session_state.msg_offset,
)

messages = data.get("messages", [])
has_more = data.get("has_more", False)
next_offset = data.get("next_offset")

# Stats bar
col_count, col_page, col_nav = st.columns([2, 1, 2])
with col_count:
    st.caption(f"Showing {len(messages)} messages (page {st.session_state.msg_page})")
with col_nav:
    nav_col1, nav_col2 = st.columns(2)
    with nav_col1:
        if st.button("⬅️ First Page", disabled=(st.session_state.msg_page <= 1)):
            st.session_state.msg_offset = ""
            st.session_state.msg_page = 1
            st.rerun()
    with nav_col2:
        if st.button("➡️ Next Page", disabled=(not has_more)):
            st.session_state.msg_offset = next_offset or ""
            st.session_state.msg_page += 1
            st.rerun()

st.markdown("---")

# Display messages
if not messages:
    st.info("No messages found. Try adjusting your filters or check that the API is running.")
else:
    for msg in messages:
        ts = msg.get("timestamp", 0)
        formatted_time = format_timestamp(ts) if ts else "Unknown"
        sender = msg.get("sender", "Unknown")
        chat_name = msg.get("chat_name", "Unknown")
        message = msg.get("message", "")
        is_group = msg.get("is_group", False)
        source_type = msg.get("source_type", "whatsapp")
        has_media = msg.get("has_media", False)

        # Message card
        with st.container():
            # Header row
            header_cols = st.columns([3, 3, 2])
            with header_cols[0]:
                st.markdown(f"**👤 {sender}**")
            with header_cols[1]:
                icon = "👥" if is_group else "💬"
                st.markdown(f"{icon} {chat_name}")
            with header_cols[2]:
                st.caption(f"🕐 {formatted_time}")

            # Message body
            if has_media:
                st.markdown(f"📎 {message}")
            elif source_type == "conversation_chunk":
                st.text(message[:500])
            else:
                st.markdown(message)

            st.markdown("---")


# Footer
st.caption("📱 Message Browser — WhatsApp RAG Assistant")
