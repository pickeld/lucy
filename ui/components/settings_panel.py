"""Settings panel component — shown in the sidebar as an expander.

Renders:
- Filter dropdowns (Chat/Group, Sender, Time Range)
- RAG configuration (k results)
- API URL configuration
- System health dashboard
- RAG statistics
"""

from typing import Optional

import streamlit as st

from utils.api import (
    check_health,
    get_chat_list,
    get_rag_stats,
    get_sender_list,
)


# Date range options
DATE_RANGE_OPTIONS = {
    "All time": None,
    "Last 24 hours": 1,
    "Last 3 days": 3,
    "Last week": 7,
    "Last month": 30,
}


def render_settings_panel() -> None:
    """Render the settings panel in the sidebar (only when toggled on)."""
    if not st.session_state.get("show_settings", False):
        return

    with st.sidebar:
        st.markdown("---")

        with st.expander("🔍 Filters", expanded=True):
            _render_filters()

        with st.expander("⚙️ Configuration", expanded=False):
            _render_config()

        with st.expander("📊 Statistics", expanded=False):
            _render_stats()

        with st.expander("🏥 System Health", expanded=False):
            _render_health()


# =========================================================================
# FILTERS
# =========================================================================

def _render_filters() -> None:
    """Render filter dropdowns for chat, sender, and time range."""
    # Chat/Group filter
    chat_list = get_chat_list()
    chat_options = [""] + chat_list
    filter_chat: str = st.selectbox(
        "Chat / Group",
        options=chat_options,
        index=0,
        format_func=lambda x: "All chats" if x == "" else x,
        key="filter_chat",
    ) or ""

    # Sender filter
    sender_list = get_sender_list()
    sender_options = [""] + sender_list
    filter_sender: str = st.selectbox(
        "Sender",
        options=sender_options,
        index=0,
        format_func=lambda x: "All senders" if x == "" else x,
        key="filter_sender",
    ) or ""

    # Time range filter
    filter_date_range: str = st.selectbox(
        "Time range",
        options=list(DATE_RANGE_OPTIONS.keys()),
        index=0,
        key="filter_date_range",
    ) or "All time"
    filter_days: Optional[int] = DATE_RANGE_OPTIONS[filter_date_range]

    # Apply filters button
    if st.button("Apply Filters", key="apply_filters", use_container_width=True):
        new_filters = {}
        if filter_chat.strip():
            new_filters["chat_name"] = filter_chat.strip()
        if filter_sender.strip():
            new_filters["sender"] = filter_sender.strip()
        if filter_days is not None:
            new_filters["days"] = str(filter_days)
        st.session_state.active_filters = new_filters
        st.rerun()


# =========================================================================
# CONFIGURATION
# =========================================================================

def _render_config() -> None:
    """Render configuration inputs."""
    api_url: str = st.text_input(
        "API URL",
        value=st.session_state.get("api_url", "http://localhost:8765"),
        key="config_api_url",
    ) or "http://localhost:8765"
    st.session_state.api_url = api_url

    k_results: int = st.slider(
        "Context documents (k)",
        min_value=1,
        max_value=50,
        value=st.session_state.get("k_results", 10),
        key="config_k_results",
    )
    st.session_state.k_results = k_results


# =========================================================================
# STATISTICS
# =========================================================================

def _render_stats() -> None:
    """Render RAG vector store statistics."""
    stats = get_rag_stats()
    if stats:
        st.metric("Total Documents", stats.get("total_documents", 0))
        collection = stats.get("collection_name", "N/A")
        st.caption(f"Collection: {collection}")
        dashboard_url = stats.get("dashboard_url")
        if dashboard_url:
            st.markdown(f"[🔗 Qdrant Dashboard]({dashboard_url})")
    else:
        st.caption("Stats unavailable")


# =========================================================================
# SYSTEM HEALTH
# =========================================================================

def _render_health() -> None:
    """Render system health dashboard."""
    health = check_health()
    deps = health.get("dependencies", {})

    for name, status in deps.items():
        is_ok = "connected" in str(status).lower()
        dot = "🟢" if is_ok else "🔴"
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:6px; "
            f"margin-bottom:4px;'>"
            f"<span>{dot}</span>"
            f"<span style='font-size:0.85rem; color:#ECECEC;'>"
            f"{name.upper()}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if not is_ok:
            st.caption(f"  {status}")

    overall = health.get("status", "unknown")
    if overall == "up":
        st.success("All systems operational", icon="✅")
    elif overall == "degraded":
        st.warning("Some services degraded", icon="⚠️")
    else:
        st.error(f"Status: {overall}", icon="🔴")
