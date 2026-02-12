"""Sidebar component — dark-themed, ChatGPT-style conversation list.

Renders:
- New Chat button
- Search / filter bar for conversations
- Time-grouped conversation list (Today, Yesterday, …)
- Three-dot (⋯) button per conversation that toggles inline rename/delete
- Settings toggle & health indicator at the bottom
"""

import streamlit as st

from utils.api import (
    check_health,
    delete_conversation,
    fetch_conversation,
    fetch_conversations,
    rename_conversation,
)
from utils.time_utils import group_conversations_by_time


def render_sidebar() -> None:
    """Render the full sidebar contents."""
    with st.sidebar:
        _render_new_chat_button()
        _render_conversation_list()
        st.markdown("---")
        _render_bottom_section()


# =========================================================================
# NEW CHAT BUTTON
# =========================================================================

def _render_new_chat_button() -> None:
    """Render the '+ New Chat' button at the top of the sidebar."""
    if st.button(
        "＋  New Chat",
        key="new_chat_btn",
        use_container_width=True,
        type="primary",
    ):
        st.session_state.conversation_id = None
        st.session_state.messages = []
        st.session_state.active_filters = {}
        st.session_state.renaming_conversation_id = None
        st.session_state.menu_open_id = None
        st.rerun()


# =========================================================================
# CONVERSATION LIST
# =========================================================================

def _render_conversation_list() -> None:
    """Fetch and render conversations grouped by time period."""
    conversations = fetch_conversations(limit=50)

    if not conversations:
        st.caption("No previous chats")
        return

    # Optional sidebar search
    search_term: str = st.text_input(
        "Search chats",
        value=st.session_state.get("sidebar_search", ""),
        placeholder="🔍 Search…",
        key="sidebar_search_input",
        label_visibility="collapsed",
    ) or ""
    st.session_state.sidebar_search = search_term

    # Filter conversations by search term
    if search_term.strip():
        needle = search_term.strip().lower()
        conversations = [
            c for c in conversations
            if needle in (c.get("title") or "").lower()
        ]

    # Group by time period
    groups = group_conversations_by_time(conversations)

    for group_label, convos in groups.items():
        # Time group header
        st.markdown(
            f'<p style="color:#888; font-size:0.72rem; text-transform:uppercase; '
            f'letter-spacing:0.05em; margin:14px 0 6px 8px; font-weight:600;">'
            f'{group_label}</p>',
            unsafe_allow_html=True,
        )

        for convo in convos:
            _render_conversation_item(convo)


def _render_conversation_item(convo: dict) -> None:
    """Render a single conversation — title + ⋯ toggle for actions."""
    convo_id = convo["id"]
    convo_title = convo.get("title") or "Untitled"
    is_active = convo_id == st.session_state.get("conversation_id")
    menu_open = st.session_state.get("menu_open_id") == convo_id

    # ── RENAME MODE ──────────────────────────────────────────────
    if st.session_state.get("renaming_conversation_id") == convo_id:
        new_title: str = st.text_input(
            "Rename",
            value=convo_title,
            key=f"rename_input_{convo_id}",
            label_visibility="collapsed",
        ) or convo_title
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Save", key=f"rename_save_{convo_id}", use_container_width=True):
                if new_title.strip():
                    rename_conversation(convo_id, new_title.strip())
                st.session_state.renaming_conversation_id = None
                st.session_state.menu_open_id = None
                st.rerun()
        with c2:
            if st.button("Cancel", key=f"rename_cancel_{convo_id}", use_container_width=True):
                st.session_state.renaming_conversation_id = None
                st.session_state.menu_open_id = None
                st.rerun()
        return

    # ── NORMAL DISPLAY: [title] [⋯] ─────────────────────────────
    display_title = convo_title[:35] + ("…" if len(convo_title) > 35 else "")

    col_title, col_dots = st.columns([9, 1])

    with col_title:
        if st.button(
            display_title,
            key=f"load_{convo_id}",
            use_container_width=True,
            disabled=is_active,
        ):
            _load_conversation(convo_id)

    with col_dots:
        if st.button("⋮", key=f"dots_{convo_id}"):
            # Toggle menu: if already open, close it; otherwise open it
            if menu_open:
                st.session_state.menu_open_id = None
            else:
                st.session_state.menu_open_id = convo_id
            st.rerun()

    # ── INLINE ACTION MENU (shown when ⋯ is toggled) ────────────
    if menu_open:
        col_rename, col_delete = st.columns(2)
        with col_rename:
            if st.button("✏️ Rename", key=f"rename_{convo_id}", use_container_width=True):
                st.session_state.renaming_conversation_id = convo_id
                st.session_state.menu_open_id = None
                st.rerun()
        with col_delete:
            if st.button("🗑️ Delete", key=f"del_{convo_id}", use_container_width=True):
                delete_conversation(convo_id)
                st.session_state.menu_open_id = None
                if convo_id == st.session_state.get("conversation_id"):
                    st.session_state.conversation_id = None
                    st.session_state.messages = []
                    st.session_state.active_filters = {}
                st.rerun()


def _load_conversation(convo_id: str) -> None:
    """Load a conversation from the API into session state."""
    loaded = fetch_conversation(convo_id)
    if loaded:
        st.session_state.conversation_id = convo_id
        st.session_state.messages = [
            {"role": m["role"], "content": m["content"]}
            for m in loaded.get("messages", [])
        ]
        st.session_state.active_filters = loaded.get("filters", {})
        st.session_state.renaming_conversation_id = None
        st.session_state.menu_open_id = None
        st.rerun()


# =========================================================================
# BOTTOM SECTION — HEALTH & SETTINGS TOGGLE
# =========================================================================

def _render_bottom_section() -> None:
    """Render health status and settings toggle at the bottom of the sidebar."""
    # Health indicator
    health = check_health()
    api_status = health.get("status", "unreachable")

    if api_status == "up":
        dot_class = "status-dot-green"
        label = "API Connected"
    elif api_status == "degraded":
        dot_class = "status-dot-yellow"
        label = "API Degraded"
    else:
        dot_class = "status-dot-red"
        label = "API Unreachable"

    st.markdown(
        f'<div style="display:flex; align-items:center; padding:4px 0;">'
        f'<span class="status-dot {dot_class}"></span>'
        f'<span style="font-size:0.8rem; color:#ECECEC;">{label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Settings toggle button
    if st.button("⚙️  Settings & Filters", key="toggle_settings", use_container_width=True):
        st.session_state.show_settings = not st.session_state.get("show_settings", False)
        st.rerun()
