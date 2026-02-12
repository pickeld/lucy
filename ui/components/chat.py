"""Chat area component — ChatGPT-style message display and input handling.

Renders:
- Empty state with suggestion cards (when no conversation is active)
- Chat message history with user/assistant bubbles
- Source citations as collapsible sections
- Active filter chips above the input
- Typing indicator while waiting for responses
- Chat input bar
"""

import ast
import json
from typing import Any, Dict, List, Optional

import streamlit as st

from utils.api import rag_query


# =========================================================================
# SUGGESTION PROMPTS — shown on empty state
# =========================================================================

SUGGESTIONS = [
    "What did everyone talk about this week?",
    "Summarize the latest group conversations",
    "Search for messages about meetings",
    "Who was the most active chatter recently?",
]


# =========================================================================
# MAIN RENDER FUNCTION
# =========================================================================

def render_chat_area() -> None:
    """Render the main chat area — empty state or conversation."""
    messages = st.session_state.get("messages", [])

    if not messages and not st.session_state.get("conversation_id"):
        _render_empty_state()
    else:
        _render_filter_chips()
        _render_messages(messages)

    _handle_chat_input()


# =========================================================================
# EMPTY STATE
# =========================================================================

def _render_empty_state() -> None:
    """Render the centered empty state with logo and suggestion cards."""
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-state-icon">🧠</div>
            <div class="empty-state-title">RAG Assistant</div>
            <div class="empty-state-subtitle">
                Ask anything about your messages and documents
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Suggestion cards as a 2×2 grid using Streamlit columns
    col1, col2 = st.columns(2)
    for i, suggestion in enumerate(SUGGESTIONS):
        col = col1 if i % 2 == 0 else col2
        with col:
            if st.button(
                suggestion,
                key=f"suggestion_{i}",
                use_container_width=True,
            ):
                # Trigger a query with this suggestion
                st.session_state.pending_question = suggestion
                st.rerun()


# =========================================================================
# FILTER CHIPS
# =========================================================================

def _render_filter_chips() -> None:
    """Show active filter chips above the chat input area."""
    filters = st.session_state.get("active_filters", {})
    if not filters:
        return

    chips_html = ""
    if filters.get("chat_name"):
        chips_html += (
            f'<span class="filter-chip">💬 {filters["chat_name"]}</span>'
        )
    if filters.get("sender"):
        chips_html += (
            f'<span class="filter-chip">👤 {filters["sender"]}</span>'
        )
    if filters.get("days"):
        chips_html += (
            f'<span class="filter-chip">📅 Last {filters["days"]}d</span>'
        )

    if chips_html:
        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(chips_html, unsafe_allow_html=True)
        with col2:
            if st.button("✕ Clear", key="clear_filters_chat", help="Clear all filters"):
                st.session_state.active_filters = {}
                st.rerun()


# =========================================================================
# CHAT MESSAGES
# =========================================================================

def _render_messages(messages: List[Dict[str, Any]]) -> None:
    """Render the chat message history."""
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources if available (stored alongside assistant messages)
            sources = message.get("sources")
            if message["role"] == "assistant" and sources:
                _render_sources(sources)


def _render_sources(sources: List[Dict[str, Any]]) -> None:
    """Render source citations as a collapsible expander."""
    if not sources:
        return

    with st.expander(f"📎 Sources ({len(sources)} messages)", expanded=False):
        for i, src in enumerate(sources):
            score = src.get("score")
            score_str = f" — {score:.0%} relevant" if score else ""
            sender = src.get("sender", "Unknown")
            chat = src.get("chat_name", "Unknown")
            content = src.get("content", "")

            st.markdown(
                f"**{i + 1}. {sender}** in _{chat}_{score_str}\n\n"
                f"> {content[:200]}{'…' if len(content) > 200 else ''}"
            )
            if i < len(sources) - 1:
                st.divider()


# =========================================================================
# CHAT INPUT HANDLING
# =========================================================================

def _handle_chat_input() -> None:
    """Handle the chat input bar and send queries."""
    # Check for a pending question from suggestion cards
    pending = st.session_state.pop("pending_question", None)

    question = st.chat_input("Ask about your messages and documents…")

    # Use pending question if available, otherwise use typed input
    active_question = pending or question

    if active_question:
        _process_question(active_question)


def _process_question(question: str) -> None:
    """Process a user question: display it, query the API, display the answer."""
    # Add user message to state and display it
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Query the RAG API
    with st.chat_message("assistant"):
        # Typing indicator
        with st.spinner(""):
            # Build query params from session state
            filters = st.session_state.get("active_filters", {})
            k_results = st.session_state.get("k_results", 10)

            data = rag_query(
                question=question,
                conversation_id=st.session_state.get("conversation_id"),
                k=k_results,
                filter_chat_name=filters.get("chat_name"),
                filter_sender=filters.get("sender"),
                filter_days=int(filters["days"]) if filters.get("days") else None,
            )

        if "error" in data:
            answer = f"❌ {data['error']}"
            st.error(answer)
            sources = []
        else:
            raw_answer = data.get("answer", "No answer received")

            # Update conversation state from response
            if data.get("conversation_id"):
                st.session_state.conversation_id = data["conversation_id"]
            if data.get("filters"):
                st.session_state.active_filters = data["filters"]

            # Parse answer (handle JSON-wrapped responses)
            answer = _parse_answer(raw_answer)
            st.markdown(answer)

            # Show sources
            sources = data.get("sources", [])
            if sources:
                _render_sources(sources)

    # Store assistant message (with sources for later re-rendering)
    msg: Dict[str, Any] = {"role": "assistant", "content": answer}
    if sources:
        msg["sources"] = sources
    st.session_state.messages.append(msg)


def _parse_answer(raw_answer: Any) -> str:
    """Parse potentially JSON-wrapped answer into a plain string."""
    if isinstance(raw_answer, str):
        stripped = raw_answer.strip()
        if (stripped.startswith("{") and stripped.endswith("}")) or (
            stripped.startswith("[") and stripped.endswith("]")
        ):
            try:
                raw_answer = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                try:
                    raw_answer = json.loads(stripped)
                except json.JSONDecodeError:
                    pass

    if isinstance(raw_answer, dict):
        return raw_answer.get("text", str(raw_answer))
    elif isinstance(raw_answer, list):
        texts = []
        for item in raw_answer:
            if isinstance(item, dict) and "text" in item:
                texts.append(item["text"])
            else:
                texts.append(str(item))
        return "\n".join(texts)
    return str(raw_answer)
