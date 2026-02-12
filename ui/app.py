"""WhatsApp RAG Assistant — ChatGPT-inspired Streamlit UI.

Single-page application that orchestrates all components:
- Dark sidebar with conversation management
- ChatGPT-style chat area with empty state
- Settings panel (toggleable in sidebar)

Usage:
    cd ui && streamlit run app.py
"""

import logging
import os
import sys

import streamlit as st

# ---------------------------------------------------------------------------
# Logging setup (stderr so Streamlit doesn't capture it)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)

log_file = os.environ.get("LOG_FILE")
if log_file:
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

# ---------------------------------------------------------------------------
# Page configuration — MUST be the first Streamlit command
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="WhatsApp RAG Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Inject custom CSS
# ---------------------------------------------------------------------------
from components.styles import inject_styles

inject_styles()

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "messages": [],
    "conversation_id": None,
    "active_filters": {},
    "api_url": "http://localhost:8765",
    "k_results": 10,
    "show_settings": False,
    "renaming_conversation_id": None,
    "sidebar_search": "",
    "menu_open_id": None,
}

for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Render components
# ---------------------------------------------------------------------------
from components.sidebar import render_sidebar
from components.chat import render_chat_area
from components.settings_panel import render_settings_panel

# Sidebar — conversations, new chat, settings toggle
render_sidebar()

# Settings panel — filters, config, health (inside sidebar, toggled)
render_settings_panel()

# Main area — chat messages, empty state, input
render_chat_area()
