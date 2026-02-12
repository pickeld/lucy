"""RAG Assistant — ChatGPT-inspired Streamlit UI.

Multi-page application that orchestrates all components:
- Dark sidebar with conversation management
- ChatGPT-style chat area with empty state
- Settings panel (toggleable in sidebar for filters/stats/health)
- Full settings page (all SQLite-backed configuration + plugin settings)

Supports multiple data sources via the plugin architecture
(WhatsApp, Telegram, Email, Paperless-NG, etc.).

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
    page_title="RAG Assistant",
    page_icon="🧠",
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

# Default API URL — can be overridden from the settings page (ui_api_url)
_DEFAULT_API_URL = os.environ.get("UI_API_URL", "http://localhost:8765")

_DEFAULTS = {
    "messages": [],
    "conversation_id": None,
    "active_filters": {},
    "api_url": _DEFAULT_API_URL,
    "k_results": 10,
    "show_settings": False,
    "show_settings_page": False,
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
from components.settings_panel import render_settings_panel, render_settings_page

# Sidebar — conversations, new chat, settings toggle
render_sidebar()

# Settings panel — filters, stats, health (inside sidebar, toggled)
render_settings_panel()

# Main area — either settings page or chat
if st.session_state.get("show_settings_page", False):
    render_settings_page()
else:
    render_chat_area()
