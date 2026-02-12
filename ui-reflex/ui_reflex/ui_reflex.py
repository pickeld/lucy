"""RAG Assistant — Reflex UI.

ChatGPT-inspired interface with dark sidebar and light chat area.
Connects to the Flask backend API at localhost:8765.

Usage:
    cd ui-reflex && reflex run
"""

import reflex as rx

from .state import AppState
from .components.layout import layout
from .components.chat_area import chat_area
from .components.settings_page import settings_page


# =========================================================================
# PAGES
# =========================================================================

def index() -> rx.Component:
    """Main chat page — sidebar + chat area."""
    return layout(chat_area())


def settings() -> rx.Component:
    """Settings page — sidebar + settings panel."""
    return layout(settings_page())


# =========================================================================
# APP
# =========================================================================

app = rx.App(
    theme=rx.theme(
        appearance="light",
        accent_color="green",
        radius="medium",
    ),
    stylesheets=["/styles.css"],
)

app.add_page(
    index,
    route="/",
    title="RAG Assistant",
    on_load=AppState.on_load,
)

app.add_page(
    settings,
    route="/settings",
    title="Settings — RAG Assistant",
    on_load=AppState.on_settings_load,
)
