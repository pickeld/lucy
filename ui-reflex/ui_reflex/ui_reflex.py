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
from .components.entities_page import entities_page
from .components.recordings_page import recordings_page
from .components.insights_page import insights_page


# =========================================================================
# PAGES
# =========================================================================

def index() -> rx.Component:
    """Main chat page — sidebar + chat area."""
    return layout(chat_area())


def settings() -> rx.Component:
    """Settings page — sidebar + settings panel."""
    return layout(settings_page())


def entities() -> rx.Component:
    """Entities page — sidebar + native entity management UI."""
    return layout(entities_page())


def recordings() -> rx.Component:
    """Recordings page — dedicated call recordings management."""
    return layout(recordings_page())


def insights() -> rx.Component:
    """Insights page — scheduled intelligence queries."""
    return layout(insights_page())


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

app.add_page(
    entities,
    route="/entities",
    title="Entities — RAG Assistant",
    on_load=AppState.on_identities_load,
)

app.add_page(
    recordings,
    route="/recordings",
    title="Recordings — RAG Assistant",
    on_load=AppState.on_recordings_load,
)

app.add_page(
    insights,
    route="/insights",
    title="Insights — RAG Assistant",
    on_load=AppState.on_insights_load,
)
