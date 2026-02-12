"""Settings page — full-page configuration management.

Uses a flat list from state to avoid nested rx.foreach.
Each item is either a "category" header or a "setting" row.
"""

import reflex as rx

from ..state import AppState


def settings_page() -> rx.Component:
    """Full settings page."""
    return rx.box(
        rx.flex(
            # Header with back button
            rx.flex(
                rx.link(
                    rx.icon_button(
                        rx.icon("arrow-left", size=18),
                        variant="ghost",
                        class_name="text-gray-500 hover:text-gray-700",
                    ),
                    href="/",
                ),
                rx.heading("Settings", size="6", class_name="text-gray-800"),
                align="center",
                gap="3",
            ),
            rx.text(
                "All configuration is stored in the database and takes effect immediately.",
                class_name="text-gray-400 text-sm mt-1 mb-6",
            ),
            # Status message
            rx.cond(
                AppState.settings_save_message != "",
                rx.box(
                    rx.text(
                        AppState.settings_save_message,
                        class_name="text-sm",
                    ),
                    class_name="mb-4 px-3 py-2 bg-gray-50 rounded-lg border border-gray-200",
                ),
                rx.fragment(),
            ),
            # Health overview
            _health_section(),
            # RAG stats
            _stats_section(),
            # Settings items (flat list)
            rx.foreach(
                AppState.settings_flat,
                _render_settings_item,
            ),
            direction="column",
            class_name="max-w-[820px] mx-auto w-full px-4 py-6",
        ),
        class_name="h-full overflow-y-auto chat-scroll",
    )


# =========================================================================
# SETTINGS ITEM RENDERER (category header or setting)
# =========================================================================

def _render_settings_item(item: dict) -> rx.Component:
    """Render either a category header or an individual setting."""
    return rx.cond(
        item["type"] == "category",
        _category_header(item),
        _setting_input(item),
    )


def _category_header(item: dict) -> rx.Component:
    """Category section header with reset button."""
    return rx.flex(
        rx.heading(
            item["label"],
            size="4",
            class_name="text-gray-700",
        ),
        rx.button(
            rx.icon("rotate-ccw", size=12, class_name="mr-1"),
            "Reset",
            on_click=AppState.reset_category(item["category"]),
            variant="ghost",
            size="1",
            class_name="text-gray-400 hover:text-gray-600",
        ),
        justify="between",
        align="center",
        class_name="mt-6 mb-3 pb-2 border-b border-gray-200",
    )


def _setting_input(item: dict) -> rx.Component:
    """Individual setting with label, description, and input."""
    return rx.box(
        rx.flex(
            rx.text(
                item["label"],
                class_name="text-sm font-medium text-gray-700",
            ),
            rx.cond(
                item["description"] != "",
                rx.tooltip(
                    rx.icon("info", size=14, class_name="text-gray-400 cursor-help"),
                    content=item["description"],
                ),
                rx.fragment(),
            ),
            align="center",
            gap="2",
            class_name="mb-1",
        ),
        # Input — use text input for all types (simplest approach for flat foreach)
        rx.cond(
            item["setting_type"] == "bool",
            # Boolean toggle
            rx.flex(
                rx.el.input(
                    type="checkbox",
                    checked=item["value"] == "true",
                    on_change=AppState.save_setting(
                        item["key"],
                        rx.cond(item["value"] == "true", "false", "true"),
                    ),
                    class_name="w-4 h-4 accent-accent cursor-pointer",
                ),
                rx.text(
                    rx.cond(item["value"] == "true", "Enabled", "Disabled"),
                    class_name="text-sm text-gray-500 ml-2",
                ),
                align="center",
            ),
            # Text / Secret / Number input
            rx.cond(
                item["setting_type"] == "secret",
                rx.el.input(
                    type="password",
                    placeholder="Enter value…",
                    default_value="",
                    class_name=(
                        "w-full bg-white border border-gray-200 rounded-lg "
                        "px-3 py-2 text-sm text-gray-700 "
                        "outline-none focus:border-accent"
                    ),
                ),
                rx.el.input(
                    type="text",
                    default_value=item["value"],
                    class_name=(
                        "w-full bg-white border border-gray-200 rounded-lg "
                        "px-3 py-2 text-sm text-gray-700 "
                        "outline-none focus:border-accent"
                    ),
                ),
            ),
        ),
        class_name="py-2",
    )


# =========================================================================
# HEALTH SECTION
# =========================================================================

def _health_section() -> rx.Component:
    """System health dashboard."""
    return rx.box(
        rx.flex(
            rx.icon("activity", size=18, class_name="text-gray-500"),
            rx.heading("System Health", size="4", class_name="text-gray-700"),
            align="center",
            gap="2",
            class_name="mb-3",
        ),
        rx.box(
            rx.flex(
                rx.box(
                    class_name=rx.cond(
                        AppState.api_status == "up",
                        "w-3 h-3 rounded-full bg-status-green",
                        rx.cond(
                            AppState.api_status == "degraded",
                            "w-3 h-3 rounded-full bg-status-yellow",
                            "w-3 h-3 rounded-full bg-status-red",
                        ),
                    ),
                ),
                rx.text(
                    AppState.health_label,
                    class_name="font-medium text-sm text-gray-700",
                ),
                align="center",
                gap="2",
            ),
            class_name="bg-gray-50 border border-gray-200 rounded-lg px-4 py-3",
        ),
        class_name="mb-6 pb-6 border-b border-gray-100",
    )


# =========================================================================
# STATS SECTION
# =========================================================================

def _stats_section() -> rx.Component:
    """RAG vector store statistics."""
    return rx.box(
        rx.flex(
            rx.icon("bar-chart-3", size=18, class_name="text-gray-500"),
            rx.heading("RAG Statistics", size="4", class_name="text-gray-700"),
            align="center",
            gap="2",
            class_name="mb-3",
        ),
        rx.grid(
            rx.box(
                rx.text(
                    "Total Documents",
                    class_name="text-xs text-gray-400 uppercase tracking-wider",
                ),
                rx.text(
                    "—",
                    class_name="text-2xl font-semibold text-gray-800 mt-1",
                ),
                class_name="bg-gray-50 border border-gray-200 rounded-lg px-4 py-3",
            ),
            columns="2",
            gap="3",
        ),
        class_name="mb-6 pb-6 border-b border-gray-100",
    )
