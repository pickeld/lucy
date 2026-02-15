"""Search toolbar — collapsible bar with source, date, content type, and sort filters.

Appears above the chat input when the user clicks the filter/tune icon.
"""

import reflex as rx

from ..state import AppState

# Known content types for the filter (matches Qdrant content_type field values)
CONTENT_TYPES = [
    {"value": "text", "label": "Messages", "icon": "💬"},
    {"value": "document", "label": "Documents", "icon": "📄"},
    {"value": "email", "label": "Emails", "icon": "📧"},
    {"value": "voice", "label": "Voice", "icon": "🎙️"},
    {"value": "image", "label": "Images", "icon": "🖼️"},
]


def search_toolbar() -> rx.Component:
    """Collapsible search toolbar with advanced filter controls."""
    return rx.cond(
        AppState.show_search_toolbar,
        rx.box(
            rx.flex(
                # Source filter section
                _source_filter_section(),
                # Date range section
                _date_range_section(),
                # Content type section
                _content_type_section(),
                # Sort order section
                _sort_order_section(),
                # Clear all button
                rx.cond(
                    AppState.has_advanced_filters,
                    rx.button(
                        rx.icon("x", size=12, class_name="mr-1"),
                        "Clear All",
                        on_click=AppState.clear_advanced_filters,
                        size="1",
                        variant="ghost",
                        class_name="text-gray-400 hover:text-red-500 ml-auto shrink-0",
                    ),
                    rx.fragment(),
                ),
                direction="row",
                align="center",
                gap="4",
                wrap="wrap",
                class_name="px-3 py-2.5",
            ),
            class_name=(
                "border border-gray-200 rounded-xl bg-gray-50/80 mb-2 "
                "shadow-sm animate-in fade-in slide-in-from-bottom-2 duration-150"
            ),
        ),
        rx.fragment(),
    )


# =========================================================================
# SOURCE FILTER
# =========================================================================

def _source_chip(source: dict) -> rx.Component:
    """Toggle chip for a single data source."""
    return rx.box(
        rx.flex(
            rx.text(source["icon"], class_name="text-xs"),
            rx.text(source["label"], class_name="text-xs font-medium"),
            align="center",
            gap="1",
        ),
        on_click=AppState.toggle_source(source["name"]),
        class_name=rx.cond(
            source["active"] == "true",
            (
                "px-2.5 py-1 rounded-full cursor-pointer transition-all duration-150 "
                "bg-accent/15 text-accent border border-accent/30 "
                "hover:bg-accent/25"
            ),
            (
                "px-2.5 py-1 rounded-full cursor-pointer transition-all duration-150 "
                "bg-white text-gray-500 border border-gray-200 "
                "hover:bg-gray-100 hover:border-gray-300"
            ),
        ),
    )


def _source_filter_section() -> rx.Component:
    """Source multi-select filter chips."""
    return rx.flex(
        rx.flex(
            rx.icon("database", size=14, class_name="text-gray-400"),
            rx.text("Sources", class_name="text-xs text-gray-500 font-medium"),
            align="center",
            gap="1",
        ),
        rx.flex(
            rx.foreach(
                AppState.available_sources,
                _source_chip,
            ),
            gap="1.5",
            wrap="wrap",
        ),
        direction="column",
        gap="1.5",
    )


# =========================================================================
# DATE RANGE
# =========================================================================

def _date_range_section() -> rx.Component:
    """Date range picker with from/to inputs."""
    return rx.flex(
        rx.flex(
            rx.icon("calendar", size=14, class_name="text-gray-400"),
            rx.text("Date Range", class_name="text-xs text-gray-500 font-medium"),
            align="center",
            gap="1",
        ),
        rx.flex(
            rx.el.input(
                type="date",
                value=AppState.filter_date_from,
                on_change=AppState.set_filter_date_from,
                class_name=(
                    "text-xs border border-gray-200 rounded-lg px-2 py-1 "
                    "bg-white text-gray-600 w-[130px] "
                    "focus:border-accent focus:outline-none"
                ),
            ),
            rx.text("–", class_name="text-gray-400 text-xs"),
            rx.el.input(
                type="date",
                value=AppState.filter_date_to,
                on_change=AppState.set_filter_date_to,
                class_name=(
                    "text-xs border border-gray-200 rounded-lg px-2 py-1 "
                    "bg-white text-gray-600 w-[130px] "
                    "focus:border-accent focus:outline-none"
                ),
            ),
            align="center",
            gap="1.5",
        ),
        direction="column",
        gap="1.5",
    )


# =========================================================================
# CONTENT TYPE
# =========================================================================

def _content_type_chip(ct: dict) -> rx.Component:
    """Toggle chip for a single content type."""
    return rx.box(
        rx.flex(
            rx.text(ct["icon"], class_name="text-xs"),
            rx.text(ct["label"], class_name="text-xs font-medium"),
            align="center",
            gap="1",
        ),
        on_click=AppState.toggle_content_type(ct["value"]),
        class_name=rx.cond(
            AppState.selected_content_types.contains(ct["value"]),
            (
                "px-2.5 py-1 rounded-full cursor-pointer transition-all duration-150 "
                "bg-blue-50 text-blue-600 border border-blue-200 "
                "hover:bg-blue-100"
            ),
            (
                "px-2.5 py-1 rounded-full cursor-pointer transition-all duration-150 "
                "bg-white text-gray-500 border border-gray-200 "
                "hover:bg-gray-100 hover:border-gray-300"
            ),
        ),
    )


def _content_type_section() -> rx.Component:
    """Content type filter chips (static list)."""
    return rx.flex(
        rx.flex(
            rx.icon("file-text", size=14, class_name="text-gray-400"),
            rx.text("Type", class_name="text-xs text-gray-500 font-medium"),
            align="center",
            gap="1",
        ),
        rx.flex(
            *[
                _static_content_type_chip(ct["value"], ct["label"], ct["icon"])
                for ct in CONTENT_TYPES
            ],
            gap="1.5",
            wrap="wrap",
        ),
        direction="column",
        gap="1.5",
    )


def _static_content_type_chip(value: str, label: str, icon: str) -> rx.Component:
    """A content-type toggle chip built from static data (not rx.foreach)."""
    return rx.box(
        rx.flex(
            rx.text(icon, class_name="text-xs"),
            rx.text(label, class_name="text-xs font-medium"),
            align="center",
            gap="1",
        ),
        on_click=AppState.toggle_content_type(value),
        class_name=rx.cond(
            AppState.selected_content_types.contains(value),
            (
                "px-2.5 py-1 rounded-full cursor-pointer transition-all duration-150 "
                "bg-blue-50 text-blue-600 border border-blue-200 "
                "hover:bg-blue-100"
            ),
            (
                "px-2.5 py-1 rounded-full cursor-pointer transition-all duration-150 "
                "bg-white text-gray-500 border border-gray-200 "
                "hover:bg-gray-100 hover:border-gray-300"
            ),
        ),
    )


# =========================================================================
# SORT ORDER
# =========================================================================

def _sort_order_section() -> rx.Component:
    """Toggle between relevance and newest-first sort."""
    return rx.flex(
        rx.flex(
            rx.icon("arrow-up-down", size=14, class_name="text-gray-400"),
            rx.text("Sort", class_name="text-xs text-gray-500 font-medium"),
            align="center",
            gap="1",
        ),
        rx.flex(
            rx.box(
                rx.text("Relevance", class_name="text-xs font-medium"),
                on_click=AppState.set_sort_order("relevance"),
                class_name=rx.cond(
                    AppState.sort_order == "relevance",
                    (
                        "px-2.5 py-1 rounded-full cursor-pointer transition-all duration-150 "
                        "bg-accent/15 text-accent border border-accent/30"
                    ),
                    (
                        "px-2.5 py-1 rounded-full cursor-pointer transition-all duration-150 "
                        "bg-white text-gray-500 border border-gray-200 hover:bg-gray-100"
                    ),
                ),
            ),
            rx.box(
                rx.text("Newest", class_name="text-xs font-medium"),
                on_click=AppState.set_sort_order("newest"),
                class_name=rx.cond(
                    AppState.sort_order == "newest",
                    (
                        "px-2.5 py-1 rounded-full cursor-pointer transition-all duration-150 "
                        "bg-accent/15 text-accent border border-accent/30"
                    ),
                    (
                        "px-2.5 py-1 rounded-full cursor-pointer transition-all duration-150 "
                        "bg-white text-gray-500 border border-gray-200 hover:bg-gray-100"
                    ),
                ),
            ),
            gap="1.5",
        ),
        direction="column",
        gap="1.5",
    )
