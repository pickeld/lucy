"""Cost tracking dashboard component for the Settings page.

Shows a summary of LLM costs: session total, 7-day total, breakdown
by type (chat/embed/whisper/image), and daily history.
"""

import reflex as rx

from ..state import AppState


def cost_dashboard() -> rx.Component:
    """Cost tracking dashboard — displayed as a tab in the settings page."""
    return rx.box(
        # Header
        rx.flex(
            rx.icon("coins", size=20, class_name="text-amber-500"),
            rx.heading("Cost Tracking", size="4", class_name="text-gray-800"),
            align="center",
            gap="2",
            class_name="mb-4",
        ),
        # Summary cards row
        rx.flex(
            _metric_card("Session", rx.cond(AppState.session_cost_display != "", AppState.session_cost_display, "$0.00"), "activity"),
            _metric_card("7-Day Total", AppState.cost_today_display, "calendar"),
            gap="4",
            wrap="wrap",
            class_name="mb-6",
        ),
        # Cost breakdown by kind
        rx.cond(
            AppState.cost_by_kind_list.length() > 0,
            rx.box(
                rx.text(
                    "Cost by Type",
                    class_name="text-sm font-semibold text-gray-600 mb-2",
                ),
                rx.foreach(
                    AppState.cost_by_kind_list,
                    _cost_kind_row,
                ),
                class_name="mb-6",
            ),
            rx.fragment(),
        ),
        # Daily history
        rx.cond(
            AppState.cost_daily_list.length() > 0,
            rx.box(
                rx.text(
                    "Daily History",
                    class_name="text-sm font-semibold text-gray-600 mb-2",
                ),
                rx.box(
                    # Header row
                    rx.flex(
                        rx.text("Date", class_name="text-xs font-medium text-gray-500 w-28"),
                        rx.text("Cost", class_name="text-xs font-medium text-gray-500 w-24 text-right"),
                        rx.text("Events", class_name="text-xs font-medium text-gray-500 w-16 text-right"),
                        class_name="px-3 py-1.5 border-b border-gray-200",
                    ),
                    rx.foreach(
                        AppState.cost_daily_list,
                        _daily_row,
                    ),
                    class_name="bg-white rounded-lg border border-gray-200",
                ),
            ),
            rx.text(
                "No cost data yet. Costs are tracked automatically when you chat.",
                class_name="text-sm text-gray-400 italic py-4",
            ),
        ),
        # Refresh button
        rx.flex(
            rx.button(
                rx.icon("refresh-cw", size=14),
                rx.text("Refresh", class_name="ml-1"),
                on_click=AppState.refresh_cost_data,
                variant="outline",
                size="1",
                class_name="text-xs",
            ),
            justify="end",
            class_name="mt-4",
        ),
        class_name="p-6",
    )


def _metric_card(label: str, value: rx.Var, icon_name: str) -> rx.Component:
    """Small metric card with icon, label, and value."""
    return rx.box(
        rx.flex(
            rx.icon(icon_name, size=16, class_name="text-amber-400"),
            rx.box(
                rx.text(label, class_name="text-[0.7rem] text-gray-500 uppercase tracking-wide"),
                rx.text(value, class_name="text-lg font-semibold text-gray-800 font-mono"),
            ),
            align="center",
            gap="2",
        ),
        class_name=(
            "bg-white border border-gray-200 rounded-lg px-4 py-3 "
            "min-w-[140px] shadow-sm"
        ),
    )


def _cost_kind_row(item: dict) -> rx.Component:
    """Single row in the cost-by-kind breakdown."""
    return rx.flex(
        rx.text(item["label"], class_name="text-sm text-gray-700 flex-1"),
        rx.text(item["cost"], class_name="text-sm text-gray-800 font-mono font-medium"),
        align="center",
        class_name="px-3 py-2 bg-white rounded-lg border border-gray-100 mb-1",
    )


def _daily_row(item: dict) -> rx.Component:
    """Single row in the daily cost history table."""
    return rx.flex(
        rx.text(item["date"], class_name="text-xs text-gray-600 w-28"),
        rx.text(item["cost"], class_name="text-xs text-gray-800 font-mono w-24 text-right"),
        rx.text(item["events"], class_name="text-xs text-gray-500 w-16 text-right"),
        class_name="px-3 py-1.5 border-b border-gray-50 hover:bg-gray-50",
    )
