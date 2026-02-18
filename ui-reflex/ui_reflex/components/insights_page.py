"""Scheduled Insights page — manage and view proactive RAG queries.

Provides:
- Task list with status indicators and actions
- Create/Edit dialog with template support
- Result history viewer with expandable entries
"""

import reflex as rx

from ..state import AppState


# =========================================================================
# SCHEDULE TYPE DISPLAY HELPERS
# =========================================================================

_SCHEDULE_LABELS = {
    "daily": "Daily",
    "weekly": "Weekly",
    "monthly": "Monthly",
    "interval": "Interval",
    "cron": "Cron",
}


# =========================================================================
# MAIN PAGE
# =========================================================================

def insights_page() -> rx.Component:
    """Full insights page layout."""
    return rx.box(
        # Header
        rx.flex(
            rx.flex(
                rx.icon("sparkles", size=24, class_name="text-amber-500"),
                rx.heading("Scheduled Insights", size="5", class_name="ml-2"),
                align="center",
            ),
            rx.flex(
                rx.button(
                    rx.icon("refresh-cw", size=14),
                    rx.text("Refresh", class_name="ml-1"),
                    on_click=AppState.refresh_insights,
                    variant="outline",
                    size="2",
                    class_name="cursor-pointer",
                ),
                rx.button(
                    rx.icon("plus", size=14),
                    rx.text("New Insight", class_name="ml-1"),
                    on_click=AppState.open_insights_create_dialog,
                    size="2",
                    class_name="cursor-pointer bg-amber-500 hover:bg-amber-600 text-white",
                ),
                gap="2",
            ),
            justify="between",
            align="center",
            class_name="px-6 py-4 border-b border-gray-200",
        ),
        # Status message
        rx.cond(
            AppState.insights_message != "",
            rx.box(
                rx.text(
                    AppState.insights_message,
                    class_name="text-sm py-2 px-4",
                ),
                class_name="bg-gray-50 border-b border-gray-200",
            ),
            rx.fragment(),
        ),
        # Content
        rx.cond(
            AppState.insights_loading,
            rx.flex(
                rx.spinner(size="3"),
                rx.text("Loading insights…", class_name="ml-2 text-gray-500"),
                align="center",
                justify="center",
                class_name="py-16",
            ),
            rx.cond(
                AppState.insights_tasks.length() > 0,
                _task_list_and_results(),
                _empty_state(),
            ),
        ),
        # Dialog
        _create_edit_dialog(),
        class_name="flex flex-col h-full overflow-auto bg-white",
    )


# =========================================================================
# EMPTY STATE
# =========================================================================

def _empty_state() -> rx.Component:
    """Shown when no tasks exist — encourage creating the first one."""
    return rx.flex(
        rx.icon("sparkles", size=48, class_name="text-amber-300 mb-4"),
        rx.heading("No Scheduled Insights Yet", size="4", class_name="text-gray-700 mb-2"),
        rx.text(
            "Create your first insight to have the system proactively "
            "query your knowledge base on a schedule.",
            class_name="text-gray-500 text-center max-w-md mb-6",
        ),
        rx.button(
            rx.icon("plus", size=16),
            rx.text("Create Your First Insight", class_name="ml-2"),
            on_click=AppState.open_insights_create_dialog,
            size="3",
            class_name="cursor-pointer bg-amber-500 hover:bg-amber-600 text-white",
        ),
        direction="column",
        align="center",
        justify="center",
        class_name="py-20",
    )


# =========================================================================
# TASK LIST + RESULTS PANEL
# =========================================================================

def _task_list_and_results() -> rx.Component:
    """Task cards + result viewer layout."""
    return rx.box(
        # Task cards
        rx.box(
            rx.foreach(
                AppState.insights_tasks,
                _render_task_card,
            ),
            class_name="p-4 space-y-3",
        ),
        # Result viewer (shown when a task is selected)
        rx.cond(
            AppState.insights_viewing_task_id > 0,
            _result_viewer(),
            rx.fragment(),
        ),
    )


def _render_task_card(task: dict) -> rx.Component:
    """Render a single insight task card."""
    task_id = task["id"]
    is_enabled = task["enabled"].to(str).lower().contains("true")
    is_viewing = AppState.insights_viewing_task_id == task["id"].to(int)

    return rx.box(
        rx.flex(
            # Left: status dot + name
            rx.flex(
                rx.box(
                    class_name=rx.cond(
                        is_enabled,
                        "w-2.5 h-2.5 rounded-full bg-green-500 mt-1.5",
                        "w-2.5 h-2.5 rounded-full bg-gray-300 mt-1.5",
                    ),
                ),
                rx.box(
                    rx.flex(
                        rx.text(
                            task["name"],
                            class_name="font-medium text-gray-900",
                        ),
                        rx.badge(
                            task["schedule_type"],
                            variant="outline",
                            size="1",
                            class_name="ml-2",
                        ),
                        align="center",
                    ),
                    rx.text(
                        task["schedule_type"], ": ", task["schedule_value"],
                        class_name="text-xs text-gray-500 mt-0.5",
                    ),
                    rx.cond(
                        task["last_run_at"] != "",
                        rx.text(
                            "Last run: ", task["last_run_at"],
                            class_name="text-xs text-gray-400 mt-0.5",
                        ),
                        rx.text(
                            "Never run",
                            class_name="text-xs text-gray-400 italic mt-0.5",
                        ),
                    ),
                    class_name="ml-3 flex-1 min-w-0",
                ),
                align="start",
                class_name="flex-1 min-w-0",
            ),
            # Right: action buttons
            rx.flex(
                rx.tooltip(
                    rx.icon_button(
                        rx.icon("eye", size=14),
                        on_click=AppState.view_insight_results(task_id),
                        variant=rx.cond(is_viewing, "solid", "outline"),
                        size="1",
                        class_name="cursor-pointer",
                    ),
                    content="View Results",
                ),
                rx.tooltip(
                    rx.icon_button(
                        rx.icon("play", size=14),
                        on_click=AppState.run_insight_task_now(task_id),
                        variant="outline",
                        size="1",
                        class_name="cursor-pointer",
                    ),
                    content="Run Now",
                ),
                rx.tooltip(
                    rx.icon_button(
                        rx.icon(
                            rx.cond(is_enabled, "pause", "play"),
                            size=14,
                        ),
                        on_click=AppState.toggle_insight_task(task_id),
                        variant="outline",
                        size="1",
                        class_name="cursor-pointer",
                    ),
                    content=rx.cond(is_enabled, "Disable", "Enable"),
                ),
                rx.tooltip(
                    rx.icon_button(
                        rx.icon("pencil", size=14),
                        on_click=AppState.open_insights_edit_dialog(task_id),
                        variant="outline",
                        size="1",
                        class_name="cursor-pointer",
                    ),
                    content="Edit",
                ),
                rx.tooltip(
                    rx.icon_button(
                        rx.icon("trash-2", size=14),
                        on_click=AppState.delete_insight_task(task_id),
                        variant="outline",
                        size="1",
                        color_scheme="red",
                        class_name="cursor-pointer",
                    ),
                    content="Delete",
                ),
                gap="1",
                align="center",
            ),
            justify="between",
            align="start",
        ),
        # Preview of latest result
        rx.cond(
            (task["latest_result"].to(str) != "") & (task["latest_result"].to(str) != "None"),
            rx.box(
                rx.text(
                    "Latest: ",
                    rx.text.span(
                        task["latest_result"].to(str)[:150],
                        class_name="text-gray-600",
                    ),
                    class_name="text-xs text-gray-400 truncate",
                ),
                class_name="mt-2 pl-5 border-t border-gray-100 pt-2",
            ),
            rx.fragment(),
        ),
        class_name=rx.cond(
            is_viewing,
            "p-4 rounded-lg border-2 border-amber-300 bg-amber-50 transition-colors",
            "p-4 rounded-lg border border-gray-200 hover:border-gray-300 bg-white transition-colors",
        ),
    )


# =========================================================================
# RESULT VIEWER
# =========================================================================

def _result_viewer() -> rx.Component:
    """Result history panel for the currently viewed task."""
    return rx.box(
        rx.flex(
            rx.flex(
                rx.icon("history", size=18, class_name="text-amber-500"),
                rx.heading(
                    "Results",
                    size="3",
                    class_name="ml-2",
                ),
                align="center",
            ),
            rx.icon_button(
                rx.icon("x", size=16),
                on_click=AppState.view_insight_results("0"),
                variant="ghost",
                size="1",
                class_name="cursor-pointer",
            ),
            justify="between",
            align="center",
            class_name="px-4 py-3 border-b border-gray-200",
        ),
        rx.cond(
            AppState.insights_results_loading,
            rx.flex(
                rx.spinner(size="2"),
                rx.text("Loading results…", class_name="ml-2 text-gray-500 text-sm"),
                align="center",
                class_name="p-4",
            ),
            rx.cond(
                AppState.insights_results.length() > 0,
                rx.box(
                    rx.foreach(
                        AppState.insights_results,
                        _render_result_item,
                    ),
                    class_name="divide-y divide-gray-100",
                ),
                rx.flex(
                    rx.text(
                        "No results yet. Click 'Run Now' to execute this insight.",
                        class_name="text-gray-500 text-sm",
                    ),
                    justify="center",
                    class_name="py-8",
                ),
            ),
        ),
        class_name="mx-4 mb-4 rounded-lg border border-gray-200 bg-white",
    )


def _render_result_item(result: dict) -> rx.Component:
    """Render a single result entry — always shows the full answer."""
    return rx.box(
        # Header with status, time, cost
        rx.flex(
            rx.flex(
                rx.cond(
                    result["status"] == "success",
                    rx.icon("circle-check", size=14, class_name="text-green-600"),
                    rx.cond(
                        result["status"] == "error",
                        rx.icon("circle-x", size=14, class_name="text-red-600"),
                        rx.icon("circle-minus", size=14, class_name="text-gray-500"),
                    ),
                ),
                rx.text(
                    result["executed_at"],
                    class_name="text-sm font-medium text-gray-700 ml-2",
                ),
                rx.cond(
                    (result["cost_usd"] != "0") & (result["cost_usd"] != "0.0"),
                    rx.badge(
                        "$", result["cost_usd"],
                        variant="outline",
                        size="1",
                        class_name="ml-2",
                    ),
                    rx.fragment(),
                ),
                align="center",
            ),
            rx.flex(
                rx.text(
                    "Duration: ", result["duration_ms"], "ms",
                    class_name="text-xs text-gray-400",
                ),
                align="center",
            ),
            justify="between",
            align="center",
            class_name="px-4 py-3",
        ),
        # Answer content — always visible
        rx.box(
            rx.markdown(
                result["answer"],
                class_name="prose prose-sm max-w-none text-gray-700",
            ),
            class_name="px-4 pb-3",
        ),
        # Error message if present
        rx.cond(
            (result["error_message"] != "") & (result["error_message"] != "None"),
            rx.box(
                rx.text(
                    "Error: ", result["error_message"],
                    class_name="text-sm text-red-600",
                ),
                class_name="px-4 pb-3",
            ),
            rx.fragment(),
        ),
    )


# =========================================================================
# CREATE / EDIT DIALOG
# =========================================================================

def _create_edit_dialog() -> rx.Component:
    """Dialog for creating or editing a scheduled insight task."""
    return rx.dialog.root(
        rx.dialog.content(
            rx.dialog.title(
                rx.cond(
                    AppState.insights_editing_id > 0,
                    "Edit Scheduled Insight",
                    "New Scheduled Insight",
                ),
            ),
            rx.dialog.description(
                "Define a prompt that will automatically query your knowledge base on a schedule.",
                class_name="text-sm text-gray-500 mb-4",
            ),
            # Template buttons (only for new tasks)
            rx.cond(
                AppState.insights_editing_id == 0,
                rx.box(
                    rx.text("Quick Templates:", class_name="text-xs font-medium text-gray-500 mb-2"),
                    rx.flex(
                        rx.foreach(
                            AppState.insights_templates,
                            _render_template_button,
                        ),
                        gap="2",
                        wrap="wrap",
                        class_name="mb-4",
                    ),
                ),
                rx.fragment(),
            ),
            # Form fields
            rx.flex(
                # Name
                rx.box(
                    rx.text("Name", class_name="text-sm font-medium text-gray-700 mb-1"),
                    rx.el.input(
                        value=AppState.insights_form_name,
                        on_change=AppState.set_insights_form_name,
                        placeholder="Daily Briefing",
                        class_name=(
                            "w-full border border-gray-300 rounded-lg px-3 py-2 text-sm "
                            "focus:outline-none focus:border-amber-500"
                        ),
                    ),
                    class_name="mb-3",
                ),
                # Description
                rx.box(
                    rx.text("Description", class_name="text-sm font-medium text-gray-700 mb-1"),
                    rx.el.input(
                        value=AppState.insights_form_description,
                        on_change=AppState.set_insights_form_description,
                        placeholder="Morning overview of your day",
                        class_name=(
                            "w-full border border-gray-300 rounded-lg px-3 py-2 text-sm "
                            "focus:outline-none focus:border-amber-500"
                        ),
                    ),
                    class_name="mb-3",
                ),
                # Prompt
                rx.box(
                    rx.text("Prompt", class_name="text-sm font-medium text-gray-700 mb-1"),
                    rx.el.textarea(
                        value=AppState.insights_form_prompt,
                        on_change=AppState.set_insights_form_prompt,
                        placeholder="What should I know about today? Check for meetings, commitments, deadlines…",
                        rows=4,
                        class_name=(
                            "w-full border border-gray-300 rounded-lg px-3 py-2 text-sm "
                            "focus:outline-none focus:border-amber-500 resize-y"
                        ),
                    ),
                    class_name="mb-3",
                ),
                # Schedule row
                rx.flex(
                    rx.box(
                        rx.text("Schedule", class_name="text-sm font-medium text-gray-700 mb-1"),
                        rx.select(
                            ["daily", "weekly", "monthly", "interval", "cron"],
                            value=AppState.insights_form_schedule_type,
                            on_change=AppState.set_insights_form_schedule_type,
                            size="2",
                        ),
                        class_name="flex-1",
                    ),
                    rx.box(
                        rx.text("Value", class_name="text-sm font-medium text-gray-700 mb-1"),
                        rx.el.input(
                            value=AppState.insights_form_schedule_value,
                            on_change=AppState.set_insights_form_schedule_value,
                            placeholder="08:00",
                            class_name=(
                                "w-full border border-gray-300 rounded-lg px-3 py-2 text-sm "
                                "focus:outline-none focus:border-amber-500"
                            ),
                        ),
                        class_name="flex-1",
                    ),
                    gap="3",
                    class_name="mb-3",
                ),
                # Filters
                rx.box(
                    rx.text("Filters (optional)", class_name="text-sm font-medium text-gray-500 mb-2"),
                    rx.flex(
                        rx.box(
                            rx.text("Days back", class_name="text-xs text-gray-500 mb-1"),
                            rx.el.input(
                                value=AppState.insights_form_filter_days,
                                on_change=AppState.set_insights_form_filter_days,
                                placeholder="30",
                                type="number",
                                class_name=(
                                    "w-full border border-gray-200 rounded px-2 py-1.5 text-sm "
                                    "focus:outline-none focus:border-amber-400"
                                ),
                            ),
                        ),
                        rx.box(
                            rx.text("Chat", class_name="text-xs text-gray-500 mb-1"),
                            rx.el.input(
                                value=AppState.insights_form_filter_chat_name,
                                on_change=AppState.set_insights_form_filter_chat_name,
                                placeholder="Any",
                                class_name=(
                                    "w-full border border-gray-200 rounded px-2 py-1.5 text-sm "
                                    "focus:outline-none focus:border-amber-400"
                                ),
                            ),
                        ),
                        rx.box(
                            rx.text("Sender", class_name="text-xs text-gray-500 mb-1"),
                            rx.el.input(
                                value=AppState.insights_form_filter_sender,
                                on_change=AppState.set_insights_form_filter_sender,
                                placeholder="Any",
                                class_name=(
                                    "w-full border border-gray-200 rounded px-2 py-1.5 text-sm "
                                    "focus:outline-none focus:border-amber-400"
                                ),
                            ),
                        ),
                        gap="2",
                    ),
                    class_name="mb-4 p-3 bg-gray-50 rounded-lg",
                ),
                direction="column",
            ),
            # Footer
            rx.flex(
                rx.dialog.close(
                    rx.button(
                        "Cancel",
                        variant="outline",
                        on_click=AppState.close_insights_dialog,
                        class_name="cursor-pointer",
                    ),
                ),
                rx.button(
                    rx.cond(
                        AppState.insights_editing_id > 0,
                        "Update",
                        "Create & Enable",
                    ),
                    on_click=AppState.save_insight_task,
                    class_name="cursor-pointer bg-amber-500 hover:bg-amber-600 text-white",
                ),
                gap="2",
                justify="end",
                class_name="mt-4",
            ),
            max_width="560px",
        ),
        open=AppState.insights_dialog_open,
    )


def _render_template_button(template: dict) -> rx.Component:
    """Render a template quick-select button."""
    return rx.button(
        rx.text(
            template["icon"], " ", template["name"],
            class_name="text-xs",
        ),
        on_click=AppState.apply_insight_template(template["name"].to(str)),
        variant="outline",
        size="1",
        class_name="cursor-pointer",
    )
