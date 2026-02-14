"""Dark sidebar component — ChatGPT-style conversation list.

Uses a flat list of items (headers + conversations) from state
to avoid nested rx.foreach — Reflex requires typed vars for iteration.
"""

import reflex as rx

from ..state import AppState


def sidebar() -> rx.Component:
    """Full sidebar: fixed 280px, dark background."""
    return rx.box(
        rx.flex(
            _new_chat_button(),
            _search_bar(),
            # Conversation list (scrollable)
            rx.box(
                rx.foreach(
                    AppState.sidebar_items,
                    _render_sidebar_item,
                ),
                rx.cond(
                    ~AppState.has_conversations,
                    rx.text(
                        "No previous chats",
                        class_name="text-sidebar-muted text-sm text-center py-8",
                    ),
                    rx.fragment(),
                ),
                class_name="flex-1 overflow-y-auto sidebar-scroll px-2",
            ),
            _bottom_section(),
            direction="column",
            class_name="h-full",
        ),
        class_name="w-[280px] min-w-[280px] h-screen bg-sidebar border-r border-sidebar-border flex flex-col",
    )


# =========================================================================
# NEW CHAT BUTTON
# =========================================================================

def _new_chat_button() -> rx.Component:
    return rx.box(
        rx.button(
            rx.icon("plus", size=16),
            rx.text("New Chat", class_name="ml-2"),
            on_click=AppState.new_chat,
            variant="outline",
            class_name=(
                "w-full justify-center border border-dashed border-sidebar-border "
                "text-sidebar-text bg-transparent hover:bg-sidebar-hover "
                "hover:border-solid rounded-lg py-2.5 cursor-pointer "
                "transition-colors duration-150"
            ),
        ),
        class_name="px-3 pt-3 pb-2",
    )


# =========================================================================
# SEARCH BAR
# =========================================================================

def _search_bar() -> rx.Component:
    return rx.box(
        rx.flex(
            rx.icon("search", size=14, class_name="text-sidebar-muted"),
            rx.el.input(
                placeholder="Search…",
                value=AppState.sidebar_search,
                on_change=AppState.set_sidebar_search,
                class_name=(
                    "bg-transparent border-none outline-none text-sidebar-text "
                    "text-sm placeholder-sidebar-muted flex-1 ml-2"
                ),
            ),
            align="center",
            class_name=(
                "bg-sidebar-hover border border-sidebar-border rounded-lg "
                "px-3 py-2"
            ),
        ),
        class_name="px-3 pb-2",
    )


# =========================================================================
# SIDEBAR ITEM RENDERER (header or conversation)
# =========================================================================

def _render_sidebar_item(item: dict) -> rx.Component:
    """Render either a time-group header or a conversation item."""
    return rx.cond(
        item["type"] == "header",
        _render_header(item),
        _render_conversation(item),
    )


def _render_header(item: dict) -> rx.Component:
    """Time group header (Today, Yesterday, etc.)."""
    return rx.text(
        item["label"],
        class_name=(
            "text-sidebar-muted text-[0.7rem] uppercase tracking-wider "
            "font-semibold px-2 pt-4 pb-1"
        ),
    )


def _render_conversation(item: dict) -> rx.Component:
    """Single conversation item with hover-to-reveal actions."""
    convo_id = item["id"]
    convo_title = item["title"]
    is_active = AppState.conversation_id == convo_id
    is_renaming = AppState.renaming_id == convo_id

    return rx.cond(
        is_renaming,
        _rename_mode(),
        _normal_mode(convo_id, convo_title, is_active),
    )


def _normal_mode(convo_id: rx.Var, convo_title: rx.Var, is_active: rx.Var) -> rx.Component:
    """Normal conversation display with hover-to-reveal actions."""
    return rx.flex(
        rx.box(
            rx.text(
                convo_title,
                class_name="text-sm text-sidebar-text truncate rtl-auto",
            ),
            on_click=AppState.load_conversation(convo_id),
            class_name="flex-1 cursor-pointer overflow-hidden",
        ),
        rx.box(
            rx.dropdown_menu.root(
                rx.dropdown_menu.trigger(
                    rx.icon_button(
                        rx.icon("ellipsis-vertical", size=14),
                        variant="ghost",
                        size="1",
                        class_name=(
                            "text-sidebar-muted hover:text-sidebar-text "
                            "hover:bg-sidebar-active cursor-pointer "
                            "!bg-transparent !p-0.5"
                        ),
                    ),
                ),
                rx.dropdown_menu.content(
                    rx.dropdown_menu.item(
                        rx.icon("pencil", size=14, class_name="mr-2"),
                        "Rename",
                        on_click=AppState.start_rename(convo_id),
                    ),
                    rx.dropdown_menu.item(
                        rx.icon("download", size=14, class_name="mr-2"),
                        "Export",
                        on_click=AppState.export_chat(convo_id),
                    ),
                    rx.dropdown_menu.separator(),
                    rx.dropdown_menu.item(
                        rx.icon("trash-2", size=14, class_name="mr-2"),
                        "Delete",
                        color="red",
                        on_click=AppState.delete_conversation(convo_id),
                    ),
                    size="1",
                ),
            ),
            class_name="conv-actions",
        ),
        align="center",
        gap="1",
        class_name=rx.cond(
            is_active,
            (
                "conv-item px-2 py-2 rounded-lg bg-sidebar-active "
                "border-l-3 border-accent cursor-default"
            ),
            (
                "conv-item px-2 py-2 rounded-lg hover:bg-sidebar-hover "
                "transition-colors duration-150 cursor-pointer"
            ),
        ),
    )


def _rename_mode() -> rx.Component:
    """Inline rename input with save/cancel buttons."""
    return rx.box(
        rx.el.input(
            value=AppState.rename_text,
            on_change=AppState.set_rename_text,
            auto_focus=True,
            class_name=(
                "w-full bg-sidebar-hover border border-sidebar-border "
                "rounded-lg px-2 py-1.5 text-sm text-sidebar-text "
                "outline-none focus:border-accent mb-1"
            ),
        ),
        rx.flex(
            rx.button(
                "Save",
                on_click=AppState.save_rename,
                size="1",
                class_name="flex-1 bg-accent text-white hover:bg-accent-hover",
            ),
            rx.button(
                "Cancel",
                on_click=AppState.cancel_rename,
                size="1",
                variant="outline",
                class_name="flex-1 text-sidebar-text border-sidebar-border",
            ),
            gap="2",
        ),
        class_name="px-2 py-2",
    )


# =========================================================================
# BOTTOM SECTION
# =========================================================================

def _bottom_section() -> rx.Component:
    """Health indicator + session cost + settings link at the bottom."""
    return rx.box(
        rx.separator(class_name="border-sidebar-border mb-2"),
        rx.flex(
            rx.box(
                class_name=rx.cond(
                    AppState.api_status == "up",
                    "w-2 h-2 rounded-full bg-status-green",
                    rx.cond(
                        AppState.api_status == "degraded",
                        "w-2 h-2 rounded-full bg-status-yellow",
                        "w-2 h-2 rounded-full bg-status-red",
                    ),
                ),
            ),
            rx.text(
                AppState.health_label,
                class_name="text-sm text-sidebar-text",
            ),
            align="center",
            gap="2",
            class_name="px-3 py-1",
        ),
        # Session cost indicator
        rx.cond(
            AppState.session_cost_display != "",
            rx.flex(
                rx.icon("coins", size=14, class_name="text-amber-400"),
                rx.text(
                    "Session: ",
                    class_name="text-xs text-sidebar-muted",
                ),
                rx.text(
                    AppState.session_cost_display,
                    class_name="text-xs text-amber-400 font-mono font-medium",
                ),
                align="center",
                gap="1",
                class_name="px-3 py-1",
            ),
            rx.fragment(),
        ),
        rx.link(
            rx.flex(
                rx.icon("settings", size=16, class_name="text-sidebar-muted"),
                rx.text("Settings", class_name="text-sm text-sidebar-text ml-2"),
                align="center",
                class_name=(
                    "px-3 py-2 rounded-lg hover:bg-sidebar-hover "
                    "transition-colors duration-150 cursor-pointer"
                ),
            ),
            href="/settings",
            underline="none",
        ),
        class_name="pb-3",
    )

