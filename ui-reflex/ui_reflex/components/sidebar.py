"""Dark sidebar component — ChatGPT-style conversation list.

Supports collapsed (icons-only, 60px) and expanded (280px) modes.
Uses a flat list of items (headers + conversations) from state
to avoid nested rx.foreach — Reflex requires typed vars for iteration.
"""

import reflex as rx

from ..state import AppState


def sidebar() -> rx.Component:
    """Collapsible sidebar: 280px expanded, 60px collapsed."""
    return rx.box(
        rx.flex(
            _toggle_button(),
            rx.cond(
                AppState.sidebar_collapsed,
                # ---- Collapsed mode: icons only ----
                rx.fragment(
                    _collapsed_new_chat(),
                    rx.box(
                        rx.foreach(
                            AppState.sidebar_items,
                            _render_collapsed_item,
                        ),
                        class_name="flex-1 overflow-y-auto sidebar-scroll px-1",
                    ),
                    _collapsed_bottom(),
                ),
                # ---- Expanded mode: full sidebar ----
                rx.fragment(
                    _new_chat_button(),
                    _search_bar(),
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
                ),
            ),
            direction="column",
            class_name="h-full",
        ),
        class_name=rx.cond(
            AppState.sidebar_collapsed,
            "sidebar-container w-[60px] min-w-[60px] h-screen bg-sidebar border-r border-sidebar-border flex flex-col",
            "sidebar-container w-[280px] min-w-[280px] h-screen bg-sidebar border-r border-sidebar-border flex flex-col",
        ),
    )


# =========================================================================
# TOGGLE BUTTON
# =========================================================================

def _toggle_button() -> rx.Component:
    """Collapse / expand toggle at the top of the sidebar."""
    return rx.box(
        rx.icon_button(
            rx.cond(
                AppState.sidebar_collapsed,
                rx.icon("panel-left-open", size=18),
                rx.icon("panel-left-close", size=18),
            ),
            on_click=AppState.toggle_sidebar,
            variant="ghost",
            size="2",
            class_name=(
                "text-sidebar-muted hover:text-sidebar-text "
                "hover:bg-sidebar-hover cursor-pointer !bg-transparent"
            ),
        ),
        class_name=rx.cond(
            AppState.sidebar_collapsed,
            "px-2 pt-3 pb-1 flex justify-center",
            "px-3 pt-3 pb-1 flex justify-end",
        ),
    )


# =========================================================================
# EXPANDED MODE — NEW CHAT BUTTON
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
        class_name="px-3 pt-1 pb-2",
    )


# =========================================================================
# COLLAPSED MODE — NEW CHAT (icon only)
# =========================================================================

def _collapsed_new_chat() -> rx.Component:
    return rx.box(
        rx.icon_button(
            rx.icon("plus", size=18),
            on_click=AppState.new_chat,
            variant="outline",
            class_name=(
                "border border-dashed border-sidebar-border "
                "text-sidebar-text bg-transparent hover:bg-sidebar-hover "
                "hover:border-solid cursor-pointer !w-9 !h-9"
            ),
        ),
        class_name="px-2 pt-1 pb-2 flex justify-center",
    )


# =========================================================================
# SEARCH BAR (expanded only)
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
# EXPANDED SIDEBAR ITEM RENDERER (header or conversation)
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
                    rx.dropdown_menu.item(
                        rx.icon("clipboard-copy", size=14, class_name="mr-2"),
                        "Copy",
                        on_click=AppState.copy_chat_to_clipboard(convo_id),
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
# COLLAPSED SIDEBAR ITEM RENDERER
# =========================================================================

def _render_collapsed_item(item: dict) -> rx.Component:
    """Render a collapsed conversation item (dot or avatar-like circle)."""
    return rx.cond(
        item["type"] == "header",
        # Skip headers in collapsed mode (just a thin separator)
        rx.separator(class_name="border-sidebar-border my-1 mx-2"),
        _render_collapsed_conversation(item),
    )


def _render_collapsed_conversation(item: dict) -> rx.Component:
    """Collapsed conversation: show first letter as circle."""
    convo_id = item["id"]
    is_active = AppState.conversation_id == convo_id

    return rx.tooltip(
        rx.box(
            rx.text(
                item["title"].to(str)[0:2].to(str).upper(),
                class_name="text-[0.65rem] font-bold text-sidebar-text",
            ),
            on_click=AppState.load_conversation(convo_id),
            class_name=rx.cond(
                is_active,
                (
                    "w-9 h-9 rounded-lg bg-sidebar-active border border-accent "
                    "flex items-center justify-center cursor-default mx-auto my-0.5"
                ),
                (
                    "w-9 h-9 rounded-lg hover:bg-sidebar-hover "
                    "flex items-center justify-center cursor-pointer mx-auto my-0.5 "
                    "transition-colors duration-150"
                ),
            ),
        ),
        content=item["title"],
        side="right",
    )


# =========================================================================
# EXPANDED BOTTOM SECTION
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
        rx.flex(
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
            rx.link(
                rx.flex(
                    rx.icon("users", size=16, class_name="text-sidebar-muted"),
                    rx.text("Entities", class_name="text-sm text-sidebar-text ml-2"),
                    align="center",
                    class_name=(
                        "px-3 py-2 rounded-lg hover:bg-sidebar-hover "
                        "transition-colors duration-150 cursor-pointer"
                    ),
                ),
                href="/entities",
                underline="none",
            ),
            rx.link(
                rx.flex(
                    rx.icon("phone", size=16, class_name="text-sidebar-muted"),
                    rx.text("Recordings", class_name="text-sm text-sidebar-text ml-2"),
                    align="center",
                    class_name=(
                        "px-3 py-2 rounded-lg hover:bg-sidebar-hover "
                        "transition-colors duration-150 cursor-pointer"
                    ),
                ),
                href="/recordings",
                underline="none",
            ),
            rx.link(
                rx.flex(
                    rx.icon("sparkles", size=16, class_name="text-sidebar-muted"),
                    rx.text("Insights", class_name="text-sm text-sidebar-text ml-2"),
                    align="center",
                    class_name=(
                        "px-3 py-2 rounded-lg hover:bg-sidebar-hover "
                        "transition-colors duration-150 cursor-pointer"
                    ),
                ),
                href="/insights",
                underline="none",
            ),
            direction="row",
            gap="0",
            wrap="wrap",
        ),
        class_name="pb-3",
    )


# =========================================================================
# COLLAPSED BOTTOM SECTION
# =========================================================================

def _collapsed_bottom() -> rx.Component:
    """Collapsed bottom: just icon links."""
    return rx.box(
        rx.separator(class_name="border-sidebar-border mb-2"),
        # Health dot (centered)
        rx.flex(
            rx.box(
                class_name=rx.cond(
                    AppState.api_status == "up",
                    "w-2.5 h-2.5 rounded-full bg-status-green",
                    rx.cond(
                        AppState.api_status == "degraded",
                        "w-2.5 h-2.5 rounded-full bg-status-yellow",
                        "w-2.5 h-2.5 rounded-full bg-status-red",
                    ),
                ),
            ),
            justify="center",
            class_name="py-1",
        ),
        # Settings icon
        rx.flex(
            rx.tooltip(
                rx.link(
                    rx.icon_button(
                        rx.icon("settings", size=18),
                        variant="ghost",
                        class_name=(
                            "text-sidebar-muted hover:text-sidebar-text "
                            "hover:bg-sidebar-hover cursor-pointer !bg-transparent"
                        ),
                    ),
                    href="/settings",
                    underline="none",
                ),
                content="Settings",
                side="right",
            ),
            justify="center",
            class_name="py-0.5",
        ),
        # Entities icon
        rx.flex(
            rx.tooltip(
                rx.link(
                    rx.icon_button(
                        rx.icon("users", size=18),
                        variant="ghost",
                        class_name=(
                            "text-sidebar-muted hover:text-sidebar-text "
                            "hover:bg-sidebar-hover cursor-pointer !bg-transparent"
                        ),
                    ),
                    href="/entities",
                    underline="none",
                ),
                content="Entities",
                side="right",
            ),
            justify="center",
            class_name="py-0.5",
        ),
        # Recordings icon
        rx.flex(
            rx.tooltip(
                rx.link(
                    rx.icon_button(
                        rx.icon("phone", size=18),
                        variant="ghost",
                        class_name=(
                            "text-sidebar-muted hover:text-sidebar-text "
                            "hover:bg-sidebar-hover cursor-pointer !bg-transparent"
                        ),
                    ),
                    href="/recordings",
                    underline="none",
                ),
                content="Recordings",
                side="right",
            ),
            justify="center",
            class_name="py-0.5",
        ),
        # Insights icon
        rx.flex(
            rx.tooltip(
                rx.link(
                    rx.icon_button(
                        rx.icon("sparkles", size=18),
                        variant="ghost",
                        class_name=(
                            "text-sidebar-muted hover:text-sidebar-text "
                            "hover:bg-sidebar-hover cursor-pointer !bg-transparent"
                        ),
                    ),
                    href="/insights",
                    underline="none",
                ),
                content="Insights",
                side="right",
            ),
            justify="center",
            class_name="py-0.5",
        ),
        class_name="pb-3",
    )
