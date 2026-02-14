"""Chat area — message list, filter chips, input bar, and typing indicator.

Main content area that shows either the empty state or the active conversation.
"""

import reflex as rx

from ..state import AppState
from .empty_state import empty_state
from .message_bubble import message_bubble, typing_indicator


def chat_area() -> rx.Component:
    """Full chat area — empty state or conversation with input."""
    return rx.flex(
        # Main content: empty state or messages
        rx.cond(
            AppState.show_chat,
            _conversation_view(),
            empty_state(),
        ),
        # Chat input (always visible)
        _chat_input_bar(),
        direction="column",
        class_name="h-full max-w-[820px] mx-auto w-full px-4",
    )


# =========================================================================
# CONVERSATION VIEW (messages + filter chips)
# =========================================================================

def _conversation_view() -> rx.Component:
    """Scrollable message list with filter chips."""
    return rx.box(
        # Filter chips
        rx.cond(
            AppState.has_filters,
            _filter_chips(),
            rx.fragment(),
        ),
        # Messages
        rx.foreach(
            AppState.messages,
            message_bubble,
        ),
        # Typing indicator
        rx.cond(
            AppState.is_loading,
            typing_indicator(),
            rx.fragment(),
        ),
        class_name="flex-1 overflow-y-auto chat-scroll pt-4 pb-2",
    )


# =========================================================================
# FILTER CHIPS
# =========================================================================

def _filter_chips() -> rx.Component:
    """Active filter chips displayed above messages."""
    return rx.flex(
        rx.foreach(
            AppState.filter_chips,
            _filter_chip,
        ),
        rx.box(
            rx.button(
                rx.icon("x", size=12, class_name="mr-1"),
                "Clear",
                on_click=AppState.clear_filters,
                size="1",
                variant="ghost",
                class_name="text-gray-400 hover:text-gray-600",
            ),
        ),
        align="center",
        gap="2",
        wrap="wrap",
        class_name="px-1 py-2 mb-2",
    )


def _filter_chip(chip: dict) -> rx.Component:
    """Single filter chip with remove button."""
    return rx.badge(
        rx.text(chip["icon"], class_name="mr-1"),
        rx.text(chip["label"]),
        rx.box(
            rx.icon(
                "x",
                size=12,
                class_name="ml-1 cursor-pointer opacity-70 hover:opacity-100",
            ),
            on_click=AppState.remove_filter(chip["key"]),
        ),
        variant="soft",
        class_name="bg-accent-light text-accent-text rounded-full px-3 py-1",
    )


# =========================================================================
# CHAT INPUT BAR
# =========================================================================

def _chat_input_bar() -> rx.Component:
    """Auto-growing textarea with send button.

    Enter submits the form; Shift+Enter inserts a newline.
    """
    # Client-side JS: intercept Enter (without Shift) in the textarea
    # and click the submit button to trigger Reflex's on_submit handler.
    # Using button.click() instead of form.requestSubmit() because Reflex
    # uses React's synthetic event system which doesn't respond to native
    # DOM submit events.
    _enter_to_submit_js = """
    (function() {
        document.addEventListener("keydown", function(e) {
            if (e.target && e.target.classList &&
                e.target.classList.contains("chat-textarea") &&
                e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                var form = e.target.closest("form");
                if (form) {
                    var btn = form.querySelector('button[type="submit"]');
                    if (btn) btn.click();
                }
            }
        });
    })();
    """
    return rx.box(
        rx.script(_enter_to_submit_js),
        rx.el.form(
            rx.flex(
                rx.el.textarea(
                    value=AppState.input_text,
                    on_change=AppState.set_input_text,
                    placeholder="Ask about your messages and documents…",
                    rows=1,
                    class_name=(
                        "chat-textarea flex-1 bg-transparent border-none "
                        "outline-none text-gray-700 text-[0.95rem] "
                        "placeholder-gray-400 py-3 px-1 rtl-auto"
                    ),
                ),
                rx.icon_button(
                    rx.icon("arrow-up", size=18),
                    type="submit",
                    size="2",
                    class_name=(
                        "bg-accent text-white hover:bg-accent-hover "
                        "rounded-lg shrink-0 cursor-pointer"
                    ),
                    disabled=AppState.is_loading,
                ),
                align="end",
                gap="2",
                class_name=(
                    "border border-gray-200 rounded-xl px-4 py-1 "
                    "bg-white shadow-sm focus-within:border-accent "
                    "focus-within:shadow-[0_0_0_2px_rgba(16,163,127,0.15)] "
                    "transition-all duration-150"
                ),
            ),
            on_submit=AppState.send_message,
            reset_on_submit=False,
        ),
        class_name="pb-4 pt-2",
    )
