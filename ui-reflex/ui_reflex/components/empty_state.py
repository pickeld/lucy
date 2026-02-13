"""Empty state — landing page with logo and suggestion cards."""

import reflex as rx

from ..state import AppState

SUGGESTIONS = [
    "What did everyone talk about this week?",
    "Summarize the latest group conversations",
    "Search for messages about meetings",
    "Who was the most active chatter recently?",
]


def empty_state() -> rx.Component:
    """Centered empty state with brain icon and 2×2 suggestion grid."""
    return rx.center(
        rx.vstack(
            # Logo
            rx.text("🧠", class_name="text-5xl opacity-80"),
            # Title
            rx.heading(
                "RAG Assistant",
                size="7",
                class_name="text-gray-800 font-semibold",
            ),
            # Subtitle
            rx.text(
                "Ask anything about your messages and documents",
                class_name="text-gray-400 text-base",
            ),
            # Spacer
            rx.box(class_name="h-6"),
            # Suggestion grid
            rx.grid(
                *[_suggestion_card(s) for s in SUGGESTIONS],
                columns="2",
                gap="3",
                class_name="w-full max-w-[500px]",
            ),
            align="center",
            spacing="3",
            class_name="py-16",
        ),
        class_name="min-h-[70vh]",
    )


def _suggestion_card(text: str) -> rx.Component:
    """Clickable suggestion card."""
    return rx.box(
        rx.text(text, class_name="text-sm text-gray-600 leading-relaxed"),
        on_click=AppState.send_suggestion(text),
        class_name=(
            "bg-gray-50 border border-gray-200 rounded-xl px-4 py-3.5 "
            "cursor-pointer hover:bg-gray-100 hover:border-gray-300 "
            "transition-colors duration-150 text-left"
        ),
    )
