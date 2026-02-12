"""Main layout — dark sidebar + light content area."""

import reflex as rx

from .sidebar import sidebar


def layout(content: rx.Component) -> rx.Component:
    """Two-panel layout: fixed dark sidebar (280px) + scrollable content."""
    return rx.flex(
        sidebar(),
        rx.box(
            content,
            class_name="flex-1 min-h-screen overflow-y-auto bg-white",
        ),
        class_name="h-screen w-screen overflow-hidden",
    )
