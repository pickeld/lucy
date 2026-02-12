"""Single chat message bubble — user or assistant.

Sources are formatted as markdown in the content string (handled by state),
so no nested data structures needed here.
"""

import reflex as rx


def message_bubble(msg: dict) -> rx.Component:
    """Render a single message bubble based on role."""
    return rx.cond(
        msg["role"] == "user",
        _user_bubble(msg),
        _assistant_bubble(msg),
    )


def _user_bubble(msg: dict) -> rx.Component:
    """User message — gray background."""
    return rx.flex(
        # Avatar
        rx.box(
            rx.icon("user", size=16, class_name="text-gray-500"),
            class_name=(
                "w-7 h-7 rounded-full bg-gray-200 flex items-center "
                "justify-center shrink-0 mt-0.5"
            ),
        ),
        # Content
        rx.box(
            rx.text(
                msg["content"],
                class_name="text-[0.95rem] leading-relaxed text-gray-700 rtl-auto whitespace-pre-wrap",
            ),
            class_name="flex-1 min-w-0",
        ),
        gap="3",
        class_name="bg-user-bubble rounded-xl px-5 py-4 mb-1",
    )


def _assistant_bubble(msg: dict) -> rx.Component:
    """Assistant message — white background, brain avatar.

    Sources (if any) are included in the content as markdown.
    """
    return rx.flex(
        # Avatar
        rx.box(
            rx.text("🧠", class_name="text-sm"),
            class_name=(
                "w-7 h-7 rounded-full bg-emerald-50 flex items-center "
                "justify-center shrink-0 mt-0.5"
            ),
        ),
        # Content (may include formatted sources)
        rx.box(
            rx.markdown(
                msg["content"],
                class_name="text-[0.95rem] leading-relaxed text-gray-700 rtl-auto prose prose-sm max-w-none",
            ),
            class_name="flex-1 min-w-0",
        ),
        gap="3",
        class_name="bg-assistant-bubble rounded-xl px-5 py-4 mb-1",
    )


def typing_indicator() -> rx.Component:
    """Animated typing indicator (three bouncing dots)."""
    return rx.flex(
        rx.box(
            rx.text("🧠", class_name="text-sm"),
            class_name=(
                "w-7 h-7 rounded-full bg-emerald-50 flex items-center "
                "justify-center shrink-0"
            ),
        ),
        rx.flex(
            rx.box(class_name="w-2 h-2 bg-gray-400 rounded-full typing-dot"),
            rx.box(class_name="w-2 h-2 bg-gray-400 rounded-full typing-dot"),
            rx.box(class_name="w-2 h-2 bg-gray-400 rounded-full typing-dot"),
            gap="1",
            align="center",
            class_name="px-3 py-2",
        ),
        gap="3",
        class_name="px-5 py-4 mb-1",
    )
