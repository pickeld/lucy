"""Single chat message bubble — user or assistant.

Sources are stored as a separate 'sources' field in the message dict
and rendered as a collapsible section (collapsed by default).
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

    Sources (if any) are rendered as a collapsible section below the answer.
    A cost badge is shown when per-query cost data is available.
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
        # Content + collapsible sources + cost badge
        rx.box(
            # Main answer
            rx.markdown(
                msg["content"],
                class_name="text-[0.95rem] leading-relaxed text-gray-700 rtl-auto prose prose-sm max-w-none",
            ),
            # Collapsible sources section (only if sources exist)
            rx.cond(
                msg["sources"] != "",
                _collapsible_sources(msg["sources"]),
                rx.fragment(),
            ),
            # Cost badge (only if cost data exists)
            rx.cond(
                msg["cost"] != "",
                rx.flex(
                    rx.icon("coins", size=12, class_name="text-amber-400"),
                    rx.text(
                        msg["cost"],
                        class_name="text-[0.7rem] text-amber-600 font-mono",
                    ),
                    align="center",
                    gap="1",
                    class_name="mt-2 opacity-60 hover:opacity-100 transition-opacity",
                ),
                rx.fragment(),
            ),
            class_name="flex-1 min-w-0",
        ),
        gap="3",
        class_name="bg-assistant-bubble rounded-xl px-5 py-4 mb-1",
    )


def _collapsible_sources(sources_md: rx.Var[str]) -> rx.Component:
    """Render sources as a collapsible details/summary section.

    Collapsed by default — user clicks to expand.
    Uses the HTML <details>/<summary> elements for native collapse behavior.
    """
    return rx.el.details(
        rx.el.summary(
            rx.flex(
                rx.icon("file-text", size=14, class_name="text-gray-400"),
                rx.text(
                    "Sources",
                    class_name="text-xs font-medium text-gray-500",
                ),
                rx.icon("chevron-right", size=12, class_name="text-gray-400 details-chevron transition-transform"),
                align="center",
                gap="1.5",
            ),
            class_name=(
                "cursor-pointer select-none list-none py-1.5 px-2 "
                "rounded-md hover:bg-gray-50 transition-colors "
                "inline-flex items-center"
            ),
        ),
        rx.box(
            rx.markdown(
                sources_md,
                class_name="text-xs leading-relaxed text-gray-600 prose prose-xs max-w-none",
            ),
            class_name="mt-1 pl-2 border-l-2 border-gray-200",
        ),
        class_name="mt-3 sources-details",
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
