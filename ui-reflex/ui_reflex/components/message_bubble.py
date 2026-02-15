"""Single chat message bubble — user or assistant.

Sources are stored as a separate 'sources' field in the message dict
and rendered as a collapsible section (collapsed by default).

Rich content blocks (images, ICS events, disambiguation buttons) are
stored as flattened string fields in the message dict and rendered
below the main answer text.
"""

import reflex as rx

from ..state import AppState


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
    Rich content (images, ICS events, buttons) are rendered between the
    answer and the sources.
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
        # Content + rich content + collapsible sources + cost badge
        rx.box(
            # Main answer
            rx.markdown(
                msg["content"],
                class_name="text-[0.95rem] leading-relaxed text-gray-700 rtl-auto prose prose-sm max-w-none",
            ),
            # Rich content: inline images
            rx.cond(
                msg["image_urls"] != "",
                _render_images(msg["image_urls"], msg["image_captions"]),
                rx.fragment(),
            ),
            # Rich content: ICS calendar event download
            rx.cond(
                msg["ics_url"] != "",
                _render_ics_download(msg["ics_url"], msg["ics_title"]),
                rx.fragment(),
            ),
            # Rich content: disambiguation/clarification buttons
            rx.cond(
                msg["button_options"] != "",
                _render_buttons(msg["button_prompt"], msg["button_options"]),
                rx.fragment(),
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


# =========================================================================
# Rich Content: Inline Images
# =========================================================================

def _render_images(image_urls: rx.Var[str], image_captions: rx.Var[str]) -> rx.Component:
    """Render inline images from pipe-separated URL and caption strings."""
    return rx.box(
        rx.foreach(
            image_urls.split("|"),
            lambda url, idx: _render_single_image(url, image_captions, idx),
        ),
        class_name="mt-3 space-y-2",
    )


def _render_single_image(url: rx.Var[str], captions: rx.Var[str], idx: rx.Var[int]) -> rx.Component:
    """Render a single inline image with optional caption."""
    return rx.box(
        rx.image(
            src=url,
            alt="Archive image",
            class_name=(
                "rounded-lg max-w-full max-h-80 object-contain "
                "border border-gray-200 shadow-sm"
            ),
            loading="lazy",
        ),
        class_name="inline-block",
    )


# =========================================================================
# Rich Content: ICS Calendar Event Download
# =========================================================================

def _render_ics_download(ics_url: rx.Var[str], ics_title: rx.Var[str]) -> rx.Component:
    """Render a calendar event download button."""
    return rx.flex(
        rx.el.a(
            rx.flex(
                rx.icon("calendar-plus", size=18, class_name="text-blue-500"),
                rx.box(
                    rx.text(
                        "📅 Download Calendar Event",
                        class_name="text-sm font-medium text-blue-600",
                    ),
                    rx.text(
                        ics_title,
                        class_name="text-xs text-gray-500 truncate",
                    ),
                ),
                align="center",
                gap="2",
            ),
            href=ics_url,
            download=True,
            class_name=(
                "inline-flex items-center px-4 py-2.5 rounded-lg "
                "bg-blue-50 hover:bg-blue-100 border border-blue-200 "
                "transition-colors cursor-pointer no-underline"
            ),
        ),
        class_name="mt-3",
    )


# =========================================================================
# Rich Content: Disambiguation / Clarification Buttons
# =========================================================================

def _render_buttons(prompt: rx.Var[str], options_str: rx.Var[str]) -> rx.Component:
    """Render disambiguation buttons from pipe-separated options string."""
    return rx.box(
        rx.flex(
            rx.foreach(
                options_str.split("|"),
                _render_single_button,
            ),
            wrap="wrap",
            gap="2",
        ),
        class_name="mt-3",
    )


def _render_single_button(option: rx.Var[str]) -> rx.Component:
    """Render a single disambiguation button that sends option as message."""
    return rx.el.button(
        rx.flex(
            rx.icon("user", size=14, class_name="text-emerald-500 shrink-0"),
            rx.text(
                option,
                class_name="text-sm text-gray-700 rtl-auto",
            ),
            align="center",
            gap="1.5",
        ),
        on_click=AppState.send_suggestion(option),
        class_name=(
            "px-4 py-2 rounded-lg bg-emerald-50 hover:bg-emerald-100 "
            "border border-emerald-200 transition-colors cursor-pointer "
            "text-left"
        ),
    )


# =========================================================================
# Collapsible Sources
# =========================================================================

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


# =========================================================================
# Typing Indicator
# =========================================================================

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
