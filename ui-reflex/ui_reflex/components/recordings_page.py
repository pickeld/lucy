"""Dedicated Call Recordings management page — /recordings.

Table-based layout with filterable, sortable columns, expandable detail
rows with speaker assignment, and full action buttons.
"""

import reflex as rx

from ..state import AppState


# =========================================================================
# MAIN PAGE
# =========================================================================


def recordings_page() -> rx.Component:
    """Full recordings management page with Active / Approved tabs."""
    return rx.box(
        rx.flex(
            _header(),
            # Status / scan message
            rx.cond(
                AppState.call_recordings_scan_message != "",
                rx.box(
                    rx.text(
                        AppState.call_recordings_scan_message,
                        class_name="text-sm",
                    ),
                    class_name="mb-3 px-3 py-2 bg-gray-50 rounded-lg border border-gray-200",
                ),
                rx.fragment(),
            ),
            # Tabs: Active (pending/transcribing/transcribed/error) | Approved
            rx.tabs.root(
                rx.tabs.list(
                    rx.tabs.trigger(
                        rx.flex(
                            rx.icon("file-audio", size=14),
                            rx.text("Active"),
                            rx.box(
                                rx.text(
                                    AppState.recordings_status_counts["pending"],
                                    class_name="text-xs font-bold",
                                ),
                                class_name="px-1.5 py-0.5 bg-yellow-100 text-yellow-700 rounded-full min-w-[20px] text-center",
                            ),
                            align="center",
                            gap="2",
                        ),
                        value="active",
                    ),
                    rx.tabs.trigger(
                        rx.flex(
                            rx.icon("circle-check", size=14),
                            rx.text("Approved"),
                            rx.box(
                                rx.text(
                                    AppState.recordings_status_counts["approved"],
                                    class_name="text-xs font-bold",
                                ),
                                class_name="px-1.5 py-0.5 bg-green-100 text-green-700 rounded-full min-w-[20px] text-center",
                            ),
                            align="center",
                            gap="2",
                        ),
                        value="approved",
                    ),
                    size="2",
                ),
                rx.tabs.content(
                    rx.box(
                        _filter_toolbar(),
                        _recordings_table(),
                        class_name="pt-4",
                    ),
                    value="active",
                ),
                rx.tabs.content(
                    rx.box(
                        _filter_toolbar(),
                        _recordings_table(),
                        class_name="pt-4",
                    ),
                    value="approved",
                ),
                value=AppState.recordings_tab,
                on_change=AppState.set_recordings_tab,
                default_value="active",
            ),
            direction="column",
            class_name="max-w-[1100px] mx-auto w-full px-4 py-6",
        ),
        class_name="h-full overflow-y-auto chat-scroll",
    )


# =========================================================================
# HEADER
# =========================================================================


def _header() -> rx.Component:
    """Page header with title, stats badges, and action buttons."""
    return rx.flex(
        # Left: back + title
        rx.flex(
            rx.link(
                rx.icon_button(
                    rx.icon("arrow-left", size=18),
                    variant="ghost",
                    class_name="text-gray-500 hover:text-gray-700",
                ),
                href="/",
            ),
            rx.heading("📞 Call Recordings", size="6", class_name="text-gray-800"),
            align="center",
            gap="3",
        ),
        # Right: action buttons
        rx.flex(
            _stat_badge("Total", AppState.recordings_status_counts["total"], "gray"),
            _stat_badge("Pending", AppState.recordings_status_counts["pending"], "yellow"),
            _stat_badge("Transcribed", AppState.recordings_status_counts["transcribed"], "blue"),
            _stat_badge("Approved", AppState.recordings_status_counts["approved"], "green"),
            _stat_badge("Errors", AppState.recordings_status_counts["error"], "red"),
            rx.button(
                rx.icon("folder-search", size=14, class_name="mr-1"),
                "Scan",
                on_click=AppState.scan_recordings,
                loading=AppState.call_recordings_files_loading,
                size="2",
                class_name="bg-blue-500 text-white hover:bg-blue-600",
            ),
            rx.button(
                rx.icon("refresh-cw", size=14, class_name="mr-1"),
                "Refresh",
                on_click=AppState.load_recording_files,
                loading=AppState.call_recordings_files_loading,
                variant="outline",
                size="2",
            ),
            # Upload button
            rx.upload(
                rx.button(
                    rx.icon("cloud-upload", size=14, class_name="mr-1"),
                    "Upload",
                    variant="outline",
                    size="2",
                    type="button",
                ),
                id="recordings_upload",
                accept={
                    "audio/*": [
                        ".mp3", ".wav", ".m4a", ".ogg", ".flac",
                        ".wma", ".aac", ".opus", ".webm",
                    ],
                },
                max_files=20,
                on_drop=AppState.upload_call_recordings(  # type: ignore[arg-type]
                    rx.upload_files(upload_id="recordings_upload"),
                ),
                no_drag=True,
                no_keyboard=True,
                border="none",
                padding="0",
            ),
            gap="2",
            align="center",
            wrap="wrap",
        ),
        justify="between",
        align="center",
        class_name="mb-4",
        wrap="wrap",
        gap="3",
    )


def _stat_badge(label: str, value: rx.Var, color: str) -> rx.Component:
    """Small stat badge for the header."""
    color_map = {
        "gray": "bg-gray-100 text-gray-700",
        "yellow": "bg-yellow-100 text-yellow-700",
        "blue": "bg-blue-100 text-blue-700",
        "green": "bg-green-100 text-green-700",
        "red": "bg-red-100 text-red-700",
    }
    cls = color_map.get(color, "bg-gray-100 text-gray-700")
    return rx.box(
        rx.flex(
            rx.text(value, class_name="text-sm font-bold"),
            rx.text(label, class_name="text-xs opacity-70"),
            align="center",
            gap="1",
        ),
        class_name=f"px-2 py-1 rounded-lg {cls}",
    )


# =========================================================================
# FILTER TOOLBAR
# =========================================================================


def _filter_toolbar() -> rx.Component:
    """Filter bar with status, date range, and search."""
    return rx.flex(
        # Status filter
        rx.select(
            ["All", "pending", "transcribing", "transcribed", "approved", "error"],
            value=rx.cond(
                AppState.call_recordings_filter_status == "",
                "All",
                AppState.call_recordings_filter_status,
            ),
            on_change=lambda v: AppState.set_call_recordings_filter_status(  # type: ignore
                rx.cond(v == "All", "", v)  # type: ignore[arg-type]
            ),
            size="2",
            class_name="w-36",
        ),
        # Date from
        rx.el.input(
            type="date",
            value=AppState.recordings_filter_date_from,
            on_change=AppState.set_recordings_filter_date_from,  # type: ignore[arg-type]
            class_name=(
                "text-sm bg-white border border-gray-200 rounded-lg "
                "px-3 py-1.5 outline-none focus:border-accent w-36"
            ),
            placeholder="From date",
        ),
        rx.text("–", class_name="text-gray-400"),
        # Date to
        rx.el.input(
            type="date",
            value=AppState.recordings_filter_date_to,
            on_change=AppState.set_recordings_filter_date_to,  # type: ignore[arg-type]
            class_name=(
                "text-sm bg-white border border-gray-200 rounded-lg "
                "px-3 py-1.5 outline-none focus:border-accent w-36"
            ),
            placeholder="To date",
        ),
        # Search
        rx.el.input(
            type="text",
            placeholder="🔍 Search name, phone, transcript…",
            value=AppState.call_recordings_filter_name,
            on_change=AppState.set_call_recordings_filter_name,  # type: ignore[arg-type]
            class_name=(
                "text-sm bg-white border border-gray-200 rounded-lg "
                "px-3 py-1.5 outline-none focus:border-accent flex-1 min-w-[200px]"
            ),
        ),
        # Clear button
        rx.icon_button(
            rx.icon("x", size=14),
            on_click=AppState.clear_recordings_filters,
            variant="ghost",
            size="1",
            class_name="text-gray-400 hover:text-gray-600",
            title="Clear filters",
        ),
        gap="2",
        align="center",
        class_name="mb-4 flex-wrap",
    )


# =========================================================================
# DATA TABLE
# =========================================================================


def _recordings_table() -> rx.Component:
    """Main data table with sortable column headers."""
    return rx.box(
        # Loading state
        rx.cond(
            AppState.call_recordings_files_loading,
            rx.flex(
                rx.spinner(size="3"),
                rx.text("Loading recordings…", class_name="text-sm text-gray-400 ml-2"),
                align="center",
                justify="center",
                class_name="py-12",
            ),
            rx.cond(
                AppState.recordings_table_data.length() > 0,  # type: ignore[union-attr]
                rx.box(
                    # Table header
                    _table_header(),
                    # Table body
                    rx.foreach(
                        AppState.recordings_table_data,
                        _table_row,
                    ),
                    class_name=(
                        "bg-white border border-gray-200 rounded-xl overflow-hidden shadow-sm"
                    ),
                ),
                rx.flex(
                    rx.icon("file-audio", size=40, class_name="text-gray-300"),
                    rx.text(
                        "No recordings found",
                        class_name="text-gray-400 text-lg mt-2",
                    ),
                    rx.text(
                        "Upload audio files or click 'Scan' to discover recordings.",
                        class_name="text-gray-300 text-sm mt-1",
                    ),
                    direction="column",
                    align="center",
                    class_name="py-16",
                ),
            ),
        ),
        # Upload message
        rx.cond(
            AppState.call_recordings_upload_message != "",
            rx.text(
                AppState.call_recordings_upload_message,
                class_name="text-sm mt-3",
            ),
            rx.fragment(),
        ),
    )


def _table_header() -> rx.Component:
    """Sticky table column headers."""
    return rx.flex(
        _col_header("Date", "modified_at", "w-[110px]"),
        _col_header("Contact", "contact_name", "flex-1 min-w-[140px]"),
        _col_header("Duration", "duration_seconds", "w-[80px]"),
        rx.text(
            "Spkrs",
            class_name="text-xs font-semibold text-gray-500 uppercase w-[50px] text-center",
        ),
        rx.text(
            "Lang",
            class_name="text-xs font-semibold text-gray-500 uppercase w-[50px] text-center",
        ),
        _col_header("Status", "status", "w-[100px]"),
        rx.text(
            "Actions",
            class_name="text-xs font-semibold text-gray-500 uppercase w-[130px] text-center",
        ),
        align="center",
        class_name=(
            "px-4 py-2.5 bg-gray-50 border-b border-gray-200 gap-3"
        ),
    )


def _col_header(label: str, sort_key: str, width_class: str) -> rx.Component:
    """Clickable sortable column header."""
    return rx.flex(
        rx.text(
            label,
            class_name="text-xs font-semibold text-gray-500 uppercase cursor-pointer hover:text-gray-700",
        ),
        rx.cond(
            AppState.recordings_sort_column == sort_key,
            rx.icon(
                rx.cond(AppState.recordings_sort_asc, "chevron-up", "chevron-down"),
                size=12,
                class_name="text-gray-500",
            ),
            rx.fragment(),
        ),
        on_click=AppState.set_recordings_sort_column(sort_key),
        align="center",
        gap="1",
        class_name=width_class,
    )


# =========================================================================
# TABLE ROW
# =========================================================================


def _table_row(item: dict) -> rx.Component:
    """Single recording row + expandable detail panel."""
    is_expanded = AppState.recordings_expanded_hash == item["content_hash"]

    return rx.box(
        # Main row (clickable to expand)
        rx.flex(
            # Date & Time
            rx.box(
                rx.text(item["date"], class_name="text-sm text-gray-800"),
                rx.text(item["time"], class_name="text-xs text-gray-400"),
                class_name="w-[110px]",
            ),
            # Contact / name
            rx.box(
                rx.text(
                    item["display_name"],
                    class_name="text-sm font-medium text-gray-800 truncate",
                ),
                rx.cond(
                    item["phone_number"] != "",
                    rx.text(
                        item["phone_number"],
                        class_name="text-xs text-gray-400",
                    ),
                    rx.fragment(),
                ),
                class_name="flex-1 min-w-[140px] cursor-pointer",
                on_click=AppState.toggle_recording_detail(item["content_hash"]),
            ),
            # Duration
            rx.text(
                item["duration_fmt"],
                class_name="text-sm text-gray-600 w-[80px] text-center",
            ),
            # Speaker count
            rx.text(
                item["speaker_count"],
                class_name="text-sm text-gray-600 w-[50px] text-center",
            ),
            # Language
            rx.cond(
                item["language"] != "",
                rx.box(
                    rx.text(
                        item["language"],
                        class_name="text-xs font-medium uppercase",
                    ),
                    class_name="px-1.5 py-0.5 bg-gray-100 text-gray-600 rounded w-[50px] text-center",
                ),
                rx.box(class_name="w-[50px]"),
            ),
            # Status badge
            _status_badge(item["status"]),
            # Actions
            _action_buttons(item),
            align="center",
            class_name="px-4 py-3 gap-3 hover:bg-gray-50 transition-colors",
        ),
        # Transcription progress (while transcribing)
        rx.cond(
            (item["status"] == "transcribing") & (item["transcription_progress"] != ""),
            rx.flex(
                rx.icon("loader-circle", size=12, class_name="animate-spin text-yellow-600"),
                rx.text(item["transcription_progress"], class_name="text-xs text-yellow-700"),
                align="center",
                gap="2",
                class_name="px-4 pb-2",
            ),
            rx.fragment(),
        ),
        # Error message
        rx.cond(
            (item["status"] == "error") & (item["error_message"] != ""),
            rx.text(
                item["error_message"],
                class_name="text-xs text-red-500 px-4 pb-2",
            ),
            rx.fragment(),
        ),
        # Expanded detail panel
        rx.cond(
            is_expanded,
            _detail_panel(item),
            rx.fragment(),
        ),
        class_name="border-b border-gray-100 last:border-b-0",
    )


# =========================================================================
# STATUS BADGE
# =========================================================================


def _status_badge(status: rx.Var[str]) -> rx.Component:
    """Colored status badge."""
    return rx.box(
        rx.text(
            status,
            class_name="text-xs font-medium capitalize",
        ),
        class_name=rx.cond(
            status == "approved",
            "px-2 py-0.5 rounded-full bg-green-100 text-green-700 w-[100px] text-center",
            rx.cond(
                status == "transcribed",
                "px-2 py-0.5 rounded-full bg-blue-100 text-blue-700 w-[100px] text-center",
                rx.cond(
                    status == "error",
                    "px-2 py-0.5 rounded-full bg-red-100 text-red-700 w-[100px] text-center",
                    rx.cond(
                        status == "transcribing",
                        "px-2 py-0.5 rounded-full bg-yellow-100 text-yellow-700 w-[100px] text-center animate-pulse",
                        "px-2 py-0.5 rounded-full bg-gray-100 text-gray-600 w-[100px] text-center",
                    ),
                ),
            ),
        ),
    )


# =========================================================================
# ACTION BUTTONS
# =========================================================================


def _action_buttons(item: dict) -> rx.Component:
    """Action icon buttons column."""
    return rx.flex(
        # Approve (transcribed only)
        rx.cond(
            item["status"] == "transcribed",
            rx.icon_button(
                rx.icon("circle-check", size=16),
                on_click=AppState.approve_recording(item["content_hash"]),
                variant="ghost",
                size="1",
                class_name="text-green-500 hover:text-green-700",
                title="Approve & index",
            ),
            rx.fragment(),
        ),
        # Restart (transcribing only — stuck jobs)
        rx.cond(
            item["status"] == "transcribing",
            rx.icon_button(
                rx.icon("rotate-ccw", size=16),
                on_click=AppState.restart_stuck_transcription(item["content_hash"]),
                variant="ghost",
                size="1",
                class_name="text-orange-500 hover:text-orange-700",
                title="Restart stuck transcription",
            ),
            rx.fragment(),
        ),
        # Transcribe / retranscribe (all except transcribing)
        rx.cond(
            item["status"] != "transcribing",
            rx.icon_button(
                rx.icon("mic", size=16),
                on_click=AppState.retry_transcription(item["content_hash"]),
                variant="ghost",
                size="1",
                class_name="text-blue-500 hover:text-blue-700",
                title=rx.cond(
                    (item["status"] == "transcribed") | (item["status"] == "approved"),
                    "Re-transcribe",
                    "Transcribe",
                ),
            ),
            rx.fragment(),
        ),
        # Expand/collapse detail
        rx.icon_button(
            rx.icon(
                rx.cond(
                    AppState.recordings_expanded_hash == item["content_hash"],
                    "chevron-up",
                    "chevron-down",
                ),
                size=16,
            ),
            on_click=AppState.toggle_recording_detail(item["content_hash"]),
            variant="ghost",
            size="1",
            class_name="text-gray-400 hover:text-gray-600",
            title="Toggle details",
        ),
        # Delete (not approved)
        rx.cond(
            item["status"] != "approved",
            rx.icon_button(
                rx.icon("trash-2", size=16),
                on_click=AppState.delete_recording(item["content_hash"]),
                variant="ghost",
                size="1",
                class_name="text-red-400 hover:text-red-600",
                title="Delete",
            ),
            rx.fragment(),
        ),
        gap="1",
        align="center",
        class_name="w-[130px] justify-center",
    )


# =========================================================================
# DETAIL PANEL (expanded row)
# =========================================================================


def _detail_panel(item: dict) -> rx.Component:
    """Expanded detail panel with transcript, speaker assignment, and metadata."""
    return rx.box(
        rx.flex(
            # Left: transcript
            rx.box(
                rx.text(
                    "Transcript",
                    class_name="text-xs font-semibold text-gray-500 uppercase mb-2",
                ),
                rx.cond(
                    item["transcript_text"] != "",
                    rx.box(
                        rx.text(
                            item["transcript_text"],
                            class_name="text-sm text-gray-700 whitespace-pre-wrap",
                        ),
                        class_name=(
                            "bg-gray-50 rounded-lg border border-gray-200 p-3 "
                            "max-h-80 overflow-y-auto"
                        ),
                    ),
                    rx.text(
                        "No transcript available yet",
                        class_name="text-sm text-gray-400 italic",
                    ),
                ),
                class_name="flex-1 min-w-0",
            ),
            # Right: speaker assignment + metadata
            rx.box(
                # Speaker Assignment
                rx.text(
                    "Speaker Assignment",
                    class_name="text-xs font-semibold text-gray-500 uppercase mb-2",
                ),
                rx.box(
                    # Speaker A
                    rx.flex(
                        rx.text("Speaker A:", class_name="text-sm text-gray-600 w-24 shrink-0"),
                        rx.select(
                            AppState.recordings_speaker_options,
                            value=AppState.recordings_speaker_a,
                            on_change=AppState.set_recordings_speaker_a,
                            size="2",
                            class_name="flex-1",
                        ),
                        align="center",
                        gap="2",
                        class_name="mb-1",
                    ),
                    # Swap button
                    rx.flex(
                        rx.icon_button(
                            rx.icon("arrow-up-down", size=14),
                            on_click=AppState.swap_speakers,
                            variant="ghost",
                            size="1",
                            class_name="text-gray-400 hover:text-accent",
                            title="Swap Speaker A ↔ B",
                        ),
                        justify="center",
                        class_name="mb-1",
                    ),
                    # Speaker B
                    rx.flex(
                        rx.text("Speaker B:", class_name="text-sm text-gray-600 w-24 shrink-0"),
                        rx.select(
                            AppState.recordings_speaker_options,
                            value=AppState.recordings_speaker_b,
                            on_change=AppState.set_recordings_speaker_b,
                            size="2",
                            class_name="flex-1",
                        ),
                        align="center",
                        gap="2",
                        class_name="mb-2",
                    ),
                    # Apply button
                    rx.button(
                        rx.icon("check", size=14, class_name="mr-1"),
                        "Apply Names",
                        on_click=AppState.save_speaker_labels(item["content_hash"]),
                        size="2",
                        class_name="bg-green-500 text-white hover:bg-green-600 w-full",
                    ),
                    class_name=(
                        "bg-gray-50 rounded-lg border border-gray-200 p-3 mb-3"
                    ),
                ),
                # Metadata
                rx.text(
                    "Metadata",
                    class_name="text-xs font-semibold text-gray-500 uppercase mb-2",
                ),
                rx.box(
                    rx.flex(
                        rx.text("Contact:", class_name="text-sm text-gray-600 w-24 shrink-0"),
                        rx.el.input(
                            type="text",
                            placeholder="Contact name…",
                            default_value=item["contact_name"],
                            on_blur=AppState.save_recording_metadata(
                                item["content_hash"], "contact_name",
                            ),
                            class_name=(
                                "text-sm bg-white border border-gray-200 rounded-lg "
                                "px-2 py-1.5 outline-none focus:border-accent flex-1"
                            ),
                        ),
                        align="center",
                        gap="2",
                        class_name="mb-2",
                    ),
                    rx.flex(
                        rx.text("Phone:", class_name="text-sm text-gray-600 w-24 shrink-0"),
                        rx.el.input(
                            type="text",
                            placeholder="Phone…",
                            default_value=item["phone_number"],
                            on_blur=AppState.save_recording_metadata(
                                item["content_hash"], "phone_number",
                            ),
                            class_name=(
                                "text-sm bg-white border border-gray-200 rounded-lg "
                                "px-2 py-1.5 outline-none focus:border-accent flex-1"
                            ),
                        ),
                        align="center",
                        gap="2",
                    ),
                    class_name=(
                        "bg-gray-50 rounded-lg border border-gray-200 p-3"
                    ),
                ),
                # File info
                rx.flex(
                    rx.text(
                        "File: " + item["filename"],  # type: ignore[operator]
                        class_name="text-xs text-gray-400",
                    ),
                    rx.cond(
                        item["confidence"] != "0.0",
                        rx.text(
                            "Confidence: " + item["confidence"],  # type: ignore[operator]
                            class_name="text-xs text-gray-400",
                        ),
                        rx.fragment(),
                    ),
                    gap="4",
                    class_name="mt-2",
                ),
                class_name="w-[320px] shrink-0",
            ),
            gap="4",
        ),
        class_name="px-4 py-4 bg-white border-t border-gray-100",
    )
