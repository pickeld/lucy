"""Settings page — redesigned tabbed configuration management.

Six tabs: AI & Models, API Keys, Knowledge Base, Plugins, System, Costs.
Each tab uses card-based sections with clear visual grouping.
Human-readable labels via SETTING_LABELS in state.py.
"""

import reflex as rx

from ..state import AppState


# =========================================================================
# MAIN SETTINGS PAGE
# =========================================================================


def settings_page() -> rx.Component:
    """Full settings page with redesigned tabbed interface."""
    return rx.box(
        rx.flex(
            # Compact header: back + title + health badge + export
            _header(),
            # Status message (save confirmation)
            rx.cond(
                AppState.settings_save_message != "",
                rx.box(
                    rx.text(
                        AppState.settings_save_message,
                        class_name="text-sm",
                    ),
                    class_name="mb-4 px-3 py-2 bg-gray-50 rounded-lg border border-gray-200",
                ),
                rx.fragment(),
            ),
            # Main tabbed interface — 6 tabs
            rx.tabs.root(
                rx.tabs.list(
                    rx.tabs.trigger("🤖 AI & Models", value="ai"),
                    rx.tabs.trigger("🔑 API Keys", value="keys"),
                    rx.tabs.trigger("📚 Knowledge Base", value="kb"),
                    rx.tabs.trigger("🔌 Plugins", value="plugins"),
                    rx.tabs.trigger("⚙️ System", value="system"),
                    rx.tabs.trigger("💰 Costs", value="costs"),
                    size="2",
                ),
                rx.tabs.content(_ai_tab(), value="ai", class_name="pt-4"),
                rx.tabs.content(_keys_tab(), value="keys", class_name="pt-4"),
                rx.tabs.content(_kb_tab(), value="kb", class_name="pt-4"),
                rx.tabs.content(_plugins_tab(), value="plugins", class_name="pt-4"),
                rx.tabs.content(_system_tab(), value="system", class_name="pt-4"),
                rx.tabs.content(_costs_tab(), value="costs", class_name="pt-4"),
                value=AppState.settings_tab,
                on_change=AppState.set_settings_tab,
                default_value="ai",
                class_name="mt-2",
            ),
            direction="column",
            class_name="max-w-[820px] mx-auto w-full px-4 py-6",
        ),
        class_name="h-full overflow-y-auto chat-scroll",
    )


# =========================================================================
# HEADER (compact — health badge inline)
# =========================================================================


def _header() -> rx.Component:
    """Compact header: back button + title + health badge + export."""
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
            rx.heading("Settings", size="6", class_name="text-gray-800"),
            align="center",
            gap="3",
        ),
        # Right: health badge + export/import
        rx.flex(
            # Inline health badge
            rx.box(
                rx.flex(
                    rx.box(
                        class_name=rx.cond(
                            AppState.api_status == "up",
                            "health-badge-dot bg-status-green",
                            rx.cond(
                                AppState.api_status == "degraded",
                                "health-badge-dot bg-status-yellow",
                                "health-badge-dot bg-status-red",
                            ),
                        ),
                    ),
                    rx.text(AppState.health_label),
                    align="center",
                    gap="1.5",
                ),
                class_name="health-badge",
            ),
            rx.button(
                rx.icon("download", size=14),
                "Export",
                on_click=AppState.export_settings,
                variant="outline",
                size="1",
            ),
            rx.upload(
                rx.button(
                    rx.icon("upload", size=14),
                    "Import",
                    variant="outline",
                    size="1",
                    type="button",
                ),
                id="settings_import",
                accept={"application/json": [".json"]},
                max_files=1,
                on_drop=AppState.import_settings(  # type: ignore[arg-type]
                    rx.upload_files(upload_id="settings_import"),
                ),
                no_drag=True,
                no_keyboard=True,
                border="none",
                padding="0",
            ),
            gap="2",
            align="center",
        ),
        justify="between",
        align="center",
        class_name="mb-4",
    )


# =========================================================================
# SECTION CARD — shared wrapper for visual grouping
# =========================================================================


def _section_card(
    title: str,
    icon: str,
    *children: rx.Component,
    reset_category: str = "",
) -> rx.Component:
    """Wrap settings in a visually distinct card with icon + title header."""
    header_items = [
        rx.flex(
            rx.icon(icon, size=16, class_name="text-gray-500"),
            rx.text(title),
            align="center",
            gap="2",
        ),
    ]
    if reset_category:
        header_items.append(
            rx.button(
                rx.icon("rotate-ccw", size=12, class_name="mr-1"),
                "Reset",
                on_click=AppState.reset_category(reset_category),
                variant="ghost",
                size="1",
                class_name="text-gray-400 hover:text-gray-600 text-xs",
            ),
        )
    return rx.box(
        rx.box(
            rx.flex(
                *header_items,
                justify="between",
                align="center",
            ),
            class_name="settings-card-header",
        ),
        *children,
        class_name="settings-card",
    )


# =========================================================================
# TAB: AI & MODELS
# =========================================================================


def _ai_tab() -> rx.Component:
    """AI & Models tab — chat provider, image generation, system prompt."""
    return rx.flex(
        # Chat Provider section
        _section_card(
            "Chat Provider", "message-square",
            rx.text(
                "Select your LLM provider and configure model settings.",
                class_name="text-xs text-gray-400 mb-3",
            ),
            rx.foreach(
                AppState.ai_chat_settings,
                _render_setting,
            ),
            reset_category="llm",
        ),
        # Image Generation section
        _section_card(
            "Image Generation", "image",
            rx.text(
                "Configure the image generation provider for visual responses.",
                class_name="text-xs text-gray-400 mb-3",
            ),
            rx.foreach(
                AppState.ai_image_settings,
                _render_setting,
            ),
        ),
        # System Prompt section
        _section_card(
            "System Prompt", "file-text",
            rx.text(
                "The system prompt sent with every LLM request. "
                "Supports {current_datetime} and {hebrew_date} placeholders.",
                class_name="text-xs text-gray-400 mb-3",
            ),
            rx.foreach(
                AppState.system_prompt_setting,
                _render_setting,
            ),
        ),
        direction="column",
    )


# =========================================================================
# TAB: API KEYS
# =========================================================================


def _keys_tab() -> rx.Component:
    """API Keys tab — all secrets in one card."""
    return rx.flex(
        _section_card(
            "API Keys & Secrets", "key",
            rx.text(
                "API keys are stored encrypted. Leave blank to keep the current value.",
                class_name="text-xs text-gray-400 mb-3",
            ),
            rx.foreach(
                AppState.secrets_settings_list,
                _render_setting,
            ),
            reset_category="secrets",
        ),
        direction="column",
    )


# =========================================================================
# TAB: KNOWLEDGE BASE
# =========================================================================


def _kb_tab() -> rx.Component:
    """Knowledge Base tab — RAG stats + retrieval settings + scoring."""
    return rx.flex(
        # Stats dashboard
        _rag_stats_section(),
        # Retrieval Settings
        _section_card(
            "Retrieval Settings", "search",
            rx.text(
                "Configure how documents are retrieved from the vector store.",
                class_name="text-xs text-gray-400 mb-3",
            ),
            rx.foreach(
                AppState.rag_retrieval_settings,
                _render_setting,
            ),
            reset_category="rag",
        ),
        # Scoring & Ranking (advanced)
        _section_card(
            "Scoring & Ranking", "sliders-horizontal",
            rx.text(
                "Advanced scoring parameters for search result ranking. "
                "Most users won't need to change these.",
                class_name="text-xs text-gray-400 mb-3",
            ),
            rx.foreach(
                AppState.rag_scoring_settings,
                _render_setting,
            ),
        ),
        direction="column",
    )


def _rag_stats_section() -> rx.Component:
    """RAG vector store statistics card."""
    return rx.box(
        rx.box(
            rx.flex(
                rx.flex(
                    rx.icon("bar-chart-3", size=16, class_name="text-gray-500"),
                    rx.text("Vector Store Statistics"),
                    align="center",
                    gap="2",
                ),
                justify="between",
                align="center",
            ),
            class_name="settings-card-header",
        ),
        rx.grid(
            _stat_tile("Total Vectors", AppState.rag_total_docs, None),
            _stat_tile("WhatsApp Messages", AppState.rag_whatsapp_count, "message-circle"),
            _stat_tile("Documents", AppState.rag_document_count, "file-text"),
            _stat_tile("Collection", AppState.rag_collection_name, None),
            columns="2",
            gap="3",
        ),
        rx.cond(
            AppState.rag_dashboard_url != "",
            rx.link(
                rx.flex(
                    rx.icon("external-link", size=14),
                    rx.text("Open Qdrant Dashboard", class_name="text-sm"),
                    align="center",
                    gap="2",
                ),
                href=AppState.rag_dashboard_url,
                is_external=True,
                class_name="text-accent mt-3 inline-flex",
            ),
            rx.fragment(),
        ),
        class_name="settings-card",
    )


def _stat_tile(
    label: str, value: rx.Var, icon_name: str | None,
) -> rx.Component:
    """Small stat tile for the RAG stats grid."""
    value_row = (
        rx.flex(
            rx.icon(icon_name, size=18, class_name="text-accent"),  # type: ignore[arg-type]
            rx.text(value, class_name="text-2xl font-semibold text-gray-800"),
            align="center",
            gap="2",
            class_name="mt-1",
        )
        if icon_name
        else rx.text(
            value,
            class_name="text-2xl font-semibold text-gray-800 mt-1",
        )
    )
    return rx.box(
        rx.text(
            label,
            class_name="text-xs text-gray-400 uppercase tracking-wider",
        ),
        value_row,
        class_name="bg-gray-50 border border-gray-200 rounded-lg px-4 py-3",
    )


# =========================================================================
# TAB: PLUGINS
# =========================================================================


def _plugins_tab() -> rx.Component:
    """Plugins tab — toggles + per-plugin accordion-style config."""
    return rx.flex(
        # Plugin enable/disable toggles
        rx.cond(
            AppState.plugins_toggle_list.length() > 0,  # type: ignore[union-attr]
            _section_card(
                "Enable / Disable Plugins", "power",
                rx.foreach(
                    AppState.plugins_toggle_list,
                    _render_setting,
                ),
            ),
            rx.fragment(),
        ),
        # Per-plugin configuration
        rx.cond(
            AppState.plugin_categories.length() > 0,  # type: ignore[union-attr]
            rx.box(
                # Plugin selector pills
                rx.flex(
                    rx.foreach(
                        AppState.plugin_categories,
                        _plugin_pill,
                    ),
                    gap="2",
                    class_name="mb-3",
                    wrap="wrap",
                ),
                # Active plugin settings in a card
                _section_card(
                    "Plugin Configuration", "settings",
                    rx.foreach(
                        AppState.active_plugin_settings,
                        _render_setting,
                    ),
                    # Paperless actions
                    rx.cond(
                        AppState.active_plugin_tab_value == "paperless",
                        _paperless_actions(),
                        rx.fragment(),
                    ),
                    # Gmail actions
                    rx.cond(
                        AppState.active_plugin_tab_value == "gmail",
                        _gmail_actions(),
                        rx.fragment(),
                    ),
                    # Call Recordings actions
                    rx.cond(
                        AppState.active_plugin_tab_value == "call_recordings",
                        _call_recordings_actions(),
                        rx.fragment(),
                    ),
                ),
            ),
            rx.text(
                "No plugin-specific settings found.",
                class_name="text-sm text-gray-400 italic",
            ),
        ),
        direction="column",
    )


def _plugin_pill(cat: rx.Var[str]) -> rx.Component:
    """Render a plugin selector pill button."""
    return rx.button(
        cat.upper(),  # type: ignore[union-attr]
        on_click=AppState.set_plugin_tab(cat),  # type: ignore[attr-defined]
        variant=rx.cond(
            AppState.active_plugin_tab_value == cat,
            "solid",
            "outline",
        ),
        size="1",
        class_name="capitalize rounded-full",
    )


def _paperless_actions() -> rx.Component:
    """Paperless-NGX test connection and sync buttons."""
    return rx.box(
        rx.flex(
            rx.button(
                rx.icon("wifi", size=14, class_name="mr-1"),
                "Test Connection",
                on_click=AppState.test_paperless_connection,
                loading=AppState.paperless_test_status == "testing",
                size="2",
                class_name="bg-blue-500 text-white hover:bg-blue-600",
            ),
            rx.button(
                rx.icon("refresh-cw", size=14, class_name="mr-1"),
                "Start Sync",
                on_click=AppState.start_paperless_sync,
                loading=AppState.paperless_sync_status == "syncing",
                size="2",
                class_name="bg-green-500 text-white hover:bg-green-600",
            ),
            gap="3",
            align="center",
        ),
        rx.cond(
            AppState.paperless_test_message != "",
            rx.text(
                AppState.paperless_test_message,
                class_name="text-sm mt-2",
            ),
            rx.fragment(),
        ),
        rx.cond(
            AppState.paperless_sync_message != "",
            rx.text(
                AppState.paperless_sync_message,
                class_name="text-sm mt-2",
            ),
            rx.fragment(),
        ),
        class_name="mt-4 pt-4 border-t border-gray-200",
    )


def _call_recordings_actions() -> rx.Component:
    """Call Recordings — scan, upload, and review table."""
    return rx.box(
        # Action buttons row
        rx.flex(
            rx.button(
                rx.icon("search", size=14, class_name="mr-1"),
                "Scan Files",
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
            gap="3",
            align="center",
        ),
        # Status / scan message
        rx.cond(
            AppState.call_recordings_scan_message != "",
            rx.text(
                AppState.call_recordings_scan_message,
                class_name="text-sm mt-2",
            ),
            rx.fragment(),
        ),
        # Upload area (compact)
        rx.box(
            rx.upload(
                rx.flex(
                    rx.icon("upload-cloud", size=20, class_name="text-gray-400"),
                    rx.text(
                        "Drop audio files here or click to browse",
                        class_name="text-sm text-gray-500",
                    ),
                    align="center",
                    gap="2",
                    class_name="py-3",
                ),
                id="call_recordings_upload",
                accept={
                    "audio/*": [
                        ".mp3", ".wav", ".m4a", ".ogg", ".flac",
                        ".wma", ".aac", ".opus", ".webm",
                    ],
                },
                max_files=20,
                on_drop=AppState.upload_call_recordings(  # type: ignore[arg-type]
                    rx.upload_files(upload_id="call_recordings_upload"),
                ),
                border="2px dashed",
                border_color="gray.200",
                border_radius="lg",
                class_name="w-full cursor-pointer hover:border-gray-400 transition-colors",
            ),
            rx.cond(
                AppState.call_recordings_upload_message != "",
                rx.text(
                    AppState.call_recordings_upload_message,
                    class_name="text-sm mt-2",
                ),
                rx.fragment(),
            ),
            class_name="mt-3",
        ),
        # Recordings table with filters
        rx.box(
            rx.flex(
                rx.text(
                    "Recordings",
                    class_name="text-sm font-medium text-gray-700",
                ),
                # Filter controls
                rx.el.input(
                    type="text",
                    placeholder="Search by name, phone…",
                    value=AppState.call_recordings_filter_name,
                    on_change=AppState.set_call_recordings_filter_name,  # type: ignore[arg-type]
                    class_name=(
                        "text-xs bg-gray-50 border border-gray-200 rounded "
                        "px-2 py-1 w-40 outline-none focus:border-accent"
                    ),
                ),
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
                    size="1",
                    variant="ghost",
                    class_name="text-xs",
                ),
                align="center",
                gap="3",
                class_name="mb-2",
            ),
            rx.cond(
                AppState.call_recordings_files_loading,
                rx.flex(
                    rx.spinner(size="2"),
                    rx.text("Loading…", class_name="text-sm text-gray-400 ml-2"),
                    align="center",
                    class_name="py-4",
                ),
                rx.cond(
                    AppState.filtered_recording_files.length() > 0,  # type: ignore[union-attr]
                    rx.box(
                        rx.foreach(
                            AppState.filtered_recording_files,
                            _recording_row,
                        ),
                    ),
                    rx.text(
                        "No recordings found. Upload files or click "
                        "'Scan Files' to discover recordings.",
                        class_name="text-sm text-gray-400 italic py-4",
                    ),
                ),
            ),
            class_name="mt-4 pt-4 border-t border-gray-100",
        ),
        class_name="mt-4 pt-4 border-t border-gray-200",
        on_mount=AppState.load_recording_files,
    )


def _status_badge(status: rx.Var[str]) -> rx.Component:
    """Colored status badge for a recording file."""
    return rx.box(
        rx.text(
            status,
            class_name="text-xs font-medium capitalize",
        ),
        class_name=rx.cond(
            status == "approved",
            "px-2 py-0.5 rounded-full bg-green-100 text-green-700 inline-block",
            rx.cond(
                status == "transcribed",
                "px-2 py-0.5 rounded-full bg-blue-100 text-blue-700 inline-block",
                rx.cond(
                    status == "error",
                    "px-2 py-0.5 rounded-full bg-red-100 text-red-700 inline-block",
                    rx.cond(
                        status == "transcribing",
                        "px-2 py-0.5 rounded-full bg-yellow-100 text-yellow-700 inline-block",
                        "px-2 py-0.5 rounded-full bg-gray-100 text-gray-600 inline-block",
                    ),
                ),
            ),
        ),
    )


def _recording_row(item: dict) -> rx.Component:
    """Render a single recording file as a card row."""
    return rx.box(
        rx.flex(
            # Left: status + filename + metadata
            rx.box(
                rx.flex(
                    _status_badge(item["status"]),
                    rx.text(
                        item["filename"],
                        class_name="text-sm font-medium text-gray-800 truncate",
                    ),
                    rx.cond(
                        item["duration_seconds"] != "0",
                        rx.text(
                            rx.cond(
                                item["duration_seconds"] != "",
                                item["duration_seconds"] + "s",
                                "",
                            ),
                            class_name="text-xs text-gray-400",
                        ),
                        rx.fragment(),
                    ),
                    rx.cond(
                        item["language"] != "",
                        rx.text(
                            item["language"],
                            class_name="text-xs text-gray-400 uppercase",
                        ),
                        rx.fragment(),
                    ),
                    align="center",
                    gap="2",
                    wrap="wrap",
                ),
                # Editable metadata row
                rx.flex(
                    rx.el.input(
                        type="text",
                        placeholder="Contact name…",
                        default_value=item["contact_name"],
                        on_blur=AppState.save_recording_metadata(
                            item["content_hash"], "contact_name",
                        ),
                        class_name=(
                            "text-xs bg-gray-50 border border-gray-200 rounded "
                            "px-2 py-1 w-36 outline-none focus:border-accent"
                        ),
                    ),
                    rx.el.input(
                        type="text",
                        placeholder="Phone…",
                        default_value=item["phone_number"],
                        on_blur=AppState.save_recording_metadata(
                            item["content_hash"], "phone_number",
                        ),
                        class_name=(
                            "text-xs bg-gray-50 border border-gray-200 rounded "
                            "px-2 py-1 w-32 outline-none focus:border-accent"
                        ),
                    ),
                    gap="2",
                    class_name="mt-1",
                ),
                # Transcript preview (expandable)
                rx.cond(
                    item["transcript_text"] != "",
                    rx.el.details(
                        rx.el.summary(
                            rx.text(
                                item["transcript_text"][:200],  # type: ignore[index]
                                class_name="text-xs text-gray-500 line-clamp-2 inline",
                            ),
                            class_name=(
                                "text-xs text-blue-500 cursor-pointer "
                                "hover:text-blue-700 mt-1 list-none "
                                "[&::-webkit-details-marker]:hidden"
                            ),
                        ),
                        rx.box(
                            rx.text(
                                item["transcript_text"],
                                class_name="text-xs text-gray-600 whitespace-pre-wrap",
                            ),
                            class_name=(
                                "mt-2 p-2 bg-gray-50 rounded border border-gray-200 "
                                "max-h-64 overflow-y-auto"
                            ),
                        ),
                        class_name="mt-1",
                    ),
                    rx.fragment(),
                ),
                # Error message
                rx.cond(
                    item["error_message"] != "",
                    rx.text(
                        item["error_message"],
                        class_name="text-xs text-red-500 mt-1",
                    ),
                    rx.fragment(),
                ),
                class_name="flex-1 min-w-0",
            ),
            # Right: action buttons
            rx.flex(
                # Approve button (only for transcribed)
                rx.cond(
                    item["status"] == "transcribed",
                    rx.icon_button(
                        rx.icon("check", size=16),
                        on_click=AppState.approve_recording(item["content_hash"]),
                        variant="ghost",
                        size="1",
                        class_name="text-green-500 hover:text-green-700",
                        title="Approve & index",
                    ),
                    rx.fragment(),
                ),
                # Transcribe / retranscribe button (all statuses except transcribing)
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
                # Delete button (always)
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
                direction="column",
                gap="1",
                align="center",
                class_name="shrink-0 ml-2",
            ),
            align="start",
            gap="3",
        ),
        class_name=(
            "px-3 py-3 border-b border-gray-100 last:border-b-0 "
            "hover:bg-gray-50 transition-colors"
        ),
    )


# =========================================================================
# TAB: SYSTEM (merged Infrastructure + App)
# =========================================================================


def _system_tab() -> rx.Component:
    """System tab — connections + application settings."""
    return rx.flex(
        rx.text(
            "Server addresses and application behaviour. "
            "Connection changes may require a restart.",
            class_name="text-sm text-gray-400 mb-3",
        ),
        # Connections section
        _section_card(
            "Connections", "server",
            rx.foreach(
                AppState.connections_settings_list,
                _render_setting,
            ),
            reset_category="infrastructure",
        ),
        # Application section
        _section_card(
            "Application", "cog",
            rx.foreach(
                AppState.application_settings_list,
                _render_setting,
            ),
            reset_category="app",
        ),
        # Danger Zone
        rx.box(
            rx.flex(
                rx.icon("alert-triangle", size=18, class_name="text-red-500"),
                rx.text(
                    "Danger Zone",
                    class_name="text-sm font-semibold text-red-600",
                ),
                align="center",
                gap="2",
                class_name="mb-3",
            ),
            rx.flex(
                rx.box(
                    rx.text(
                        "Delete all chat conversations",
                        class_name="text-sm font-medium text-gray-700",
                    ),
                    rx.text(
                        "Permanently remove all conversations and chat history from the database. "
                        "This does not affect the RAG knowledge base.",
                        class_name="text-xs text-gray-400 mt-0.5",
                    ),
                    class_name="flex-1",
                ),
                rx.button(
                    rx.icon("trash-2", size=14, class_name="mr-1"),
                    "Delete All Chats",
                    on_click=AppState.delete_all_chats,
                    size="2",
                    variant="outline",
                    color_scheme="red",
                    class_name="shrink-0",
                ),
                align="center",
                gap="4",
            ),
            class_name=(
                "bg-white rounded-xl border border-red-200 shadow-sm p-4 mt-4"
            ),
        ),
        direction="column",
    )


# =========================================================================
# TAB: COSTS
# =========================================================================


def _costs_tab() -> rx.Component:
    """Cost tracking dashboard tab."""
    from .cost_display import cost_dashboard

    return cost_dashboard()


# =========================================================================
# SHARED: SETTING RENDERER
# =========================================================================


def _render_setting(item: dict) -> rx.Component:
    """Render a single setting based on its type.

    Handles: text, secret, bool, select, int, float.
    System prompt renders as a textarea.
    """
    return rx.box(
        # Label + description tooltip
        rx.flex(
            rx.text(
                item["label"],
                class_name="text-sm font-medium text-gray-700",
            ),
            rx.cond(
                item["description"] != "",
                rx.tooltip(
                    rx.icon(
                        "info",
                        size=14,
                        class_name="text-gray-400 cursor-help",
                    ),
                    content=item["description"],
                ),
                rx.fragment(),
            ),
            align="center",
            gap="2",
            class_name="mb-1",
        ),
        # Input — branch by setting_type (with special-case overrides)
        rx.cond(
            item["key"] == "paperless_sync_tags",
            _paperless_tags_input(),
            rx.cond(
                item["key"] == "gmail_sync_folders",
                _gmail_folders_input(),
                rx.cond(
                    item["setting_type"] == "bool",
                    _bool_input(item),
                    rx.cond(
                        item["setting_type"] == "select",
                        _select_input(item),
                        rx.cond(
                            item["setting_type"] == "secret",
                            _secret_input(item),
                            rx.cond(
                                item["key"] == "system_prompt",
                                _textarea_input(item),
                                _text_input(item),
                            ),
                        ),
                    ),
                ),
            ),
        ),
        class_name="settings-field",
    )


# =========================================================================
# SHARED: INPUT WIDGETS
# =========================================================================

_INPUT_CLASS = (
    "w-full bg-white border border-gray-200 rounded-lg "
    "px-3 py-2 text-sm text-gray-700 "
    "outline-none focus:border-accent"
)


def _bool_input(item: dict) -> rx.Component:
    """Boolean toggle switch."""
    return rx.flex(
        rx.switch(
            checked=item["value"] == "true",
            on_change=AppState.save_setting(
                item["key"],
                rx.cond(item["value"] == "true", "false", "true"),
            ),
        ),
        rx.text(
            rx.cond(item["value"] == "true", "Enabled", "Disabled"),
            class_name="text-sm text-gray-500 ml-2",
        ),
        align="center",
    )


def _select_input(item: dict) -> rx.Component:
    """Dropdown select for enum-type settings."""
    return rx.select(
        item["options"].split("|"),  # type: ignore[union-attr]
        value=item["value"],
        on_change=AppState.save_setting(item["key"]),
        size="2",
        class_name="w-full",
    )


def _secret_input(item: dict) -> rx.Component:
    """Password input for secret values with eye toggle and save button."""
    is_revealed = AppState.revealed_secrets.contains(item["key"])
    has_value = item["value"] != ""

    return rx.flex(
        rx.cond(
            is_revealed,
            rx.el.input(
                type="text",
                placeholder="Enter new value…",
                default_value=AppState.revealed_secret_values[item["key"]],
                on_change=AppState.set_pending_change(item["key"]),  # type: ignore[arg-type]
                class_name=_INPUT_CLASS + " flex-1",
            ),
            rx.el.input(
                type="password",
                placeholder=rx.cond(has_value, item["value"], "Enter new value…"),
                on_change=AppState.set_pending_change(item["key"]),  # type: ignore[arg-type]
                class_name=_INPUT_CLASS + " flex-1",
            ),
        ),
        rx.icon_button(
            rx.cond(
                is_revealed,
                rx.icon("eye-off", size=16),
                rx.icon("eye", size=16),
            ),
            on_click=AppState.toggle_secret_visibility(item["key"]),
            variant="ghost",
            size="1",
            class_name="text-gray-400 hover:text-gray-600 shrink-0",
        ),
        rx.cond(
            AppState.pending_changes.contains(item["key"]),
            rx.button(
                rx.icon("save", size=14, class_name="mr-1"),
                "Save",
                on_click=AppState.save_pending_change(item["key"]),
                size="1",
                class_name="bg-green-500 text-white hover:bg-green-600 shrink-0",
            ),
            rx.fragment(),
        ),
        align="center",
        gap="2",
    )


def _text_input(item: dict) -> rx.Component:
    """Text input with explicit save button."""
    return rx.flex(
        rx.el.input(
            type="text",
            default_value=item["value"],
            on_change=AppState.set_pending_change(item["key"]),  # type: ignore[arg-type]
            class_name=_INPUT_CLASS + " flex-1",
        ),
        rx.cond(
            AppState.pending_changes.contains(item["key"]),
            rx.button(
                rx.icon("save", size=14, class_name="mr-1"),
                "Save",
                on_click=AppState.save_pending_change(item["key"]),
                size="1",
                class_name="bg-green-500 text-white hover:bg-green-600 shrink-0",
            ),
            rx.fragment(),
        ),
        align="center",
        gap="2",
    )


def _textarea_input(item: dict) -> rx.Component:
    """Textarea with explicit save button."""
    return rx.flex(
        rx.el.textarea(
            default_value=item["value"],
            rows=8,
            on_change=AppState.set_pending_change(item["key"]),  # type: ignore[arg-type]
            class_name=_INPUT_CLASS + " resize-y flex-1",
        ),
        rx.cond(
            AppState.pending_changes.contains(item["key"]),
            rx.button(
                rx.icon("save", size=14, class_name="mr-1"),
                "Save",
                on_click=AppState.save_pending_change(item["key"]),
                size="1",
                class_name="bg-green-500 text-white hover:bg-green-600 shrink-0 self-start",
            ),
            rx.fragment(),
        ),
        align="start",
        gap="2",
    )


# =========================================================================
# PAPERLESS TAGS — multi-select with bubble display
# =========================================================================


def _paperless_tag_bubble(tag: rx.Var[dict]) -> rx.Component:
    """Render a single selected tag as a removable bubble with × inside."""
    return rx.box(
        rx.flex(
            rx.icon(
                "x",
                size=14,
                class_name="shrink-0 cursor-pointer opacity-60 hover:opacity-100",
                on_click=AppState.remove_paperless_tag(tag["name"]),
            ),
            rx.text(
                tag["name"],
                class_name="text-sm font-medium",
            ),
            align="center",
            gap="1.5",
        ),
        class_name="px-3 py-1 rounded-full border inline-flex",
        style={
            "background_color": rx.cond(
                tag["color"] != "",
                tag["color"] + "30",  # 30 = ~19% opacity hex suffix
                "#a6cee330",
            ),
            "border_color": rx.cond(
                tag["color"] != "",
                tag["color"] + "80",  # 80 = ~50% opacity
                "#a6cee380",
            ),
            "color": "#374151",
        },
    )


def _paperless_tag_option(tag: rx.Var[dict]) -> rx.Component:
    """Render a single tag option in the dropdown list."""
    return rx.box(
        rx.flex(
            rx.box(
                class_name="w-3 h-3 rounded-full shrink-0",
                style={
                    "background_color": rx.cond(
                        tag["color"] != "",
                        tag["color"],
                        "#a6cee3",
                    ),
                },
            ),
            rx.text(
                tag["name"],
                class_name="text-sm text-gray-700",
            ),
            align="center",
            gap="2",
        ),
        on_click=AppState.add_paperless_tag(tag["name"]),
        class_name=(
            "px-3 py-2 cursor-pointer hover:bg-gray-50 "
            "border-b border-gray-100 last:border-b-0"
        ),
    )


def _paperless_tags_input() -> rx.Component:
    """Multi-select tag input with bubble display for paperless_sync_tags.

    Shows selected tags as colored bubbles with × remove buttons.
    A dropdown chevron fetches and shows all available tags from Paperless-NGX.
    """
    return rx.box(
        rx.flex(
            # Selected tags area (bubbles)
            rx.box(
                rx.cond(
                    AppState.paperless_selected_tags.length() > 0,  # type: ignore[union-attr]
                    rx.flex(
                        rx.foreach(
                            AppState.paperless_selected_tag_items,
                            _paperless_tag_bubble,
                        ),
                        direction="column",
                        gap="2",
                        class_name="py-2 px-2",
                    ),
                    rx.text(
                        "No tags selected (all documents will sync)",
                        class_name="text-sm text-gray-400 italic py-2 px-3",
                    ),
                ),
                class_name="flex-1 min-w-0",
            ),
            # Right side: clear all + dropdown toggle
            rx.flex(
                rx.cond(
                    AppState.paperless_selected_tags.length() > 0,  # type: ignore[union-attr]
                    rx.icon_button(
                        rx.icon("x", size=14),
                        on_click=AppState.clear_all_paperless_tags,
                        variant="ghost",
                        size="1",
                        class_name="text-gray-400 hover:text-gray-600",
                    ),
                    rx.fragment(),
                ),
                rx.icon_button(
                    rx.icon("chevron-down", size=16),
                    on_click=AppState.load_paperless_tags,
                    variant="ghost",
                    size="1",
                    class_name="text-gray-400 hover:text-gray-600",
                ),
                align="center",
                gap="1",
                class_name="shrink-0 pr-1",
            ),
            align="start",
            justify="between",
            class_name=(
                "w-full bg-white border border-gray-200 rounded-lg "
                "min-h-[42px]"
            ),
        ),
        # Dropdown list of available tags
        rx.cond(
            AppState.paperless_tag_dropdown_open,
            rx.box(
                rx.cond(
                    AppState.paperless_tags_loading,
                    rx.flex(
                        rx.spinner(size="2"),
                        rx.text("Loading tags…", class_name="text-sm text-gray-400 ml-2"),
                        align="center",
                        class_name="px-3 py-3",
                    ),
                    rx.cond(
                        AppState.paperless_unselected_tags.length() > 0,  # type: ignore[union-attr]
                        rx.box(
                            rx.foreach(
                                AppState.paperless_unselected_tags,
                                _paperless_tag_option,
                            ),
                            class_name="max-h-[200px] overflow-y-auto",
                        ),
                        rx.text(
                            "No more tags available",
                            class_name="text-sm text-gray-400 italic px-3 py-2",
                        ),
                    ),
                ),
                class_name=(
                    "mt-1 bg-white border border-gray-200 rounded-lg shadow-lg "
                    "overflow-hidden z-50"
                ),
            ),
            rx.fragment(),
        ),
        class_name="relative w-full",
    )


# =========================================================================
# GMAIL ACTIONS — auth, test, sync
# =========================================================================


def _gmail_actions() -> rx.Component:
    """Gmail authorization, test connection, and sync buttons."""
    return rx.box(
        # Auth section
        rx.flex(
            rx.button(
                rx.icon("key-round", size=14, class_name="mr-1"),
                "Authorize Gmail",
                on_click=AppState.gmail_start_auth,
                loading=AppState.gmail_auth_status == "pending",
                size="2",
                class_name="bg-blue-500 text-white hover:bg-blue-600",
            ),
            rx.button(
                rx.icon("wifi", size=14, class_name="mr-1"),
                "Test Connection",
                on_click=AppState.gmail_test_connection,
                loading=AppState.gmail_test_status == "testing",
                size="2",
                class_name="bg-blue-500 text-white hover:bg-blue-600",
            ),
            rx.button(
                rx.icon("refresh-cw", size=14, class_name="mr-1"),
                "Start Sync",
                on_click=AppState.start_gmail_sync,
                loading=AppState.gmail_sync_status == "syncing",
                size="2",
                class_name="bg-green-500 text-white hover:bg-green-600",
            ),
            gap="3",
            align="center",
            wrap="wrap",
        ),
        # Auth URL display (when authorization is pending)
        rx.cond(
            AppState.gmail_auth_url != "",
            rx.box(
                rx.text(
                    "Open this URL in your browser to authorize:",
                    class_name="text-sm text-gray-600 mb-2",
                ),
                rx.box(
                    rx.text(
                        AppState.gmail_auth_url,
                        class_name="text-xs text-blue-600 break-all",
                    ),
                    class_name="bg-gray-50 border border-gray-200 rounded p-2 mb-3",
                ),
                rx.text(
                    "After approving, paste the authorization code below:",
                    class_name="text-sm text-gray-600 mb-2",
                ),
                rx.flex(
                    rx.el.input(
                        type="text",
                        placeholder="Paste authorization code here…",
                        on_change=AppState.set_gmail_auth_code_input,
                        class_name=(
                            "flex-1 bg-white border border-gray-200 rounded-lg "
                            "px-3 py-2 text-sm text-gray-700 "
                            "outline-none focus:border-accent"
                        ),
                    ),
                    rx.button(
                        rx.icon("check", size=14, class_name="mr-1"),
                        "Submit Code",
                        on_click=AppState.gmail_submit_auth_code,
                        size="2",
                        class_name="bg-green-500 text-white hover:bg-green-600 shrink-0",
                    ),
                    gap="2",
                    align="center",
                ),
                class_name="mt-3",
            ),
            rx.fragment(),
        ),
        # Auth status message
        rx.cond(
            AppState.gmail_auth_message != "",
            rx.text(
                AppState.gmail_auth_message,
                class_name="text-sm mt-2",
            ),
            rx.fragment(),
        ),
        # Test status message
        rx.cond(
            AppState.gmail_test_message != "",
            rx.text(
                AppState.gmail_test_message,
                class_name="text-sm mt-2",
            ),
            rx.fragment(),
        ),
        # Sync status message
        rx.cond(
            AppState.gmail_sync_message != "",
            rx.text(
                AppState.gmail_sync_message,
                class_name="text-sm mt-2",
            ),
            rx.fragment(),
        ),
        class_name="mt-4 pt-4 border-t border-gray-200",
    )


# =========================================================================
# GMAIL FOLDERS — multi-select with bubble display
# =========================================================================


def _gmail_folder_bubble(folder: rx.Var[dict]) -> rx.Component:
    """Render a single selected folder as a removable bubble."""
    return rx.box(
        rx.flex(
            rx.icon(
                "x",
                size=14,
                class_name="shrink-0 cursor-pointer opacity-60 hover:opacity-100",
                on_click=AppState.remove_gmail_folder(folder["name"]),
            ),
            rx.text(
                folder["name"],
                class_name="text-sm font-medium",
            ),
            align="center",
            gap="1.5",
        ),
        class_name="px-3 py-1 rounded-full border inline-flex",
        style={
            "background_color": rx.cond(
                folder["type"] == "system",
                "#dbeafe",   # Light blue for system folders
                "#f3e8ff",   # Light purple for user labels
            ),
            "border_color": rx.cond(
                folder["type"] == "system",
                "#93c5fd",
                "#c4b5fd",
            ),
            "color": "#374151",
        },
    )


def _gmail_folder_option(folder: rx.Var[dict]) -> rx.Component:
    """Render a single folder option in the dropdown list."""
    return rx.box(
        rx.flex(
            rx.icon(
                rx.cond(
                    folder["type"] == "system",
                    "folder",
                    "tag",
                ),
                size=14,
                class_name="text-gray-400 shrink-0",
            ),
            rx.text(
                folder["name"],
                class_name="text-sm text-gray-700",
            ),
            align="center",
            gap="2",
        ),
        on_click=AppState.add_gmail_folder(folder["name"]),
        class_name=(
            "px-3 py-2 cursor-pointer hover:bg-gray-50 "
            "border-b border-gray-100 last:border-b-0"
        ),
    )


def _gmail_folders_input() -> rx.Component:
    """Multi-select folder input with bubble display for gmail_sync_folders.

    Shows selected folders as colored bubbles with × remove buttons.
    A dropdown chevron fetches and shows all available Gmail labels.
    """
    return rx.box(
        rx.flex(
            # Selected folders area (bubbles)
            rx.box(
                rx.cond(
                    AppState.gmail_selected_folders.length() > 0,  # type: ignore[union-attr]
                    rx.flex(
                        rx.foreach(
                            AppState.gmail_selected_folder_items,
                            _gmail_folder_bubble,
                        ),
                        direction="column",
                        gap="2",
                        class_name="py-2 px-2",
                    ),
                    rx.text(
                        "No folders selected (defaults to INBOX)",
                        class_name="text-sm text-gray-400 italic py-2 px-3",
                    ),
                ),
                class_name="flex-1 min-w-0",
            ),
            # Right side: clear all + dropdown toggle
            rx.flex(
                rx.cond(
                    AppState.gmail_selected_folders.length() > 0,  # type: ignore[union-attr]
                    rx.icon_button(
                        rx.icon("x", size=14),
                        on_click=AppState.clear_all_gmail_folders,
                        variant="ghost",
                        size="1",
                        class_name="text-gray-400 hover:text-gray-600",
                    ),
                    rx.fragment(),
                ),
                rx.icon_button(
                    rx.icon("chevron-down", size=16),
                    on_click=AppState.load_gmail_folders,
                    variant="ghost",
                    size="1",
                    class_name="text-gray-400 hover:text-gray-600",
                ),
                align="center",
                gap="1",
                class_name="shrink-0 pr-1",
            ),
            align="start",
            justify="between",
            class_name=(
                "w-full bg-white border border-gray-200 rounded-lg "
                "min-h-[42px]"
            ),
        ),
        # Dropdown list of available folders
        rx.cond(
            AppState.gmail_folder_dropdown_open,
            rx.box(
                rx.cond(
                    AppState.gmail_folders_loading,
                    rx.flex(
                        rx.spinner(size="2"),
                        rx.text("Loading folders…", class_name="text-sm text-gray-400 ml-2"),
                        align="center",
                        class_name="px-3 py-3",
                    ),
                    rx.cond(
                        AppState.gmail_unselected_folders.length() > 0,  # type: ignore[union-attr]
                        rx.box(
                            rx.foreach(
                                AppState.gmail_unselected_folders,
                                _gmail_folder_option,
                            ),
                            class_name="max-h-[200px] overflow-y-auto",
                        ),
                        rx.text(
                            "No more folders available",
                            class_name="text-sm text-gray-400 italic px-3 py-2",
                        ),
                    ),
                ),
                class_name=(
                    "mt-1 bg-white border border-gray-200 rounded-lg shadow-lg "
                    "overflow-hidden z-50"
                ),
            ),
            rx.fragment(),
        ),
        class_name="relative w-full",
    )
