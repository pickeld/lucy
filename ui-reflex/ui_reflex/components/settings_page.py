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
        # Input — branch by setting_type
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
