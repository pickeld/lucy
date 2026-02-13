"""Settings page — tabbed configuration management.

Replaces the flat scrollable list with a modern tabbed interface.
Six main tabs: LLM, Keys, RAG, Plugins, Infra, App.
Nested plugin sub-tabs for per-plugin configuration.
"""

import reflex as rx

from ..state import AppState


# =========================================================================
# MAIN SETTINGS PAGE
# =========================================================================


def settings_page() -> rx.Component:
    """Full settings page with tabbed interface."""
    return rx.box(
        rx.flex(
            # Header with back button + Export/Import
            _header(),
            # Status message
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
            # Health overview
            _health_section(),
            # Main tabbed interface
            rx.tabs.root(
                rx.tabs.list(
                    rx.tabs.trigger("🤖 LLM", value="llm"),
                    rx.tabs.trigger("🔑 Keys", value="keys"),
                    rx.tabs.trigger("🔍 RAG", value="rag"),
                    rx.tabs.trigger("🔌 Plugins", value="plugins"),
                    rx.tabs.trigger("🏗️ Infra", value="infra"),
                    rx.tabs.trigger("🔧 App", value="app"),
                    size="2",
                ),
                rx.tabs.content(_llm_tab(), value="llm", class_name="pt-4"),
                rx.tabs.content(_keys_tab(), value="keys", class_name="pt-4"),
                rx.tabs.content(_rag_tab(), value="rag", class_name="pt-4"),
                rx.tabs.content(_plugins_tab(), value="plugins", class_name="pt-4"),
                rx.tabs.content(_infra_tab(), value="infra", class_name="pt-4"),
                rx.tabs.content(_app_tab(), value="app", class_name="pt-4"),
                default_value="llm",
                class_name="mt-4",
            ),
            direction="column",
            class_name="max-w-[820px] mx-auto w-full px-4 py-6",
        ),
        class_name="h-full overflow-y-auto chat-scroll",
    )


# =========================================================================
# HEADER
# =========================================================================


def _header() -> rx.Component:
    """Header with back button, title, and Export/Import buttons."""
    return rx.flex(
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
        rx.flex(
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
        class_name="mb-2",
    )


# =========================================================================
# HEALTH SECTION (unchanged from original)
# =========================================================================


def _health_section() -> rx.Component:
    """System health dashboard."""
    return rx.box(
        rx.flex(
            rx.icon("activity", size=18, class_name="text-gray-500"),
            rx.heading("System Health", size="4", class_name="text-gray-700"),
            align="center",
            gap="2",
            class_name="mb-3",
        ),
        rx.box(
            rx.flex(
                rx.box(
                    class_name=rx.cond(
                        AppState.api_status == "up",
                        "w-3 h-3 rounded-full bg-status-green",
                        rx.cond(
                            AppState.api_status == "degraded",
                            "w-3 h-3 rounded-full bg-status-yellow",
                            "w-3 h-3 rounded-full bg-status-red",
                        ),
                    ),
                ),
                rx.text(
                    AppState.health_label,
                    class_name="font-medium text-sm text-gray-700",
                ),
                align="center",
                gap="2",
            ),
            class_name="bg-gray-50 border border-gray-200 rounded-lg px-4 py-3",
        ),
        class_name="mb-6 pb-6 border-b border-gray-100",
    )


# =========================================================================
# TAB CONTENT: LLM
# =========================================================================


def _llm_tab() -> rx.Component:
    """LLM configuration tab — providers, models, system prompt."""
    return rx.flex(
        _tab_header("🤖 LLM Configuration", "llm"),
        rx.text(
            "Configure your LLM and image generation providers. "
            "Only settings for the selected provider are shown.",
            class_name="text-sm text-gray-400 mb-4",
        ),
        rx.foreach(
            AppState.llm_settings_list,
            _render_setting,
        ),
        direction="column",
    )


# =========================================================================
# TAB CONTENT: KEYS
# =========================================================================


def _keys_tab() -> rx.Component:
    """API Keys & Secrets tab."""
    return rx.flex(
        _tab_header("🔑 API Keys & Secrets", "secrets"),
        rx.text(
            "API keys are stored encrypted. Leave blank to keep the current value.",
            class_name="text-sm text-gray-400 mb-4",
        ),
        rx.foreach(
            AppState.secrets_settings_list,
            _render_setting,
        ),
        direction="column",
    )


# =========================================================================
# TAB CONTENT: RAG
# =========================================================================


def _rag_tab() -> rx.Component:
    """RAG configuration tab — stats + settings."""
    return rx.flex(
        _tab_header("🔍 RAG Configuration", "rag"),
        # RAG Statistics section
        _rag_stats_section(),
        # RAG settings
        rx.foreach(
            AppState.rag_settings_list,
            _render_setting,
        ),
        direction="column",
    )


def _rag_stats_section() -> rx.Component:
    """RAG vector store statistics — real data from rag_stats."""
    return rx.box(
        rx.flex(
            rx.icon("bar-chart-3", size=16, class_name="text-gray-500"),
            rx.text(
                "Vector Store Statistics",
                class_name="text-sm font-medium text-gray-600",
            ),
            align="center",
            gap="2",
            class_name="mb-3",
        ),
        rx.grid(
            rx.box(
                rx.text(
                    "Total Documents",
                    class_name="text-xs text-gray-400 uppercase tracking-wider",
                ),
                rx.text(
                    AppState.rag_total_docs,
                    class_name="text-2xl font-semibold text-gray-800 mt-1",
                ),
                class_name="bg-gray-50 border border-gray-200 rounded-lg px-4 py-3",
            ),
            rx.box(
                rx.text(
                    "Collection",
                    class_name="text-xs text-gray-400 uppercase tracking-wider",
                ),
                rx.text(
                    AppState.rag_collection_name,
                    class_name="text-2xl font-semibold text-gray-800 mt-1",
                ),
                class_name="bg-gray-50 border border-gray-200 rounded-lg px-4 py-3",
            ),
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
                class_name="text-accent mt-2 inline-flex",
            ),
            rx.fragment(),
        ),
        class_name="mb-6 pb-4 border-b border-gray-100",
    )


# =========================================================================
# TAB CONTENT: PLUGINS
# =========================================================================


def _plugins_tab() -> rx.Component:
    """Plugins tab — toggles + per-plugin sub-tabs."""
    return rx.flex(
        _tab_header("🔌 Plugins", "plugins"),
        # Plugin enable/disable toggles
        rx.cond(
            AppState.plugins_toggle_list.length() > 0,  # type: ignore[union-attr]
            rx.box(
                rx.text(
                    "Enable / Disable",
                    class_name="text-sm font-medium text-gray-600 mb-2",
                ),
                rx.foreach(
                    AppState.plugins_toggle_list,
                    _render_setting,
                ),
                class_name="mb-4 pb-4 border-b border-gray-100",
            ),
            rx.fragment(),
        ),
        # Plugin sub-tabs (per-plugin configuration)
        rx.cond(
            AppState.plugin_categories.length() > 0,  # type: ignore[union-attr]
            rx.box(
                rx.text(
                    "Plugin Configuration",
                    class_name="text-sm font-medium text-gray-600 mb-3",
                ),
                # Sub-tab buttons
                rx.flex(
                    rx.foreach(
                        AppState.plugin_categories,
                        _plugin_tab_button,
                    ),
                    gap="2",
                    class_name="mb-4",
                    wrap="wrap",
                ),
                # Active plugin settings
                rx.foreach(
                    AppState.active_plugin_settings,
                    _render_setting,
                ),
                # Paperless test connection button
                rx.cond(
                    AppState.active_plugin_tab_value == "paperless",
                    rx.box(
                        rx.button(
                            rx.icon("wifi", size=14, class_name="mr-1"),
                            "Test Connection",
                            on_click=AppState.test_paperless_connection,
                            loading=AppState.paperless_test_status == "testing",
                            size="2",
                            class_name="bg-blue-500 text-white hover:bg-blue-600",
                        ),
                        rx.cond(
                            AppState.paperless_test_message != "",
                            rx.text(
                                AppState.paperless_test_message,
                                class_name="text-sm mt-2",
                            ),
                            rx.fragment(),
                        ),
                        class_name="mt-4 pt-4 border-t border-gray-200",
                    ),
                    rx.fragment(),
                ),
            ),
            rx.text(
                "No plugin-specific settings found.",
                class_name="text-sm text-gray-400 italic",
            ),
        ),
        direction="column",
    )


def _plugin_tab_button(cat: rx.Var[str]) -> rx.Component:
    """Render a plugin sub-tab button."""
    return rx.button(
        cat.upper(),  # type: ignore[union-attr]
        on_click=AppState.set_plugin_tab(cat),  # type: ignore[attr-defined]
        variant=rx.cond(
            AppState.active_plugin_tab_value == cat,
            "solid",
            "outline",
        ),
        size="1",
        class_name="capitalize",
    )


# =========================================================================
# TAB CONTENT: INFRASTRUCTURE
# =========================================================================


def _infra_tab() -> rx.Component:
    """Infrastructure configuration tab."""
    return rx.flex(
        _tab_header("🏗️ Infrastructure", "infrastructure"),
        rx.text(
            "Server addresses and connection settings. "
            "Changes may require a restart to take effect.",
            class_name="text-sm text-gray-400 mb-4",
        ),
        rx.foreach(
            AppState.infra_settings_list,
            _render_setting,
        ),
        direction="column",
    )


# =========================================================================
# TAB CONTENT: APP
# =========================================================================


def _app_tab() -> rx.Component:
    """App configuration tab."""
    return rx.flex(
        _tab_header("🔧 App Configuration", "app"),
        rx.foreach(
            AppState.app_settings_list,
            _render_setting,
        ),
        direction="column",
    )


# =========================================================================
# SHARED: TAB HEADER WITH RESET BUTTON
# =========================================================================


def _tab_header(title: str, category: str) -> rx.Component:
    """Tab section header with reset button."""
    return rx.flex(
        rx.heading(
            title,
            size="4",
            class_name="text-gray-700",
        ),
        rx.button(
            rx.icon("rotate-ccw", size=12, class_name="mr-1"),
            "Reset to Defaults",
            on_click=AppState.reset_category(category),
            variant="ghost",
            size="1",
            class_name="text-gray-400 hover:text-gray-600",
        ),
        justify="between",
        align="center",
        class_name="mb-3 pb-2 border-b border-gray-200",
    )


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
        class_name="py-2",
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
    """Password input for secret values with eye toggle and save button.

    Always shows dots (via type=password) when a value is saved.
    Clicking the eye toggle switches to type=text to reveal the full value.
    """
    return rx.flex(
        rx.el.input(
            type=rx.cond(
                AppState.revealed_secrets.contains(item["key"]),
                "text",
                "password",
            ),
            placeholder="Enter new value…",
            default_value=item["value"],
            on_change=AppState.set_pending_change(item["key"]),  # type: ignore[arg-type]
            class_name=_INPUT_CLASS + " flex-1",
        ),
        rx.icon_button(
            rx.cond(
                AppState.revealed_secrets.contains(item["key"]),
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
