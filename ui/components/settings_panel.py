"""Settings page — full-page view for all application configuration.

Renders all settings from the SQLite database grouped by category,
with proper input widgets based on setting type (text, secret, int,
float, bool, select). Includes plugin enable/disable toggles and
per-plugin configuration sections.

Also includes the original sidebar panels: filters, stats, and health.
"""

from typing import Any, Dict, List, Optional

import streamlit as st

from utils.api import (
    check_health,
    fetch_config,
    fetch_config_meta,
    fetch_plugins,
    get_chat_list,
    get_rag_stats,
    get_sender_list,
    reset_config,
    save_config,
)


# Date range options for the filter panel
DATE_RANGE_OPTIONS = {
    "All time": None,
    "Last 24 hours": 1,
    "Last 3 days": 3,
    "Last week": 7,
    "Last month": 30,
}


# =========================================================================
# MAIN ENTRY POINT
# =========================================================================

def render_settings_panel() -> None:
    """Render the settings panel in the sidebar (only when toggled on)."""
    if not st.session_state.get("show_settings", False):
        return

    with st.sidebar:
        st.markdown("---")

        with st.expander("🔍 Filters", expanded=True):
            _render_filters()

        with st.expander("📊 Statistics", expanded=False):
            _render_stats()

        with st.expander("🏥 System Health", expanded=False):
            _render_health()


def render_settings_page() -> None:
    """Render the full settings page (main area).

    Called from app.py when the user navigates to the settings page.
    Fetches all settings and metadata from the backend API and renders
    category-based sections with typed input widgets.
    """
    st.title("⚙️ Settings")
    st.caption("All configuration is stored in the database and takes effect immediately.")

    # Fetch settings and metadata from backend
    all_settings = fetch_config()
    meta = fetch_config_meta()

    if not all_settings:
        st.warning("Could not load settings from the backend API. Is the server running?")
        return

    category_meta = meta.get("category_meta", {})
    select_options = meta.get("select_options", {})

    # Sort categories by their order in category_meta
    categories = sorted(
        all_settings.keys(),
        key=lambda c: float(category_meta.get(c, {}).get("order", "99")),
    )

    # Render each category
    for category in categories:
        settings_in_cat = all_settings[category]
        label = category_meta.get(category, {}).get("label", f"📁 {category.title()}")

        # Special handling for the plugins category
        if category == "plugins":
            _render_plugins_section(label, settings_in_cat, select_options)
        else:
            _render_category_section(category, label, settings_in_cat, select_options)


# =========================================================================
# CATEGORY SECTION RENDERER
# =========================================================================

def _render_category_section(
    category: str,
    label: str,
    settings_dict: Dict[str, Dict[str, Any]],
    select_options: Dict[str, List[str]],
) -> None:
    """Render a single category section with all its settings.

    Args:
        category: Category key (e.g. 'llm', 'rag')
        label: Display label with icon (e.g. '🤖 LLM Configuration')
        settings_dict: Dict of key -> {value, type, description, updated_at}
        select_options: Dict of key -> list of allowed values for select types
    """
    with st.expander(label, expanded=False):
        changes: Dict[str, str] = {}

        for key, info in settings_dict.items():
            value = info.get("value", "")
            setting_type = info.get("type", "text")
            description = info.get("description", "")

            new_value = _render_setting_input(
                key=key,
                value=value,
                setting_type=setting_type,
                description=description,
                select_options=select_options,
                category=category,
            )

            if new_value is not None and str(new_value) != str(value):
                changes[key] = str(new_value)

        # Action buttons
        col_save, col_reset = st.columns(2)
        with col_save:
            if st.button(
                "💾 Save",
                key=f"save_{category}",
                use_container_width=True,
                disabled=len(changes) == 0,
            ):
                result = save_config(changes)
                if "error" in result:
                    st.error(f"Save failed: {result['error']}")
                else:
                    updated = result.get("updated", [])
                    st.success(f"Saved {len(updated)} setting(s)")
                    st.rerun()

        with col_reset:
            if st.button(
                "↩️ Reset",
                key=f"reset_{category}",
                use_container_width=True,
            ):
                result = reset_config(category=category)
                if "error" in result:
                    st.error(f"Reset failed: {result['error']}")
                else:
                    count = result.get("reset_count", 0)
                    st.success(f"Reset {count} setting(s) to defaults")
                    st.rerun()


# =========================================================================
# PLUGINS SECTION
# =========================================================================

def _render_plugins_section(
    label: str,
    settings_dict: Dict[str, Dict[str, Any]],
    select_options: Dict[str, List[str]],
) -> None:
    """Render the plugins section with enable/disable toggles.

    Shows each discovered plugin with a toggle and links to its
    settings category section.

    Args:
        label: Display label (e.g. '🔌 Plugins')
        settings_dict: Plugin enable/disable settings
        select_options: Select options dict
    """
    with st.expander(label, expanded=True):
        # Fetch plugin info for display names and icons
        plugins_data = fetch_plugins()
        plugins_info = plugins_data.get("plugins", {})

        changes: Dict[str, str] = {}

        if not settings_dict:
            st.caption("No plugins discovered.")
            return

        for key, info in settings_dict.items():
            value = info.get("value", "false")
            description = info.get("description", "")

            # Extract plugin name from key (plugin_<name>_enabled)
            plugin_name = key.replace("plugin_", "").replace("_enabled", "")
            plugin_info = plugins_info.get(plugin_name, {})
            icon = plugin_info.get("icon", "🔌")
            display_name = plugin_info.get("display_name", plugin_name.title())
            version = plugin_info.get("version", "")

            # Render toggle
            current_val = value.lower() == "true"
            col_toggle, col_info = st.columns([1, 3])
            with col_toggle:
                new_val = st.toggle(
                    f"{icon} {display_name}",
                    value=current_val,
                    key=f"plugin_toggle_{key}",
                )
            with col_info:
                version_str = f" v{version}" if version else ""
                st.caption(f"{description}{version_str}")

            new_str = "true" if new_val else "false"
            if new_str != value:
                changes[key] = new_str

        if changes:
            if st.button(
                "💾 Save Plugin Settings",
                key="save_plugins",
                use_container_width=True,
            ):
                result = save_config(changes)
                if "error" in result:
                    st.error(f"Save failed: {result['error']}")
                else:
                    st.success("Plugin settings saved. Restart may be required.")
                    st.rerun()


# =========================================================================
# INDIVIDUAL SETTING INPUT RENDERER
# =========================================================================

def _render_setting_input(
    key: str,
    value: str,
    setting_type: str,
    description: str,
    select_options: Dict[str, List[str]],
    category: str,
) -> Optional[str]:
    """Render the appropriate input widget for a setting based on its type.

    Args:
        key: Setting key
        value: Current value (always string)
        setting_type: One of 'text', 'secret', 'int', 'float', 'bool', 'select'
        description: Help text for the setting
        select_options: Dict of key -> list of allowed values
        category: Category name (used for unique widget keys)

    Returns:
        New value as string, or None if unchanged
    """
    widget_key = f"setting_{category}_{key}"
    label = _format_label(key)

    if setting_type == "secret":
        # Password input — show masked value as placeholder
        new_value = st.text_input(
            label,
            value="",
            type="password",
            help=f"{description}\nCurrent: {value}" if value else description,
            placeholder=value if value else "Enter value…",
            key=widget_key,
        )
        # Only return if user typed something (empty means no change)
        return new_value if new_value else None

    elif setting_type == "bool":
        current_bool = value.lower() in ("true", "1", "yes")
        new_bool = st.toggle(
            label,
            value=current_bool,
            help=description,
            key=widget_key,
        )
        return "true" if new_bool else "false"

    elif setting_type == "select":
        options = select_options.get(key, [value])
        try:
            current_index = options.index(value)
        except ValueError:
            options = [value] + options
            current_index = 0
        new_value = st.selectbox(
            label,
            options=options,
            index=current_index,
            help=description,
            key=widget_key,
        )
        return str(new_value)

    elif setting_type == "int":
        try:
            int_val = int(value)
        except (ValueError, TypeError):
            int_val = 0
        new_value = st.number_input(
            label,
            value=int_val,
            step=1,
            help=description,
            key=widget_key,
        )
        return str(int(new_value))

    elif setting_type == "float":
        try:
            float_val = float(value)
        except (ValueError, TypeError):
            float_val = 0.0
        new_value = st.number_input(
            label,
            value=float_val,
            step=0.1,
            format="%.2f",
            help=description,
            key=widget_key,
        )
        return f"{float(new_value):.2f}"

    else:
        # Default: text input (also handles 'text' type)
        # Use text_area for long values (like system_prompt)
        if len(value) > 200 or key in ("system_prompt",):
            new_value = st.text_area(
                label,
                value=value,
                help=description,
                key=widget_key,
                height=200,
            )
        else:
            new_value = st.text_input(
                label,
                value=value,
                help=description,
                key=widget_key,
            )
        return str(new_value)


def _format_label(key: str) -> str:
    """Convert a snake_case setting key to a human-readable label.

    Args:
        key: Setting key (e.g. 'openai_model')

    Returns:
        Formatted label (e.g. 'Openai Model')
    """
    return key.replace("_", " ").title()


# =========================================================================
# SIDEBAR PANELS (filters, stats, health)
# =========================================================================

def _render_filters() -> None:
    """Render filter dropdowns for chat, sender, and time range."""
    # Chat/Group filter
    chat_list = get_chat_list()
    chat_options = [""] + chat_list
    filter_chat: str = st.selectbox(
        "Chat / Group",
        options=chat_options,
        index=0,
        format_func=lambda x: "All chats" if x == "" else x,
        key="filter_chat",
    ) or ""

    # Sender filter
    sender_list = get_sender_list()
    sender_options = [""] + sender_list
    filter_sender: str = st.selectbox(
        "Sender",
        options=sender_options,
        index=0,
        format_func=lambda x: "All senders" if x == "" else x,
        key="filter_sender",
    ) or ""

    # Time range filter
    filter_date_range: str = st.selectbox(
        "Time range",
        options=list(DATE_RANGE_OPTIONS.keys()),
        index=0,
        key="filter_date_range",
    ) or "All time"
    filter_days: Optional[int] = DATE_RANGE_OPTIONS[filter_date_range]

    # Apply filters button
    if st.button("Apply Filters", key="apply_filters", use_container_width=True):
        new_filters = {}
        if filter_chat.strip():
            new_filters["chat_name"] = filter_chat.strip()
        if filter_sender.strip():
            new_filters["sender"] = filter_sender.strip()
        if filter_days is not None:
            new_filters["days"] = str(filter_days)
        st.session_state.active_filters = new_filters
        st.rerun()


def _render_stats() -> None:
    """Render RAG vector store statistics."""
    stats = get_rag_stats()
    if stats:
        st.metric("Total Documents", stats.get("total_documents", 0))
        collection = stats.get("collection_name", "N/A")
        st.caption(f"Collection: {collection}")
        dashboard_url = stats.get("dashboard_url")
        if dashboard_url:
            st.markdown(f"[🔗 Qdrant Dashboard]({dashboard_url})")
    else:
        st.caption("Stats unavailable")


def _render_health() -> None:
    """Render system health dashboard."""
    health = check_health()
    deps = health.get("dependencies", {})

    for name, status in deps.items():
        is_ok = "connected" in str(status).lower()
        dot = "🟢" if is_ok else "🔴"
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:6px; "
            f"margin-bottom:4px;'>"
            f"<span>{dot}</span>"
            f"<span style='font-size:0.85rem; color:#ECECEC;'>"
            f"{name.upper()}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if not is_ok:
            st.caption(f"  {status}")

    overall = health.get("status", "unknown")
    if overall == "up":
        st.success("All systems operational", icon="✅")
    elif overall == "degraded":
        st.warning("Some services degraded", icon="⚠️")
    else:
        st.error(f"Status: {overall}", icon="🔴")
