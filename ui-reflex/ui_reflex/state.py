"""Application state management for the RAG Assistant Reflex UI.

Replaces Streamlit's st.session_state with reactive Reflex state.
All API calls happen in async event handlers — UI updates via yield.
"""

from __future__ import annotations

import ast
import json
import logging
from typing import Any

import reflex as rx

from . import api_client
from .utils.time_utils import group_conversations_by_time

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Human-readable display labels for settings keys
# ---------------------------------------------------------------------------

SETTING_LABELS: dict[str, str] = {
    # LLM / AI
    "llm_provider": "Provider",
    "openai_model": "Model",
    "openai_temperature": "Temperature",
    "gemini_model": "Model",
    "gemini_temperature": "Temperature",
    "image_provider": "Provider",
    "imagen_model": "Imagen Model",
    "dalle_model": "DALL-E Model",
    "system_prompt": "System Prompt",
    # Secrets
    "openai_api_key": "OpenAI API Key",
    "google_api_key": "Google API Key",
    "waha_api_key": "WAHA API Key",
    # RAG
    "rag_collection_name": "Collection Name",
    "rag_min_score": "Minimum Similarity Score",
    "rag_max_context_tokens": "Max Context Tokens",
    "rag_default_k": "Documents to Retrieve",
    "rag_context_window_seconds": "Context Time Window (seconds)",
    "embedding_model": "Embedding Model",
    "rag_vector_size": "Vector Dimensions",
    "rag_rrf_k": "RRF Smoothing Constant",
    "rag_fulltext_score_sender": "Sender Match Score",
    "rag_fulltext_score_chat_name": "Chat Name Match Score",
    "rag_fulltext_score_message": "Message Content Score",
    # Infrastructure / Connections
    "redis_host": "Redis Host",
    "redis_port": "Redis Port",
    "qdrant_host": "Qdrant Host",
    "qdrant_port": "Qdrant Port",
    "waha_base_url": "WAHA Server URL",
    "webhook_url": "Webhook URL",
    "ui_api_url": "UI API URL",
    # App
    "log_level": "Log Level",
    "timezone": "Timezone",
    "redis_ttl": "Cache TTL (seconds)",
    "session_ttl_minutes": "Session Timeout (minutes)",
    "session_max_history": "Max History Turns",
    "cost_tracking_enabled": "Cost Tracking",
    # WhatsApp plugin
    "chat_prefix": "Chat Trigger Prefix",
    "dalle_prefix": "Image Generation Prefix",
    "waha_session_name": "Session Name",
    # Paperless plugin
    "paperless_url": "Server URL",
    "paperless_token": "API Token",
    "paperless_sync_interval": "Sync Interval (seconds)",
    "paperless_sync_tags": "Sync Tags",
    "paperless_max_docs": "Max Documents per Sync",
    # Gmail plugin
    "gmail_client_id": "OAuth2 Client ID",
    "gmail_client_secret": "OAuth2 Client Secret",
    "gmail_refresh_token": "OAuth2 Refresh Token",
    "gmail_sync_folders": "Sync Folders",
    "gmail_sync_interval": "Sync Interval (seconds)",
    "gmail_max_emails": "Max Emails per Sync",
    "gmail_processed_label": "Processed Label",
    "gmail_include_attachments": "Include Attachments",
}


class AppState(rx.State):
    """Root application state."""

    # --- Conversation list ---
    conversations: list[dict[str, str]] = []

    # --- Active conversation ---
    conversation_id: str = ""
    messages: list[dict[str, str]] = []  # {role, content} — all string values

    # --- Filters ---
    active_filters: dict[str, str] = {}

    # --- Advanced search filters ---
    selected_sources: list[str] = []       # e.g. ["whatsapp", "gmail"]
    filter_date_from: str = ""             # ISO date string, e.g. "2026-01-01"
    filter_date_to: str = ""               # ISO date string, e.g. "2026-02-15"
    selected_content_types: list[str] = [] # e.g. ["text", "document"]
    sort_order: str = "relevance"          # "relevance" or "newest"
    show_search_toolbar: bool = False      # Toggle for the search toolbar

    # --- UI state ---
    is_loading: bool = False
    sidebar_search: str = ""
    show_settings: bool = False
    renaming_id: str = ""
    rename_text: str = ""
    input_text: str = ""

    # --- Health ---
    api_status: str = "unknown"
    health_deps: dict[str, str] = {}

    # --- Chat / sender lists (for filter dropdowns) ---
    chat_list: list[str] = []
    sender_list: list[str] = []

    # --- Settings ---
    all_settings: dict[str, Any] = {}
    config_meta: dict[str, Any] = {}
    plugins_data: dict[str, Any] = {}
    settings_save_message: str = ""

    # --- Cost tracking ---
    session_cost: float = 0.0
    last_query_cost: float = 0.0
    cost_summary: dict[str, Any] = {}
    cost_breakdown: dict[str, Any] = {}

    # --- Pending changes (explicit save) ---
    pending_changes: dict[str, str] = {}  # key -> new_value

    # --- Secret visibility ---
    revealed_secrets: list[str] = []  # Keys of secrets currently revealed
    revealed_secret_values: dict[str, str] = {}  # key -> unmasked value

    # --- Paperless test state ---
    paperless_test_status: str = ""   # "", "testing", "success", "error"
    paperless_test_message: str = ""

    # --- Paperless sync state ---
    paperless_sync_status: str = ""   # "", "syncing", "complete", "error"
    paperless_sync_message: str = ""

    # --- Paperless tags (multi-select) ---
    paperless_available_tags: list[dict[str, str]] = []  # [{name, color}, ...]
    paperless_selected_tags: list[str] = []  # tag names currently selected
    paperless_tags_loading: bool = False
    paperless_tag_dropdown_open: bool = False

    # --- Gmail auth state ---
    gmail_auth_url: str = ""
    gmail_auth_status: str = ""   # "", "pending", "success", "error"
    gmail_auth_message: str = ""
    gmail_auth_code_input: str = ""

    # --- Gmail test state ---
    gmail_test_status: str = ""   # "", "testing", "success", "error"
    gmail_test_message: str = ""

    # --- Gmail sync state ---
    gmail_sync_status: str = ""   # "", "syncing", "complete", "error"
    gmail_sync_message: str = ""

    # --- Gmail folders (multi-select) ---
    gmail_available_folders: list[dict[str, str]] = []  # [{id, name, type}, ...]
    gmail_selected_folders: list[str] = []  # folder display names
    gmail_folders_loading: bool = False
    gmail_folder_dropdown_open: bool = False

    # --- Tab state ---
    settings_tab: str = "ai"   # Active main tab
    plugin_tab: str = ""       # Active plugin sub-tab (empty = first plugin)

    # --- RAG stats ---
    rag_stats: dict[str, Any] = {}

    # =====================================================================
    # EXPLICIT SETTERS (avoid deprecated state_auto_setters)
    # =====================================================================

    def set_input_text(self, value: str):
        """Set the chat input text."""
        self.input_text = value

    def set_sidebar_search(self, value: str):
        """Set the sidebar search filter text."""
        self.sidebar_search = value

    def set_rename_text(self, value: str):
        """Set the conversation rename text."""
        self.rename_text = value

    def set_settings_tab(self, value: str):
        """Set the active settings tab."""
        self.settings_tab = value

    def set_plugin_tab(self, value: str):
        """Set the active plugin sub-tab."""
        self.plugin_tab = value

    def set_filter_date_from(self, value: str):
        """Set the date-from filter."""
        self.filter_date_from = value

    def set_filter_date_to(self, value: str):
        """Set the date-to filter."""
        self.filter_date_to = value

    def set_sort_order(self, value: str):
        """Set sort order ('relevance' or 'newest')."""
        self.sort_order = value

    def toggle_search_toolbar(self):
        """Toggle the search toolbar visibility."""
        self.show_search_toolbar = not self.show_search_toolbar

    def toggle_source(self, source: str):
        """Toggle a source in/out of the selected sources list."""
        new = list(self.selected_sources)
        if source in new:
            new.remove(source)
        else:
            new.append(source)
        self.selected_sources = new

    def toggle_content_type(self, ct: str):
        """Toggle a content type in/out of the selected content types list."""
        new = list(self.selected_content_types)
        if ct in new:
            new.remove(ct)
        else:
            new.append(ct)
        self.selected_content_types = new

    def clear_advanced_filters(self):
        """Reset all advanced search filters to defaults."""
        self.selected_sources = []
        self.filter_date_from = ""
        self.filter_date_to = ""
        self.selected_content_types = []
        self.sort_order = "relevance"

    # =====================================================================
    # COMPUTED VARS
    # =====================================================================

    @rx.var(cache=True)
    def show_chat(self) -> bool:
        """Whether to show the chat view (vs empty state)."""
        return bool(self.messages) or bool(self.conversation_id)

    @rx.var(cache=True)
    def has_filters(self) -> bool:
        return (
            len(self.active_filters) > 0
            or bool(self.selected_sources)
            or bool(self.filter_date_from)
            or bool(self.filter_date_to)
            or bool(self.selected_content_types)
            or self.sort_order != "relevance"
        )

    @rx.var(cache=True)
    def filter_chips(self) -> list[dict[str, str]]:
        """Active filters as a list for rx.foreach rendering."""
        chips: list[dict[str, str]] = []
        if self.active_filters.get("chat_name"):
            chips.append({
                "key": "chat_name",
                "icon": "💬",
                "label": self.active_filters["chat_name"],
            })
        if self.active_filters.get("sender"):
            chips.append({
                "key": "sender",
                "icon": "👤",
                "label": self.active_filters["sender"],
            })
        if self.active_filters.get("days"):
            chips.append({
                "key": "days",
                "icon": "📅",
                "label": f"Last {self.active_filters['days']}d",
            })
        # Advanced filter chips
        if self.selected_sources:
            chips.append({
                "key": "sources",
                "icon": "📦",
                "label": ", ".join(s.title() for s in self.selected_sources),
            })
        if self.filter_date_from or self.filter_date_to:
            date_label = ""
            if self.filter_date_from and self.filter_date_to:
                date_label = f"{self.filter_date_from} – {self.filter_date_to}"
            elif self.filter_date_from:
                date_label = f"From {self.filter_date_from}"
            else:
                date_label = f"Until {self.filter_date_to}"
            chips.append({
                "key": "date_range",
                "icon": "📅",
                "label": date_label,
            })
        if self.selected_content_types:
            chips.append({
                "key": "content_types",
                "icon": "📄",
                "label": ", ".join(ct.title() for ct in self.selected_content_types),
            })
        if self.sort_order == "newest":
            chips.append({
                "key": "sort_order",
                "icon": "🕐",
                "label": "Newest First",
            })
        return chips

    @rx.var(cache=True)
    def has_advanced_filters(self) -> bool:
        """Whether any advanced filters are active."""
        return bool(
            self.selected_sources
            or self.filter_date_from
            or self.filter_date_to
            or self.selected_content_types
            or self.sort_order != "relevance"
        )

    @rx.var(cache=True)
    def available_sources(self) -> list[dict[str, str]]:
        """Available data sources from plugins for the source filter.

        Returns a list of dicts with 'name', 'label', 'icon', 'active' keys.
        """
        plugins = self.plugins_data.get("plugins", {})
        selected = set(self.selected_sources)
        result: list[dict[str, str]] = []
        for name, info in plugins.items():
            result.append({
                "name": name,
                "label": info.get("label", name.title()),
                "icon": info.get("icon", "📦"),
                "active": "true" if name in selected else "false",
            })
        return result

    @rx.var(cache=True)
    def sidebar_items(self) -> list[dict[str, str]]:
        """Flat list of sidebar items: headers + conversations for rx.foreach.

        Each item has: {type, label, id, title}
        type="header" → group label row
        type="conv"   → conversation row
        """
        convos = self.conversations
        if self.sidebar_search.strip():
            needle = self.sidebar_search.strip().lower()
            convos = [
                c for c in convos
                if needle in (c.get("title") or "").lower()
            ]
        groups = group_conversations_by_time(convos)
        flat: list[dict[str, str]] = []
        for group in groups:
            flat.append({
                "type": "header",
                "label": group["label"],
                "id": "",
                "title": "",
            })
            for c in group["conversations"]:
                flat.append({
                    "type": "conv",
                    "label": "",
                    "id": c.get("id", ""),
                    "title": c.get("title", "Untitled"),
                })
        return flat

    @rx.var(cache=True)
    def has_conversations(self) -> bool:
        return len(self.conversations) > 0

    @rx.var(cache=True)
    def health_label(self) -> str:
        if self.api_status == "up":
            return "API Connected"
        elif self.api_status == "degraded":
            return "API Degraded"
        return "API Unreachable"

    @rx.var(cache=True)
    def settings_flat(self) -> list[dict[str, str]]:
        """Flat list of settings items for the settings page.

        Each item has: {type, category, label, key, value, setting_type, description}
        type="category" → section header row
        type="setting"  → individual setting row
        """
        cat_meta = self.config_meta.get("category_meta", {})
        categories = sorted(
            self.all_settings.keys(),
            key=lambda c: float(cat_meta.get(c, {}).get("order", "99")),
        )
        flat: list[dict[str, str]] = []
        for cat in categories:
            settings_in_cat = self.all_settings[cat]
            label = cat_meta.get(cat, {}).get("label", f"📁 {cat.title()}")
            flat.append({
                "type": "category",
                "category": cat,
                "label": label,
                "key": "",
                "value": "",
                "setting_type": "",
                "description": "",
            })
            for key, info in settings_in_cat.items():
                flat.append({
                    "type": "setting",
                    "category": cat,
                    "label": key.replace("_", " ").title(),
                    "key": key,
                    "value": str(info.get("value", "")),
                    "setting_type": info.get("type", "text"),
                    "description": info.get("description", ""),
                })
        return flat

    @rx.var(cache=True)
    def rag_total_docs(self) -> str:
        """Extract total documents from rag_stats."""
        return str(self.rag_stats.get("total_documents", "—"))

    @rx.var(cache=True)
    def rag_whatsapp_count(self) -> str:
        """Extract WhatsApp message count from rag_stats."""
        return str(self.rag_stats.get("whatsapp_messages", "—"))

    @rx.var(cache=True)
    def rag_document_count(self) -> str:
        """Extract document count (Paperless etc.) from rag_stats."""
        return str(self.rag_stats.get("documents", "—"))

    @rx.var(cache=True)
    def rag_collection_name(self) -> str:
        """Extract collection name from rag_stats."""
        return str(self.rag_stats.get("collection_name", "—"))

    @rx.var(cache=True)
    def rag_dashboard_url(self) -> str:
        """Extract Qdrant dashboard URL from rag_stats."""
        return str(self.rag_stats.get("dashboard_url", ""))

    @rx.var(cache=True)
    def select_options(self) -> dict:
        """Get select-type option lists from config_meta."""
        return self.config_meta.get("select_options", {})

    @rx.var(cache=True)
    def plugin_categories(self) -> list[str]:
        """Get list of plugin-specific setting categories (for sub-tabs)."""
        plugin_cats = []
        for cat in self.all_settings.keys():
            if cat != "plugins" and cat not in [
                "secrets", "llm", "rag", "infrastructure", "app",
            ]:
                plugin_cats.append(cat)
        return sorted(plugin_cats)

    @rx.var(cache=True)
    def current_llm_provider(self) -> str:
        """Get the current LLM provider selection."""
        llm_settings = self.all_settings.get("llm", {})
        return str(llm_settings.get("llm_provider", {}).get("value", "openai"))

    @rx.var(cache=True)
    def current_image_provider(self) -> str:
        """Get the current image provider selection."""
        llm_settings = self.all_settings.get("llm", {})
        return str(llm_settings.get("image_provider", {}).get("value", "openai"))

    # ----- Per-category setting lists for tabbed UI -----

    def _cat_settings(self, category: str) -> list[dict[str, str]]:
        """Return settings for a category as a flat list of dicts.

        Each dict has: key, label, value, setting_type, description, category, options.
        The ``options`` field is a ``|``-separated string of valid choices for
        select-type settings (empty for non-select types).
        Uses SETTING_LABELS for human-readable display names.
        """
        settings = self.all_settings.get(category, {})
        opts = self.config_meta.get("select_options", {})
        result: list[dict[str, str]] = []
        for key, info in settings.items():
            options_list = opts.get(key, [])
            result.append({
                "key": key,
                "label": SETTING_LABELS.get(key, key.replace("_", " ").title()),
                "value": str(info.get("value", "")),
                "setting_type": info.get("type", "text"),
                "description": info.get("description", ""),
                "category": category,
                "options": "|".join(options_list) if options_list else "",
            })
        return result

    def _pick_settings(
        self, keys: list[tuple[str, str]],
    ) -> list[dict[str, str]]:
        """Pick specific settings by (category, key) pairs, returned in order.

        Allows pulling settings from multiple backend categories into a single
        UI section.  Uses SETTING_LABELS for human-readable display names.
        """
        opts = self.config_meta.get("select_options", {})
        result: list[dict[str, str]] = []
        for category, key in keys:
            cat_settings = self.all_settings.get(category, {})
            info = cat_settings.get(key)
            if info:
                options_list = opts.get(key, [])
                result.append({
                    "key": key,
                    "label": SETTING_LABELS.get(key, key.replace("_", " ").title()),
                    "value": str(info.get("value", "")),
                    "setting_type": info.get("type", "text"),
                    "description": info.get("description", ""),
                    "category": category,
                    "options": "|".join(options_list) if options_list else "",
                })
        return result

    @rx.var(cache=True)
    def llm_settings_list(self) -> list[dict[str, str]]:
        """LLM settings filtered by current provider selections."""
        all_llm = self._cat_settings("llm")
        provider = self.current_llm_provider
        img_provider = self.current_image_provider
        hide_keys: set[str] = set()
        if provider != "openai":
            hide_keys.update(["openai_model", "openai_temperature"])
        if provider != "gemini":
            hide_keys.update(["gemini_model", "gemini_temperature"])
        if img_provider != "google":
            hide_keys.add("imagen_model")
        return [s for s in all_llm if s["key"] not in hide_keys]

    @rx.var(cache=True)
    def secrets_settings_list(self) -> list[dict[str, str]]:
        """Settings for the Keys tab."""
        return self._cat_settings("secrets")

    @rx.var(cache=True)
    def rag_settings_list(self) -> list[dict[str, str]]:
        """Settings for the RAG tab."""
        return self._cat_settings("rag")

    @rx.var(cache=True)
    def infra_settings_list(self) -> list[dict[str, str]]:
        """Settings for the Infrastructure tab."""
        return self._cat_settings("infrastructure")

    @rx.var(cache=True)
    def app_settings_list(self) -> list[dict[str, str]]:
        """Settings for the App tab."""
        return self._cat_settings("app")

    @rx.var(cache=True)
    def plugins_toggle_list(self) -> list[dict[str, str]]:
        """Plugin enable/disable toggles from plugins_data."""
        result: list[dict[str, str]] = []
        plugins = self.plugins_data.get("plugins", {})
        for name, info in plugins.items():
            result.append({
                "key": f"plugin_{name}_enabled",
                "label": info.get("label", name.title()),
                "value": str(info.get("enabled", False)).lower(),
                "setting_type": "bool",
                "description": info.get("description", ""),
                "category": "plugins",
            })
        return result

    @rx.var(cache=True)
    def active_plugin_settings(self) -> list[dict[str, str]]:
        """Settings for the currently selected plugin sub-tab."""
        cat = self.plugin_tab
        if not cat and self.plugin_categories:
            cat = self.plugin_categories[0]
        return self._cat_settings(cat) if cat else []

    @rx.var(cache=True)
    def active_plugin_tab_value(self) -> str:
        """The resolved plugin sub-tab value (uses first if empty)."""
        if self.plugin_tab:
            return self.plugin_tab
        cats = self.plugin_categories
        return cats[0] if cats else ""

    # ----- Reorganized settings for the new tabbed UI -----

    @rx.var(cache=True)
    def ai_chat_settings(self) -> list[dict[str, str]]:
        """Chat provider settings filtered by current provider selection."""
        provider = self.current_llm_provider
        keys: list[tuple[str, str]] = [("llm", "llm_provider")]
        if provider == "openai":
            keys += [("llm", "openai_model"), ("llm", "openai_temperature")]
        elif provider == "gemini":
            keys += [("llm", "gemini_model"), ("llm", "gemini_temperature")]
        return self._pick_settings(keys)

    @rx.var(cache=True)
    def ai_image_settings(self) -> list[dict[str, str]]:
        """Image provider settings filtered by current provider selection."""
        img_provider = self.current_image_provider
        keys: list[tuple[str, str]] = [("llm", "image_provider")]
        if img_provider == "openai":
            keys.append(("whatsapp", "dalle_model"))
        elif img_provider == "google":
            keys.append(("llm", "imagen_model"))
        return self._pick_settings(keys)

    @rx.var(cache=True)
    def system_prompt_setting(self) -> list[dict[str, str]]:
        """Just the system_prompt setting for its own card."""
        return self._pick_settings([("llm", "system_prompt")])

    @rx.var(cache=True)
    def rag_retrieval_settings(self) -> list[dict[str, str]]:
        """Core RAG retrieval settings."""
        return self._pick_settings([
            ("rag", "rag_collection_name"),
            ("rag", "embedding_model"),
            ("rag", "rag_vector_size"),
            ("rag", "rag_default_k"),
            ("rag", "rag_max_context_tokens"),
            ("rag", "rag_context_window_seconds"),
            ("rag", "rag_min_score"),
        ])

    @rx.var(cache=True)
    def rag_scoring_settings(self) -> list[dict[str, str]]:
        """Advanced RAG scoring / ranking settings."""
        return self._pick_settings([
            ("rag", "rag_rrf_k"),
            ("rag", "rag_fulltext_score_sender"),
            ("rag", "rag_fulltext_score_chat_name"),
            ("rag", "rag_fulltext_score_message"),
        ])

    @rx.var(cache=True)
    def connections_settings_list(self) -> list[dict[str, str]]:
        """Infrastructure / connection settings for the System tab."""
        return self._pick_settings([
            ("infrastructure", "redis_host"),
            ("infrastructure", "redis_port"),
            ("infrastructure", "qdrant_host"),
            ("infrastructure", "qdrant_port"),
            ("infrastructure", "waha_base_url"),
            ("infrastructure", "webhook_url"),
            ("app", "ui_api_url"),
        ])

    @rx.var(cache=True)
    def application_settings_list(self) -> list[dict[str, str]]:
        """Application behaviour settings for the System tab."""
        return self._pick_settings([
            ("app", "log_level"),
            ("app", "timezone"),
            ("app", "redis_ttl"),
            ("app", "session_ttl_minutes"),
            ("app", "session_max_history"),
            ("app", "cost_tracking_enabled"),
        ])

    # =====================================================================
    # LIFECYCLE EVENTS
    # =====================================================================

    async def on_load(self):
        """Called when the page loads — fetch initial data."""
        await self._refresh_conversations()
        await self._check_health()
        # Load plugin data so available_sources is populated for the search toolbar
        if not self.plugins_data:
            self.plugins_data = await api_client.fetch_plugins()

    async def on_settings_load(self):
        """Called when settings page loads."""
        await self.on_load()
        await self._load_settings()
        await self._load_cost_data()

    # =====================================================================
    # CONVERSATION MANAGEMENT
    # =====================================================================

    async def _refresh_conversations(self):
        raw = await api_client.fetch_conversations(limit=50)
        # Normalize to list[dict[str, str]] — ensure all values are strings
        self.conversations = [
            {k: str(v) if v is not None else "" for k, v in c.items()}
            for c in raw
        ]

    async def new_chat(self):
        """Start a new conversation."""
        self.conversation_id = ""
        self.messages = []
        self.active_filters = {}
        self.input_text = ""
        self.renaming_id = ""
        # Reset advanced filters
        self.selected_sources = []
        self.filter_date_from = ""
        self.filter_date_to = ""
        self.selected_content_types = []
        self.sort_order = "relevance"

    async def load_conversation(self, convo_id: str):
        """Load a conversation by ID."""
        loaded = await api_client.fetch_conversation(convo_id)
        if loaded:
            self.conversation_id = convo_id
            msgs: list[dict[str, str]] = []
            for m in loaded.get("messages", []):
                sources_md = ""
                raw_sources = m.get("sources", "")
                if raw_sources and m.get("role") == "assistant":
                    try:
                        src_list = json.loads(raw_sources)
                        sources_md = _format_sources(src_list) if src_list else ""
                    except (json.JSONDecodeError, TypeError):
                        pass
                msgs.append({
                    "role": m["role"],
                    "content": m["content"],
                    "sources": sources_md,
                    "cost": "",
                })
            self.messages = msgs
            self.active_filters = loaded.get("filters", {})
            self.renaming_id = ""
            self.input_text = ""
            # Restore advanced filters from persisted conversation filters
            conv_filters = loaded.get("filters", {})
            sources_str = conv_filters.get("sources", "")
            self.selected_sources = [s.strip() for s in sources_str.split(",") if s.strip()] if sources_str else []
            self.filter_date_from = conv_filters.get("date_from", "")
            self.filter_date_to = conv_filters.get("date_to", "")
            ct_str = conv_filters.get("content_types", "")
            self.selected_content_types = [c.strip() for c in ct_str.split(",") if c.strip()] if ct_str else []
            self.sort_order = conv_filters.get("sort_order", "relevance")
            # Navigate to chat view when conversation is selected
            return rx.redirect("/")

    async def delete_conversation(self, convo_id: str):
        """Delete a conversation."""
        await api_client.delete_conversation(convo_id)
        if convo_id == self.conversation_id:
            self.conversation_id = ""
            self.messages = []
            self.active_filters = {}
        await self._refresh_conversations()

    def start_rename(self, convo_id: str):
        """Enter rename mode for a conversation."""
        self.renaming_id = convo_id
        for c in self.conversations:
            if c.get("id") == convo_id:
                self.rename_text = c.get("title", "")
                break

    def cancel_rename(self):
        """Cancel rename mode."""
        self.renaming_id = ""
        self.rename_text = ""

    async def save_rename(self):
        """Save the new conversation title."""
        if self.renaming_id and self.rename_text.strip():
            await api_client.rename_conversation(
                self.renaming_id, self.rename_text.strip()
            )
        self.renaming_id = ""
        self.rename_text = ""
        await self._refresh_conversations()

    async def export_chat(self, convo_id: str):
        """Export a conversation as a Markdown file download."""
        try:
            data = await api_client.export_conversation(convo_id)
            if "error" in data:
                logger.error(f"Export chat failed: {data['error']}")
                return
            title = data.get("title", "chat")
            markdown = data.get("markdown", "")
            # Sanitize title for filename
            safe_title = "".join(
                c if c.isalnum() or c in (" ", "-", "_") else ""
                for c in title
            ).strip().replace(" ", "-")[:60] or "chat"
            filename = f"{safe_title}.md"
            return rx.download(
                data=markdown.encode("utf-8"),
                filename=filename,
            )
        except Exception as e:
            logger.error(f"Export chat error: {e}")

    # =====================================================================
    # CHAT / QUERY
    # =====================================================================

    async def send_message(self, form_data: dict | None = None):
        """Send a user message and get an AI response."""
        question = self.input_text.strip()
        if not question:
            return

        # Add user message immediately
        self.messages.append({"role": "user", "content": question, "sources": "", "cost": ""})
        self.input_text = ""
        self.is_loading = True
        yield  # Update UI

        # Call RAG API
        filters = self.active_filters
        data = await api_client.rag_query(
            question=question,
            conversation_id=self.conversation_id or None,
            k=10,
            filter_chat_name=filters.get("chat_name"),
            filter_sender=filters.get("sender"),
            filter_days=int(filters["days"]) if filters.get("days") else None,
            filter_sources=self.selected_sources or None,
            filter_date_from=self.filter_date_from or None,
            filter_date_to=self.filter_date_to or None,
            filter_content_types=self.selected_content_types or None,
            sort_order=self.sort_order,
        )

        self.is_loading = False

        if "error" in data:
            self.messages.append({
                "role": "assistant",
                "content": f"❌ {data['error']}",
                "sources": "",
                "cost": "",
            })
        else:
            raw_answer = data.get("answer", "No answer received")
            answer = _parse_answer(raw_answer)

            if data.get("conversation_id"):
                self.conversation_id = data["conversation_id"]
            if data.get("filters"):
                self.active_filters = data["filters"]

            # Extract cost info from response
            cost_data = data.get("cost", {})
            query_cost = cost_data.get("query_cost_usd", 0.0)
            self.last_query_cost = query_cost
            self.session_cost = cost_data.get("session_total_usd", self.session_cost)
            cost_str = f"${query_cost:.4f}" if query_cost > 0 else ""

            # Store sources as a separate field (rendered as collapsible in UI)
            sources = data.get("sources", [])
            sources_md = _format_sources(sources) if sources else ""

            self.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources_md,
                "cost": cost_str,
            })

        # Refresh sidebar conversations
        await self._refresh_conversations()

    async def send_suggestion(self, suggestion: str):
        """Send a suggestion prompt as a message."""
        self.input_text = suggestion
        async for _ in self.send_message():
            yield

    # =====================================================================
    # FILTERS
    # =====================================================================

    def remove_filter(self, key: str):
        """Remove a single filter by key — handles both classic and advanced filters."""
        # Advanced filter keys
        if key == "sources":
            self.selected_sources = []
        elif key == "date_range":
            self.filter_date_from = ""
            self.filter_date_to = ""
        elif key == "content_types":
            self.selected_content_types = []
        elif key == "sort_order":
            self.sort_order = "relevance"
        else:
            # Classic filters stored in active_filters dict
            new_filters = dict(self.active_filters)
            new_filters.pop(key, None)
            self.active_filters = new_filters

    def clear_filters(self):
        """Clear all active filters (classic and advanced)."""
        self.active_filters = {}
        self.selected_sources = []
        self.filter_date_from = ""
        self.filter_date_to = ""
        self.selected_content_types = []
        self.sort_order = "relevance"

    def set_filter(self, key: str, value: str):
        """Set a single filter."""
        if value:
            new_filters = dict(self.active_filters)
            new_filters[key] = value
            self.active_filters = new_filters
        else:
            self.remove_filter(key)

    # =====================================================================
    # HEALTH
    # =====================================================================

    async def _check_health(self):
        health = await api_client.check_health()
        self.api_status = health.get("status", "unreachable")
        self.health_deps = health.get("dependencies", {})

    # =====================================================================
    # COST TRACKING
    # =====================================================================

    async def _load_cost_data(self):
        """Load cost summary and breakdown from the backend."""
        self.cost_summary = await api_client.get_cost_summary(days=7)
        self.cost_breakdown = await api_client.get_cost_breakdown(days=7)
        session_data = await api_client.get_cost_session(n=20)
        self.session_cost = session_data.get("session_total_usd", 0.0)

    async def refresh_cost_data(self):
        """Public wrapper for _load_cost_data — usable as an on_click handler."""
        await self._load_cost_data()

    @rx.var(cache=True)
    def session_cost_display(self) -> str:
        """Formatted session cost for display."""
        if self.session_cost <= 0:
            return ""
        return f"${self.session_cost:.4f}"

    @rx.var(cache=True)
    def cost_today_display(self) -> str:
        """Today's total cost for display."""
        total = self.cost_summary.get("total_cost_usd", 0.0)
        if total <= 0:
            return "$0.00"
        return f"${total:.4f}"

    @rx.var(cache=True)
    def cost_by_kind_list(self) -> list[dict[str, str]]:
        """Cost breakdown by kind (chat/embed/whisper/image) for display."""
        by_kind = self.cost_summary.get("by_kind", {})
        if not by_kind:
            return []
        result: list[dict[str, str]] = []
        kind_labels = {
            "chat": "💬 Chat (LLM)",
            "embed": "🔢 Embeddings",
            "whisper": "🎙️ Transcription",
            "image": "🖼️ Image Gen",
        }
        for kind, cost in by_kind.items():
            result.append({
                "kind": kind,
                "label": kind_labels.get(kind) or kind.title(),
                "cost": f"${cost:.4f}" if cost else "$0.00",
            })
        return result

    @rx.var(cache=True)
    def cost_daily_list(self) -> list[dict[str, str]]:
        """Daily cost totals for display."""
        daily = self.cost_summary.get("daily", [])
        result: list[dict[str, str]] = []
        for day in daily:
            result.append({
                "date": str(day.get("date", "")),
                "cost": f"${day.get('total_cost', 0):.4f}",
                "events": str(day.get("event_count", 0)),
            })
        return result

    # =====================================================================
    # SETTINGS
    # =====================================================================

    async def _load_settings(self):
        # Fetch masked settings — secrets show as "sk-a...xyz"
        # Unmasked values are fetched on-demand when the user clicks reveal
        self.all_settings = await api_client.fetch_config(unmask=False)
        self.config_meta = await api_client.fetch_config_meta()
        self.plugins_data = await api_client.fetch_plugins()
        self.rag_stats = await api_client.get_rag_stats()
        self.chat_list = await api_client.get_chat_list()
        self.sender_list = await api_client.get_sender_list()
        # Clear revealed secrets so stale unmasked values don't persist
        self.revealed_secrets = []
        self.revealed_secret_values = {}
        # Initialize paperless selected tags from saved setting
        paperless_settings = self.all_settings.get("paperless", {})
        tags_str = str(paperless_settings.get("paperless_sync_tags", {}).get("value", ""))
        self.paperless_selected_tags = [
            t.strip() for t in tags_str.split(",") if t.strip()
        ] if tags_str else []
        # Initialize gmail selected folders from saved setting
        gmail_settings = self.all_settings.get("gmail", {})
        folders_str = str(gmail_settings.get("gmail_sync_folders", {}).get("value", ""))
        self.gmail_selected_folders = [
            f.strip() for f in folders_str.split(",") if f.strip()
        ] if folders_str else []

    async def save_setting(self, key: str, value: str):
        """Save a single setting."""
        result = await api_client.save_config({key: value})
        if "error" in result:
            self.settings_save_message = f"❌ {result['error']}"
        else:
            self.settings_save_message = "✅ Saved"
        await self._load_settings()

    # ----- Pending changes (explicit save button) -----

    def set_pending_change(self, key: str, value: str):
        """Track a pending change before explicit save."""
        new_pending = dict(self.pending_changes)
        new_pending[key] = value
        self.pending_changes = new_pending

    async def save_pending_change(self, key: str):
        """Save a single pending change."""
        if key in self.pending_changes:
            value = self.pending_changes[key]
            await self.save_setting(key, value)
            new_pending = dict(self.pending_changes)
            del new_pending[key]
            self.pending_changes = new_pending

    # ----- Secret visibility -----

    async def toggle_secret_visibility(self, key: str):
        """Toggle visibility of a secret field.

        When revealing, fetches the unmasked value from the API.
        When hiding, removes the cached unmasked value.
        """
        new_revealed = list(self.revealed_secrets)
        if key in new_revealed:
            # Hide: remove from revealed list and clear cached value
            new_revealed.remove(key)
            new_values = dict(self.revealed_secret_values)
            new_values.pop(key, None)
            self.revealed_secret_values = new_values
        else:
            # Reveal: fetch unmasked value from API
            result = await api_client.fetch_secret_value(key)
            if "error" not in result:
                new_values = dict(self.revealed_secret_values)
                new_values[key] = result.get("value", "")
                self.revealed_secret_values = new_values
            new_revealed.append(key)
        self.revealed_secrets = new_revealed

    # ----- Paperless test -----

    async def test_paperless_connection(self):
        """Test Paperless-NGX connection."""
        self.paperless_test_status = "testing"
        self.paperless_test_message = "Testing connection..."
        yield

        result = await api_client.test_paperless_connection()
        if "error" in result:
            self.paperless_test_status = "error"
            self.paperless_test_message = f"❌ {result['error']}"
        elif result.get("status") == "connected":
            self.paperless_test_status = "success"
            self.paperless_test_message = "✅ Connected successfully"
        else:
            self.paperless_test_status = "error"
            self.paperless_test_message = "❌ Unexpected response"

    # ----- Paperless sync -----

    async def start_paperless_sync(self):
        """Trigger Paperless-NGX document sync to RAG.

        The backend auto-detects an empty Qdrant collection and
        switches to force mode (skipping processed-tag exclusion)
        so this single button works after a collection reset too.
        """
        self.paperless_sync_status = "syncing"
        self.paperless_sync_message = "⏳ Syncing documents…"
        yield

        result = await api_client.start_paperless_sync()
        if "error" in result:
            self.paperless_sync_status = "error"
            self.paperless_sync_message = f"❌ {result['error']}"
        elif result.get("status") == "complete":
            synced = result.get("synced", 0)
            tagged = result.get("tagged", 0)
            skipped = result.get("skipped", 0)
            errors = result.get("errors", 0)
            self.paperless_sync_status = "complete"
            self.paperless_sync_message = (
                f"✅ Sync complete — {synced} indexed, "
                f"{tagged} tagged, {skipped} skipped, {errors} errors"
            )
            # Refresh RAG stats to reflect new document count
            self.rag_stats = await api_client.get_rag_stats()
        elif result.get("status") == "already_running":
            self.paperless_sync_status = "syncing"
            self.paperless_sync_message = "⏳ Sync already in progress"
        else:
            self.paperless_sync_status = "error"
            self.paperless_sync_message = f"❌ Unexpected response: {result}"

    # ----- Paperless tags (multi-select) -----

    @rx.var(cache=True)
    def paperless_unselected_tags(self) -> list[dict[str, str]]:
        """Available tags that are NOT yet selected — for the dropdown."""
        selected = set(self.paperless_selected_tags)
        return [
            t for t in self.paperless_available_tags
            if t.get("name", "") not in selected
        ]

    @rx.var(cache=True)
    def paperless_selected_tag_items(self) -> list[dict[str, str]]:
        """Selected tags with their color info for bubble rendering."""
        tag_map = {t["name"]: t for t in self.paperless_available_tags}
        result: list[dict[str, str]] = []
        for name in self.paperless_selected_tags:
            info = tag_map.get(name, {"name": name, "color": "#a6cee3"})
            result.append({"name": info.get("name", name), "color": info.get("color", "#a6cee3")})
        return result

    async def load_paperless_tags(self):
        """Fetch all tags from Paperless-NGX and populate the dropdown."""
        self.paperless_tags_loading = True
        self.paperless_tag_dropdown_open = True
        yield

        raw_tags = await api_client.fetch_paperless_tags()
        self.paperless_available_tags = [
            {"name": t.get("name", ""), "color": t.get("color", "#a6cee3")}
            for t in raw_tags
        ]
        self.paperless_tags_loading = False

    async def add_paperless_tag(self, tag_name: str):
        """Add a tag to the selected list and save to backend."""
        if tag_name and tag_name not in self.paperless_selected_tags:
            new_selected = list(self.paperless_selected_tags)
            new_selected.append(tag_name)
            self.paperless_selected_tags = new_selected
            # Save comma-separated list to the backend setting
            await self.save_setting(
                "paperless_sync_tags", ",".join(new_selected),
            )

    async def remove_paperless_tag(self, tag_name: str):
        """Remove a tag from the selected list and save to backend."""
        new_selected = [t for t in self.paperless_selected_tags if t != tag_name]
        self.paperless_selected_tags = new_selected
        await self.save_setting(
            "paperless_sync_tags", ",".join(new_selected),
        )

    async def clear_all_paperless_tags(self):
        """Remove all selected tags and save to backend."""
        self.paperless_selected_tags = []
        await self.save_setting("paperless_sync_tags", "")

    def toggle_paperless_tag_dropdown(self):
        """Toggle the tag dropdown open/closed."""
        self.paperless_tag_dropdown_open = not self.paperless_tag_dropdown_open

    def close_paperless_tag_dropdown(self):
        """Close the tag dropdown."""
        self.paperless_tag_dropdown_open = False

    # ----- Gmail auth -----

    def set_gmail_auth_code_input(self, value: str):
        """Set the Gmail auth code input text."""
        self.gmail_auth_code_input = value

    async def gmail_start_auth(self):
        """Request a Gmail OAuth2 authorization URL from the backend."""
        self.gmail_auth_status = "pending"
        self.gmail_auth_message = "Generating authorization URL…"
        self.gmail_auth_url = ""
        yield

        result = await api_client.gmail_get_auth_url()
        if "error" in result:
            self.gmail_auth_status = "error"
            self.gmail_auth_message = f"❌ {result['error']}"
        elif result.get("auth_url"):
            self.gmail_auth_status = "pending"
            self.gmail_auth_url = result["auth_url"]
            self.gmail_auth_message = (
                "Open the URL below in your browser, sign in, "
                "approve the permissions, then paste the authorization code."
            )
        else:
            self.gmail_auth_status = "error"
            self.gmail_auth_message = "❌ Unexpected response"

    async def gmail_submit_auth_code(self):
        """Submit the OAuth2 authorization code to the backend."""
        code = self.gmail_auth_code_input.strip()
        if not code:
            self.gmail_auth_message = "❌ Please enter the authorization code"
            return

        self.gmail_auth_status = "pending"
        self.gmail_auth_message = "⏳ Exchanging code for tokens…"
        yield

        result = await api_client.gmail_submit_auth_code(code)
        if "error" in result:
            self.gmail_auth_status = "error"
            self.gmail_auth_message = f"❌ {result['error']}"
        elif result.get("status") == "authorized":
            self.gmail_auth_status = "success"
            self.gmail_auth_message = "✅ Gmail authorized successfully!"
            self.gmail_auth_url = ""
            self.gmail_auth_code_input = ""
            # Refresh settings to reflect the new token
            await self._load_settings()
        else:
            self.gmail_auth_status = "error"
            self.gmail_auth_message = f"❌ Unexpected response: {result}"

    # ----- Gmail test -----

    async def gmail_test_connection(self):
        """Test Gmail connection."""
        self.gmail_test_status = "testing"
        self.gmail_test_message = "Testing connection…"
        yield

        result = await api_client.gmail_test_connection()
        if "error" in result:
            self.gmail_test_status = "error"
            self.gmail_test_message = f"❌ {result['error']}"
        elif result.get("status") == "connected":
            email = result.get("email", "")
            total = result.get("total_messages", 0)
            self.gmail_test_status = "success"
            self.gmail_test_message = f"✅ Connected as {email} ({total:,} messages)"
        else:
            self.gmail_test_status = "error"
            self.gmail_test_message = "❌ Unexpected response"

    # ----- Gmail sync -----

    async def start_gmail_sync(self):
        """Trigger Gmail email sync to RAG."""
        self.gmail_sync_status = "syncing"
        self.gmail_sync_message = "⏳ Syncing emails…"
        yield

        result = await api_client.start_gmail_sync()
        if "error" in result:
            self.gmail_sync_status = "error"
            self.gmail_sync_message = f"❌ {result['error']}"
        elif result.get("status") == "complete":
            synced = result.get("synced", 0)
            labeled = result.get("labeled", 0)
            skipped = result.get("skipped", 0)
            errors = result.get("errors", 0)
            attachments = result.get("attachments", 0)
            self.gmail_sync_status = "complete"
            self.gmail_sync_message = (
                f"✅ Sync complete — {synced} emails indexed, "
                f"{attachments} attachments, {labeled} labeled, "
                f"{skipped} skipped, {errors} errors"
            )
            # Refresh RAG stats
            self.rag_stats = await api_client.get_rag_stats()
        elif result.get("status") == "already_running":
            self.gmail_sync_status = "syncing"
            self.gmail_sync_message = "⏳ Sync already in progress"
        else:
            self.gmail_sync_status = "error"
            self.gmail_sync_message = f"❌ Unexpected response: {result}"

    # ----- Gmail folders (multi-select) -----

    @rx.var(cache=True)
    def gmail_unselected_folders(self) -> list[dict[str, str]]:
        """Available folders that are NOT yet selected — for the dropdown."""
        selected = set(self.gmail_selected_folders)
        return [
            f for f in self.gmail_available_folders
            if f.get("name", "") not in selected
        ]

    @rx.var(cache=True)
    def gmail_selected_folder_items(self) -> list[dict[str, str]]:
        """Selected folders with their info for bubble rendering."""
        folder_map = {f["name"]: f for f in self.gmail_available_folders}
        result: list[dict[str, str]] = []
        for name in self.gmail_selected_folders:
            info = folder_map.get(name, {"name": name, "id": name, "type": "user"})
            result.append({
                "name": info.get("name", name),
                "type": info.get("type", "user"),
            })
        return result

    async def load_gmail_folders(self):
        """Fetch all labels from Gmail and populate the dropdown."""
        self.gmail_folders_loading = True
        self.gmail_folder_dropdown_open = True
        yield

        raw_folders = await api_client.fetch_gmail_folders()
        self.gmail_available_folders = [
            {"name": f.get("name", ""), "id": f.get("id", ""), "type": f.get("type", "user")}
            for f in raw_folders
        ]
        self.gmail_folders_loading = False

    async def add_gmail_folder(self, folder_name: str):
        """Add a folder to the selected list and save to backend."""
        if folder_name and folder_name not in self.gmail_selected_folders:
            new_selected = list(self.gmail_selected_folders)
            new_selected.append(folder_name)
            self.gmail_selected_folders = new_selected
            await self.save_setting(
                "gmail_sync_folders", ",".join(new_selected),
            )

    async def remove_gmail_folder(self, folder_name: str):
        """Remove a folder from the selected list and save to backend."""
        new_selected = [f for f in self.gmail_selected_folders if f != folder_name]
        self.gmail_selected_folders = new_selected
        await self.save_setting(
            "gmail_sync_folders", ",".join(new_selected),
        )

    async def clear_all_gmail_folders(self):
        """Remove all selected folders and save to backend."""
        self.gmail_selected_folders = []
        await self.save_setting("gmail_sync_folders", "")

    def toggle_gmail_folder_dropdown(self):
        """Toggle the folder dropdown open/closed."""
        self.gmail_folder_dropdown_open = not self.gmail_folder_dropdown_open

    def close_gmail_folder_dropdown(self):
        """Close the folder dropdown."""
        self.gmail_folder_dropdown_open = False

    async def reset_category(self, category: str):
        """Reset a settings category to defaults."""
        result = await api_client.reset_config(category=category)
        if "error" in result:
            self.settings_save_message = f"❌ {result['error']}"
        else:
            count = result.get("reset_count", 0)
            self.settings_save_message = f"✅ Reset {count} settings"
        await self._load_settings()

    async def export_settings(self):
        """Export settings as JSON file download."""
        try:
            data = await api_client.export_config()
            if "error" in data:
                self.settings_save_message = f"❌ Export failed: {data['error']}"
            else:
                import json as _json
                json_str = _json.dumps(data, indent=2)
                return rx.download(
                    data=json_str.encode(),
                    filename="lucy-settings.json",
                )
        except Exception as e:
            self.settings_save_message = f"❌ Export error: {str(e)}"

    async def import_settings(self, files: list[rx.UploadFile]):
        """Import settings from uploaded JSON file."""
        try:
            if not files:
                self.settings_save_message = "❌ No file selected"
                return

            file = files[0]
            content = await file.read()

            import json as _json
            data = _json.loads(content.decode())

            result = await api_client.import_config(data)
            if "error" in result:
                self.settings_save_message = f"❌ Import failed: {result['error']}"
            else:
                count = result.get("count", 0)
                self.settings_save_message = f"✅ Imported {count} settings"
                await self._load_settings()
        except json.JSONDecodeError:
            self.settings_save_message = "❌ Invalid JSON file"
        except Exception as e:
            self.settings_save_message = f"❌ Import error: {str(e)}"


# =========================================================================
# HELPERS
# =========================================================================

def _parse_answer(raw_answer: Any) -> str:
    """Parse potentially JSON-wrapped answer into a plain string."""
    if isinstance(raw_answer, str):
        stripped = raw_answer.strip()
        if (stripped.startswith("{") and stripped.endswith("}")) or (
            stripped.startswith("[") and stripped.endswith("]")
        ):
            try:
                raw_answer = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                try:
                    raw_answer = json.loads(stripped)
                except json.JSONDecodeError:
                    pass

    if isinstance(raw_answer, dict):
        return raw_answer.get("text", str(raw_answer))
    elif isinstance(raw_answer, list):
        texts = []
        for item in raw_answer:
            if isinstance(item, dict) and "text" in item:
                texts.append(item["text"])
            else:
                texts.append(str(item))
        return "\n".join(texts)
    return str(raw_answer)


def _format_sources(sources: list[dict]) -> str:
    """Format source citations as markdown for the collapsible sources section."""
    if not sources:
        return ""
    lines: list[str] = []
    for i, src in enumerate(sources):
        sender = src.get("sender", "")
        chat_name = src.get("chat_name", "")
        content = src.get("content", "")[:200]
        score = src.get("score")
        score_str = f" — {score:.0%}" if score else ""

        if sender:
            header = f"**{i + 1}. {sender}** in _{chat_name}_{score_str}"
        elif chat_name:
            header = f"**{i + 1}.** _{chat_name}_{score_str}"
        else:
            header = f"**{i + 1}.** _Source_{score_str}"

        lines.append(header)
        if content:
            lines.append(f"> {content}{'…' if len(content) >= 200 else ''}\n")
    return "\n".join(lines)
