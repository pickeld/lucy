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


class AppState(rx.State):
    """Root application state."""

    # --- Conversation list ---
    conversations: list[dict[str, str]] = []

    # --- Active conversation ---
    conversation_id: str = ""
    messages: list[dict[str, str]] = []  # {role, content} — all string values

    # --- Filters ---
    active_filters: dict[str, str] = {}

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

    # --- Tab state ---
    settings_tab: str = "llm"  # Active main tab
    plugin_tab: str = ""       # Active plugin sub-tab (empty = first plugin)

    # --- RAG stats ---
    rag_stats: dict[str, Any] = {}

    # =====================================================================
    # COMPUTED VARS
    # =====================================================================

    @rx.var(cache=True)
    def show_chat(self) -> bool:
        """Whether to show the chat view (vs empty state)."""
        return bool(self.messages) or bool(self.conversation_id)

    @rx.var(cache=True)
    def has_filters(self) -> bool:
        return len(self.active_filters) > 0

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
        return chips

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
        """
        settings = self.all_settings.get(category, {})
        opts = self.config_meta.get("select_options", {})
        result: list[dict[str, str]] = []
        for key, info in settings.items():
            options_list = opts.get(key, [])
            result.append({
                "key": key,
                "label": key.replace("_", " ").title(),
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

    # =====================================================================
    # LIFECYCLE EVENTS
    # =====================================================================

    async def on_load(self):
        """Called when the page loads — fetch initial data."""
        await self._refresh_conversations()
        await self._check_health()

    async def on_settings_load(self):
        """Called when settings page loads."""
        await self.on_load()
        await self._load_settings()

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

    async def load_conversation(self, convo_id: str):
        """Load a conversation by ID."""
        loaded = await api_client.fetch_conversation(convo_id)
        if loaded:
            self.conversation_id = convo_id
            self.messages = [
                {"role": m["role"], "content": m["content"], "sources": ""}
                for m in loaded.get("messages", [])
            ]
            self.active_filters = loaded.get("filters", {})
            self.renaming_id = ""
            self.input_text = ""
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

    # =====================================================================
    # CHAT / QUERY
    # =====================================================================

    async def send_message(self, form_data: dict | None = None):
        """Send a user message and get an AI response."""
        question = self.input_text.strip()
        if not question:
            return

        # Add user message immediately
        self.messages.append({"role": "user", "content": question, "sources": ""})
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
        )

        self.is_loading = False

        if "error" in data:
            self.messages.append({
                "role": "assistant",
                "content": f"❌ {data['error']}",
                "sources": "",
            })
        else:
            raw_answer = data.get("answer", "No answer received")
            answer = _parse_answer(raw_answer)

            if data.get("conversation_id"):
                self.conversation_id = data["conversation_id"]
            if data.get("filters"):
                self.active_filters = data["filters"]

            # Store sources as a separate field (rendered as collapsible in UI)
            sources = data.get("sources", [])
            sources_md = _format_sources(sources) if sources else ""

            self.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources_md,
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
        """Remove a single filter by key."""
        new_filters = dict(self.active_filters)
        new_filters.pop(key, None)
        self.active_filters = new_filters

    def clear_filters(self):
        """Clear all active filters."""
        self.active_filters = {}

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
        """Trigger Paperless-NGX document sync to RAG."""
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
                    filename="whatsapp-gpt-settings.json",
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
