"""Application state management for the RAG Assistant Reflex UI.

Replaces Streamlit's st.session_state with reactive Reflex state.
All API calls happen in async event handlers — UI updates via yield.
"""

from __future__ import annotations

import ast
import json
import logging
import os
from typing import Any

import reflex as rx

try:
    import plotly.graph_objects as go
    import plotly.graph_objs  # noqa: F401 — needed for forward ref resolution
except ImportError:
    go = None  # type: ignore[assignment]

from . import api_client
from .utils.time_utils import group_conversations_by_time

# API base URL for constructing browser-facing media URLs.
# PUBLIC_API_URL is the URL the browser can reach (e.g. http://localhost:8765).
# Falls back to API_URL (server-to-server) which works when not in Docker.
_API_PUBLIC_URL = os.environ.get(
    "PUBLIC_API_URL",
    os.environ.get("API_URL", "http://localhost:8765"),
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Human-readable display labels for settings keys
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Human-readable labels for entity fact keys
# ---------------------------------------------------------------------------

FACT_LABELS: dict[str, str] = {
    "birth_date": "Birthday",
    "gender": "Gender",
    "city": "City",
    "job_title": "Job Title",
    "employer": "Employer",
    "marital_status": "Marital Status",
    "email": "Email",
    "id_number": "ID Number",
    "is_business": "Business Account",
    "age": "Age",
    "address": "Address",
    "country": "Country",
    "industry": "Industry",
    "phone": "Phone",
    "recent_topic": "Recent Topic",
    "recent_mood": "Recent Mood",
}

# Fact keys grouped by semantic category for the entity detail panel.
# Order matters — categories are displayed top-to-bottom.
FACT_CATEGORIES: list[dict[str, object]] = [
    {"name": "Identity", "icon": "tag", "keys": ["gender", "birth_date", "age", "id_number"]},
    {"name": "Location", "icon": "map-pin", "keys": ["city", "address", "country"]},
    {"name": "Work", "icon": "briefcase", "keys": ["job_title", "employer", "industry"]},
    {"name": "Contact", "icon": "mail", "keys": ["email", "phone"]},
    {"name": "Business", "icon": "building", "keys": ["is_business"]},
]

# All keys that belong to a named category — anything else goes to "Other"
_CATEGORIZED_KEYS: set[str] = set()
for _cat in FACT_CATEGORIES:
    _CATEGORIZED_KEYS.update(_cat["keys"])  # type: ignore[arg-type]


def _format_source_ref(source_ref: str, source_type: str = "") -> str:
    """Convert a raw source_ref string into a human-readable label.

    Examples:
        "chat:972501234567@c.us:1708012345" → "WhatsApp · 2024-02-15"
        "paperless:42" → "Paperless #42"
        "" / None → "Manual" (if source_type == "manual") or ""
    """
    if not source_ref:
        if source_type == "manual":
            return "Manual entry"
        return ""

    # WhatsApp message: "chat:{chat_id}:{unix_timestamp}"
    if source_ref.startswith("chat:"):
        parts = source_ref.split(":")
        if len(parts) >= 3:
            try:
                from datetime import datetime, timezone
                ts = int(parts[-1])
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                return f"WhatsApp · {dt.strftime('%Y-%m-%d')}"
            except (ValueError, OSError):
                pass
        return "WhatsApp"

    # Paperless document: "paperless:{doc_id}"
    if source_ref.startswith("paperless:"):
        doc_id = source_ref.replace("paperless:", "")
        return f"Paperless #{doc_id}"

    # Gmail
    if source_ref.startswith("gmail:"):
        return "Gmail"

    # Call recording
    if source_ref.startswith("call:"):
        return "Call Recording"

    # Fallback: show as-is (truncated)
    return source_ref[:40]


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
    # Source display filtering
    "source_display_filter_enabled": "Filter Sources for Relevance",
    "source_display_min_score": "Min Source Display Score",
    "source_display_max_count": "Max Sources Displayed",
    "source_display_answer_filter": "Answer-Relevance Filter",
    "chat_entity_extraction_enabled": "Learn Facts from Chat",
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
    # Call Recordings plugin
    "call_recordings_source_path": "Source Path",
    "call_recordings_transcription_provider": "Transcription Provider",
    "call_recordings_whisper_model": "Whisper Model Size",
    "call_recordings_compute_type": "Compute Type",
    "call_recordings_file_extensions": "Audio File Extensions",
    "call_recordings_max_files": "Max Files per Sync",
    "call_recordings_sync_interval": "Sync Interval (seconds)",
    "call_recordings_enable_diarization": "Speaker Diarization",
    "call_recordings_assemblyai_model": "AssemblyAI Model",
    "call_recordings_auto_transcribe": "Auto-Transcribe",
    "call_recordings_my_name": "My Name (Speaker Default)",
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
    sidebar_collapsed: bool = False
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

    # --- Call Recordings test state ---
    call_recordings_test_status: str = ""   # "", "testing", "success", "error"
    call_recordings_test_message: str = ""

    # --- Call Recordings sync state ---
    call_recordings_sync_status: str = ""   # "", "syncing", "complete", "error"
    call_recordings_sync_message: str = ""

    # --- Call Recordings upload state ---
    call_recordings_upload_message: str = ""

    # --- Call Recordings files table ---
    call_recordings_files: list[dict[str, str]] = []
    call_recordings_files_loading: bool = False
    call_recordings_counts: dict[str, str] = {}
    call_recordings_scan_message: str = ""
    call_recordings_filter_name: str = ""

    # --- Recordings page (dedicated /recordings) ---
    recordings_expanded_hash: str = ""            # Which row is expanded
    recordings_speaker_map: list[dict[str, str]] = []  # [{old_label, new_name}, ...]
    recordings_entity_names: list[str] = []       # Entity names for dropdown
    recordings_filter_date_from: str = ""         # Date filter from
    recordings_filter_date_to: str = ""           # Date filter to
    recordings_sort_column: str = "modified_at"   # Sort column
    recordings_sort_asc: bool = False             # Sort direction (desc by default)
    recordings_my_name: str = ""                  # Cached "My Name" setting
    recordings_auto_transcribe: bool = True       # Auto-transcribe toggle
    recordings_active_statuses: list[str] = []    # Multi-select status filter (empty = all)

    # --- Tab state ---
    settings_tab: str = "ai"   # Active main tab
    plugin_tab: str = ""       # Active plugin sub-tab (empty = first plugin)

    # --- RAG stats ---
    rag_stats: dict[str, Any] = {}

    # --- Entity store ---
    entity_persons: list[dict[str, str]] = []
    entity_stats: dict[str, str] = {}
    entity_search: str = ""
    entity_selected_id: int = 0
    entity_detail: dict[str, Any] = {}
    entity_tab: str = "people"
    entity_loading: bool = False
    entity_detail_loading: bool = False
    entity_save_message: str = ""
    entity_new_fact_key: str = ""
    entity_new_fact_value: str = ""
    entity_new_alias: str = ""
    entity_editing_fact_key: str = ""
    entity_editing_fact_value: str = ""
    entity_editing_name: bool = False
    entity_editing_name_value: str = ""
    entity_all_facts: list[dict[str, str]] = []
    entity_fact_keys: list[str] = []
    entity_fact_key_filter: str = ""
    entity_seed_status: str = ""
    entity_seed_message: str = ""

    # --- Entity merge ---
    entity_merge_mode: bool = False
    entity_merge_selection: list[str] = []  # person IDs as strings
    entity_merge_candidates: list[dict[str, Any]] = []
    entity_candidates_loading: bool = False

    # --- Entity graph ---
    entity_graph_nodes: list[dict[str, Any]] = []
    entity_graph_edges: list[dict[str, Any]] = []
    entity_graph_loading: bool = False
    entity_graph_html: str = ""
    # --- Full graph (interactive visualization) ---
    full_graph_nodes: list[dict[str, Any]] = []
    full_graph_edges: list[dict[str, Any]] = []
    full_graph_loading: bool = False
    full_graph_stats: dict[str, str] = {}
    full_graph_figure_data: dict[str, Any] = {}

    # --- Scheduled Insights ---
    insights_tasks: list[dict[str, Any]] = []
    insights_loading: bool = False
    insights_templates: list[dict[str, Any]] = []
    insights_message: str = ""
    # Create/Edit dialog state
    insights_dialog_open: bool = False
    insights_editing_id: int = 0       # 0 = creating new, >0 = editing
    insights_form_name: str = ""
    insights_form_description: str = ""
    insights_form_prompt: str = ""
    insights_form_schedule_type: str = "daily"
    insights_form_schedule_value: str = "08:00"
    insights_form_enabled: bool = True
    insights_form_filter_days: str = ""
    insights_form_filter_chat_name: str = ""
    insights_form_filter_sender: str = ""
    # Result viewer state
    insights_viewing_task_id: int = 0
    insights_viewing_task: dict[str, Any] = {}
    insights_results: list[dict[str, Any]] = []
    insights_results_loading: bool = False
    insights_expanded_result_id: int = 0  # Which result is expanded

    # =====================================================================
    # EXPLICIT SETTERS (avoid deprecated state_auto_setters)
    # =====================================================================

    def set_input_text(self, value: str):
        """Set the chat input text."""
        self.input_text = value

    def set_sidebar_search(self, value: str):
        """Set the sidebar search filter text."""
        self.sidebar_search = value

    def toggle_sidebar(self):
        """Toggle sidebar between collapsed (icons-only) and expanded."""
        self.sidebar_collapsed = not self.sidebar_collapsed

    def set_call_recordings_filter_name(self, value: str):
        """Set the call recordings name filter."""
        self.call_recordings_filter_name = value

    def toggle_recording_status(self, status: str):
        """Toggle a status in/out of the active status filter list."""
        new = list(self.recordings_active_statuses)
        if status in new:
            new.remove(status)
        else:
            new.append(status)
        self.recordings_active_statuses = new

    # --- Recordings page setters ---

    def set_recordings_filter_date_from(self, value: str):
        """Set the recordings date-from filter."""
        self.recordings_filter_date_from = value

    def set_recordings_filter_date_to(self, value: str):
        """Set the recordings date-to filter."""
        self.recordings_filter_date_to = value

    def set_speaker_name(self, old_label: str, new_name: str):
        """Set the new name for a specific speaker label in the speaker map."""
        new_map = list(self.recordings_speaker_map)
        for entry in new_map:
            if entry.get("old_label") == old_label:
                entry["new_name"] = new_name
                break
        self.recordings_speaker_map = new_map

    def set_recordings_sort_column(self, value: str):
        """Set sort column and toggle direction if same column clicked."""
        if self.recordings_sort_column == value:
            self.recordings_sort_asc = not self.recordings_sort_asc
        else:
            self.recordings_sort_column = value
            self.recordings_sort_asc = False

    # --- Insights setters ---

    def set_insights_form_name(self, value: str):
        self.insights_form_name = value

    def set_insights_form_description(self, value: str):
        self.insights_form_description = value

    def set_insights_form_prompt(self, value: str):
        self.insights_form_prompt = value

    def set_insights_form_schedule_type(self, value: str):
        self.insights_form_schedule_type = value

    def set_insights_form_schedule_value(self, value: str):
        self.insights_form_schedule_value = value

    def set_insights_form_filter_days(self, value: float):
        self.insights_form_filter_days = str(int(value)) if value else ""

    def set_insights_form_filter_chat_name(self, value: str):
        self.insights_form_filter_chat_name = value

    def set_insights_form_filter_sender(self, value: str):
        self.insights_form_filter_sender = value

    def toggle_insights_form_enabled(self, value: bool):
        self.insights_form_enabled = value

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

    # --- Entity setters ---

    def set_entity_search(self, value: str):
        """Set entity search text."""
        self.entity_search = value

    def set_entity_tab(self, value: str):
        """Set the active entity tab."""
        self.entity_tab = value

    def set_entity_new_fact_key(self, value: str):
        self.entity_new_fact_key = value

    def set_entity_new_fact_value(self, value: str):
        self.entity_new_fact_value = value

    def set_entity_new_alias(self, value: str):
        self.entity_new_alias = value

    def set_entity_editing_fact_value(self, value: str):
        self.entity_editing_fact_value = value

    def set_entity_fact_key_filter(self, value: str):
        self.entity_fact_key_filter = value

    def toggle_merge_mode(self):
        """Toggle entity merge selection mode on/off."""
        self.entity_merge_mode = not self.entity_merge_mode
        if not self.entity_merge_mode:
            self.entity_merge_selection = []

    def toggle_merge_selection(self, person_id: str):
        """Toggle a person in/out of the merge selection."""
        new_sel = list(self.entity_merge_selection)
        if person_id in new_sel:
            new_sel.remove(person_id)
        else:
            new_sel.append(person_id)
        self.entity_merge_selection = new_sel

    # =====================================================================
    # COMPUTED VARS
    # =====================================================================

    @rx.var(cache=True)
    def safe_messages(self) -> list[dict[str, str]]:
        """Messages with all rich content fields guaranteed to exist.
        
        Normalizes message dicts so that old messages (loaded from DB
        before the rich content feature) have empty string defaults
        for image_urls, ics_url, button_options, etc. — preventing
        JavaScript 'undefined.split()' errors in the Reflex component.
        """
        _required_keys = (
            "role", "content", "sources", "cost", "rich_content",
            "image_urls", "image_captions",
            "ics_url", "ics_title",
            "button_prompt", "button_options",
        )
        result: list[dict[str, str]] = []
        for msg in self.messages:
            normalized = dict(msg)
            for key in _required_keys:
                if key not in normalized:
                    normalized[key] = ""
            result.append(normalized)
        return result

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
    def filtered_recording_files(self) -> list[dict[str, str]]:
        """Call recording files filtered by name and active statuses."""
        files = self.call_recordings_files
        needle = self.call_recordings_filter_name.strip().lower()
        active = self.recordings_active_statuses

        if needle:
            files = [
                f for f in files
                if needle in (f.get("filename", "") or "").lower()
                or needle in (f.get("contact_name", "") or "").lower()
                or needle in (f.get("phone_number", "") or "").lower()
            ]

        if active:
            files = [f for f in files if f.get("status", "") in active]

        return files

    @rx.var(cache=True)
    def recordings_table_data(self) -> list[dict[str, str]]:
        """Recording files filtered/sorted for the dedicated recordings page.

        Enriches each record with parsed date, time, speaker count,
        and formatted duration for table display.
        """
        import re as _re

        files = self.call_recordings_files

        # Apply multi-select status filter (empty = show all)
        active = self.recordings_active_statuses
        if active:
            files = [f for f in files if f.get("status", "") in active]

        needle = self.call_recordings_filter_name.strip().lower()

        # Apply text search filter
        if needle:
            files = [
                f for f in files
                if needle in (f.get("filename", "") or "").lower()
                or needle in (f.get("contact_name", "") or "").lower()
                or needle in (f.get("phone_number", "") or "").lower()
                or needle in (f.get("transcript_text", "") or "").lower()
            ]

        # Apply date range filter
        date_from = self.recordings_filter_date_from.strip()
        date_to = self.recordings_filter_date_to.strip()
        if date_from or date_to:
            filtered = []
            for f in files:
                mod = (f.get("modified_at", "") or "")[:10]  # YYYY-MM-DD
                if date_from and mod < date_from:
                    continue
                if date_to and mod > date_to:
                    continue
                filtered.append(f)
            files = filtered

        # Enrich with parsed fields
        result: list[dict[str, str]] = []
        for f in files:
            enriched = dict(f)

            # Parse date and time — prefer re-parsing from filename for accuracy
            # (existing DB records may have wrong dates from the old DDMMYY parser)
            filename = f.get("filename", "") or ""
            date_str = ""
            time_str = ""

            # Try to extract date/time from Call_recording filename (YYMMDD_HHMMSS)
            fn_match = _re.search(
                r"(?:call[_\s-]*recording[_\s-]*.+?)[_\s-](\d{6})[_\s-](\d{6})\.",
                filename,
                _re.IGNORECASE,
            )
            if fn_match:
                raw_date = fn_match.group(1)  # YYMMDD
                raw_time = fn_match.group(2)  # HHMMSS
                try:
                    yy = int(raw_date[0:2])
                    mm = int(raw_date[2:4])
                    dd = int(raw_date[4:6])
                    yyyy = 2000 + yy if yy < 50 else 1900 + yy
                    date_str = f"{yyyy:04d}-{mm:02d}-{dd:02d}"
                    time_str = f"{raw_time[0:2]}:{raw_time[2:4]}"
                except (ValueError, IndexError):
                    pass

            # Fallback to modified_at or created_at from DB
            if not date_str:
                mod_at = (f.get("modified_at", "") or "").strip()
                if not mod_at or len(mod_at) < 10:
                    mod_at = (f.get("created_at", "") or "").strip()
                date_str = mod_at[:10] if len(mod_at) >= 10 else ""
                if len(mod_at) >= 16:
                    tp = mod_at[11:16]
                    time_str = tp if ":" in tp else ""

            # Convert YYYY-MM-DD → DD/MM/YYYY for display
            if date_str and len(date_str) == 10 and date_str[4] == "-":
                parts = date_str.split("-")
                enriched["date"] = f"{parts[2]}/{parts[1]}/{parts[0]}"
            else:
                enriched["date"] = date_str
            enriched["time"] = time_str

            # Format duration as M:SS
            try:
                dur = int(f.get("duration_seconds", "0") or "0")
                mins, secs = divmod(dur, 60)
                enriched["duration_fmt"] = f"{mins}:{secs:02d}"
            except (ValueError, TypeError):
                enriched["duration_fmt"] = ""

            # Count speakers from transcript text
            transcript = f.get("transcript_text", "") or ""
            speaker_labels = set(_re.findall(r"^(.+?)(?:\s*:)", transcript, _re.MULTILINE))
            enriched["speaker_count"] = str(len(speaker_labels)) if speaker_labels else "0"

            # Use contact_name or filename stem as display name
            enriched["display_name"] = (
                f.get("contact_name", "")
                or f.get("filename", "").rsplit(".", 1)[0]
            )

            result.append(enriched)

        # Sort
        sort_col = self.recordings_sort_column
        reverse = not self.recordings_sort_asc
        if sort_col:
            result.sort(
                key=lambda r: (r.get(sort_col, "") or "").lower(),
                reverse=reverse,
            )

        return result

    @rx.var(cache=True)
    def recordings_status_counts(self) -> dict[str, str]:
        """Status counts for the recordings page header badges.

        Always includes all expected status keys with '0' defaults
        so that the UI badges never hit a missing-key error.
        """
        counts: dict[str, int] = {
            "pending": 0,
            "transcribing": 0,
            "transcribed": 0,
            "approved": 0,
            "error": 0,
            "total": 0,
        }
        for f in self.call_recordings_files:
            s = f.get("status", "unknown")
            counts[s] = counts.get(s, 0) + 1
        counts["total"] = len(self.call_recordings_files)
        return {k: str(v) for k, v in counts.items()}

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
    def rag_gmail_count(self) -> str:
        """Extract Gmail email count from rag_stats."""
        return str(self.rag_stats.get("gmail_emails", "—"))

    @rx.var(cache=True)
    def rag_call_recording_count(self) -> str:
        """Extract call recording count from rag_stats."""
        return str(self.rag_stats.get("call_recordings", "—"))

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
        # Settings removed from the codebase but may still linger in the DB
        _HIDDEN_KEYS = {"call_recordings_default_participants"}
        result: list[dict[str, str]] = []
        for key, info in settings.items():
            if key in _HIDDEN_KEYS:
                continue
            options_list = opts.get(key, [])
            setting_type = info.get("type", "text")
            # Guard: if a select setting has no options, render as text
            # to avoid Radix UI Select.Item empty-value crash
            if setting_type == "select" and not options_list:
                setting_type = "text"
            result.append({
                "key": key,
                "label": SETTING_LABELS.get(key, key.replace("_", " ").title()),
                "value": str(info.get("value", "")),
                "setting_type": setting_type,
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
            ("rag", "asset_neighborhood_expansion_enabled"),
            ("rag", "pii_redaction_enabled"),
            ("rag", "source_display_filter_enabled"),
            ("rag", "source_display_min_score"),
            ("rag", "source_display_max_count"),
            ("rag", "source_display_answer_filter"),
            ("rag", "chat_entity_extraction_enabled"),
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

    # ----- Entity computed vars -----

    @rx.var(cache=True)
    def entity_stats_persons(self) -> str:
        return str(self.entity_stats.get("persons", "0"))

    @rx.var(cache=True)
    def entity_stats_aliases(self) -> str:
        return str(self.entity_stats.get("aliases", "0"))

    @rx.var(cache=True)
    def entity_stats_facts(self) -> str:
        return str(self.entity_stats.get("facts", "0"))

    @rx.var(cache=True)
    def entity_stats_relationships(self) -> str:
        return str(self.entity_stats.get("relationships", "0"))

    @rx.var(cache=True)
    def entity_merge_count(self) -> int:
        """Number of persons currently selected for merge."""
        return len(self.entity_merge_selection)

    @rx.var(cache=True)
    def entity_can_merge(self) -> bool:
        """Whether enough persons are selected to perform a merge (≥2)."""
        return len(self.entity_merge_selection) >= 2

    @rx.var(cache=True)
    def entity_has_detail(self) -> bool:
        return self.entity_selected_id > 0

    @rx.var(cache=True)
    def entity_detail_name(self) -> str:
        """Display name — uses bilingual display_name if available."""
        display = self.entity_detail.get("display_name", "")
        if display:
            return str(display)
        return str(self.entity_detail.get("canonical_name", ""))

    @rx.var(cache=True)
    def entity_detail_phone(self) -> str:
        return str(self.entity_detail.get("phone", "") or "")

    @rx.var(cache=True)
    def entity_detail_whatsapp(self) -> str:
        return str(self.entity_detail.get("whatsapp_id", "") or "")

    @rx.var(cache=True)
    def entity_aliases_list(self) -> list[dict[str, str]]:
        """Aliases from entity_detail with id, alias, script, source."""
        aliases = self.entity_detail.get("aliases", [])
        result: list[dict[str, str]] = []
        for a in aliases:
            result.append({
                "id": str(a.get("id", "")),
                "alias": str(a.get("alias", "")),
                "script": str(a.get("script", "")),
                "source": str(a.get("source", "")),
            })
        return result

    @rx.var(cache=True)
    def entity_relationships_list(self) -> list[dict[str, str]]:
        """Relationships from entity_detail."""
        rels = self.entity_detail.get("relationships", [])
        result: list[dict[str, str]] = []
        for r in rels:
            result.append({
                "type": str(r.get("relationship_type", "")),
                "related_name": str(r.get("related_name", "")),
                "confidence": str(r.get("confidence", "")),
            })
        return result

    @rx.var(cache=True)
    def entity_facts_grouped(self) -> list[dict[str, str]]:
        """Facts grouped into flat list with category headers for rx.foreach.

        Each item: {type: "header"|"fact", category, icon, key, label, value,
                     confidence, source_type, source_ref, source_quote, fact_key}
        """
        facts_detail = self.entity_detail.get("facts_detail", [])
        if not facts_detail:
            return []

        # Build fact lookup by key
        fact_map: dict[str, dict] = {}
        for f in facts_detail:
            fact_map[f.get("fact_key", "")] = f

        result: list[dict[str, str]] = []
        used_keys: set[str] = set()

        # Named categories
        for cat in FACT_CATEGORIES:
            cat_facts: list[dict[str, str]] = []
            for key in cat["keys"]:  # type: ignore[union-attr]
                if key in fact_map:
                    f = fact_map[key]
                    conf = f.get("confidence")
                    conf_str = f"{float(conf) * 100:.0f}%" if conf is not None else ""
                    raw_ref = str(f.get("source_ref", "") or "")
                    src_type = str(f.get("source_type", "") or "")
                    cat_facts.append({
                        "type": "fact",
                        "category": str(cat["name"]),
                        "icon": str(cat["icon"]),
                        "key": key,
                        "label": FACT_LABELS.get(key, key.replace("_", " ").title()),
                        "value": str(f.get("fact_value", "")),
                        "confidence": conf_str,
                        "source_type": src_type,
                        "source_ref": _format_source_ref(raw_ref, src_type),
                        "source_quote": str(f.get("source_quote", "") or ""),
                        "fact_key": key,
                    })
                    used_keys.add(key)
            if cat_facts:
                result.append({
                    "type": "header",
                    "category": str(cat["name"]),
                    "icon": str(cat["icon"]),
                    "key": "", "label": "", "value": "",
                    "confidence": "", "source_type": "", "source_ref": "", "source_quote": "", "fact_key": "",
                })
                result.extend(cat_facts)

        # "Other" category for uncategorized facts
        other_facts: list[dict[str, str]] = []
        for key, f in fact_map.items():
            if key not in used_keys:
                conf = f.get("confidence")
                conf_str = f"{float(conf) * 100:.0f}%" if conf is not None else ""
                raw_ref = str(f.get("source_ref", "") or "")
                src_type = str(f.get("source_type", "") or "")
                other_facts.append({
                    "type": "fact",
                    "category": "Other",
                    "icon": "file-text",
                    "key": key,
                    "label": FACT_LABELS.get(key, key.replace("_", " ").title()),
                    "value": str(f.get("fact_value", "")),
                    "confidence": conf_str,
                    "source_type": src_type,
                    "source_ref": _format_source_ref(raw_ref, src_type),
                    "source_quote": str(f.get("source_quote", "") or ""),
                    "fact_key": key,
                })
        if other_facts:
            result.append({
                "type": "header",
                "category": "Other",
                "icon": "file-text",
                "key": "", "label": "", "value": "",
                "confidence": "", "source_type": "", "source_ref": "", "source_quote": "", "fact_key": "",
            })
            result.extend(other_facts)

        return result

    # =====================================================================
    # LIFECYCLE EVENTS
    # =====================================================================

    async def on_load(self):
        """Called when the page loads — fetch initial data.

        Runs independent API calls in parallel via asyncio.gather()
        to reduce total page-load latency.
        """
        import asyncio

        convos_task = api_client.fetch_conversations(limit=50)
        health_task = api_client.check_health()
        plugins_task = api_client.fetch_plugins() if not self.plugins_data else asyncio.sleep(0)

        results = await asyncio.gather(
            convos_task, health_task, plugins_task,
            return_exceptions=True,
        )

        # Process conversations
        raw: list = results[0] if not isinstance(results[0], BaseException) else []
        self.conversations = [
            {k: str(v) if v is not None else "" for k, v in c.items()}
            for c in raw
        ]

        # Process health
        health = results[1] if not isinstance(results[1], BaseException) else {}
        self.api_status = health.get("status", "unreachable") if isinstance(health, dict) else "unreachable"
        self.health_deps = health.get("dependencies", {}) if isinstance(health, dict) else {}

        # Process plugins
        if not self.plugins_data and isinstance(results[2], dict):
            self.plugins_data = results[2]

    async def on_settings_load(self):
        """Called when settings page loads.

        Runs on_load and settings/cost fetches in parallel where possible.
        """
        import asyncio

        # Phase 1: Load page basics + settings data in parallel
        await asyncio.gather(
            self.on_load(),
            self._load_settings(),
            self._load_cost_data(),
        )

    async def on_entities_load(self):
        """Called when entities page loads.

        Runs on_load and entity fetches in parallel.
        """
        import asyncio

        results = await asyncio.gather(
            self.on_load(),
            self._load_entity_list(),
            api_client.fetch_entity_stats(),
            return_exceptions=True,
        )
        # Assign stats from the third result
        stats = results[2] if not isinstance(results[2], BaseException) else {}
        self.entity_stats = {k: str(v) for k, v in stats.items()}

    async def on_recordings_load(self):
        """Called when the /recordings page loads.

        Loads recordings files, entity names for speaker dropdowns,
        and the 'My Name' setting in parallel.
        """
        import asyncio

        results = await asyncio.gather(
            self.on_load(),
            self._load_recording_files(),
            api_client.fetch_entities(),
            api_client.fetch_config(unmask=False),
            return_exceptions=True,
        )

        # Entity names for speaker dropdown
        entities = results[2] if not isinstance(results[2], BaseException) else []
        self.recordings_entity_names = [
            str(p.get("display_name", "") or p.get("canonical_name", ""))
            for p in entities
            if p.get("canonical_name")
        ][:50]

        # My Name + Auto-Transcribe settings
        config = results[3] if not isinstance(results[3], BaseException) else {}
        cr_settings = config.get("call_recordings", {}) if isinstance(config, dict) else {}
        my_name_info = cr_settings.get("call_recordings_my_name", {})
        self.recordings_my_name = str(my_name_info.get("value", "")) if isinstance(my_name_info, dict) else ""
        auto_info = cr_settings.get("call_recordings_auto_transcribe", {})
        auto_val = str(auto_info.get("value", "true")) if isinstance(auto_info, dict) else "true"
        self.recordings_auto_transcribe = auto_val.lower() in ("true", "1", "yes")

        # Auto-transcribe pending recordings if enabled
        if self.recordings_auto_transcribe:
            await self._auto_transcribe_pending()

    async def on_insights_load(self):
        """Called when the /insights page loads.

        Loads all scheduled tasks and templates in parallel.
        """
        import asyncio

        self.insights_loading = True
        yield

        results = await asyncio.gather(
            self.on_load(),
            api_client.fetch_scheduled_tasks(),
            api_client.fetch_insight_templates(),
            return_exceptions=True,
        )

        tasks_data = results[1] if not isinstance(results[1], BaseException) else {}
        templates_data = results[2] if not isinstance(results[2], BaseException) else {}

        raw_tasks = tasks_data.get("tasks", []) if isinstance(tasks_data, dict) else []
        # Normalize all values to strings for Reflex rendering
        self.insights_tasks = [
            {k: str(v) if v is not None else "" for k, v in t.items()}
            for t in raw_tasks
        ]
        self.insights_templates = templates_data.get("templates", []) if isinstance(templates_data, dict) else []
        self.insights_loading = False

    async def toggle_auto_transcribe(self):
        """Toggle auto-transcribe on/off and save to backend."""
        new_val = not self.recordings_auto_transcribe
        self.recordings_auto_transcribe = new_val
        await api_client.save_config({
            "call_recordings_auto_transcribe": "true" if new_val else "false",
        })
        # If turned on, immediately queue pending
        if new_val:
            await self._auto_transcribe_pending()

    async def _auto_transcribe_pending(self):
        """Auto-queue pending recordings for transcription (max 3 concurrent).

        Counts how many are currently 'transcribing', then queues
        enough pending files to reach 3 total concurrent transcriptions.
        Each transcribe request is fire-and-forget (returns 202).
        """
        MAX_CONCURRENT = 3

        transcribing_count = sum(
            1 for f in self.call_recordings_files
            if f.get("status") == "transcribing"
        )

        if transcribing_count >= MAX_CONCURRENT:
            return  # Already at capacity

        slots = MAX_CONCURRENT - transcribing_count
        pending = [
            f.get("content_hash", "")
            for f in self.call_recordings_files
            if f.get("status") == "pending" and f.get("content_hash")
        ]

        if not pending:
            return

        queued = 0
        for content_hash in pending[:slots]:
            result = await api_client.transcribe_recording(content_hash)
            if result.get("status") == "queued" or not result.get("error"):
                queued += 1

        if queued:
            self.call_recordings_scan_message = (
                f"⏳ Auto-queued {queued} recording(s) for transcription — "
                f"click Refresh to check status"
            )
            await self._load_recording_files()

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
                msg = _empty_msg(m["role"], m["content"])
                msg["sources"] = sources_md
                # Restore rich content (images, ICS events, buttons) from DB
                raw_rich = m.get("rich_content", "")
                if raw_rich and m.get("role") == "assistant":
                    try:
                        rich_list = json.loads(raw_rich)
                        if rich_list:
                            rich_fields = _flatten_rich_content(rich_list)
                            msg.update(rich_fields)
                    except (json.JSONDecodeError, TypeError):
                        pass
                msgs.append(msg)
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

    async def delete_all_chats(self):
        """Delete all conversations and chat history."""
        result = await api_client.delete_all_conversations()
        if "error" in result:
            self.settings_save_message = f"❌ {result['error']}"
        else:
            deleted = result.get("deleted", 0)
            self.settings_save_message = f"✅ Deleted {deleted} conversation(s)"
            self.conversation_id = ""
            self.messages = []
            self.active_filters = {}
            self.conversations = []
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

    async def copy_chat_to_clipboard(self, convo_id: str):
        """Copy a conversation as Markdown text to the clipboard."""
        try:
            data = await api_client.export_conversation(convo_id)
            if "error" in data:
                logger.error(f"Copy chat failed: {data['error']}")
                return rx.toast.error("Failed to copy chat")
            markdown = data.get("markdown", "")
            return [
                rx.set_clipboard(markdown),
                rx.toast.success("Chat copied to clipboard"),
            ]
        except Exception as e:
            logger.error(f"Copy chat error: {e}")
            return rx.toast.error("Failed to copy chat")

    # =====================================================================
    # CHAT / QUERY
    # =====================================================================

    async def send_message(self, form_data: dict | None = None):
        """Send a user message and get an AI response."""
        question = self.input_text.strip()
        if not question:
            return

        # Add user message immediately
        self.messages.append(_empty_msg("user", question))
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
            self.messages.append(_empty_msg("assistant", f"❌ {data['error']}"))
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

            # Flatten rich content into message-level fields for Reflex rendering
            rich_content = data.get("rich_content", [])
            rich_fields = _flatten_rich_content(rich_content)

            msg = _empty_msg("assistant", answer)
            msg["sources"] = sources_md
            msg["cost"] = cost_str
            msg.update(rich_fields)
            self.messages.append(msg)

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
        """Fetch all settings, metadata, plugin info, and filter lists.

        All 6 API calls are independent — run them in parallel to reduce
        total settings-page load time from ~1–2s (serial) to ~300ms.
        """
        import asyncio

        (
            all_settings,
            config_meta,
            plugins_data,
            rag_stats,
            chat_list,
            sender_list,
        ) = await asyncio.gather(
            api_client.fetch_config(unmask=False),
            api_client.fetch_config_meta(),
            api_client.fetch_plugins(),
            api_client.get_rag_stats(),
            api_client.get_chat_list(),
            api_client.get_sender_list(),
        )

        self.all_settings = all_settings
        self.config_meta = config_meta
        self.plugins_data = plugins_data
        self.rag_stats = rag_stats
        self.chat_list = chat_list
        self.sender_list = sender_list
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

    # ----- Call Recordings test -----

    async def test_call_recordings_connection(self):
        """Test call recordings source connectivity."""
        self.call_recordings_test_status = "testing"
        self.call_recordings_test_message = "Testing source access…"
        yield

        result = await api_client.test_call_recordings_connection()
        if "error" in result:
            self.call_recordings_test_status = "error"
            self.call_recordings_test_message = f"❌ {result['error']}"
        elif result.get("status") == "connected":
            source = result.get("source", "")
            path = result.get("path", "")
            self.call_recordings_test_status = "success"
            self.call_recordings_test_message = (
                f"✅ Connected — {source} source at {path}"
            )
        else:
            self.call_recordings_test_status = "error"
            self.call_recordings_test_message = "❌ Unexpected response"

    # ----- Call Recordings sync -----

    async def start_call_recordings_sync(self):
        """Trigger call recordings sync (scan → transcribe → index)."""
        self.call_recordings_sync_status = "syncing"
        self.call_recordings_sync_message = "⏳ Scanning and transcribing recordings…"
        yield

        result = await api_client.start_call_recordings_sync()
        if "error" in result:
            self.call_recordings_sync_status = "error"
            self.call_recordings_sync_message = f"❌ {result['error']}"
        elif result.get("status") == "complete":
            synced = result.get("synced", 0)
            skipped = result.get("skipped", 0)
            errors = result.get("errors", 0)
            self.call_recordings_sync_status = "complete"
            self.call_recordings_sync_message = (
                f"✅ Sync complete — {synced} transcribed & indexed, "
                f"{skipped} skipped, {errors} errors"
            )
            # Refresh RAG stats
            self.rag_stats = await api_client.get_rag_stats()
        elif result.get("status") == "already_running":
            self.call_recordings_sync_status = "syncing"
            self.call_recordings_sync_message = "⏳ Sync already in progress"
        else:
            self.call_recordings_sync_status = "error"
            self.call_recordings_sync_message = f"❌ Unexpected response: {result}"

    # ----- Call Recordings upload -----

    async def upload_call_recordings(self, files: list[rx.UploadFile]):
        """Upload audio files to the call recordings plugin.

        Files are sent to the backend which saves them to the configured
        source path. After upload, automatically triggers a sync.
        """
        if not files:
            self.call_recordings_upload_message = "❌ No files selected"
            return

        self.call_recordings_upload_message = f"⏳ Uploading {len(files)} file(s)…"
        yield

        try:
            file_data: list[tuple[str, bytes]] = []
            for f in files:
                content = await f.read()
                file_data.append((f.filename or "recording.mp3", content))

            result = await api_client.upload_call_recordings(file_data)

            if "error" in result:
                self.call_recordings_upload_message = f"❌ {result['error']}"
            else:
                saved = result.get("saved", 0)
                errors = result.get("errors", [])
                filenames = result.get("filenames", [])
                msg = f"✅ Uploaded {saved} file(s): {', '.join(filenames)}"
                if errors:
                    msg += f" — Errors: {'; '.join(errors)}"
                self.call_recordings_upload_message = msg
                # Refresh the files table — transcription runs in background
                self.call_recordings_scan_message = (
                    "⏳ Transcription in progress — status updates automatically"
                )
                await self._load_recording_files()
        except Exception as e:
            self.call_recordings_upload_message = f"❌ Upload error: {str(e)}"

    # ----- Call Recordings files table -----

    async def _load_recording_files(self) -> bool:
        """Fetch recording files from the backend and populate the table.

        Returns:
            True if any files are currently in 'transcribing' state.
        """
        self.call_recordings_files_loading = True
        result = await api_client.fetch_call_recording_files()
        has_transcribing = False
        if "error" in result and result["error"]:
            self.call_recordings_files = []
            self.call_recordings_counts = {}
        else:
            raw_files = result.get("files", [])
            # Normalize all values to strings for Reflex rendering
            self.call_recordings_files = [
                {k: str(v) if v is not None else "" for k, v in f.items()}
                for f in raw_files
            ]
            counts = result.get("counts", {})
            self.call_recordings_counts = {k: str(v) for k, v in counts.items()}
            # Check if any files are still transcribing
            has_transcribing = any(
                f.get("status") == "transcribing" for f in raw_files
            )
        self.call_recordings_files_loading = False
        return has_transcribing

    async def load_recording_files(self):
        """Public handler to load/refresh the recordings table."""
        await self._load_recording_files()
        # Auto-queue pending recordings if auto-transcribe is enabled
        if self.recordings_auto_transcribe:
            await self._auto_transcribe_pending()

    async def scan_recordings(self):
        """Scan for new files (without auto-transcribe — too slow for many files)."""
        self.call_recordings_scan_message = "⏳ Scanning for new files…"
        self.call_recordings_files_loading = True
        yield

        result = await api_client.scan_call_recordings(auto_transcribe=False)
        if "error" in result:
            self.call_recordings_scan_message = f"❌ {result['error']}"
        elif result.get("status") == "complete":
            discovered = result.get("discovered", 0)
            new = result.get("new", 0)
            transcribed = result.get("transcribed", 0)
            errors = result.get("errors", 0)
            self.call_recordings_scan_message = (
                f"✅ Scan complete — {discovered} found, {new} new, "
                f"{transcribed} transcribed, {errors} errors"
            )
        elif result.get("status") == "already_running":
            self.call_recordings_scan_message = "⏳ Scan already in progress"
        else:
            self.call_recordings_scan_message = f"❌ Unexpected: {result}"

        await self._load_recording_files()

    async def approve_recording(self, content_hash: str):
        """Approve a transcribed recording: apply speaker names then index.

        Uses the dynamic speaker_map if the row is expanded, otherwise
        auto-assigns from my_name + contact_name for the first two speakers.
        """
        self.call_recordings_scan_message = "⏳ Applying speaker names & approving…"
        yield

        # Build speaker mappings from the speaker_map state (if expanded)
        # or auto-assign for the first two speakers
        speakers: list[dict[str, str]] = []
        if self.recordings_expanded_hash == content_hash and self.recordings_speaker_map:
            speakers = [
                {"old": e.get("old_label", ""), "new": e.get("new_name", "")}
                for e in self.recordings_speaker_map
                if e.get("new_name", "").strip()
            ]
        else:
            # Auto-assign: first speaker = my_name, second = contact_name
            my_name = self.recordings_my_name or "Me"
            contact = ""
            for f in self.call_recordings_files:
                if f.get("content_hash") == content_hash:
                    contact = f.get("contact_name", "") or "Unknown"
                    break
            speakers = [
                {"old": "Speaker A", "new": my_name},
                {"old": "Speaker B", "new": contact},
            ]

        if speakers:
            label_result = await api_client.update_speaker_labels(
                content_hash, speakers=speakers,
            )
            if "error" in label_result:
                self.call_recordings_scan_message = f"⚠️ Speaker names failed: {label_result['error']} — approving anyway"
                yield

        result = await api_client.approve_recording(content_hash)
        if "error" in result:
            self.call_recordings_scan_message = f"❌ {result['error']}"
        else:
            self.call_recordings_scan_message = "✅ Recording approved and indexed (with speaker names)"
            self.rag_stats = await api_client.get_rag_stats()
        await self._load_recording_files()

    async def delete_recording(self, content_hash: str):
        """Delete a recording file."""
        result = await api_client.delete_recording(content_hash)
        if "error" in result:
            self.call_recordings_scan_message = f"❌ {result['error']}"
        else:
            self.call_recordings_scan_message = "✅ Recording deleted"
        await self._load_recording_files()

    async def retry_transcription(self, content_hash: str):
        """Trigger transcription for a recording (fire-and-forget).

        Sends the transcription request to the backend and returns
        immediately. The user can click Refresh to check progress.
        """
        self.call_recordings_scan_message = "⏳ Transcription queued…"
        yield

        result = await api_client.transcribe_recording(content_hash)
        if "error" in result and result.get("status") != "queued":
            self.call_recordings_scan_message = f"❌ {result['error']}"
        else:
            self.call_recordings_scan_message = (
                "⏳ Transcription in progress — click Refresh to check status"
            )
        await self._load_recording_files()

    async def restart_stuck_transcription(self, content_hash: str):
        """Restart a stuck transcription (fire-and-forget).

        Resets the file status and re-queues transcription, then
        returns immediately.
        """
        self.call_recordings_scan_message = "⏳ Restarting transcription…"
        yield

        result = await api_client.restart_recording(content_hash)
        if "error" in result and result.get("status") != "restarted":
            self.call_recordings_scan_message = f"❌ {result['error']}"
        else:
            self.call_recordings_scan_message = (
                "⏳ Transcription restarted — click Refresh to check status"
            )
        await self._load_recording_files()

    async def _wait_for_transcription(self, content_hash: str):
        """Block until the Celery worker signals transcription completion.

        Uses the Flask ``/wait`` endpoint which does a Redis BLPOP —
        truly push-based, no polling.  Falls back gracefully on timeout.
        """
        wait_result = await api_client.wait_for_transcription(
            content_hash, timeout=300,
        )
        # Refresh the file list with the final state
        await self._load_recording_files()

        status = wait_result.get("status", "unknown")
        if status == "transcribed":
            self.call_recordings_scan_message = "✅ Transcription complete"
        elif status == "error":
            error = wait_result.get("error", "")
            self.call_recordings_scan_message = f"❌ Transcription failed{': ' + error if error else ''}"
        elif status == "timeout":
            self.call_recordings_scan_message = (
                "⏳ Transcription still running — use Refresh to check"
            )
        else:
            self.call_recordings_scan_message = f"✅ Transcription finished ({status})"

    async def save_recording_metadata(self, content_hash: str, field: str, value: str):
        """Save an edited metadata field for a recording."""
        kwargs: dict = {}
        if field == "contact_name":
            kwargs["contact_name"] = value
        elif field == "phone_number":
            kwargs["phone_number"] = value
        else:
            return
        await api_client.update_recording_metadata(content_hash, **kwargs)
        await self._load_recording_files()

    # ----- Recordings page event handlers -----

    def toggle_recording_detail(self, content_hash: str):
        """Expand/collapse a recording row in the dedicated recordings page.

        Extracts all unique speaker labels from the transcript and builds
        a dynamic speaker map with suggested names (my_name for first,
        contact_name for second, empty for the rest).
        """
        import re as _re

        if self.recordings_expanded_hash == content_hash:
            self.recordings_expanded_hash = ""
            self.recordings_speaker_map = []
        else:
            self.recordings_expanded_hash = content_hash
            # Find the file record
            contact = ""
            transcript = ""
            for f in self.call_recordings_files:
                if f.get("content_hash") == content_hash:
                    contact = f.get("contact_name", "") or ""
                    transcript = f.get("transcript_text", "") or ""
                    break

            # Extract all unique speaker labels from transcript, sorted alphabetically
            # Matches lines starting with "SomeLabel:" or "SomeLabel :"
            labels = sorted(set(
                _re.findall(r"^(.+?)\s*:", transcript, _re.MULTILINE)
            ))

            # Build speaker map with auto-suggested names
            my_name = self.recordings_my_name or "Me"
            speaker_map: list[dict[str, str]] = []
            for i, label in enumerate(labels):
                if i == 0:
                    suggested = my_name
                elif i == 1:
                    suggested = contact or ""
                else:
                    suggested = ""
                speaker_map.append({
                    "old_label": label,
                    "new_name": suggested,
                })
            self.recordings_speaker_map = speaker_map

    def swap_speakers(self):
        """Swap the first two speaker names in the map."""
        if len(self.recordings_speaker_map) >= 2:
            new_map = list(self.recordings_speaker_map)
            # Swap the new_name values of the first two entries
            name_0 = new_map[0].get("new_name", "")
            name_1 = new_map[1].get("new_name", "")
            new_map[0] = {**new_map[0], "new_name": name_1}
            new_map[1] = {**new_map[1], "new_name": name_0}
            self.recordings_speaker_map = new_map

    def clear_recordings_filters(self):
        """Reset all recordings page filters."""
        self.call_recordings_filter_name = ""
        self.recordings_active_statuses = []
        self.recordings_filter_date_from = ""
        self.recordings_filter_date_to = ""

    # =================================================================
    # SCHEDULED INSIGHTS HANDLERS
    # =================================================================

    async def _load_insights_tasks(self):
        """Reload the insights task list from the API."""
        data = await api_client.fetch_scheduled_tasks()
        raw_tasks = data.get("tasks", []) if isinstance(data, dict) else []
        self.insights_tasks = [
            {k: str(v) if v is not None else "" for k, v in t.items()}
            for t in raw_tasks
        ]

    def open_insights_create_dialog(self):
        """Open the dialog for creating a new insight task."""
        self.insights_editing_id = 0
        self.insights_form_name = ""
        self.insights_form_description = ""
        self.insights_form_prompt = ""
        self.insights_form_schedule_type = "daily"
        self.insights_form_schedule_value = "08:00"
        self.insights_form_enabled = True
        self.insights_form_filter_days = ""
        self.insights_form_filter_chat_name = ""
        self.insights_form_filter_sender = ""
        self.insights_dialog_open = True

    def open_insights_edit_dialog(self, task_id: str):
        """Open the dialog for editing an existing insight task."""
        tid = int(task_id)
        self.insights_editing_id = tid
        # Find the task in the current list
        for t in self.insights_tasks:
            if str(t.get("id", "")) == str(tid):
                self.insights_form_name = str(t.get("name", ""))
                self.insights_form_description = str(t.get("description", ""))
                self.insights_form_prompt = str(t.get("prompt", ""))
                self.insights_form_schedule_type = str(t.get("schedule_type", "daily"))
                self.insights_form_schedule_value = str(t.get("schedule_value", "08:00"))
                self.insights_form_enabled = str(t.get("enabled", "True")).lower() in ("true", "1", "yes")
                # Parse filters
                filters_str = str(t.get("filters", "{}"))
                try:
                    import json as _json
                    filters = _json.loads(filters_str) if filters_str and filters_str != "{}" else {}
                except Exception:
                    filters = {}
                self.insights_form_filter_days = str(filters.get("days", ""))
                self.insights_form_filter_chat_name = str(filters.get("chat_name", ""))
                self.insights_form_filter_sender = str(filters.get("sender", ""))
                break
        self.insights_dialog_open = True

    def close_insights_dialog(self):
        """Close the create/edit dialog."""
        self.insights_dialog_open = False

    def apply_insight_template(self, template_name: str):
        """Fill the dialog form from a template by name."""
        for tmpl in self.insights_templates:
            if str(tmpl.get("name", "")) == template_name:
                self.insights_form_name = str(tmpl.get("name", ""))
                self.insights_form_description = str(tmpl.get("description", ""))
                self.insights_form_prompt = str(tmpl.get("prompt", ""))
                self.insights_form_schedule_type = str(tmpl.get("schedule_type", "daily"))
                self.insights_form_schedule_value = str(tmpl.get("schedule_value", "08:00"))
                filters = tmpl.get("filters", {})
                if isinstance(filters, dict):
                    self.insights_form_filter_days = str(filters.get("days", ""))
                break

    async def save_insight_task(self):
        """Save the insight task (create or update) from dialog form."""
        if not self.insights_form_name.strip() or not self.insights_form_prompt.strip():
            self.insights_message = "❌ Name and prompt are required"
            return

        # Build filters dict
        filters: dict[str, Any] = {}
        filter_days_str = str(self.insights_form_filter_days).strip()
        if filter_days_str:
            try:
                filters["days"] = int(filter_days_str)
            except ValueError:
                pass
        if self.insights_form_filter_chat_name.strip():
            filters["chat_name"] = self.insights_form_filter_chat_name.strip()
        if self.insights_form_filter_sender.strip():
            filters["sender"] = self.insights_form_filter_sender.strip()

        data = {
            "name": self.insights_form_name.strip(),
            "description": self.insights_form_description.strip(),
            "prompt": self.insights_form_prompt.strip(),
            "schedule_type": self.insights_form_schedule_type,
            "schedule_value": self.insights_form_schedule_value.strip(),
            "enabled": self.insights_form_enabled,
            "filters": filters,
        }

        self.insights_message = "⏳ Saving…"
        yield

        if self.insights_editing_id > 0:
            result = await api_client.update_scheduled_task(self.insights_editing_id, data)
        else:
            result = await api_client.create_scheduled_task(data)

        if "error" in result:
            self.insights_message = f"❌ {result['error']}"
        else:
            name = result.get("name", data["name"])
            action = "updated" if self.insights_editing_id > 0 else "created"
            self.insights_message = f"✅ '{name}' {action}"
            self.insights_dialog_open = False
            await self._load_insights_tasks()

    async def delete_insight_task(self, task_id: str):
        """Delete a scheduled insight task."""
        tid = int(task_id)
        result = await api_client.delete_scheduled_task(tid)
        if "error" in result:
            self.insights_message = f"❌ {result['error']}"
        else:
            self.insights_message = "✅ Task deleted"
            if self.insights_viewing_task_id == tid:
                self.insights_viewing_task_id = 0
                self.insights_viewing_task = {}
                self.insights_results = []
            await self._load_insights_tasks()

    async def toggle_insight_task(self, task_id: str):
        """Toggle a task's enabled/disabled state."""
        tid = int(task_id)
        result = await api_client.toggle_scheduled_task(tid)
        if "error" in result:
            self.insights_message = f"❌ {result['error']}"
        else:
            enabled = result.get("enabled", False)
            self.insights_message = f"✅ Task {'enabled' if enabled else 'disabled'}"
            await self._load_insights_tasks()

    async def run_insight_task_now(self, task_id: str):
        """Manually trigger a scheduled insight task."""
        tid = int(task_id)
        self.insights_message = "⏳ Running insight…"
        yield

        result = await api_client.run_scheduled_task(tid)
        if "error" in result:
            self.insights_message = f"❌ {result['error']}"
        else:
            self.insights_message = f"✅ {result.get('message', 'Task dispatched')}"

    async def view_insight_results(self, task_id: str):
        """Open the result history for a task."""
        tid = int(task_id)
        if self.insights_viewing_task_id == tid:
            # Toggle off
            self.insights_viewing_task_id = 0
            self.insights_viewing_task = {}
            self.insights_results = []
            self.insights_message = ""
            yield
            return

        self.insights_viewing_task_id = tid
        self.insights_results_loading = True
        self.insights_message = "⏳ Loading results…"
        yield

        try:
            results_data = await api_client.fetch_scheduled_task_results(tid, limit=20)
            task_data = await api_client.fetch_scheduled_task(tid)

            if isinstance(task_data, dict) and task_data:
                self.insights_viewing_task = {
                    k: str(v) if v is not None else "" for k, v in task_data.items()
                }
            else:
                self.insights_viewing_task = {}

            raw_results = results_data.get("results", []) if isinstance(results_data, dict) else []
            self.insights_results = [
                {k: str(v) if v is not None else "" for k, v in r.items()}
                for r in raw_results
            ]
            self.insights_results_loading = False
            self.insights_message = ""
        except Exception as e:
            self.insights_results_loading = False
            self.insights_message = f"❌ Failed to load results: {e}"

    def toggle_insight_result_expand(self, result_id: str):
        """Expand/collapse a specific result in the result viewer."""
        rid = int(result_id)
        if self.insights_expanded_result_id == rid:
            self.insights_expanded_result_id = 0
        else:
            self.insights_expanded_result_id = rid

    async def refresh_insights(self):
        """Refresh the insights task list and currently viewed results."""
        self.insights_loading = True
        yield

        await self._load_insights_tasks()

        if self.insights_viewing_task_id > 0:
            results_data = await api_client.fetch_scheduled_task_results(
                self.insights_viewing_task_id, limit=20,
            )
            raw_results = results_data.get("results", []) if isinstance(results_data, dict) else []
            self.insights_results = [
                {k: str(v) if v is not None else "" for k, v in r.items()}
                for r in raw_results
            ]

        self.insights_loading = False
        self.insights_message = "✅ Refreshed"

    async def rate_insight_result(self, result_id: str, rating: str):
        """Rate an insight result (thumbs up/down).

        Args:
            result_id: The result ID to rate
            rating: "1" for thumbs up, "-1" for thumbs down, "0" to clear
        """
        rid = int(result_id)
        rate_val = int(rating)
        result = await api_client.rate_insight_result(rid, rate_val)
        if "error" not in result:
            # Update the rating in the local results list
            for r in self.insights_results:
                if r.get("id") == result_id or r.get("id") == str(rid):
                    r["rating"] = str(rate_val)
                    break
            self.insights_message = "👍 Rated" if rate_val == 1 else (
                "👎 Rated" if rate_val == -1 else "Rating cleared"
            )

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

    # =====================================================================
    # ENTITY STORE
    # =====================================================================

    async def _load_entity_list(self, query: str | None = None):
        """Fetch person list from backend."""
        self.entity_loading = True
        raw = await api_client.fetch_entities(query=query or None)
        processed: list[dict[str, str]] = []
        for p in raw:
            # Compute alias count and preview from list
            aliases_raw = p.get("aliases", [])
            if isinstance(aliases_raw, list):
                alias_count = len(aliases_raw)
                # Filter to name-like aliases (not pure numeric/phone)
                name_aliases = [
                    a for a in aliases_raw
                    if isinstance(a, str) and not a.replace("+", "").isdigit()
                ]
                aliases_preview = ", ".join(name_aliases[:3])
                if len(name_aliases) > 3:
                    aliases_preview += f" +{len(name_aliases) - 3}"
            else:
                alias_count = p.get("alias_count", 0)
                aliases_preview = ""

            # Compute fact count from dict or use provided count
            facts_raw = p.get("facts", {})
            if isinstance(facts_raw, dict):
                fact_count = len(facts_raw)
            else:
                fact_count = p.get("fact_count", 0)

            # Use display_name (bilingual) if available, else canonical_name
            # For list cards, show only English part; Hebrew goes to aliases preview
            raw_display = str(p.get("display_name", "") or p.get("canonical_name", ""))
            if " / " in raw_display:
                # Bilingual: "English Name / Hebrew Name" — show English in card
                parts = raw_display.split(" / ", 1)
                display_name = parts[0].strip()
                # Prepend Hebrew part to aliases preview
                hebrew_part = parts[1].strip() if len(parts) > 1 else ""
                if hebrew_part and hebrew_part not in aliases_preview:
                    aliases_preview = hebrew_part + (", " + aliases_preview if aliases_preview else "")
            else:
                display_name = raw_display

            processed.append({
                "id": str(p.get("id", "")),
                "canonical_name": display_name,
                "phone": str(p.get("phone", "") or ""),
                "whatsapp_id": str(p.get("whatsapp_id", "") or ""),
                "alias_count": str(alias_count),
                "fact_count": str(fact_count),
                "aliases_preview": aliases_preview,
            })
        self.entity_persons = processed
        self.entity_loading = False

    async def search_entities(self):
        """Search entities using current entity_search value.
        
        When in merge mode, preserves already-selected persons at the top
        of the list even if they don't match the current search query.
        """
        q = self.entity_search.strip()
        
        if self.entity_merge_mode and self.entity_merge_selection:
            # Save current selection before reloading
            saved_selection = list(self.entity_merge_selection)
            
            # Load search results
            await self._load_entity_list(q if q else None)
            
            # Restore selection (it's preserved in state, but the persons
            # may not be in the filtered list — fetch them individually)
            self.entity_merge_selection = saved_selection
            
            # Ensure selected persons appear in the list
            current_ids = {p["id"] for p in self.entity_persons}
            missing_ids = [pid for pid in saved_selection if pid not in current_ids]
            
            if missing_ids:
                for pid in missing_ids:
                    data = await api_client.fetch_entity(int(pid))
                    if "error" not in data:
                        display_name = str(data.get("display_name", "") or data.get("canonical_name", ""))
                        aliases_raw = data.get("aliases", [])
                        alias_count = len(aliases_raw) if isinstance(aliases_raw, list) else 0
                        facts_raw = data.get("facts", {})
                        fact_count = len(facts_raw) if isinstance(facts_raw, dict) else 0
                        
                        self.entity_persons.insert(0, {
                            "id": str(data.get("id", "")),
                            "canonical_name": display_name,
                            "phone": str(data.get("phone", "") or ""),
                            "whatsapp_id": str(data.get("whatsapp_id", "") or ""),
                            "alias_count": str(alias_count),
                            "fact_count": str(fact_count),
                            "aliases_preview": "",
                        })
        else:
            await self._load_entity_list(q if q else None)

    async def select_entity(self, person_id: str):
        """Load full person detail for the side panel."""
        pid = int(person_id)
        self.entity_selected_id = pid
        self.entity_detail_loading = True
        self.entity_editing_fact_key = ""
        yield
        data = await api_client.fetch_entity(pid)
        if "error" not in data:
            self.entity_detail = data
        else:
            self.entity_save_message = f"❌ {data['error']}"
        self.entity_detail_loading = False

    def close_entity_detail(self):
        """Close the detail side panel."""
        self.entity_selected_id = 0
        self.entity_detail = {}
        self.entity_editing_fact_key = ""
        self.entity_editing_name = False
        self.entity_editing_name_value = ""
        self.entity_new_fact_key = ""
        self.entity_new_fact_value = ""
        self.entity_new_alias = ""

    # --- Entity name editing ---

    def start_edit_name(self):
        """Enter inline edit mode for the person's canonical name."""
        current_name = self.entity_detail.get("canonical_name", "")
        self.entity_editing_name = True
        self.entity_editing_name_value = current_name

    def cancel_edit_name(self):
        """Cancel inline name editing."""
        self.entity_editing_name = False
        self.entity_editing_name_value = ""

    async def save_name_edit(self):
        """Save the edited person name via API."""
        new_name = self.entity_editing_name_value.strip()
        if not new_name or self.entity_selected_id <= 0:
            return
        result = await api_client.rename_entity(self.entity_selected_id, new_name)
        if "error" in result:
            self.entity_save_message = f"❌ {result['error']}"
        else:
            self.entity_save_message = f"✅ Renamed to '{new_name}'"
            self.entity_editing_name = False
            self.entity_editing_name_value = ""
            # Refresh detail and person list
            data = await api_client.fetch_entity(self.entity_selected_id)
            if "error" not in data:
                self.entity_detail = data
            await self._load_entity_list()

    # --- All Facts tab CRUD ---

    async def delete_fact_from_all_tab(self, person_id: str, fact_key: str):
        """Delete a fact from the All Facts tab."""
        pid = int(person_id)
        result = await api_client.delete_entity_fact(pid, fact_key)
        if "error" in result:
            self.entity_save_message = f"❌ {result['error']}"
        else:
            self.entity_save_message = f"✅ Fact '{fact_key}' deleted"
        # Refresh the All Facts table
        await self.load_all_entity_facts()
        # Also refresh detail if this person is selected
        if self.entity_selected_id == pid:
            data = await api_client.fetch_entity(pid)
            if "error" not in data:
                self.entity_detail = data
        stats = await api_client.fetch_entity_stats()
        self.entity_stats = {k: str(v) for k, v in stats.items()}

    async def edit_fact_from_all_tab(self, person_id: str, fact_key: str, current_value: str):
        """Navigate to person detail with inline edit open for a specific fact."""
        pid = int(person_id)
        self.entity_tab = "people"
        self.entity_selected_id = pid
        self.entity_detail_loading = True
        yield  # type: ignore[misc]
        data = await api_client.fetch_entity(pid)
        if "error" not in data:
            self.entity_detail = data
            self.entity_editing_fact_key = fact_key
            self.entity_editing_fact_value = current_value
        else:
            self.entity_save_message = f"❌ {data['error']}"
        self.entity_detail_loading = False

    async def delete_entity(self):
        """Delete the currently selected person."""
        if self.entity_selected_id <= 0:
            return
        pid = self.entity_selected_id
        result = await api_client.delete_entity(pid)
        if "error" in result:
            self.entity_save_message = f"❌ {result['error']}"
        else:
            self.entity_save_message = "✅ Person deleted"
            self.close_entity_detail()
            await self._load_entity_list()
            stats = await api_client.fetch_entity_stats()
            self.entity_stats = {k: str(v) for k, v in stats.items()}

    async def add_entity_fact(self):
        """Add a new fact to the selected person."""
        key = self.entity_new_fact_key.strip()
        value = self.entity_new_fact_value.strip()
        if not key or not value or self.entity_selected_id <= 0:
            return
        result = await api_client.add_entity_fact(
            self.entity_selected_id, key, value,
        )
        if "error" in result:
            self.entity_save_message = f"❌ {result['error']}"
        else:
            self.entity_save_message = f"✅ Fact '{key}' saved"
            self.entity_new_fact_key = ""
            self.entity_new_fact_value = ""
            # Refresh detail
            data = await api_client.fetch_entity(self.entity_selected_id)
            if "error" not in data:
                self.entity_detail = data
            stats = await api_client.fetch_entity_stats()
            self.entity_stats = {k: str(v) for k, v in stats.items()}

    async def save_entity_fact_edit(self):
        """Save an inline fact edit."""
        key = self.entity_editing_fact_key.strip()
        value = self.entity_editing_fact_value.strip()
        if not key or not value or self.entity_selected_id <= 0:
            return
        result = await api_client.add_entity_fact(
            self.entity_selected_id, key, value,
        )
        if "error" in result:
            self.entity_save_message = f"❌ {result['error']}"
        else:
            self.entity_save_message = f"✅ Fact '{key}' updated"
        self.entity_editing_fact_key = ""
        self.entity_editing_fact_value = ""
        # Refresh detail
        data = await api_client.fetch_entity(self.entity_selected_id)
        if "error" not in data:
            self.entity_detail = data

    def start_edit_fact(self, fact_key: str, current_value: str):
        """Enter inline edit mode for a fact."""
        self.entity_editing_fact_key = fact_key
        self.entity_editing_fact_value = current_value

    def cancel_edit_fact(self):
        """Cancel inline fact editing."""
        self.entity_editing_fact_key = ""
        self.entity_editing_fact_value = ""

    async def delete_entity_fact(self, fact_key: str):
        """Delete a fact from the selected person."""
        if self.entity_selected_id <= 0:
            return
        result = await api_client.delete_entity_fact(
            self.entity_selected_id, fact_key,
        )
        if "error" in result:
            self.entity_save_message = f"❌ {result['error']}"
        else:
            self.entity_save_message = f"✅ Fact deleted"
        # Refresh detail
        data = await api_client.fetch_entity(self.entity_selected_id)
        if "error" not in data:
            self.entity_detail = data
        stats = await api_client.fetch_entity_stats()
        self.entity_stats = {k: str(v) for k, v in stats.items()}

    async def add_entity_alias(self):
        """Add a new alias to the selected person."""
        alias = self.entity_new_alias.strip()
        if not alias or self.entity_selected_id <= 0:
            return
        result = await api_client.add_entity_alias(
            self.entity_selected_id, alias,
        )
        if "error" in result:
            self.entity_save_message = f"❌ {result['error']}"
        else:
            self.entity_save_message = f"✅ Alias '{alias}' added"
            self.entity_new_alias = ""
        # Refresh detail
        data = await api_client.fetch_entity(self.entity_selected_id)
        if "error" not in data:
            self.entity_detail = data
        stats = await api_client.fetch_entity_stats()
        self.entity_stats = {k: str(v) for k, v in stats.items()}

    async def delete_entity_alias(self, alias_id: str):
        """Delete an alias from the selected person."""
        if self.entity_selected_id <= 0:
            return
        result = await api_client.delete_entity_alias(
            self.entity_selected_id, int(alias_id),
        )
        if "error" in result:
            self.entity_save_message = f"❌ {result['error']}"
        else:
            self.entity_save_message = "✅ Alias deleted"
        # Refresh detail
        data = await api_client.fetch_entity(self.entity_selected_id)
        if "error" not in data:
            self.entity_detail = data
        stats = await api_client.fetch_entity_stats()
        self.entity_stats = {k: str(v) for k, v in stats.items()}

    async def seed_entities(self):
        """Seed entity store from WhatsApp contacts."""
        self.entity_seed_status = "seeding"
        self.entity_seed_message = "⏳ Seeding from WhatsApp contacts…"
        yield
        result = await api_client.seed_entities()
        if "error" in result:
            self.entity_seed_status = "error"
            self.entity_seed_message = f"❌ {result['error']}"
        else:
            r = result.get("result", {})
            created = r.get("created", 0)
            updated = r.get("updated", 0)
            skipped = r.get("skipped", 0)
            self.entity_seed_status = "complete"
            self.entity_seed_message = (
                f"✅ Seeded: {created} created, {updated} updated, {skipped} skipped"
            )
            await self._load_entity_list()
            stats = await api_client.fetch_entity_stats()
            self.entity_stats = {k: str(v) for k, v in stats.items()}

    async def load_all_entity_facts(self):
        """Load all facts across all persons for the All Facts tab."""
        key_filter = self.entity_fact_key_filter or None
        data = await api_client.fetch_all_entity_facts(key=key_filter)
        raw_facts = data.get("facts", [])
        result: list[dict[str, str]] = []
        for f in raw_facts:
            row = {k: str(v) if v is not None else "" for k, v in f.items()}
            # Format source_ref into human-readable cause text
            raw_ref = str(f.get("source_ref", "") or "")
            src_type = str(f.get("source_type", "") or "")
            row["source_ref"] = _format_source_ref(raw_ref, src_type)
            result.append(row)
        self.entity_all_facts = result
        keys = data.get("available_keys", [])
        self.entity_fact_keys = [str(k) for k in keys]

    async def refresh_entities(self):
        """Refresh entity list and stats."""
        await self._load_entity_list()
        stats = await api_client.fetch_entity_stats()
        self.entity_stats = {k: str(v) for k, v in stats.items()}

    async def cleanup_entities(self):
        """Remove persons with garbage/invalid names."""
        self.entity_save_message = "⏳ Cleaning up…"
        yield
        result = await api_client.cleanup_entities()
        if "error" in result:
            self.entity_save_message = f"❌ {result['error']}"
        else:
            deleted = result.get("deleted", 0)
            self.entity_save_message = f"✅ Removed {deleted} garbage entries"
            await self._load_entity_list()
            stats = await api_client.fetch_entity_stats()
            self.entity_stats = {k: str(v) for k, v in stats.items()}

    async def execute_merge(self):
        """Merge selected persons — first selected becomes the target."""
        if len(self.entity_merge_selection) < 2:
            self.entity_save_message = "❌ Select at least 2 persons to merge"
            return

        target_id = int(self.entity_merge_selection[0])
        source_ids = [int(s) for s in self.entity_merge_selection[1:]]

        self.entity_save_message = "⏳ Merging…"
        yield

        result = await api_client.merge_entities(target_id, source_ids)
        if "error" in result:
            self.entity_save_message = f"❌ {result['error']}"
        else:
            merged = result.get("sources_deleted", 0)
            aliases = result.get("aliases_moved", 0)
            facts = result.get("facts_moved", 0)
            name = result.get("display_name", "")
            self.entity_save_message = (
                f"✅ Merged {merged + 1} persons into \"{name}\" "
                f"({aliases} aliases, {facts} facts moved)"
            )
            self.entity_merge_selection = []
            self.entity_merge_mode = False
            # Close detail if the selected entity was a merge source
            if self.entity_selected_id > 0 and str(self.entity_selected_id) in [str(s) for s in source_ids]:
                self.close_entity_detail()
            await self._load_entity_list()
            stats = await api_client.fetch_entity_stats()
            self.entity_stats = {k: str(v) for k, v in stats.items()}
            # Reload detail if target was selected
            if self.entity_selected_id == target_id:
                data = await api_client.fetch_entity(target_id)
                if "error" not in data:
                    self.entity_detail = data

    async def update_entity_display_name(self):
        """Recalculate bilingual display name for the selected person."""
        if self.entity_selected_id <= 0:
            return
        result = await api_client.update_entity_display_name(self.entity_selected_id)
        if "error" in result:
            self.entity_save_message = f"❌ {result['error']}"
        elif result.get("status") == "ok":
            new_name = result.get("display_name", "")
            self.entity_save_message = f"✅ Display name updated: {new_name}"
            # Refresh detail and list
            data = await api_client.fetch_entity(self.entity_selected_id)
            if "error" not in data:
                self.entity_detail = data
            await self._load_entity_list()
        else:
            self.entity_save_message = result.get("message", "No change needed")

    async def load_entity_graph(self):
        """Fetch person-relationship-asset graph data for display."""
        self.entity_graph_loading = True
        yield
        data = await api_client.fetch_entity_graph(limit=200)
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        # Filter: only show persons with relationships, assets, or facts
        edge_person_ids: set = set()
        for e in edges:
            edge_person_ids.add(str(e.get("source_id", "")))
            edge_person_ids.add(str(e.get("target_id", "")))

        filtered_nodes: list = []
        for n in nodes:
            nid = n.get("id", "")
            total = int(n.get("total_assets", "0"))
            facts = int(n.get("fact_count", "0"))
            has_rel = str(nid) in edge_person_ids
            if total > 0 or has_rel or facts > 0:
                # Add a badge field for display
                badges = []
                if has_rel:
                    badges.append("🔗 has relationships")
                summary = n.get("asset_summary", "")
                if summary:
                    badges.append(f"📦 {summary}")
                if facts > 0:
                    badges.append(f"📋 {n.get('fact_count', '0')} facts")
                n["badges"] = " · ".join(badges) if badges else ""
                filtered_nodes.append(n)

        self.entity_graph_nodes = filtered_nodes  # type: ignore[assignment]
        self.entity_graph_edges = edges  # type: ignore[assignment]
        self.entity_graph_loading = False

    async def load_full_entity_graph(self):
        """Fetch full graph data (persons + assets + all edge types) for interactive visualization."""
        self.full_graph_loading = True
        yield
        data = await api_client.fetch_full_entity_graph(
            limit_persons=100,
            limit_assets=10,
            include_asset_edges=True,
        )
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        # Compute stats
        person_count = sum(1 for n in nodes if n.get("type") == "person")
        asset_count = sum(1 for n in nodes if n.get("type") == "asset")
        ii_edges = sum(1 for e in edges if e.get("edge_category") == "identity_identity")
        ia_edges = sum(1 for e in edges if e.get("edge_category") == "identity_asset")
        aa_edges = sum(1 for e in edges if e.get("edge_category") == "asset_asset")

        self.full_graph_nodes = nodes  # type: ignore[assignment]
        self.full_graph_edges = edges  # type: ignore[assignment]
        self.full_graph_stats = {
            "persons": str(person_count),
            "assets": str(asset_count),
            "identity_edges": str(ii_edges),
            "asset_links": str(ia_edges),
            "asset_edges": str(aa_edges),
            "total_nodes": str(len(nodes)),
            "total_edges": str(len(edges)),
        }

        # Store figure data as dict (state vars must be serializable)
        fig_dict = self._build_plotly_graph(nodes, edges)
        if fig_dict and fig_dict.get("data"):
            self.full_graph_figure_data = fig_dict
            self.entity_graph_html = "loaded"  # Flag that graph is ready
        else:
            self.full_graph_figure_data = {}
            self.entity_graph_html = ""
        self.full_graph_loading = False

    @rx.var(cache=True)
    def full_graph_figure(self) -> go.Figure:
        """Computed var: convert stored dict to a Plotly Figure for rx.plotly()."""
        if not self.full_graph_figure_data or not self.full_graph_figure_data.get("data"):
            return go.Figure()
        return go.Figure(self.full_graph_figure_data)

    @staticmethod
    def _build_plotly_graph(nodes: list, edges: list) -> dict:
        """Generate a Plotly network graph figure as a dict.

        Uses a spring layout to position nodes and renders them as
        a Plotly scatter plot (nodes) + line traces (edges).
        Returns a dict suitable for rx.plotly(data=...).
        """
        import math
        import random

        # Color map by node type / asset_type
        color_map = {
            "person": "#4F81BD",
            "whatsapp_msg": "#25D366",
            "document": "#FF8C00",
            "call_recording": "#9B59B6",
            "gmail": "#EA4335",
            "asset": "#95A5A6",
            "linked": "#BDC3C7",
        }
        # Edge color map by category
        edge_color_map = {
            "identity_identity": "rgba(231,76,60,0.4)",
            "identity_asset": "rgba(52,152,219,0.3)",
            "asset_asset": "rgba(46,204,113,0.3)",
        }

        if not nodes:
            return {}

        # Build adjacency and node index
        node_ids = [n.get("id", "") for n in nodes]
        node_idx = {nid: i for i, nid in enumerate(node_ids)}

        # Simple force-directed layout (Fruchterman-Reingold approximation)
        n = len(nodes)
        random.seed(42)
        pos_x = [random.uniform(-1, 1) for _ in range(n)]
        pos_y = [random.uniform(-1, 1) for _ in range(n)]

        # Build edge list for layout
        edge_pairs = []
        for e in edges:
            s = node_idx.get(e.get("source", ""))
            t = node_idx.get(e.get("target", ""))
            if s is not None and t is not None and s != t:
                edge_pairs.append((s, t))

        # Run simple spring layout iterations
        k = 1.0 / math.sqrt(max(n, 1))  # Optimal distance
        for _ in range(80):
            # Repulsion between all pairs (approximated for up to ~500 nodes)
            dx = [0.0] * n
            dy = [0.0] * n
            for i in range(n):
                for j in range(i + 1, min(n, i + 50)):  # Limit pairs for speed
                    diffx = pos_x[i] - pos_x[j]
                    diffy = pos_y[i] - pos_y[j]
                    dist = math.sqrt(diffx * diffx + diffy * diffy) + 0.001
                    force = k * k / dist * 0.1
                    fx = diffx / dist * force
                    fy = diffy / dist * force
                    dx[i] += fx
                    dy[i] += fy
                    dx[j] -= fx
                    dy[j] -= fy

            # Attraction along edges
            for s, t in edge_pairs:
                diffx = pos_x[s] - pos_x[t]
                diffy = pos_y[s] - pos_y[t]
                dist = math.sqrt(diffx * diffx + diffy * diffy) + 0.001
                force = dist * dist / k * 0.01
                fx = diffx / dist * force
                fy = diffy / dist * force
                dx[s] -= fx
                dy[s] -= fy
                dx[t] += fx
                dy[t] += fy

            # Apply with damping
            for i in range(n):
                disp = math.sqrt(dx[i] * dx[i] + dy[i] * dy[i]) + 0.001
                cap = min(disp, 0.1) / disp
                pos_x[i] += dx[i] * cap
                pos_y[i] += dy[i] * cap

        # Build edge traces (one per edge category for legend)
        edge_traces = {}
        for e in edges:
            cat = e.get("edge_category", "other")
            s = node_idx.get(e.get("source", ""))
            t = node_idx.get(e.get("target", ""))
            if s is None or t is None:
                continue
            if cat not in edge_traces:
                edge_traces[cat] = {"x": [], "y": [], "color": edge_color_map.get(cat, "rgba(150,150,150,0.3)")}
            trace = edge_traces[cat]
            trace["x"].extend([pos_x[s], pos_x[t], None])
            trace["y"].extend([pos_y[s], pos_y[t], None])

        # Build Plotly traces
        traces = []

        # Edge traces
        cat_labels = {
            "identity_identity": "Person↔Person",
            "identity_asset": "Person↔Asset",
            "asset_asset": "Asset↔Asset",
        }
        for cat, trace_data in edge_traces.items():
            traces.append({
                "x": trace_data["x"],
                "y": trace_data["y"],
                "mode": "lines",
                "type": "scatter",
                "line": {"width": 1, "color": trace_data["color"]},
                "hoverinfo": "none",
                "name": cat_labels.get(cat, cat),
                "showlegend": True,
            })

        # Group nodes by type for colored legend
        node_groups: dict = {}
        for i, nd in enumerate(nodes):
            ntype = nd.get("type", "asset")
            atype = nd.get("asset_type", "")
            group_key = ntype if ntype == "person" else (atype or "asset")
            if group_key not in node_groups:
                node_groups[group_key] = {"x": [], "y": [], "text": [], "hover": [], "color": color_map.get(group_key, "#95A5A6"), "size": []}
            g = node_groups[group_key]
            g["x"].append(pos_x[i])
            g["y"].append(pos_y[i])

            label = nd.get("label", "?")
            if len(label) > 20:
                label = label[:17] + "…"
            g["text"].append(label)

            # Hover text
            if ntype == "person":
                hover = f"<b>{nd.get('label', '')}</b><br>Facts: {nd.get('fact_count', 0)}<br>Assets: {nd.get('total_assets', 0)}"
            else:
                hover = f"<b>{nd.get('label', '')}</b><br>Type: {atype}"
            g["hover"].append(hover)
            g["size"].append(18 if ntype == "person" else 10)

        group_labels = {
            "person": "👤 Person",
            "whatsapp_msg": "💬 WhatsApp",
            "document": "📄 Document",
            "call_recording": "📞 Call",
            "gmail": "📧 Email",
            "asset": "📦 Asset",
            "linked": "🔗 Linked",
        }
        for group_key, g in node_groups.items():
            traces.append({
                "x": g["x"],
                "y": g["y"],
                "mode": "markers+text",
                "type": "scatter",
                "marker": {
                    "size": g["size"],
                    "color": g["color"],
                    "line": {"width": 1, "color": "#fff"},
                },
                "text": g["text"],
                "textposition": "top center",
                "textfont": {"size": 9},
                "hovertext": g["hover"],
                "hoverinfo": "text",
                "name": group_labels.get(group_key, group_key),
                "showlegend": True,
            })

        fig = {
            "data": traces,
            "layout": {
                "showlegend": True,
                "legend": {"x": 0, "y": 1, "bgcolor": "rgba(255,255,255,0.8)"},
                "hovermode": "closest",
                "xaxis": {"showgrid": False, "zeroline": False, "showticklabels": False, "visible": False},
                "yaxis": {"showgrid": False, "zeroline": False, "showticklabels": False, "visible": False},
                "margin": {"l": 10, "r": 10, "t": 10, "b": 10},
                "paper_bgcolor": "#fafafa",
                "plot_bgcolor": "#fafafa",
                "height": 600,
                "dragmode": "pan",
            },
        }

        return fig

    async def load_merge_candidates(self):
        """Fetch merge suggestions from the backend."""
        self.entity_candidates_loading = True
        yield
        data = await api_client.fetch_merge_candidates(limit=50)
        raw_candidates = data.get("candidates", [])
        # Flatten candidates into renderable format
        processed: list[dict[str, str]] = []
        for group in raw_candidates:
            reason = group.get("reason", "")
            persons = group.get("persons", [])
            if len(persons) < 2:
                continue
            # Build a flat dict per candidate group
            person_ids = [str(p.get("id", "")) for p in persons]
            person_names = [str(p.get("canonical_name", "")) for p in persons]
            person_details = []
            for p in persons:
                facts = p.get("fact_count", 0)
                aliases = p.get("alias_count", 0)
                phone = str(p.get("phone", "") or "")
                email = str(p.get("email", "") or "")
                detail = str(p.get("canonical_name", ""))
                extras = []
                if phone:
                    extras.append(phone)
                if email:
                    extras.append(email)
                if extras:
                    detail += f" ({', '.join(extras)})"
                detail += f" [{facts}f, {aliases}a]"
                person_details.append(detail)

            processed.append({
                "reason": reason,
                "ids": ",".join(person_ids),
                "names": " ↔ ".join(person_names[:4]),
                "details": " | ".join(person_details[:4]),
                "count": str(len(persons)),
                "target_id": person_ids[0] if person_ids else "",
                "source_ids": ",".join(person_ids[1:]),
            })
        self.entity_merge_candidates = processed  # type: ignore[assignment]
        self.entity_candidates_loading = False

    async def merge_candidate_group(self, target_id: str, source_ids_str: str):
        """Merge a suggested candidate group with one click."""
        if not target_id or not source_ids_str:
            return
        source_ids = [int(s) for s in source_ids_str.split(",") if s]
        self.entity_save_message = "⏳ Merging…"
        yield
        result = await api_client.merge_entities(int(target_id), source_ids)
        if "error" in result:
            self.entity_save_message = f"❌ {result['error']}"
        else:
            merged = result.get("sources_deleted", 0)
            name = result.get("display_name", "")
            self.entity_save_message = f"✅ Merged {merged + 1} → \"{name}\""
            # Refresh suggestions and entity list
            await self._reload_merge_candidates()
            await self._load_entity_list()
            stats = await api_client.fetch_entity_stats()
            self.entity_stats = {k: str(v) for k, v in stats.items()}

    async def _reload_merge_candidates(self):
        """Internal: reload merge candidates without yield (non-generator)."""
        data = await api_client.fetch_merge_candidates(limit=50)
        raw_candidates = data.get("candidates", [])
        processed: list[dict[str, str]] = []
        for group in raw_candidates:
            reason = group.get("reason", "")
            persons = group.get("persons", [])
            if len(persons) < 2:
                continue
            person_ids = [str(p.get("id", "")) for p in persons]
            person_names = [str(p.get("canonical_name", "")) for p in persons]
            person_details = []
            for p in persons:
                facts = p.get("fact_count", 0)
                aliases = p.get("alias_count", 0)
                phone = str(p.get("phone", "") or "")
                email = str(p.get("email", "") or "")
                detail = str(p.get("canonical_name", ""))
                extras = []
                if phone:
                    extras.append(phone)
                if email:
                    extras.append(email)
                if extras:
                    detail += f" ({', '.join(extras)})"
                detail += f" [{facts}f, {aliases}a]"
                person_details.append(detail)
            processed.append({
                "reason": reason,
                "ids": ",".join(person_ids),
                "names": " ↔ ".join(person_names[:4]),
                "details": " | ".join(person_details[:4]),
                "count": str(len(persons)),
                "target_id": person_ids[0] if person_ids else "",
                "source_ids": ",".join(person_ids[1:]),
            })
        self.entity_merge_candidates = processed  # type: ignore[assignment]


# =========================================================================
# HELPERS
# =========================================================================

# All rich content field keys — used to create empty message dicts
_RICH_FIELDS = (
    "image_urls", "image_captions",
    "ics_url", "ics_title",
    "button_prompt", "button_options",
)


def _empty_msg(role: str, content: str) -> dict[str, str]:
    """Create a message dict with all required fields initialized to empty."""
    msg: dict[str, str] = {
        "role": role,
        "content": content,
        "sources": "",
        "cost": "",
        "rich_content": "",
    }
    for field in _RICH_FIELDS:
        msg[field] = ""
    return msg


def _flatten_rich_content(rich_content: list[dict]) -> dict[str, str]:
    """Flatten a list of rich content blocks into message-level string fields.

    Converts structured rich content blocks into pipe-separated string fields
    that Reflex can consume directly in component templates.

    Returns dict with keys: image_urls, image_captions, ics_url, ics_title,
    button_prompt, button_options.
    """
    result: dict[str, str] = {field: "" for field in _RICH_FIELDS}

    if not rich_content:
        return result

    image_urls: list[str] = []
    image_captions: list[str] = []

    for block in rich_content:
        block_type = block.get("type", "")

        if block_type == "image":
            url = block.get("url", "")
            if url:
                # Prefix with public API URL so the browser can reach the backend
                image_urls.append(f"{_API_PUBLIC_URL}{url}")
                image_captions.append(block.get("caption", ""))

        elif block_type == "ics_event":
            # Take the first event only
            if not result["ics_url"]:
                url = block.get("download_url", "")
                result["ics_url"] = f"{_API_PUBLIC_URL}{url}" if url else ""
                result["ics_title"] = block.get("title", "Calendar Event")

        elif block_type == "buttons":
            options = block.get("options", [])
            if options:
                result["button_prompt"] = block.get("prompt", "")
                result["button_options"] = "|".join(
                    opt.get("label", opt.get("value", ""))
                    for opt in options
                )

    if image_urls:
        result["image_urls"] = "|".join(image_urls)
        result["image_captions"] = "|".join(image_captions)

    # Also keep the raw JSON for any future use
    result["rich_content"] = json.dumps(rich_content)

    return result


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
    """Format source citations as markdown for the collapsible sources section.

    The ``content`` field already contains a cleanly formatted string from
    the RAG layer in the form ``Source | date | item text``, so we only
    need to number the entries and append a relevance score.

    Scores > 1 are raw timestamps from recency search and are not shown.
    Scores between 0 and 1 are displayed as percentages.
    """
    if not sources:
        return ""
    lines: list[str] = []
    for i, src in enumerate(sources):
        content = src.get("content", "")[:300]
        score = src.get("score")
        # Only show score when it's a meaningful relevance value (0–1 range).
        # Scores > 1 are Unix timestamps from recency search — skip those.
        score_str = f" ({score:.0%})" if score and 0 < score <= 1 else ""
        if content:
            lines.append(f"**{i + 1}.** {content}{'…' if len(content) >= 300 else ''}{score_str}\n")
    return "\n".join(lines)
