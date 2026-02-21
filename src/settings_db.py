"""SQLite-backed settings database for Lucy.

All application configuration is stored in a SQLite database. On first run,
defaults are seeded and any values found in environment variables (from .env)
are overlaid. After that, all config is read from and written to SQLite,
editable through the Settings UI page.

Database location: data/settings.db (relative to project root)
"""

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Database path resolution
# ---------------------------------------------------------------------------

def _resolve_db_path() -> str:
    """Resolve the SQLite database file path.
    
    Searches for a 'data' directory relative to this file's location,
    walking up to the project root. Creates the directory if needed.
    
    Returns:
        Absolute path to settings.db
    """
    # Check for explicit env override
    explicit = os.environ.get("SETTINGS_DB_PATH")
    if explicit:
        Path(explicit).parent.mkdir(parents=True, exist_ok=True)
        return explicit
    
    # Walk up from this file to find the project root (where docker-compose.yml lives)
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "docker-compose.yml").exists() or (parent / ".git").exists():
            db_dir = parent / "data"
            db_dir.mkdir(parents=True, exist_ok=True)
            return str(db_dir / "settings.db")
    
    # Docker container: /app/data is the standard volume mount path
    docker_data = Path("/app/data")
    if docker_data.exists() or Path("/app/src").exists():
        docker_data.mkdir(parents=True, exist_ok=True)
        return str(docker_data / "settings.db")
    
    # Fallback: create data/ next to this file
    db_dir = current / "data"
    db_dir.mkdir(parents=True, exist_ok=True)
    return str(db_dir / "settings.db")


DB_PATH = _resolve_db_path()

# ---------------------------------------------------------------------------
# Mapping: SQLite key -> environment variable name (for first-run seed)
# ---------------------------------------------------------------------------

ENV_KEY_MAP: Dict[str, str] = {
    # Secrets
    "openai_api_key": "OPENAI_API_KEY",
    "google_api_key": "GOOGLE_API_KEY",
    "waha_api_key": "WAHA_API_KEY",
    "cohere_api_key": "COHERE_API_KEY",
    # LLM
    "llm_provider": "LLM_PROVIDER",
    "openai_model": "OPENAI_MODEL",
    "openai_temperature": "OPENAI_TEMPERATURE",
    "gemini_model": "GEMINI_MODEL",
    "gemini_temperature": "GEMINI_TEMPERATURE",
    "image_provider": "IMAGE_PROVIDER",
    "imagen_model": "IMAGEN_MODEL",
    "system_prompt": "SYSTEM_PROMPT",
    # RAG
    "rag_collection_name": "RAG_COLLECTION_NAME",
    "rag_min_score": "RAG_MIN_SCORE",
    "rag_max_context_tokens": "RAG_MAX_CONTEXT_TOKENS",
    "rag_default_k": "RAG_DEFAULT_K",
    "embedding_model": "EMBEDDING_MODEL",
    "rag_vector_size": "RAG_VECTOR_SIZE",
    "rag_fulltext_score_sender": "RAG_FULLTEXT_SCORE_SENDER",
    "rag_fulltext_score_chat_name": "RAG_FULLTEXT_SCORE_CHAT_NAME",
    "rag_fulltext_score_message": "RAG_FULLTEXT_SCORE_MESSAGE",
    # WhatsApp
    "chat_prefix": "CHAT_PREFIX",
    "dalle_prefix": "DALLE_PREFIX",
    "waha_session_name": "WAHA_SESSION_NAME",
    "dalle_model": "DALLE_MODEL",
    # Infrastructure
    "redis_host": "REDIS_HOST",
    "redis_port": "REDIS_PORT",
    "qdrant_host": "QDRANT_HOST",
    "qdrant_port": "QDRANT_PORT",
    "waha_base_url": "WAHA_BASE_URL",
    "webhook_url": "WEBHOOK_URL",
    # App
    "log_level": "LOG_LEVEL",
    "redis_ttl": "REDIS_TTL",
    "session_ttl_minutes": "SESSION_TTL_MINUTES",
    "session_max_history": "SESSION_MAX_HISTORY",
    "timezone": "TIMEZONE",
    "ui_api_url": "UI_API_URL",
    "cost_tracking_enabled": "COST_TRACKING_ENABLED",
}

# Reverse map for looking up env var name from SQLite key
_REVERSE_ENV_MAP: Dict[str, str] = {v: k for k, v in ENV_KEY_MAP.items()}

# ---------------------------------------------------------------------------
# Default settings (seeded on first run)
# ---------------------------------------------------------------------------

# Default system prompt template — {current_datetime} and {hebrew_date} are
# injected at runtime by LlamaIndexRAG._build_system_prompt().
# The known contacts list is appended dynamically after template formatting.
_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant for a personal knowledge base and message archive search system.\n"
    "You have access to retrieved messages and documents from multiple sources "
    "(messaging platforms, documents, emails, etc.) that will be provided as context.\n\n"
    "Current Date/Time: {current_datetime}\n"
    "תאריך ושעה נוכחיים: {hebrew_date}\n\n"
    "Instructions:\n"
    "1. ANALYZE the retrieved messages to find information relevant to the question.\n"
    "2. CITE specific messages when possible — mention who said what and when.\n"
    "3. If multiple messages are relevant, SYNTHESIZE them into a coherent answer.\n"
    "4. For follow-up questions, USE information from earlier in this conversation. "
    "If you already provided an answer about a topic, build on it — do NOT say "
    "\"no information found\" when you discussed it in a previous turn.\n"
    "5. Only say you lack information when BOTH the retrieved context AND the "
    "conversation history don't contain what's needed. Do NOT fabricate information.\n"
    "6. If the question is general (like \"what day is today?\"), answer directly "
    "without referencing the archive.\n"
    "7. Answer in the SAME LANGUAGE as the question.\n"
    "8. Be concise but thorough. Prefer specific facts over vague summaries.\n"
    "9. DISAMBIGUATION: When the user mentions a person's name (first name only) "
    "that matches multiple people in the known contacts list below, ASK the user "
    "to clarify which person they mean BEFORE answering. Present the matching "
    "names as numbered options. For example: 'I found multiple people named Doron: "
    "1) Doron Yazkirovich 2) דורון עלאני — which one did you mean?' "
    "Note: names may appear in different languages/scripts (Hebrew and English) "
    "but refer to the same first name (e.g., דורון = Doron, דוד = David). "
    "Only ask if there is genuine ambiguity — if the user provided a full name "
    "or enough context to identify the person, answer directly."
)

# Each tuple: (key, default_value, category, type, description)
DEFAULT_SETTINGS: List[Tuple[str, str, str, str, str]] = [
    # Secrets
    ("openai_api_key", "", "secrets", "secret", "OpenAI API key"),
    ("google_api_key", "", "secrets", "secret", "Google Gemini API key"),
    ("waha_api_key", "", "secrets", "secret", "WAHA API key"),
    ("cohere_api_key", "", "secrets", "secret", "Cohere API key (for multilingual reranking)"),
    # LLM
    ("llm_provider", "openai", "llm", "select", "LLM provider: openai or gemini"),
    ("openai_model", "gpt-4o", "llm", "text", "OpenAI model name"),
    ("openai_temperature", "0.7", "llm", "float", "OpenAI temperature (0.0-2.0)"),
    ("gemini_model", "gemini-pro", "llm", "text", "Gemini model name"),
    ("gemini_temperature", "0.7", "llm", "float", "Gemini temperature (0.0-2.0)"),
    ("image_provider", "openai", "llm", "select", "Image generation provider: openai or google"),
    ("imagen_model", "imagen-3.0-generate-002", "llm", "text", "Google Imagen model name"),
    ("system_prompt", _DEFAULT_SYSTEM_PROMPT, "llm", "text", "System prompt template for the AI assistant (supports {current_datetime} and {hebrew_date} placeholders)"),
    # RAG
    ("rag_collection_name", "knowledge_base", "rag", "text", "Qdrant collection name"),
    ("rag_min_score", "0.2", "rag", "float", "Minimum similarity score threshold (0.0-1.0)"),
    ("rag_max_context_tokens", "3000", "rag", "int", "Max tokens for RAG context window"),
    ("rag_default_k", "10", "rag", "int", "Default number of context documents"),
    ("rag_context_window_seconds", "1800", "rag", "int", "Time window (seconds) around matched messages for context expansion (default: 1800 = 30 min)"),
    ("embedding_model", "text-embedding-3-large", "rag", "text", "OpenAI embedding model (text-embedding-3-large recommended for Hebrew+English)"),
    ("rag_vector_size", "1024", "rag", "int", "Embedding vector dimensions (1024 recommended for text-embedding-3-large)"),
    ("rag_fulltext_score_sender", "0.95", "rag", "float", "Full-text search score for sender field matches (0.0-1.0)"),
    ("rag_fulltext_score_chat_name", "0.85", "rag", "float", "Full-text search score for chat name field matches (0.0-1.0)"),
    ("rag_fulltext_score_message", "0.75", "rag", "float", "Full-text search score for message content matches (0.0-1.0)"),
    ("rag_morphology_prefixes", "\u05d4\u05d1\u05dc\u05de\u05e9\u05db\u05d5", "rag", "text", "Single-letter prefixes to strip during fulltext tokenization (e.g. Hebrew הבלמשכו). Leave empty to disable."),
    # rag_min_solo_embed_chars removed — all messages are now always embedded
    # individually.  The per-message cost is negligible (~$0.000005) and the
    # previous threshold (80 chars) silently dropped semantically rich Hebrew
    # messages from the RAG index entirely.
    ("rag_chunk_max_messages", "5", "rag", "int", "Max messages per conversation chunk before flushing"),
    ("rag_chunk_buffer_ttl", "120", "rag", "int", "Conversation chunk buffer TTL in seconds"),
    ("rag_chunk_overlap_messages", "1", "rag", "int", "Number of messages to keep as overlap between consecutive conversation chunks"),
    ("gmail_signature_markers", "-- ,--,---", "rag", "text", "Comma-separated email signature delimiters (content after these is stripped)"),
    # RAG — LlamaIndex feature toggles
    ("rag_embedding_cache_enabled", "true", "rag", "bool", "Enable Redis-backed embedding cache (avoids re-embedding unchanged content during re-syncs)"),
    ("rag_rerank_top_n", "10", "rag", "int", "Number of results to keep after Cohere reranking (reranking is auto-enabled when cohere_api_key is set)"),
    ("rag_rerank_model", "rerank-multilingual-v3.0", "rag", "text", "Cohere rerank model (rerank-multilingual-v3.0 supports Hebrew). Reranking auto-activates when cohere_api_key is set."),
    ("rag_query_fusion_num_queries", "3", "rag", "int", "Number of query variants to generate for QueryFusionRetriever (always active)"),
    ("rag_entity_extraction_in_pipeline", "false", "rag", "bool", "Run entity extraction as part of the LlamaIndex ingestion pipeline (instead of standalone)"),
    ("asset_neighborhood_expansion_enabled", "false", "rag", "bool", "Enable asset neighborhood expansion: follow thread/attachment/parent edges at retrieval time for cross-channel coherence"),
    ("pii_redaction_enabled", "false", "rag", "bool", "Enable PII detection and redaction (requires presidio-analyzer and presidio-anonymizer packages)"),
    # RAG — Source display filtering (controls which sources are shown to the user)
    ("source_display_filter_enabled", "true", "rag", "bool", "Filter sources shown to user for relevance (hides context-only nodes, low-score noise)"),
    ("source_display_min_score", "0.5", "rag", "float", "Minimum score for a source to be displayed (0.0-1.0). Sources below this are hidden from the user."),
    ("source_display_max_count", "8", "rag", "int", "Maximum number of sources to show per response"),
    ("source_display_answer_filter", "true", "rag", "bool", "Only show sources whose sender/chat_name/content is referenced in the LLM answer"),
    # Insights — Scheduled Insights quality settings
    ("insight_default_k", "20", "insights", "int", "Documents per sub-query for insights (higher = more thorough, default 20)"),
    ("insight_max_context_tokens", "8000", "insights", "int", "Max context tokens for insight LLM calls (higher than chat default for thorough analysis)"),
    ("insight_llm_model", "", "insights", "text", "LLM model override for insights (empty = use main model). Recommended: o3-mini for analytical depth"),
    ("insight_llm_temperature", "0.1", "insights", "float", "Temperature for insight LLM calls (lower = more factual, 0.0-2.0)"),
    ("insight_decompose_with_llm", "true", "insights", "bool", "Use LLM to decompose custom prompts into sub-queries (templates use predefined sub-queries)"),
    # WhatsApp
    ("chat_prefix", "??", "whatsapp", "text", "Prefix to trigger AI chat response"),
    ("dalle_prefix", "!!", "whatsapp", "text", "Prefix to trigger DALL-E image generation"),
    ("waha_session_name", "default", "whatsapp", "text", "WAHA WhatsApp session name"),
    ("dalle_model", "dall-e-3", "whatsapp", "text", "DALL-E model version"),
    # Infrastructure
    ("redis_host", "redis", "infrastructure", "text", "Redis server hostname"),
    ("redis_port", "6379", "infrastructure", "int", "Redis server port"),
    ("qdrant_host", "qdrant", "infrastructure", "text", "Qdrant server hostname"),
    ("qdrant_port", "6333", "infrastructure", "int", "Qdrant server port"),
    ("waha_base_url", "http://waha:3000", "infrastructure", "text", "WAHA server URL"),
    ("webhook_url", "http://app:8765/webhook", "infrastructure", "text", "Webhook callback URL"),
    # App
    ("log_level", "DEBUG", "app", "select", "Application log level"),
    ("redis_ttl", "604800", "app", "int", "Redis key TTL in seconds"),
    ("session_ttl_minutes", "30", "app", "int", "Conversation session timeout in minutes"),
    ("session_max_history", "20", "app", "int", "Max conversation history turns to keep"),
    ("timezone", "Asia/Jerusalem", "app", "text", "Timezone for date/time display (e.g. Asia/Jerusalem, US/Eastern)"),
    ("ui_api_url", "http://localhost:8765", "app", "text", "Backend API URL for the Streamlit UI"),
    ("cost_tracking_enabled", "true", "app", "bool", "Enable real-time LLM cost tracking and logging"),
]

# Category display order and labels
CATEGORY_META: Dict[str, Dict[str, str]] = {
    "plugins": {"label": "🔌 Plugins", "order": "-1"},
    "secrets": {"label": "🔑 API Keys & Secrets", "order": "0"},
    "llm": {"label": "🤖 LLM Configuration", "order": "1"},
    "rag": {"label": "🔍 RAG Configuration", "order": "2"},
    "insights": {"label": "✨ Scheduled Insights", "order": "2.5"},
    "whatsapp": {"label": "💬 WhatsApp Configuration", "order": "3"},
    "infrastructure": {"label": "🏗️ Infrastructure", "order": "4"},
    "app": {"label": "🔧 App Configuration", "order": "5"},
}

# Select-type options
SELECT_OPTIONS: Dict[str, List[str]] = {
    "llm_provider": ["openai", "gemini"],
    "image_provider": ["openai", "google"],
    "log_level": ["DEBUG", "INFO", "WARNING", "ERROR"],
}


# ---------------------------------------------------------------------------
# Database connection helper
# ---------------------------------------------------------------------------

# Thread-local storage for connection reuse — avoids opening/closing a fresh
# SQLite connection on every single get_setting_value() call.
import threading as _threading
_local = _threading.local()


def _get_connection() -> sqlite3.Connection:
    """Get a thread-local SQLite connection (reused within the same thread).

    Uses threading.local() so each gunicorn worker thread gets its own
    persistent connection, eliminating the overhead of open/close on every call.

    Returns:
        sqlite3.Connection with row_factory set to sqlite3.Row
    """
    conn = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.execute("SELECT 1")  # quick liveness check
            return conn
        except Exception:
            # Connection went stale — close and recreate
            try:
                conn.close()
            except Exception:
                pass
            _local.conn = None

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent read performance
    _local.conn = conn
    return conn


# ---------------------------------------------------------------------------
# Initialization and seeding
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Initialize the settings database.
    
    Creates the settings table if it doesn't exist, then seeds defaults
    and overlays environment variable values on first run (when table is empty).
    
    For existing databases, calls _seed_missing_defaults() to add any new
    settings that were added in code updates (uses INSERT OR IGNORE so
    existing user-modified values are never overwritten).
    """
    conn = _get_connection()
    conn.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'app',
                type TEXT NOT NULL DEFAULT 'text',
                description TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    conn.commit()
    
    # Check if table is empty (first run)
    row = conn.execute("SELECT COUNT(*) as cnt FROM settings").fetchone()
    if row["cnt"] == 0:
        _seed_defaults(conn)
        _seed_from_env(conn)
    else:
        # Existing database — add any new settings from code updates
        _seed_missing_defaults(conn)


def _seed_defaults(conn: sqlite3.Connection) -> None:
    """Seed the database with default settings.
    
    Args:
        conn: Active database connection
    """
    conn.executemany(
        """INSERT OR IGNORE INTO settings (key, value, category, type, description)
           VALUES (?, ?, ?, ?, ?)""",
        DEFAULT_SETTINGS
    )
    conn.commit()


def _seed_missing_defaults(conn: sqlite3.Connection) -> None:
    """Add any new settings that don't yet exist in an existing database.
    
    Uses INSERT OR IGNORE so existing user-modified values are never
    overwritten. This handles the case where new settings are added to
    DEFAULT_SETTINGS in a code update but the database already has data.
    
    Args:
        conn: Active database connection
    """
    conn.executemany(
        """INSERT OR IGNORE INTO settings (key, value, category, type, description)
           VALUES (?, ?, ?, ?, ?)""",
        DEFAULT_SETTINGS
    )
    conn.commit()


def _seed_from_env(conn: sqlite3.Connection) -> None:
    """Overlay environment variable values onto seeded defaults.
    
    Reads from os.environ (which includes .env via python-dotenv)
    and updates any matching settings in the database.
    
    Args:
        conn: Active database connection
    """
    # Ensure .env is loaded into os.environ
    try:
        from dotenv import load_dotenv
        # Find .env by walking up from this file
        current = Path(__file__).resolve().parent
        for parent in [current] + list(current.parents):
            env_path = parent / ".env"
            if env_path.is_file():
                load_dotenv(env_path, override=False)
                break
            if (parent / "docker-compose.yml").exists():
                break
    except ImportError:
        pass  # dotenv not available, rely on existing os.environ
    
    updates = []
    for sqlite_key, env_key in ENV_KEY_MAP.items():
        env_value = os.environ.get(env_key)
        if env_value is not None and env_value.strip():
            updates.append((env_value.strip(), sqlite_key))
    
    if updates:
        conn.executemany(
            "UPDATE settings SET value = ?, updated_at = CURRENT_TIMESTAMP WHERE key = ?",
            updates
        )
        conn.commit()


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------

# Infrastructure keys where environment variables should always win.
# This allows Docker to override hosts/ports via docker-compose `environment:`
# while local dev uses .env values — without touching SQLite.
_ENV_OVERRIDE_KEYS = frozenset({
    "redis_host", "redis_port", "qdrant_host", "qdrant_port",
    "waha_base_url", "webhook_url",
})


# ---------------------------------------------------------------------------
# In-process TTL cache for get_setting_value()
# ---------------------------------------------------------------------------
# Eliminates 10-20+ SQLite round-trips per request.  Settings change rarely
# (only via the UI Settings page), so a 60-second TTL is a good trade-off.

import time as _time

_SETTINGS_CACHE: dict[str, tuple[Optional[str], float]] = {}
_SETTINGS_CACHE_TTL: float = 60.0  # seconds


def invalidate_settings_cache(key: str | None = None) -> None:
    """Clear the in-process settings cache.

    Called automatically by set_settings() and delete_setting() so that
    UI config changes take effect immediately within the same process.

    Args:
        key: If provided, only invalidate that key.  If ``None``, flush all.
    """
    if key is not None:
        _SETTINGS_CACHE.pop(key, None)
    else:
        _SETTINGS_CACHE.clear()


def get_setting_value(key: str) -> Optional[str]:
    """Get a single setting value by key.
    
    Uses an in-process TTL cache (60 s) to avoid repeated SQLite lookups
    for the same key within a short window.  Cache is automatically
    invalidated when settings are modified via set_settings() or
    delete_setting().

    For infrastructure settings (hosts, ports, URLs), environment variables
    take precedence over SQLite. This allows Docker and local dev to coexist
    with the same SQLite database.
    
    Args:
        key: The setting key (e.g., 'openai_model')
        
    Returns:
        The setting value as string, or None if not found
    """
    # For infrastructure keys, check env var first (bypass cache)
    if key in _ENV_OVERRIDE_KEYS:
        env_key = ENV_KEY_MAP.get(key)
        if env_key:
            env_value = os.environ.get(env_key)
            if env_value is not None and env_value.strip():
                return env_value.strip()

    # Check in-process cache
    now = _time.monotonic()
    cached = _SETTINGS_CACHE.get(key)
    if cached is not None:
        value, ts = cached
        if (now - ts) < _SETTINGS_CACHE_TTL:
            return value

    # Cache miss — query SQLite
    conn = _get_connection()
    row = conn.execute(
        "SELECT value FROM settings WHERE key = ?", (key,)
    ).fetchone()
    value = row["value"] if row else None

    # Store in cache
    _SETTINGS_CACHE[key] = (value, now)
    return value


def get_setting_row(key: str) -> Optional[Dict[str, Any]]:
    """Get a single setting with all metadata.
    
    Args:
        key: The setting key
        
    Returns:
        Dict with key, value, category, type, description, updated_at
        or None if not found
    """
    conn = _get_connection()
    row = conn.execute(
        "SELECT * FROM settings WHERE key = ?", (key,)
    ).fetchone()
    return dict(row) if row else None


def set_setting(key: str, value: str) -> bool:
    """Update a setting value.
    
    Args:
        key: The setting key
        value: The new value (always stored as string)
        
    Returns:
        True if the setting was updated, False if key not found
    """
    conn = _get_connection()
    cursor = conn.execute(
        "UPDATE settings SET value = ?, updated_at = CURRENT_TIMESTAMP WHERE key = ?",
        (value, key)
    )
    conn.commit()
    invalidate_settings_cache(key)
    return cursor.rowcount > 0


def delete_setting(key: str) -> bool:
    """Delete a setting from the database.

    Used to clean up obsolete settings that have been removed from plugins.

    Args:
        key: The setting key to delete

    Returns:
        True if the setting was deleted, False if key not found
    """
    conn = _get_connection()
    cursor = conn.execute("DELETE FROM settings WHERE key = ?", (key,))
    conn.commit()
    invalidate_settings_cache(key)
    return cursor.rowcount > 0


def set_settings(updates: Dict[str, str]) -> List[str]:
    """Update multiple settings at once.
    
    Args:
        updates: Dict of key -> value pairs to update
        
    Returns:
        List of keys that were successfully updated
    """
    conn = _get_connection()
    updated_keys = []
    for key, value in updates.items():
        cursor = conn.execute(
            "UPDATE settings SET value = ?, updated_at = CURRENT_TIMESTAMP WHERE key = ?",
            (str(value), key)
        )
        if cursor.rowcount > 0:
            updated_keys.append(key)
    conn.commit()
    # Invalidate cache for all updated keys
    invalidate_settings_cache()
    return updated_keys


def get_all_settings() -> Dict[str, Dict[str, Any]]:
    """Get all settings grouped by category.
    
    Returns:
        Dict of category -> {key: {value, type, description, updated_at}}
    """
    conn = _get_connection()
    rows = conn.execute(
        "SELECT * FROM settings ORDER BY category, key"
    ).fetchall()
    
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        category = row["category"]
        if category not in grouped:
            grouped[category] = {}
        grouped[category][row["key"]] = {
            "value": row["value"],
            "type": row["type"],
            "description": row["description"],
            "updated_at": row["updated_at"],
        }
    
    return grouped


def get_settings_by_category(category: str) -> Dict[str, Dict[str, Any]]:
    """Get all settings for a specific category.
    
    Args:
        category: Category name (e.g., 'llm', 'rag', 'secrets')
        
    Returns:
        Dict of key -> {value, type, description, updated_at}
    """
    conn = _get_connection()
    rows = conn.execute(
        "SELECT * FROM settings WHERE category = ? ORDER BY key",
        (category,)
    ).fetchall()
    
    return {
        row["key"]: {
            "value": row["value"],
            "type": row["type"],
            "description": row["description"],
            "updated_at": row["updated_at"],
        }
        for row in rows
    }


def get_categories() -> List[str]:
    """Get all distinct categories.
    
    Returns:
        Sorted list of category names
    """
    conn = _get_connection()
    rows = conn.execute(
        "SELECT DISTINCT category FROM settings ORDER BY category"
    ).fetchall()
    
    # Sort by CATEGORY_META order, then alphabetically for unknown categories
    categories = [row["category"] for row in rows]
    return sorted(
        categories,
        key=lambda c: CATEGORY_META.get(c, {}).get("order", "99")
    )


def reset_to_defaults(category: Optional[str] = None) -> int:
    """Reset settings to their default values.
    
    Args:
        category: If provided, only reset settings in this category.
                  If None, reset all settings.
        
    Returns:
        Number of settings reset
    """
    conn = _get_connection()
    count = 0
    for key, default_value, cat, type_, desc in DEFAULT_SETTINGS:
        if category and cat != category:
            continue
        cursor = conn.execute(
            "UPDATE settings SET value = ?, updated_at = CURRENT_TIMESTAMP WHERE key = ?",
            (default_value, key)
        )
        count += cursor.rowcount
    conn.commit()
    invalidate_settings_cache()
    return count


def mask_secret(value: str) -> str:
    """Mask a secret value for display, showing only first 4 and last 3 chars.
    
    Args:
        value: The secret value to mask
        
    Returns:
        Masked string like 'sk-a...xyz' or '****' if too short
    """
    if not value or len(value) < 8:
        return "****" if value else ""
    return f"{value[:4]}...{value[-3:]}"


def get_all_settings_masked() -> Dict[str, Dict[str, Any]]:
    """Get all settings grouped by category, with secrets masked.
    
    Same as get_all_settings() but secret-type values are masked.
    
    Returns:
        Dict of category -> {key: {value, type, description, updated_at}}
    """
    all_settings = get_all_settings()
    
    for category, settings in all_settings.items():
        for key, info in settings.items():
            if info["type"] == "secret":
                info["value"] = mask_secret(info["value"])
    
    return all_settings


# ---------------------------------------------------------------------------
# Plugin settings registration
# ---------------------------------------------------------------------------

def register_plugin_settings(
    settings: List[Tuple[str, str, str, str, str]],
    category_meta: Optional[Dict[str, Dict[str, str]]] = None,
    env_key_map: Optional[Dict[str, str]] = None,
) -> int:
    """Register plugin settings (INSERT OR IGNORE to preserve existing values).
    
    Called by the PluginRegistry during plugin discovery. Uses INSERT OR IGNORE
    so that existing user-modified values are never overwritten.
    
    Also registers category metadata for UI display and env var mappings
    for first-run seeding.
    
    Args:
        settings: List of (key, default_value, category, type, description) tuples
        category_meta: Optional dict of category -> {"label": "...", "order": "N"}
        env_key_map: Optional dict of sqlite_key -> ENV_VAR_NAME
        
    Returns:
        Number of new settings registered (0 if all already existed)
    """
    if category_meta:
        CATEGORY_META.update(category_meta)
    
    if env_key_map:
        ENV_KEY_MAP.update(env_key_map)
        # Also update reverse map
        global _REVERSE_ENV_MAP
        _REVERSE_ENV_MAP = {v: k for k, v in ENV_KEY_MAP.items()}
    
    conn = _get_connection()
    count = 0
    for key, default_value, category, type_, description in settings:
        cursor = conn.execute(
            """INSERT OR IGNORE INTO settings (key, value, category, type, description)
               VALUES (?, ?, ?, ?, ?)""",
            (key, default_value, category, type_, description)
        )
        count += cursor.rowcount
    
    # Overlay env values for any newly inserted settings
    if env_key_map and count > 0:
        for sqlite_key, env_key in env_key_map.items():
            env_value = os.environ.get(env_key)
            if env_value is not None and env_value.strip():
                conn.execute(
                    "UPDATE settings SET value = ?, updated_at = CURRENT_TIMESTAMP WHERE key = ? AND value = ''",
                    (env_value.strip(), sqlite_key)
                )
    
    conn.commit()
    return count


# ---------------------------------------------------------------------------
# Module-level initialization
# ---------------------------------------------------------------------------

# Auto-initialize on import
init_db()
