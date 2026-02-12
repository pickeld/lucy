"""SQLite-backed settings database for WhatsApp-GPT.

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
    "tavily_api_key": "TAVILY_API_KEY",
    "waha_api_key": "WAHA_API_KEY",
    "langchain_api_key": "LANGCHAIN_API_KEY",
    # LLM
    "llm_provider": "LLM_PROVIDER",
    "openai_model": "OPENAI_MODEL",
    "openai_temperature": "OPENAI_TEMPERATURE",
    "gemini_model": "GEMINI_MODEL",
    "gemini_temperature": "GEMINI_TEMPERATURE",
    # RAG
    "rag_collection_name": "RAG_COLLECTION_NAME",
    "rag_min_score": "RAG_MIN_SCORE",
    "rag_max_context_tokens": "RAG_MAX_CONTEXT_TOKENS",
    "rag_default_k": "RAG_DEFAULT_K",
    "embedding_model": "EMBEDDING_MODEL",
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
    # Tracing
    "langchain_tracing_v2": "LANGCHAIN_TRACING_V2",
    "langchain_project": "LANGCHAIN_PROJECT",
}

# Reverse map for looking up env var name from SQLite key
_REVERSE_ENV_MAP: Dict[str, str] = {v: k for k, v in ENV_KEY_MAP.items()}

# ---------------------------------------------------------------------------
# Default settings (seeded on first run)
# ---------------------------------------------------------------------------

# Each tuple: (key, default_value, category, type, description)
DEFAULT_SETTINGS: List[Tuple[str, str, str, str, str]] = [
    # Secrets
    ("openai_api_key", "", "secrets", "secret", "OpenAI API key"),
    ("google_api_key", "", "secrets", "secret", "Google Gemini API key"),
    ("tavily_api_key", "", "secrets", "secret", "Tavily search API key"),
    ("waha_api_key", "", "secrets", "secret", "WAHA API key"),
    ("langchain_api_key", "", "secrets", "secret", "LangSmith API key"),
    # LLM
    ("llm_provider", "openai", "llm", "select", "LLM provider: openai or gemini"),
    ("openai_model", "gpt-4o", "llm", "text", "OpenAI model name"),
    ("openai_temperature", "0.7", "llm", "float", "OpenAI temperature (0.0-2.0)"),
    ("gemini_model", "gemini-pro", "llm", "text", "Gemini model name"),
    ("gemini_temperature", "0.7", "llm", "float", "Gemini temperature (0.0-2.0)"),
    # RAG
    ("rag_collection_name", "knowledge_base", "rag", "text", "Qdrant collection name"),
    ("rag_min_score", "0.2", "rag", "float", "Minimum similarity score threshold (0.0-1.0)"),
    ("rag_max_context_tokens", "3000", "rag", "int", "Max tokens for RAG context window"),
    ("rag_default_k", "10", "rag", "int", "Default number of context documents"),
    ("embedding_model", "text-embedding-3-large", "rag", "text", "OpenAI embedding model (text-embedding-3-large recommended for Hebrew+English)"),
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
    # Tracing
    ("langchain_tracing_v2", "false", "tracing", "bool", "Enable LangSmith tracing"),
    ("langchain_project", "whatsapp-gpt", "tracing", "text", "LangSmith project name"),
]

# Category display order and labels
CATEGORY_META: Dict[str, Dict[str, str]] = {
    "secrets": {"label": "🔑 API Keys & Secrets", "order": "0"},
    "llm": {"label": "🤖 LLM Configuration", "order": "1"},
    "rag": {"label": "🔍 RAG Configuration", "order": "2"},
    "whatsapp": {"label": "💬 WhatsApp Configuration", "order": "3"},
    "infrastructure": {"label": "🏗️ Infrastructure", "order": "4"},
    "app": {"label": "🔧 App Configuration", "order": "5"},
    "tracing": {"label": "📊 Tracing — LangSmith", "order": "6"},
}

# Select-type options
SELECT_OPTIONS: Dict[str, List[str]] = {
    "llm_provider": ["openai", "gemini"],
    "log_level": ["DEBUG", "INFO", "WARNING", "ERROR"],
}


# ---------------------------------------------------------------------------
# Database connection helper
# ---------------------------------------------------------------------------

def _get_connection() -> sqlite3.Connection:
    """Get a new SQLite connection (connection-per-request for thread safety).
    
    Returns:
        sqlite3.Connection with row_factory set to sqlite3.Row
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent read performance
    return conn


# ---------------------------------------------------------------------------
# Initialization and seeding
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Initialize the settings database.
    
    Creates the settings table if it doesn't exist, then seeds defaults
    and overlays environment variable values on first run (when table is empty).
    """
    conn = _get_connection()
    try:
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
    finally:
        conn.close()


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

def get_setting_value(key: str) -> Optional[str]:
    """Get a single setting value by key.
    
    Args:
        key: The setting key (e.g., 'openai_model')
        
    Returns:
        The setting value as string, or None if not found
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None
    finally:
        conn.close()


def get_setting_row(key: str) -> Optional[Dict[str, Any]]:
    """Get a single setting with all metadata.
    
    Args:
        key: The setting key
        
    Returns:
        Dict with key, value, category, type, description, updated_at
        or None if not found
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM settings WHERE key = ?", (key,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def set_setting(key: str, value: str) -> bool:
    """Update a setting value.
    
    Args:
        key: The setting key
        value: The new value (always stored as string)
        
    Returns:
        True if the setting was updated, False if key not found
    """
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "UPDATE settings SET value = ?, updated_at = CURRENT_TIMESTAMP WHERE key = ?",
            (value, key)
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def set_settings(updates: Dict[str, str]) -> List[str]:
    """Update multiple settings at once.
    
    Args:
        updates: Dict of key -> value pairs to update
        
    Returns:
        List of keys that were successfully updated
    """
    conn = _get_connection()
    updated_keys = []
    try:
        for key, value in updates.items():
            cursor = conn.execute(
                "UPDATE settings SET value = ?, updated_at = CURRENT_TIMESTAMP WHERE key = ?",
                (str(value), key)
            )
            if cursor.rowcount > 0:
                updated_keys.append(key)
        conn.commit()
        return updated_keys
    finally:
        conn.close()


def get_all_settings() -> Dict[str, Dict[str, Any]]:
    """Get all settings grouped by category.
    
    Returns:
        Dict of category -> {key: {value, type, description, updated_at}}
    """
    conn = _get_connection()
    try:
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
    finally:
        conn.close()


def get_settings_by_category(category: str) -> Dict[str, Dict[str, Any]]:
    """Get all settings for a specific category.
    
    Args:
        category: Category name (e.g., 'llm', 'rag', 'secrets')
        
    Returns:
        Dict of key -> {value, type, description, updated_at}
    """
    conn = _get_connection()
    try:
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
    finally:
        conn.close()


def get_categories() -> List[str]:
    """Get all distinct categories.
    
    Returns:
        Sorted list of category names
    """
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT DISTINCT category FROM settings ORDER BY category"
        ).fetchall()
        
        # Sort by CATEGORY_META order, then alphabetically for unknown categories
        categories = [row["category"] for row in rows]
        return sorted(
            categories,
            key=lambda c: CATEGORY_META.get(c, {}).get("order", "99")
        )
    finally:
        conn.close()


def reset_to_defaults(category: Optional[str] = None) -> int:
    """Reset settings to their default values.
    
    Args:
        category: If provided, only reset settings in this category.
                  If None, reset all settings.
        
    Returns:
        Number of settings reset
    """
    conn = _get_connection()
    try:
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
        return count
    finally:
        conn.close()


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
# Module-level initialization
# ---------------------------------------------------------------------------

# Auto-initialize on import
init_db()
