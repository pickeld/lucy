"""Configuration management for WhatsApp-GPT application.

All configuration is stored in SQLite (managed by settings_db module).
On first run, settings_db seeds the database from .env values.
After that, SQLite is the single source of truth.

Usage:
    from config import settings
    
    model = settings.openai_model          # returns str
    temperature = settings.openai_temperature  # returns str, cast as needed
    api_key = settings.openai_api_key
"""

from typing import Any


class Settings:
    """Attribute-style access to SQLite-backed settings.
    
    Reads every attribute lookup from the SQLite settings database.
    
    Usage:
        settings.openai_model       -> "gpt-4o"
        settings.redis_host         -> "redis"
        settings.openai_api_key     -> "sk-..."
    """
    
    def __getattr__(self, name: str) -> str:
        """Look up a setting by attribute name from SQLite.
        
        Args:
            name: Setting key (e.g., 'openai_model')
            
        Returns:
            Setting value as string
            
        Raises:
            AttributeError: If the setting key does not exist in SQLite
        """
        # Skip private/dunder attributes to avoid infinite recursion
        if name.startswith("_"):
            raise AttributeError(f"'Settings' object has no attribute '{name}'")
        
        from settings_db import get_setting_value
        value = get_setting_value(name)
        if value is not None:
            return value
        
        # Also try lowercase version of the name
        value = get_setting_value(name.lower())
        if value is not None:
            return value
        
        raise AttributeError(
            f"Setting '{name}' not found in database. "
            f"Check that it exists in settings_db.DEFAULT_SETTINGS."
        )
    
    def get(self, name: str, default: Any = None) -> Any:
        """Get a setting with a fallback default.
        
        Args:
            name: Setting key
            default: Value to return if setting not found
            
        Returns:
            Setting value or default
        """
        try:
            return getattr(self, name)
        except AttributeError:
            return default


# Singleton instance — import this everywhere
settings = Settings()
