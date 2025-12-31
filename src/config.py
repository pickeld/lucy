"""Configuration management for WhatsApp-GPT application.

Loads configuration from environment variables and .env file.
Supports both local development and Docker deployments.
"""

import os
from pathlib import Path
from typing import Any, Optional


def find_env_file() -> Optional[Path]:
    """Find the .env file by searching up the directory tree.
    
    Searches from the current file's directory upward to find .env file.
    This handles both running from src/ directory and project root.
    
    Returns:
        Path to .env file if found, None otherwise
    """
    # Start from the directory containing this file
    current_dir = Path(__file__).parent.resolve()
    
    # Search up the directory tree
    for parent in [current_dir] + list(current_dir.parents):
        env_path = parent / ".env"
        if env_path.is_file():
            return env_path
        # Stop at project root indicators
        if (parent / "docker-compose.yml").exists() or (parent / ".git").exists():
            # Check one more time in this directory
            if env_path.is_file():
                return env_path
            break
    
    return None


class Config:
    """Configuration manager that loads from .env file and environment variables.
    
    Environment variables take precedence over .env file values.
    Attribute access is case-insensitive for convenience.
    """
    
    def __init__(self, env_file: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            env_file: Optional path to .env file. If not provided, will search
                      for .env file in parent directories.
        """
        self._attributes: dict[str, str] = {}
        
        # Find and load .env file
        if env_file:
            env_path = Path(env_file)
            if not env_path.is_absolute():
                # Resolve relative to this file's directory
                env_path = Path(__file__).parent / env_file
        else:
            env_path = find_env_file()
        
        if env_path and env_path.is_file():
            self._load_env_file(env_path)
        else:
            # Fall back to environment variables only
            self._load_from_environ()

    def _load_env_file(self, path: Path) -> None:
        """Load configuration from .env file.
        
        Args:
            path: Path to the .env file
        """
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")

                # Only set in os.environ if not already set (env vars take precedence)
                if key not in os.environ:
                    os.environ[key] = value
                
                # Store both original case and lowercase for case-insensitive access
                self._attributes[key] = os.environ.get(key, value)
                self._attributes[key.lower()] = os.environ.get(key, value)
    
    def _load_from_environ(self) -> None:
        """Load configuration from environment variables only."""
        # Load common config keys from environment
        common_keys = [
            "LOG_LEVEL", "WEBHOOK_URL", "DALLE_PREFIX", "CHAT_PREFIX",
            "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB",
            "REDIS_HOST", "REDIS_PORT", "REDIS_TTL",
            "WAHA_SESSION_NAME", "WAHA_BASE_URL", "WAHA_API_KEY",
            "LLM_PROVIDER", "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_TEMPERATURE",
            "GOOGLE_API_KEY", "GEMINI_MODEL", "GEMINI_TEMPERATURE",
            "QDRANT_HOST", "QDRANT_PORT",
            "LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT"
        ]
        
        for key in common_keys:
            value = os.environ.get(key)
            if value:
                self._attributes[key] = value
                self._attributes[key.lower()] = value

    def __getattr__(self, name: str) -> str:
        """Get configuration value by attribute name.
        
        Args:
            name: Configuration key (case-insensitive)
            
        Returns:
            Configuration value as string
            
        Raises:
            AttributeError: If configuration key is not found
        """
        # Skip private attributes
        if name.startswith("_"):
            raise AttributeError(f"'Config' object has no attribute '{name}'")
        
        # Try exact match first, then lowercase
        if name in self._attributes:
            return self._attributes[name]
        if name.lower() in self._attributes:
            return self._attributes[name.lower()]
        
        # Fall back to environment variable
        env_value = os.environ.get(name) or os.environ.get(name.upper())
        if env_value:
            return env_value
        
        raise AttributeError(f"'Config' object has no attribute '{name}'")
    
    def get(self, name: str, default: Any = None) -> Any:
        """Get configuration value with optional default.
        
        Args:
            name: Configuration key (case-insensitive)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            return getattr(self, name)
        except AttributeError:
            return default


# Create singleton config instance
config = Config()
