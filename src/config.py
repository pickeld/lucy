"""Configuration management for WhatsApp-GPT application.

Loads configuration from environment variables and .env file using python-dotenv.
Supports both local development and Docker deployments.
"""

import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv


def find_and_load_env() -> bool:
    """Find and load the .env file by searching up the directory tree.
    
    Searches from the current file's directory upward to find .env file.
    This handles both running from src/ directory and project root.
    
    Returns:
        True if .env file was found and loaded, False otherwise
    """
    # Start from the directory containing this file
    current_dir = Path(__file__).parent.resolve()
    
    # Search up the directory tree
    for parent in [current_dir] + list(current_dir.parents):
        env_path = parent / ".env"
        if env_path.is_file():
            load_dotenv(env_path, override=False)  # env vars take precedence
            return True
        # Stop at project root indicators
        if (parent / "docker-compose.yml").exists() or (parent / ".git").exists():
            if env_path.is_file():
                load_dotenv(env_path, override=False)
                return True
            break
    
    return False


class Config:
    """Configuration manager that reads from environment variables.
    
    Uses python-dotenv to load .env file into os.environ, then provides
    attribute-style access to configuration values. Environment variables
    always take precedence over .env file values.
    
    Attribute access is case-insensitive for convenience.
    """
    
    def __init__(self):
        """Initialize configuration by loading .env file."""
        find_and_load_env()

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
        
        # Try exact match first, then uppercase
        value = os.environ.get(name) or os.environ.get(name.upper())
        if value is not None:
            return value
        
        # Try lowercase
        value = os.environ.get(name.lower())
        if value is not None:
            return value
        
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
