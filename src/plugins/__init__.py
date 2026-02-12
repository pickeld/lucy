"""Plugin framework for multi-channel data ingestion.

This package provides the base classes and registry for channel plugins
(WhatsApp, Telegram, Email, Paperless-NG, etc.).

Each plugin lives in a subdirectory of src/plugins/ and implements
the ChannelPlugin abstract base class.

Usage:
    from plugins.registry import plugin_registry
    
    # Discover and load plugins
    plugin_registry.discover_plugins()
    plugin_registry.load_enabled_plugins(app)
    
    # Access enabled plugins
    for plugin in plugin_registry.enabled_plugins():
        print(plugin.display_name)
"""

from .base import ChannelPlugin
from .registry import PluginRegistry, plugin_registry

__all__ = [
    "ChannelPlugin",
    "PluginRegistry",
    "plugin_registry",
]
