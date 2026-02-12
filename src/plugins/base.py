"""Abstract base class for channel plugins.

All channel plugins (WhatsApp, Telegram, Email, Paperless-NG, etc.)
must implement this interface to integrate with the core application.

A plugin provides:
    - Identity (name, display_name, icon, version)
    - Settings (default settings registered in settings_db)
    - Lifecycle (initialize/shutdown)
    - Flask routes (Blueprint with webhook, setup, cache endpoints)
    - Health checks (dependency connectivity)
    - Webhook processing (parse payload → BaseRAGDocument)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from flask import Blueprint, Flask


class ChannelPlugin(ABC):
    """Base class all channel plugins must implement.
    
    Plugins are discovered by the PluginRegistry by scanning
    src/plugins/*/plugin.py for classes that subclass ChannelPlugin.
    
    Lifecycle:
        1. Discovery: Registry finds the plugin class
        2. Settings registration: get_default_settings() called, inserted into DB
        3. Instantiation: Plugin class is instantiated
        4. Initialization: initialize(app) called when plugin is enabled
        5. Running: Blueprint routes are active, webhooks are processed
        6. Shutdown: shutdown() called when plugin is disabled
    """

    # -------------------------------------------------------------------------
    # Identity
    # -------------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique plugin identifier (lowercase, no spaces).
        
        Used as:
        - Settings category name
        - Blueprint URL prefix: /plugins/<name>/
        - Enable setting key: plugin_<name>_enabled
        - Redis key prefix for plugin-specific caches
        
        Examples: 'whatsapp', 'telegram', 'email', 'paperless'
        """

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name for UI display.
        
        Examples: 'WhatsApp', 'Telegram', 'Email', 'Paperless-NG'
        """

    @property
    @abstractmethod
    def icon(self) -> str:
        """Emoji or icon string for UI display.
        
        Examples: '💬', '✈️', '📧', '📄'
        """

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version string (semver recommended).
        
        Examples: '1.0.0', '0.2.1'
        """

    @property
    def description(self) -> str:
        """Optional longer description for the settings UI.
        
        Override to provide a more detailed description.
        """
        return f"{self.display_name} integration plugin"

    # -------------------------------------------------------------------------
    # Settings
    # -------------------------------------------------------------------------

    @abstractmethod
    def get_default_settings(self) -> List[Tuple[str, str, str, str, str]]:
        """Return default settings for this plugin.
        
        Each tuple: (key, default_value, category, type, description)
        
        Category should be the plugin name (e.g. 'whatsapp').
        These are registered in settings_db on plugin discovery using
        INSERT OR IGNORE, so existing user-modified values are preserved.
        
        Type can be: 'text', 'secret', 'int', 'float', 'bool', 'select'
        
        Returns:
            List of setting tuples
            
        Example:
            return [
                ("waha_session_name", "default", "whatsapp", "text", "WAHA session name"),
                ("waha_api_key", "", "whatsapp", "secret", "WAHA API key"),
            ]
        """

    def get_select_options(self) -> Dict[str, List[str]]:
        """Return select-type option lists for settings.
        
        Override if any settings have type='select'.
        
        Returns:
            Dict of setting_key -> list of allowed values
            
        Example:
            return {"log_level": ["DEBUG", "INFO", "WARNING", "ERROR"]}
        """
        return {}

    def get_env_key_map(self) -> Dict[str, str]:
        """Return mapping of setting keys to environment variable names.
        
        Used during first-run seeding to overlay .env values onto defaults.
        Override to support environment variable configuration.
        
        Returns:
            Dict of sqlite_key -> ENV_VAR_NAME
            
        Example:
            return {
                "waha_api_key": "WAHA_API_KEY",
                "waha_session_name": "WAHA_SESSION_NAME",
            }
        """
        return {}

    def get_category_meta(self) -> Dict[str, Dict[str, str]]:
        """Return category metadata for settings UI display.
        
        Override to customize the category label and sort order.
        
        Returns:
            Dict of category_name -> {"label": "...", "order": "N"}
            
        Example:
            return {"whatsapp": {"label": "💬 WhatsApp", "order": "10"}}
        """
        return {
            self.name: {
                "label": f"{self.icon} {self.display_name}",
                "order": "10"
            }
        }

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    @abstractmethod
    def initialize(self, app: Flask) -> None:
        """Called when the plugin is enabled.
        
        Set up connections, caches, background tasks, etc.
        The Flask app is provided for access to app context.
        
        This is called:
        - At startup for plugins that are already enabled
        - At runtime when a plugin is enabled via settings UI
        
        Args:
            app: The Flask application instance
        """

    @abstractmethod
    def shutdown(self) -> None:
        """Called when the plugin is disabled.
        
        Clean up resources, close connections, stop background tasks.
        
        This is called:
        - At app shutdown for all enabled plugins
        - At runtime when a plugin is disabled via settings UI
        """

    # -------------------------------------------------------------------------
    # Flask Integration
    # -------------------------------------------------------------------------

    @abstractmethod
    def get_blueprint(self) -> Blueprint:
        """Return a Flask Blueprint with all plugin-specific routes.
        
        The blueprint is mounted at /plugins/<name>/ by the registry.
        
        Should include:
        - Webhook endpoint(s) (e.g. /webhook)
        - Setup/pairing endpoints (e.g. /pair, /qr_code for WhatsApp)
        - Cache management endpoints
        - Any plugin-specific API routes
        
        The blueprint's url_prefix should be set to /plugins/<name>.
        
        Returns:
            Flask Blueprint instance
            
        Example:
            bp = Blueprint('whatsapp', __name__, url_prefix='/plugins/whatsapp')
            
            @bp.route('/webhook', methods=['POST'])
            def webhook():
                ...
            
            return bp
        """

    def get_legacy_routes(self) -> List[Tuple[str, str, str]]:
        """Return legacy route aliases for backward compatibility.
        
        Override to register additional routes outside the /plugins/ prefix.
        Each tuple: (rule, endpoint_name, methods_comma_separated)
        
        Returns:
            List of (rule, endpoint, methods) tuples
            
        Example:
            return [
                ("/webhook", "whatsapp.webhook", "POST"),
            ]
        """
        return []

    # -------------------------------------------------------------------------
    # Health
    # -------------------------------------------------------------------------

    @abstractmethod
    def health_check(self) -> Dict[str, str]:
        """Check plugin-specific dependencies.
        
        Called by the core /health endpoint for each enabled plugin.
        
        Returns:
            Dict of dependency_name -> status_string
            Status should be "connected" on success, or "error: <details>" on failure.
            
        Example:
            return {"waha": "connected"}
            return {"waha": "error: connection refused"}
        """

    # -------------------------------------------------------------------------
    # Webhook Processing
    # -------------------------------------------------------------------------

    def should_process(self, payload: Dict[str, Any]) -> bool:
        """Filter/validate an incoming webhook payload.
        
        Override to implement plugin-specific filtering logic.
        Return False to silently drop the payload (e.g. ack events,
        system notifications).
        
        Args:
            payload: The raw webhook payload
            
        Returns:
            True if the payload should be processed, False to skip
        """
        return True

    @abstractmethod
    def process_webhook(self, payload: Dict[str, Any]) -> Optional[Any]:
        """Process an incoming webhook payload.
        
        Should:
        1. Parse the payload into a domain object
        2. Extract message content, sender, chat info
        3. Convert to a BaseRAGDocument subclass
        4. Return the document (caller stores in RAG)
        
        Return None if the payload should be ignored (e.g. after filtering).
        
        Args:
            payload: The webhook payload dictionary
            
        Returns:
            BaseRAGDocument instance, or None if payload should be ignored
        """

    # -------------------------------------------------------------------------
    # String representation
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} v{self.version}>"

    def __str__(self) -> str:
        return f"{self.icon} {self.display_name} v{self.version}"
