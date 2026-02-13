"""Plugin registry for discovering, loading, and managing channel plugins.

The PluginRegistry is responsible for:
1. Discovery — scanning src/plugins/*/ for plugin.py files
2. Settings registration — inserting plugin settings into settings_db
3. Loading — instantiating enabled plugins, calling initialize()
4. Runtime toggle — enabling/disabling plugins via settings
5. Iteration — providing access to enabled plugins for health checks, etc.

Usage:
    from plugins.registry import plugin_registry
    
    # During app startup:
    plugin_registry.discover_plugins()
    plugin_registry.load_enabled_plugins(app)
    
    # At runtime:
    for plugin in plugin_registry.enabled_plugins():
        status = plugin.health_check()
"""

import importlib
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Type

from flask import Flask

from utils.logger import logger

if TYPE_CHECKING:
    from plugins.base import ChannelPlugin


class PluginRegistry:
    """Registry for discovering and managing channel plugins.
    
    Maintains a mapping of plugin name -> plugin class/instance,
    and handles lifecycle management (init/shutdown).
    """
    
    def __init__(self):
        """Initialize an empty registry."""
        # plugin_name -> plugin class (all discovered)
        self._discovered: Dict[str, Type["ChannelPlugin"]] = {}
        # plugin_name -> plugin instance (enabled and initialized)
        self._enabled: Dict[str, "ChannelPlugin"] = {}
        # Track which blueprints are registered
        self._registered_blueprints: set = set()
    
    # -------------------------------------------------------------------------
    # Discovery
    # -------------------------------------------------------------------------
    
    def discover_plugins(self) -> List[str]:
        """Scan src/plugins/*/ for plugin.py files with ChannelPlugin subclasses.
        
        Each plugin must have a plugin.py file that defines a class
        inheriting from ChannelPlugin. The class is found by checking
        all module-level names for ChannelPlugin subclasses.
        
        Also registers each plugin's default settings in settings_db.
        
        Returns:
            List of discovered plugin names
        """
        from plugins.base import ChannelPlugin
        
        plugins_dir = Path(__file__).parent
        discovered_names = []
        
        for entry in sorted(plugins_dir.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name.startswith("_"):
                continue
            
            plugin_file = entry / "plugin.py"
            if not plugin_file.exists():
                continue
            
            try:
                # Import the plugin module
                module_name = f"plugins.{entry.name}.plugin"
                
                # Ensure the plugin package is importable
                init_file = entry / "__init__.py"
                if not init_file.exists():
                    init_file.touch()
                
                module = importlib.import_module(module_name)
                
                # Find the ChannelPlugin subclass
                plugin_class = None
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, ChannelPlugin)
                        and attr is not ChannelPlugin
                    ):
                        plugin_class = attr
                        break
                
                if plugin_class is None:
                    logger.warning(
                        f"Plugin {entry.name}: plugin.py found but no "
                        f"ChannelPlugin subclass defined"
                    )
                    continue
                
                # Instantiate to get metadata
                instance = plugin_class()
                name = instance.name
                
                self._discovered[name] = plugin_class
                discovered_names.append(name)
                
                # Register plugin settings in settings_db
                self._register_plugin_settings(instance)
                
                logger.info(f"Discovered plugin: {instance}")
                
            except Exception as e:
                logger.error(f"Failed to discover plugin in {entry.name}: {e}")
                continue
        
        logger.info(f"Plugin discovery complete: {len(discovered_names)} plugins found")
        return discovered_names
    
    def _register_plugin_settings(self, plugin) -> None:
        """Register a plugin's default settings and enable toggle in settings_db.
        
        Uses INSERT OR IGNORE to preserve existing user-modified values.
        
        Args:
            plugin: ChannelPlugin instance
        """
        try:
            import settings_db
            
            # Register the plugin enable/disable setting
            enable_key = f"plugin_{plugin.name}_enabled"
            enable_settings = [
                (
                    enable_key,
                    "true",  # Enabled by default for first plugin (WhatsApp)
                    "plugins",
                    "bool",
                    f"Enable {plugin.display_name} integration"
                )
            ]
            
            # Get plugin-specific settings
            plugin_settings = plugin.get_default_settings()
            
            # Combine and register
            all_settings = enable_settings + plugin_settings
            settings_db.register_plugin_settings(
                settings=all_settings,
                category_meta=plugin.get_category_meta(),
                env_key_map=plugin.get_env_key_map(),
            )
            
            # Register select options
            select_options = plugin.get_select_options()
            if select_options:
                settings_db.SELECT_OPTIONS.update(select_options)
            
            logger.debug(f"Registered {len(all_settings)} settings for plugin {plugin.name}")
            
        except Exception as e:
            logger.error(f"Failed to register settings for plugin {plugin.name}: {e}")
    
    # -------------------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------------------
    
    def load_enabled_plugins(self, app: Flask) -> List[str]:
        """Load and initialize all enabled plugins.
        
        Checks the plugin_<name>_enabled setting for each discovered plugin.
        For enabled plugins: instantiates, calls initialize(), registers Blueprint.
        
        Args:
            app: The Flask application instance
            
        Returns:
            List of enabled plugin names
        """
        from config import settings
        
        enabled_names = []
        
        for name, plugin_class in self._discovered.items():
            try:
                enable_key = f"plugin_{name}_enabled"
                is_enabled = settings.get(enable_key, "false").lower() == "true"
                
                if not is_enabled:
                    logger.info(f"Plugin {name} is disabled, skipping")
                    continue
                
                # Instantiate and initialize
                instance = plugin_class()
                instance.initialize(app)
                
                # Register Blueprint
                blueprint = instance.get_blueprint()
                if blueprint.name not in self._registered_blueprints:
                    app.register_blueprint(blueprint)
                    self._registered_blueprints.add(blueprint.name)
                    logger.info(f"Registered blueprint for plugin {name}")
                
                # Register legacy routes if any
                for rule, endpoint, methods in instance.get_legacy_routes():
                    try:
                        # Check if rule already exists
                        existing_rules = [r.rule for r in app.url_map.iter_rules()]
                        if rule not in existing_rules:
                            # Find the view function from the blueprint
                            view_func = app.view_functions.get(endpoint)
                            if view_func:
                                app.add_url_rule(
                                    rule,
                                    endpoint=f"legacy_{endpoint}",
                                    view_func=view_func,
                                    methods=methods.split(",")
                                )
                                logger.info(f"Registered legacy route: {rule} -> {endpoint}")
                    except Exception as e:
                        logger.warning(f"Failed to register legacy route {rule}: {e}")
                
                self._enabled[name] = instance
                enabled_names.append(name)
                logger.info(f"Loaded and initialized plugin: {instance}")
                
            except Exception as e:
                logger.error(f"Failed to load plugin {name}: {e}")
                continue
        
        logger.info(f"Plugin loading complete: {len(enabled_names)} plugins enabled")
        return enabled_names
    
    # -------------------------------------------------------------------------
    # Runtime toggle
    # -------------------------------------------------------------------------
    
    def enable_plugin(self, name: str, app: Flask) -> bool:
        """Enable a plugin at runtime.
        
        Args:
            name: Plugin name to enable
            app: Flask app instance
            
        Returns:
            True if successfully enabled
        """
        if name in self._enabled:
            logger.info(f"Plugin {name} is already enabled")
            return True
        
        if name not in self._discovered:
            logger.error(f"Plugin {name} not found in discovered plugins")
            return False
        
        try:
            plugin_class = self._discovered[name]
            instance = plugin_class()
            instance.initialize(app)
            
            # Register Blueprint if not already registered
            blueprint = instance.get_blueprint()
            if blueprint.name not in self._registered_blueprints:
                app.register_blueprint(blueprint)
                self._registered_blueprints.add(blueprint.name)
            
            self._enabled[name] = instance
            
            # Update setting
            import settings_db
            settings_db.set_setting(f"plugin_{name}_enabled", "true")
            
            logger.info(f"Enabled plugin: {instance}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to enable plugin {name}: {e}")
            return False
    
    def disable_plugin(self, name: str) -> bool:
        """Disable a plugin at runtime.
        
        Calls shutdown() on the plugin and removes it from enabled list.
        Note: Flask Blueprints cannot be truly unregistered, so the routes
        will still exist but the plugin's process_webhook will not be called.
        
        Args:
            name: Plugin name to disable
            
        Returns:
            True if successfully disabled
        """
        if name not in self._enabled:
            logger.info(f"Plugin {name} is not currently enabled")
            return True
        
        try:
            instance = self._enabled[name]
            instance.shutdown()
            del self._enabled[name]
            
            # Update setting
            import settings_db
            settings_db.set_setting(f"plugin_{name}_enabled", "false")
            
            logger.info(f"Disabled plugin: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to disable plugin {name}: {e}")
            return False
    
    # -------------------------------------------------------------------------
    # Access
    # -------------------------------------------------------------------------
    
    def enabled_plugins(self) -> List:
        """Get all enabled plugin instances.
        
        Returns:
            List of ChannelPlugin instances
        """
        return list(self._enabled.values())
    
    def get_plugin(self, name: str) -> Optional[object]:
        """Get an enabled plugin by name.
        
        Args:
            name: Plugin name
            
        Returns:
            ChannelPlugin instance or None if not enabled
        """
        return self._enabled.get(name)
    
    def is_enabled(self, name: str) -> bool:
        """Check if a plugin is currently enabled.
        
        Args:
            name: Plugin name
            
        Returns:
            True if the plugin is enabled
        """
        return name in self._enabled
    
    def discovered_plugins(self) -> Dict[str, dict]:
        """Get info about all discovered plugins (enabled and disabled).
        
        Returns:
            Dict of name -> {display_name, icon, version, enabled, description}
        """
        from config import settings
        
        result = {}
        for name, plugin_class in self._discovered.items():
            try:
                instance = plugin_class()
                enable_key = f"plugin_{name}_enabled"
                is_enabled = settings.get(enable_key, "false").lower() == "true"
                
                result[name] = {
                    "display_name": instance.display_name,
                    "icon": instance.icon,
                    "version": instance.version,
                    "description": instance.description,
                    "enabled": is_enabled,
                }
            except Exception as e:
                result[name] = {
                    "display_name": name,
                    "icon": "❓",
                    "version": "unknown",
                    "description": f"Error: {e}",
                    "enabled": False,
                }
        
        return result
    
    # -------------------------------------------------------------------------
    # Health
    # -------------------------------------------------------------------------
    
    def health_check_all(self) -> Dict[str, Dict[str, str]]:
        """Run health checks on all enabled plugins.
        
        Returns:
            Dict of plugin_name -> {dependency: status}
        """
        results = {}
        for name, instance in self._enabled.items():
            try:
                results[name] = instance.health_check()
            except Exception as e:
                results[name] = {"plugin": f"error: {e}"}
        return results
    
    # -------------------------------------------------------------------------
    # Shutdown
    # -------------------------------------------------------------------------
    
    def shutdown_all(self) -> None:
        """Shutdown all enabled plugins. Called during app teardown."""
        for name in list(self._enabled.keys()):
            try:
                self._enabled[name].shutdown()
                logger.info(f"Shut down plugin: {name}")
            except Exception as e:
                logger.error(f"Error shutting down plugin {name}: {e}")
        self._enabled.clear()


# Singleton instance
plugin_registry = PluginRegistry()
