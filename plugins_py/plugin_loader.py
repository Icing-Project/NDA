"""
Python Plugin Loader for NDA
Dynamically loads Python plugins
"""

import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional
from base_plugin import BasePlugin, PluginInfo


class PluginLoader:
    """Loads and manages Python plugins"""

    def __init__(self, plugin_dir: str = "plugins_py"):
        self.plugin_dir = Path(plugin_dir)
        self.loaded_plugins: Dict[str, BasePlugin] = {}

    def discover_plugins(self) -> List[str]:
        """Discover all Python plugin files in the plugin directory"""
        plugins = []

        if not self.plugin_dir.exists():
            print(f"Plugin directory not found: {self.plugin_dir}")
            return plugins

        for plugin_file in self.plugin_dir.glob("*.py"):
            if plugin_file.name.startswith("_") or plugin_file.name == "base_plugin.py":
                continue

            plugins.append(plugin_file.stem)

        return plugins

    def load_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Load a plugin by name"""
        plugin_path = self.plugin_dir / f"{plugin_name}.py"

        if not plugin_path.exists():
            print(f"Plugin not found: {plugin_path}")
            return None

        try:
            # Load module dynamically
            spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
            if spec is None or spec.loader is None:
                print(f"Failed to load spec for: {plugin_name}")
                return None

            module = importlib.util.module_from_spec(spec)
            sys.modules[plugin_name] = module
            spec.loader.exec_module(module)

            # Get the create_plugin factory function
            if not hasattr(module, 'create_plugin'):
                print(f"Plugin {plugin_name} does not have create_plugin() function")
                return None

            # Create plugin instance
            plugin = module.create_plugin()

            if not isinstance(plugin, BasePlugin):
                print(f"Plugin {plugin_name} does not inherit from BasePlugin")
                return None

            self.loaded_plugins[plugin_name] = plugin
            print(f"[PluginLoader] Loaded plugin: {plugin_name}")

            return plugin

        except Exception as e:
            print(f"Failed to load plugin {plugin_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin"""
        if plugin_name not in self.loaded_plugins:
            return False

        plugin = self.loaded_plugins[plugin_name]
        plugin.shutdown()

        del self.loaded_plugins[plugin_name]
        print(f"[PluginLoader] Unloaded plugin: {plugin_name}")

        return True

    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Get a loaded plugin by name"""
        return self.loaded_plugins.get(plugin_name)

    def list_loaded_plugins(self) -> List[str]:
        """List all loaded plugins"""
        return list(self.loaded_plugins.keys())

    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """Get info for a loaded plugin"""
        plugin = self.get_plugin(plugin_name)
        if plugin:
            return plugin.get_info()
        return None

    def unload_all(self):
        """Unload all plugins"""
        for plugin_name in list(self.loaded_plugins.keys()):
            self.unload_plugin(plugin_name)
