"""
Python Plugin Loader for NDA
Dynamically loads Python plugins with optional Cython auto-compilation.

When Cython is available, plugins are automatically compiled to native
extensions for 10-50x speedup. Falls back silently to pure Python if
Cython is not installed or compilation fails.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional
from base_plugin import BasePlugin, PluginInfo

# Cython compiler integration (optional)
try:
    from cython_compiler import (
        compile_plugin,
        get_compiled_path,
        get_cache_dir,
        is_cache_valid,
        cleanup_stale_cache,
        find_cached_module,
        is_cython_available,
    )
    CYTHON_AVAILABLE = is_cython_available()
except ImportError:
    CYTHON_AVAILABLE = False


class PluginLoader:
    """Loads and manages Python plugins with transparent Cython compilation"""

    def __init__(self, plugin_dir: str = "plugins_py", enable_cython: bool = True):
        self.plugin_dir = Path(plugin_dir)
        self.loaded_plugins: Dict[str, BasePlugin] = {}
        self.enable_cython = enable_cython and CYTHON_AVAILABLE

        # Initialize cache directory and cleanup on startup
        if self.enable_cython:
            self._cache_dir = get_cache_dir(self.plugin_dir)
            cleanup_stale_cache(self._cache_dir, self.plugin_dir)
        else:
            self._cache_dir = None

    def discover_plugins(self) -> List[str]:
        """Discover all Python plugin files in the plugin directory"""
        plugins = []

        if not self.plugin_dir.exists():
            print(f"Plugin directory not found: {self.plugin_dir}")
            return plugins

        # Files to exclude from plugin discovery
        exclude_files = {
            'base_plugin.py',
            'plugin_loader.py',
            'cython_compiler.py',
        }

        for plugin_file in self.plugin_dir.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue
            if plugin_file.name in exclude_files:
                continue

            plugins.append(plugin_file.stem)

        return plugins

    def _try_load_compiled(self, plugin_name: str, plugin_path: Path) -> Optional[BasePlugin]:
        """
        Attempt to load a compiled Cython version of the plugin.

        Args:
            plugin_name: Name of the plugin (without extension)
            plugin_path: Path to the source .py file

        Returns:
            Plugin instance if successful, None otherwise
        """
        if not self.enable_cython or not self._cache_dir:
            return None

        # Check for existing valid cache or compile
        compiled_path = find_cached_module(plugin_path, self._cache_dir)

        if not compiled_path:
            # Attempt compilation
            compiled_path = compile_plugin(plugin_path, self._cache_dir)
            if not compiled_path:
                return None  # Compilation failed, caller will fall back

        # Load the compiled module
        try:
            spec = importlib.util.spec_from_file_location(
                plugin_name,
                compiled_path
            )
            if spec is None or spec.loader is None:
                return None

            module = importlib.util.module_from_spec(spec)
            sys.modules[plugin_name] = module
            spec.loader.exec_module(module)

            # Get the create_plugin factory function
            if not hasattr(module, 'create_plugin'):
                print(f"Compiled plugin {plugin_name} missing create_plugin()")
                return None

            # Create plugin instance
            plugin = module.create_plugin()

            if not isinstance(plugin, BasePlugin):
                print(f"Compiled plugin {plugin_name} does not inherit BasePlugin")
                return None

            return plugin

        except Exception:
            # Silent fallback - don't print exception details
            return None

    def load_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """
        Load a plugin by name, preferring compiled Cython version.

        First attempts to load a compiled .so/.pyd if Cython is enabled.
        Falls back to pure Python if compilation fails or Cython unavailable.
        """
        plugin_path = self.plugin_dir / f"{plugin_name}.py"

        print("Loading plugin:", plugin_name)

        if not plugin_path.exists():
            print(f"Plugin not found: {plugin_path}")
            return None

        # Try compiled version first (if Cython enabled)
        if self.enable_cython:
            plugin = self._try_load_compiled(plugin_name, plugin_path)
            if plugin:
                self.loaded_plugins[plugin_name] = plugin
                print(f"[PluginLoader] Loaded COMPILED plugin: {plugin_name}")
                return plugin

        # Fallback to pure Python version
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
