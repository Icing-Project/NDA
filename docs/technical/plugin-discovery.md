# NDA Plugin Discovery System

**Version**: 2.1
**Last Updated**: 2026-01-10
**Status**: Production

---

## Overview

The NDA Plugin Discovery System provides robust, priority-based automatic discovery of both C++ and Python plugins across different build configurations (Visual Studio multi-config, Ninja/Make single-config) and deployment scenarios.

This system ensures that plugins are reliably discovered whether running:
- From a development build directory (in-tree builds)
- From a deployed/installed package (readytoship/)
- With custom plugin locations (via environment variable)

---

## Architecture

### Design Principles

1. **Priority-Based Search**: Environment variables override defaults, with intelligent fallback paths
2. **Relative Path Resolution**: All paths are relative to the application executable location
3. **Build-System Agnostic**: Works with Visual Studio, Ninja, Make, and other CMake generators
4. **Diagnostic Logging**: Clear console output shows which paths are searched and where plugins are found
5. **Performance Optimized**: Stops searching after finding plugins in the first valid directory

### 4-Tier Discovery System

```
┌─────────────────────────────────────────────────────────┐
│ Tier 1: Environment Variable Override                  │
│ Priority: HIGHEST                                       │
│ - NDA_PLUGIN_PATH                                       │
└─────────────────────────────────────────────────────────┘
                        ↓ (if not set)
┌─────────────────────────────────────────────────────────┐
│ Tier 2: Application-Relative Paths                     │
│ Priority: NORMAL (Primary Runtime)                     │
│ - <appdir>/plugins                                      │
│ - <appdir>/../plugins                                   │
└─────────────────────────────────────────────────────────┘
                        ↓ (if not found)
┌─────────────────────────────────────────────────────────┐
│ Tier 3: Build-Tree Paths                               │
│ Priority: LOW (Development Fallback)                   │
│ - <appdir>/../plugins/Release  (legacy MSVC)           │
│ - <appdir>/../plugins/Debug    (legacy MSVC)           │
│ - <appdir>/../build/plugins    (single-config)         │
│ - <appdir>/../../build/plugins (deep builds)           │
└─────────────────────────────────────────────────────────┘
                        ↓ (if not found)
┌─────────────────────────────────────────────────────────┐
│ Tier 4: Source-Tree Fallback                           │
│ Priority: LOWEST (Python Only)                         │
│ - <appdir>/../../plugins_py                             │
└─────────────────────────────────────────────────────────┘
```

---

## C++ Plugin Discovery

### Search Paths (In Priority Order)

When `autoLoadPlugins()` is called on startup, the system searches for `*.dll` (Windows) or `*.so` (Linux/Mac) files in:

1. **`$NDA_PLUGIN_PATH`** (if environment variable is set)
2. **`<appdir>/plugins`** - Deployed layout: `readytoship/bin/../plugins`
3. **`<appdir>/../plugins`** - Alternative deployed layout
4. **`<appdir>/../plugins/Release`** - Legacy MSVC multi-config Release builds
5. **`<appdir>/../plugins/Debug`** - Legacy MSVC multi-config Debug builds
6. **`<appdir>/../build/plugins`** - Single-config Ninja/Make builds
7. **`<appdir>/../../build/plugins`** - Deep build tree structures

Where `<appdir>` = `QCoreApplication::applicationDirPath()` (directory containing NDA.exe)

### Build Output Location

As of v2.1, **all plugins are built to a flat directory** regardless of build configuration:

```bash
# Visual Studio (multi-config)
build-msvc/plugins/*.dll          # ← All configs output here (no Release/ subdir)

# Ninja/Make (single-config)
build/plugins/*.dll               # ← Single output location
```

This is enforced by CMake properties in `add_nda_plugin()`:
```cmake
RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PLUGIN_OUTPUT_DIR}
RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PLUGIN_OUTPUT_DIR}
```

### Expected Plugins (v2.1)

The following C++ plugins should be auto-discovered on Windows:

- **WindowsMicrophoneSourcePlugin.dll** - WASAPI microphone capture (Bridge Mode TX)
- **WindowsSpeakerSinkPlugin.dll** - WASAPI speaker output (Bridge Mode RX)
- **AIOCSourcePlugin.dll** - AIOC device microphone input (Bridge Mode RX)
- **AIOCSinkPlugin.dll** - AIOC device speaker output (Bridge Mode TX)
- **SineWaveSourcePlugin.dll** - Test signal generator
- **NullSinkPlugin.dll** - Null output for testing
- **WavFileSinkPlugin.dll** - WAV file recording

---

## Python Plugin Discovery

### Search Paths (In Priority Order)

When Python support is enabled (`NDA_ENABLE_PYTHON=ON`), the system searches for `*.py` files (excluding base/test/setup files) in:

1. **`$NDA_PLUGIN_PATH`** (if environment variable is set)
2. **`<appdir>/plugins_py`** - Deployed alongside bin/
3. **`<appdir>/../plugins_py`** - Alternative deployed layout
4. **`<appdir>/../../plugins_py`** - Source tree from build directory

### Excluded Files

The following Python files are NOT loaded as plugins:
- `base_plugin.py` - Base class definition
- `test_plugins.py` - Test suite
- `cython_compiler.py` - Build utility
- `__init__.py` - Package initialization
- `setup_*.py` - Setup scripts

### Python Plugin Requirements

Python plugins must:
1. Inherit from `BasePlugin` class
2. Implement required methods: `initialize()`, `getName()`, `getVersion()`, etc.
3. Be located in a directory where Python can import them
4. Have NumPy installed in the Python environment (for audio buffer handling)

---

## Environment Variable Override

### Usage

Set the `NDA_PLUGIN_PATH` environment variable to override automatic discovery:

**Windows (PowerShell)**:
```powershell
$env:NDA_PLUGIN_PATH = "C:\custom\plugins"
.\NDA.exe
```

**Windows (CMD)**:
```cmd
set NDA_PLUGIN_PATH=C:\custom\plugins
NDA.exe
```

**Linux/Mac (Bash)**:
```bash
export NDA_PLUGIN_PATH="/custom/plugins"
./NDA
```

### Behavior

- If set, `NDA_PLUGIN_PATH` is searched **first** (highest priority)
- Fallback paths are still searched if custom path is empty or doesn't exist
- Console output shows: `[PluginPaths] Using NDA_PLUGIN_PATH: <path>`

### Use Cases

1. **Testing custom plugin builds** without modifying deployment
2. **Debugging plugin loading issues** by isolating plugin directories
3. **Multi-version plugin management** (e.g., stable vs experimental)
4. **CI/CD pipelines** where build artifacts are in non-standard locations

---

## Console Diagnostic Output

### Normal Startup (All Plugins Found)

```
[PluginPaths] C++ plugin search order:
  - C:/Users/Dev/NDA/build-msvc/Release/plugins [not found]
  - C:/Users/Dev/NDA/build-msvc/plugins [EXISTS]
  - C:/Users/Dev/NDA/build-msvc/plugins/Release [not found]
  - C:/Users/Dev/NDA/build-msvc/plugins/Debug [not found]
  - C:/Users/Dev/NDA/build-msvc/build/plugins [not found]
[MainWindow] Found 7 C++ plugins in: C:/Users/Dev/NDA/build-msvc/plugins
[PluginManager] Loaded C++ plugin: Windows Microphone Source
[PluginManager] Loaded C++ plugin: Windows Speaker Sink
[PluginManager] Loaded C++ plugin: AIOC Source
[PluginManager] Loaded C++ plugin: AIOC Sink
[PluginManager] Loaded C++ plugin: Sine Wave Source
[PluginManager] Loaded C++ plugin: Null Sink
[PluginManager] Loaded C++ plugin: WAV File Sink
[MainWindow] Auto-loaded 7 total plugins
```

### No Plugins Found

```
[PluginPaths] C++ plugin search order:
  - C:/Users/Dev/NDA/build-msvc/Release/plugins [not found]
  - C:/Users/Dev/NDA/build-msvc/plugins [not found]
  - C:/Users/Dev/NDA/build-msvc/plugins/Release [not found]
  - C:/Users/Dev/NDA/build-msvc/plugins/Debug [not found]
  - C:/Users/Dev/NDA/build-msvc/build/plugins [not found]
[MainWindow] Found 0 C++ plugins in: <none>
[MainWindow] Auto-loaded 0 total plugins
```

### Environment Variable Override

```
[PluginPaths] Using NDA_PLUGIN_PATH: D:/custom-plugins
[PluginPaths] C++ plugin search order:
  - D:/custom-plugins [EXISTS]
  - C:/Users/Dev/NDA/build-msvc/Release/plugins [not found]
  - C:/Users/Dev/NDA/build-msvc/plugins [EXISTS]
[MainWindow] Found 3 C++ plugins in: D:/custom-plugins
[MainWindow] Auto-loaded 3 total plugins
```

---

## Troubleshooting

### Problem: Plugins Not Showing Up in UI

**Symptoms**:
- Dropdown menus (TX Source, RX Sink) are empty or missing expected plugins
- Console shows: `[MainWindow] Auto-loaded 0 total plugins`

**Diagnosis**:
1. Check console output for `[PluginPaths]` lines - which paths were searched?
2. Verify at least one path shows `[EXISTS]`
3. Check if plugin DLLs actually exist in the expected directory

**Solutions**:
- **Fresh build**: Run clean rebuild to ensure plugins are compiled
  ```bash
  cd /path/to/NDA
  rm -rf build-msvc
  scripts/build_windows.bat
  ```

- **Check build output**: Verify plugins were built successfully
  ```bash
  ls -la build-msvc/plugins/
  # Expected: 7 DLL files (Windows)
  ```

- **Set environment override**: Test with explicit plugin path
  ```bash
  set NDA_PLUGIN_PATH=C:\path\to\NDA\build-msvc\plugins
  NDA.exe
  ```

### Problem: Python Plugins Not Loading

**Symptoms**:
- Console shows: `[MainWindow] Python plugin support disabled in this build`
- No Python plugins appear even though files exist

**Diagnosis**:
- Check if NDA was built with `NDA_ENABLE_PYTHON=ON`
- Verify Python DLL is in same directory as NDA.exe (deployment only)
- Check NumPy is installed: `python -c "import numpy; print(numpy.__version__)"`

**Solutions**:
- **Enable Python support**: Reconfigure CMake
  ```bash
  cmake -B build-msvc -S . -DNDA_ENABLE_PYTHON=ON
  cmake --build build-msvc --config Release
  ```

- **Install NumPy**: Ensure it's available in the Python environment
  ```bash
  pip install numpy
  ```

- **Check Python path**: Verify plugins_py directory exists
  ```bash
  ls -la plugins_py/
  # Expected: base_plugin.py, sine_wave_source.py, etc.
  ```

### Problem: Deployed Build Can't Find Plugins

**Symptoms**:
- Plugins work in development but not in `readytoship/` package
- Console shows all paths `[not found]`

**Diagnosis**:
1. Check `readytoship/plugins/` directory exists and contains DLLs
2. Verify `readytoship/bin/NDA.exe` is present
3. Check launcher scripts (NDA.bat) are being used

**Solutions**:
- **Redeploy**: Run deployment script again
  ```bash
  scripts/deploy_windows.bat
  ```

- **Manual copy**: Copy plugins to correct location
  ```bash
  cp build-msvc/plugins/*.dll readytoship/plugins/
  ```

- **Use launcher**: Run via `NDA.bat`, not directly
  ```bash
  cd readytoship
  NDA.bat
  ```

---

## Implementation Details

### Source Files

- **`include/plugins/PluginPaths.h`** - Public API for path resolution
- **`src/plugins/PluginPaths.cpp`** - Implementation of discovery logic
- **`src/ui/MainWindow.cpp`** - Integration with auto-load system

### Key Functions

```cpp
// Get C++ plugin search paths in priority order
QStringList PluginPaths::getCppPluginSearchPaths();

// Get Python plugin search paths in priority order
QStringList PluginPaths::getPythonPluginSearchPaths();

// Get environment variable override path (if set)
QString PluginPaths::getEnvironmentPluginPath();
```

### Integration Pattern

```cpp
// In MainWindow::autoLoadPlugins()
auto cppPaths = nda::PluginPaths::getCppPluginSearchPaths();
for (const auto& dir : cppPaths) {
    if (QDir(dir).exists()) {
        auto pluginFiles = pluginManager_->scanPluginDirectory(dir.toStdString());
        for (const auto& path : pluginFiles) {
            if (pluginManager_->loadPlugin(path)) {
                loadedCount++;
            }
        }
        if (!pluginFiles.empty()) {
            break;  // Stop after first successful directory
        }
    }
}
```

---

## Related Documentation

- **Plugin Development**: `docs/development/plugins.md`
- **Python Plugins**: `docs/development/python-plugins.md`
- **Build System**: `docs/development/building.md`
- **Deployment**: `README.md` (Deployment section)

---

## Version History

| Version | Date       | Changes                                      |
|---------|------------|----------------------------------------------|
| 2.1     | 2026-01-10 | Initial centralized plugin discovery system  |
|         |            | - Fixed MSVC multi-config plugin layout      |
|         |            | - Added NDA_PLUGIN_PATH environment variable |
|         |            | - Improved diagnostic logging                |
|         |            | - 4-tier priority-based search               |

---

## Future Enhancements

The following features may be added in future releases:

1. **Plugin manifest system** - JSON/YAML metadata for plugin versioning/dependencies
2. **Hot-reload support** - Load/unload plugins without restarting application
3. **Plugin configuration files** - Persistent plugin settings (`.ini` format)
4. **Plugin registry** - Central database of available plugins with metadata
5. **XDG Base Directory compliance** - Use `~/.config/nda/` for user plugin directories (Linux/Mac)

These are **NOT required for v2.1** but are documented for future roadmap consideration.
