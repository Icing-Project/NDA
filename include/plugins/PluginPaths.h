#ifndef PLUGIN_PATHS_H
#define PLUGIN_PATHS_H

#include <QString>
#include <QStringList>

namespace nda {

/**
 * @brief Centralized plugin path resolution system
 *
 * Provides a 4-tier priority-based discovery system for C++ and Python plugins:
 * 1. Environment variable override (NDA_PLUGIN_PATH)
 * 2. Application-relative paths (deployed layout)
 * 3. Build-tree paths (development)
 * 4. Source-tree fallback (Python plugins)
 *
 * This class follows the existing NDA_* environment variable pattern and ensures
 * plugins can be discovered across different build configurations (MSVC multi-config,
 * Ninja single-config) and deployment scenarios.
 */
class PluginPaths {
public:
    /**
     * @brief Get ordered list of directories to search for C++ plugins
     * @return QStringList of absolute paths in priority order
     *
     * Searches in this order:
     * - $NDA_PLUGIN_PATH (if set)
     * - <appdir>/plugins (deployed layout)
     * - <appdir>/../plugins (alternative deployed layout)
     * - <appdir>/../plugins/Release (legacy MSVC multi-config)
     * - <appdir>/../plugins/Debug (legacy MSVC multi-config)
     * - <appdir>/../build/plugins (single-config builds)
     * - <appdir>/../../build/plugins (deep build trees)
     */
    static QStringList getCppPluginSearchPaths();

    /**
     * @brief Get ordered list of directories to search for Python plugins
     * @return QStringList of absolute paths in priority order
     *
     * Searches in this order:
     * - $NDA_PLUGIN_PATH (if set)
     * - <appdir>/plugins_py (deployed alongside bin)
     * - <appdir>/../plugins_py (alternative deployed layout)
     * - <appdir>/../../plugins_py (source tree from build dir)
     */
    static QStringList getPythonPluginSearchPaths();

    /**
     * @brief Get plugin path from NDA_PLUGIN_PATH environment variable
     * @return QString containing the path, or empty string if not set
     *
     * Follows existing NDA_* environment variable pattern used for:
     * - NDA_PROFILE, NDA_PIPELINE_FRAME_SIZE, NDA_RESAMPLER_QUALITY, etc.
     */
    static QString getEnvironmentPluginPath();

    /**
     * @brief Check if directory exists and log the result
     * @param path The directory path to validate
     * @param type Description of path type for logging (e.g., "C++ plugin")
     * @return true if directory exists, false otherwise
     */
    static bool validateAndLogPath(const QString& path, const QString& type);
};

} // namespace nda

#endif // PLUGIN_PATHS_H
