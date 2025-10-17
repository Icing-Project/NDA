#ifndef BASEPLUGIN_H
#define BASEPLUGIN_H

#include "PluginTypes.h"
#include "audio/AudioBuffer.h"
#include <string>
#include <map>

#define NADE_PLUGIN_API_VERSION 1

namespace NADE {

struct PluginInfo {
    std::string name;
    std::string version;
    std::string author;
    std::string description;
    PluginType type;
    int apiVersion;
};

class BasePlugin {
public:
    virtual ~BasePlugin() = default;

    // Plugin lifecycle
    virtual bool initialize() = 0;
    virtual void shutdown() = 0;
    virtual bool start() = 0;
    virtual void stop() = 0;

    // Plugin information
    virtual PluginInfo getInfo() const = 0;
    virtual PluginType getType() const = 0;

    // Configuration
    virtual void setParameter(const std::string& key, const std::string& value) = 0;
    virtual std::string getParameter(const std::string& key) const = 0;

    // State
    virtual PluginState getState() const = 0;
};

} // namespace NADE

// Plugin factory function signatures
typedef NADE::BasePlugin* (*CreatePluginFunc)();
typedef void (*DestroyPluginFunc)(NADE::BasePlugin*);

// Export macros for plugins
#ifdef _WIN32
    #define NADE_PLUGIN_EXPORT extern "C" __declspec(dllexport)
#else
    #define NADE_PLUGIN_EXPORT extern "C" __attribute__((visibility("default")))
#endif

#define NADE_DECLARE_PLUGIN(PluginClass) \
    NADE_PLUGIN_EXPORT NADE::BasePlugin* createPlugin() { \
        return new PluginClass(); \
    } \
    NADE_PLUGIN_EXPORT void destroyPlugin(NADE::BasePlugin* plugin) { \
        delete plugin; \
    } \
    NADE_PLUGIN_EXPORT int getPluginApiVersion() { \
        return NADE_PLUGIN_API_VERSION; \
    }

#endif // BASEPLUGIN_H
