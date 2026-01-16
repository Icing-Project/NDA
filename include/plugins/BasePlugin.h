#ifndef BASEPLUGIN_H
#define BASEPLUGIN_H

#include "PluginTypes.h"
#include "audio/AudioBuffer.h"
#include <string>
#include <map>

#define NDA_PLUGIN_API_VERSION 1

namespace nda {

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

} // namespace nda

// Plugin factory function signatures
typedef nda::BasePlugin* (*CreatePluginFunc)();
typedef void (*DestroyPluginFunc)(nda::BasePlugin*);

// Export macros for plugins
#ifdef _WIN32
    #define NDA_PLUGIN_EXPORT extern "C" __declspec(dllexport)
#else
    #define NDA_PLUGIN_EXPORT extern "C" __attribute__((visibility("default")))
#endif

#define NDA_DECLARE_PLUGIN(PluginClass) \
    NDA_PLUGIN_EXPORT nda::BasePlugin* createPlugin() { \
        return new PluginClass(); \
    } \
    NDA_PLUGIN_EXPORT void destroyPlugin(nda::BasePlugin* plugin) { \
        delete plugin; \
    } \
    NDA_PLUGIN_EXPORT int getPluginApiVersion() { \
        return NDA_PLUGIN_API_VERSION; \
    }

#endif // BASEPLUGIN_H
