#ifndef PLUGINMANAGER_H
#define PLUGINMANAGER_H

#include "BasePlugin.h"
#include "AudioSourcePlugin.h"
#include "BearerPlugin.h"
#include "EncryptorPlugin.h"
#include "AudioSinkPlugin.h"
#include <string>
#include <vector>
#include <memory>
#include <map>

namespace nda {

struct LoadedPlugin {
    std::string path;
    std::string name;
    void* libraryHandle;
    std::shared_ptr<BasePlugin> instance;
    PluginInfo info;
    PluginType type;
};

class PluginManager
{
public:
    PluginManager();
    ~PluginManager();

    // Plugin loading
    bool loadPlugin(const std::string& path);
    bool unloadPlugin(const std::string& name);
    void unloadAll();

    // Plugin discovery
    std::vector<std::string> scanPluginDirectory(const std::string& directory);

    // Plugin access by type
    std::vector<LoadedPlugin> getPluginsByType(PluginType type) const;
    std::shared_ptr<BasePlugin> getPlugin(const std::string& name);
    std::vector<LoadedPlugin> getAllPlugins() const;

    // Typed plugin getters
    std::shared_ptr<AudioSourcePlugin> getAudioSourcePlugin(const std::string& name);
    std::shared_ptr<BearerPlugin> getBearerPlugin(const std::string& name);
    std::shared_ptr<EncryptorPlugin> getEncryptorPlugin(const std::string& name);
    std::shared_ptr<AudioSinkPlugin> getAudioSinkPlugin(const std::string& name);

private:
    bool validatePlugin(BasePlugin* plugin);
    void* loadLibrary(const std::string& path);
    void unloadLibrary(void* handle);

    // Plugin loading helpers
    bool loadCppPlugin(const std::string& path);
#ifdef NDA_ENABLE_PYTHON
    bool loadPythonPlugin(const std::string& path);
#endif

    std::map<std::string, LoadedPlugin> plugins_;
};

} // namespace nda

#endif // PLUGINMANAGER_H
