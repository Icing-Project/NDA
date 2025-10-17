#include "plugins/PluginManager.h"
#include <filesystem>
#include <memory>
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#ifdef NADE_ENABLE_PYTHON
#include "plugins/PythonPluginBridge.h"
#endif

namespace NADE {

PluginManager::PluginManager()
{
}

PluginManager::~PluginManager()
{
    unloadAll();
}

bool PluginManager::loadPlugin(const std::string& path)
{
    std::filesystem::path pluginPath(path);
    std::string extension = pluginPath.extension().string();

    // Check if it's a Python plugin
    if (extension == ".py") {
#ifdef NADE_ENABLE_PYTHON
        return loadPythonPlugin(path);
#else
        std::cerr << "[PluginManager] Python support not enabled. Cannot load: " << path << std::endl;
        return false;
#endif
    }

    // Otherwise, load as C++ dynamic library
    return loadCppPlugin(path);
}

bool PluginManager::loadCppPlugin(const std::string& path)
{
    // Load the dynamic library
    void* libraryHandle = loadLibrary(path);
    if (!libraryHandle) {
        std::cerr << "[PluginManager] Failed to load library: " << path << std::endl;
        return false;
    }

    // Get the factory functions
    CreatePluginFunc createFunc = nullptr;
    DestroyPluginFunc destroyFunc = nullptr;

#ifdef _WIN32
    createFunc = reinterpret_cast<CreatePluginFunc>(
        GetProcAddress(static_cast<HMODULE>(libraryHandle), "createPlugin"));
    destroyFunc = reinterpret_cast<DestroyPluginFunc>(
        GetProcAddress(static_cast<HMODULE>(libraryHandle), "destroyPlugin"));
#else
    createFunc = reinterpret_cast<CreatePluginFunc>(
        dlsym(libraryHandle, "createPlugin"));
    destroyFunc = reinterpret_cast<DestroyPluginFunc>(
        dlsym(libraryHandle, "destroyPlugin"));
#endif

    if (!createFunc || !destroyFunc) {
        std::cerr << "[PluginManager] Plugin factory functions not found in: " << path << std::endl;
        unloadLibrary(libraryHandle);
        return false;
    }

    // Create plugin instance
    BasePlugin* rawInstance = createFunc();
    if (!rawInstance || !validatePlugin(rawInstance)) {
        std::cerr << "[PluginManager] Plugin validation failed: " << path << std::endl;
        if (rawInstance) {
            destroyFunc(rawInstance);
        }
        unloadLibrary(libraryHandle);
        return false;
    }

    // Wrap in shared_ptr with custom deleter
    std::shared_ptr<BasePlugin> instance(rawInstance, [destroyFunc](BasePlugin* p) {
        destroyFunc(p);
    });

    // Initialize plugin
    if (!instance->initialize()) {
        std::cerr << "[PluginManager] Plugin initialization failed: " << path << std::endl;
        unloadLibrary(libraryHandle);
        return false;
    }

    // Store plugin
    LoadedPlugin loadedPlugin;
    loadedPlugin.path = path;
    loadedPlugin.name = instance->getInfo().name;
    loadedPlugin.libraryHandle = libraryHandle;
    loadedPlugin.instance = instance;
    loadedPlugin.info = instance->getInfo();
    loadedPlugin.type = instance->getType();

    plugins_[loadedPlugin.name] = loadedPlugin;

    std::cout << "[PluginManager] Loaded C++ plugin: " << loadedPlugin.name << std::endl;
    return true;
}

#ifdef NADE_ENABLE_PYTHON
bool PluginManager::loadPythonPlugin(const std::string& path)
{
    std::filesystem::path pluginPath(path);
    std::string filename = pluginPath.filename().string();
    std::string directory = pluginPath.parent_path().string();

    if (directory.empty()) {
        directory = "plugins_py";
    }

    // Create Python plugin bridge
    PythonPluginBridge* bridge = PythonPluginFactory::createPlugin(filename, directory);
    if (!bridge) {
        std::cerr << "[PluginManager] Failed to load Python plugin: " << path << std::endl;
        return false;
    }

    // Initialize plugin
    if (!bridge->initialize()) {
        std::cerr << "[PluginManager] Python plugin initialization failed: " << path << std::endl;
        delete bridge;
        return false;
    }

    // Wrap in shared_ptr
    std::shared_ptr<BasePlugin> instance(bridge);

    // Store plugin
    LoadedPlugin loadedPlugin;
    loadedPlugin.path = path;
    loadedPlugin.name = instance->getInfo().name;
    loadedPlugin.libraryHandle = nullptr; // No library handle for Python plugins
    loadedPlugin.instance = instance;
    loadedPlugin.info = instance->getInfo();
    loadedPlugin.type = instance->getType();

    plugins_[loadedPlugin.name] = loadedPlugin;

    std::cout << "[PluginManager] Loaded Python plugin: " << loadedPlugin.name << std::endl;
    return true;
}
#endif

bool PluginManager::unloadPlugin(const std::string& name)
{
    auto it = plugins_.find(name);
    if (it == plugins_.end()) {
        return false;
    }

    LoadedPlugin& plugin = it->second;

    // Shutdown plugin
    plugin.instance->shutdown();

    // Release shared_ptr (will call custom deleter)
    plugin.instance.reset();

    // Unload library
    unloadLibrary(plugin.libraryHandle);

    plugins_.erase(it);
    return true;
}

void PluginManager::unloadAll()
{
    std::vector<std::string> names;
    for (const auto& pair : plugins_) {
        names.push_back(pair.first);
    }

    for (const auto& name : names) {
        unloadPlugin(name);
    }
}

std::vector<std::string> PluginManager::scanPluginDirectory(const std::string& directory)
{
    std::vector<std::string> pluginPaths;

    try {
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::string filename = entry.path().filename().string();

                // Check for C++ plugins
#ifdef _WIN32
                bool isCppPlugin = (ext == ".dll");
#else
                bool isCppPlugin = (ext == ".so");
#endif

                // Check for Python plugins
                bool isPythonPlugin = false;
#ifdef NADE_ENABLE_PYTHON
                bool isSetupFile = (filename.size() >= 6 && filename.substr(0, 6) == "setup_");
                isPythonPlugin = (ext == ".py" &&
                                 filename != "base_plugin.py" &&
                                 filename != "plugin_loader.py" &&
                                 filename != "test_plugins.py" &&
                                 filename != "__init__.py" &&
                                 !isSetupFile);
#endif

                if (isCppPlugin || isPythonPlugin) {
                    pluginPaths.push_back(entry.path().string());
                }
            }
        }
    } catch (...) {
        // Directory doesn't exist or can't be read
    }

    return pluginPaths;
}

std::shared_ptr<BasePlugin> PluginManager::getPlugin(const std::string& name)
{
    auto it = plugins_.find(name);
    if (it != plugins_.end()) {
        return it->second.instance;
    }
    return nullptr;
}

std::vector<LoadedPlugin> PluginManager::getPluginsByType(PluginType type) const
{
    std::vector<LoadedPlugin> result;
    for (const auto& pair : plugins_) {
        if (pair.second.type == type) {
            result.push_back(pair.second);
        }
    }
    return result;
}

std::vector<LoadedPlugin> PluginManager::getAllPlugins() const
{
    std::vector<LoadedPlugin> result;
    for (const auto& pair : plugins_) {
        result.push_back(pair.second);
    }
    return result;
}

std::shared_ptr<AudioSourcePlugin> PluginManager::getAudioSourcePlugin(const std::string& name)
{
    auto plugin = getPlugin(name);
    if (plugin && plugin->getType() == PluginType::AudioSource) {
        return std::dynamic_pointer_cast<AudioSourcePlugin>(plugin);
    }
    return nullptr;
}

std::shared_ptr<BearerPlugin> PluginManager::getBearerPlugin(const std::string& name)
{
    auto plugin = getPlugin(name);
    if (plugin && plugin->getType() == PluginType::Bearer) {
        return std::dynamic_pointer_cast<BearerPlugin>(plugin);
    }
    return nullptr;
}

std::shared_ptr<EncryptorPlugin> PluginManager::getEncryptorPlugin(const std::string& name)
{
    auto plugin = getPlugin(name);
    if (plugin && plugin->getType() == PluginType::Encryptor) {
        return std::dynamic_pointer_cast<EncryptorPlugin>(plugin);
    }
    return nullptr;
}

std::shared_ptr<AudioSinkPlugin> PluginManager::getAudioSinkPlugin(const std::string& name)
{
    auto plugin = getPlugin(name);
    if (plugin && plugin->getType() == PluginType::AudioSink) {
        return std::dynamic_pointer_cast<AudioSinkPlugin>(plugin);
    }
    return nullptr;
}

bool PluginManager::validatePlugin(BasePlugin* plugin)
{
    if (!plugin) {
        return false;
    }

    PluginInfo info = plugin->getInfo();
    if (info.apiVersion != NADE_PLUGIN_API_VERSION) {
        return false;
    }

    return true;
}

void* PluginManager::loadLibrary(const std::string& path)
{
#ifdef _WIN32
    return LoadLibraryA(path.c_str());
#else
    return dlopen(path.c_str(), RTLD_LAZY);
#endif
}

void PluginManager::unloadLibrary(void* handle)
{
    if (!handle) return;

#ifdef _WIN32
    FreeLibrary(static_cast<HMODULE>(handle));
#else
    dlclose(handle);
#endif
}

} // namespace NADE
