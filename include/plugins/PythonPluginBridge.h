#ifndef PYTHONPLUGINBRIDGE_H
#define PYTHONPLUGINBRIDGE_H

#include "BasePlugin.h"
#include "AudioSourcePlugin.h"
#include "AudioSinkPlugin.h"
#include "AudioProcessorPlugin.h"

// Only compile if Python support is enabled
#ifdef NDA_ENABLE_PYTHON

#define PY_SSIZE_T_CLEAN
#ifdef _MSC_VER
// On Windows, CPython headers enable debug ABI (`Py_DEBUG`) when `_DEBUG` is set,
// which requires a debug-build CPython (python3x_d.lib + debug-only symbols).
// NDA typically embeds a standard release CPython, so temporarily hide `_DEBUG`
// while including Python headers and then restore it.
#ifdef _DEBUG
#define NDA_RESTORE_DEBUG 1
#undef _DEBUG
#endif
#endif
#include <Python.h>
#ifdef _MSC_VER
#ifdef NDA_RESTORE_DEBUG
#define _DEBUG
#undef NDA_RESTORE_DEBUG
#endif
#endif

// Include numpy for audio buffer conversion
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

namespace nda {

/**
 * @brief Bridge class to load and use Python plugins from C++
 *
 * This class wraps Python plugins and exposes them through the
 * standard NDA plugin interface. It handles:
 * - Loading Python modules
 * - Converting between C++ and Python data structures
 * - Managing Python interpreter lifecycle
 *
 * Note: This class implements all plugin interfaces so it can act as any type
 * 
 * @note v2.0: Now includes AudioProcessorPlugin for Python processor support
 */
class PythonPluginBridge : public AudioSourcePlugin, 
                           public AudioSinkPlugin,
                           public AudioProcessorPlugin {  // v2.0: Added processor support
public:
    PythonPluginBridge();
    ~PythonPluginBridge() override;

    /**
     * @brief Load a Python plugin from file
     * @param pluginPath Path to the Python plugin file (e.g., "sine_wave_source.py")
     * @param pluginDir Directory containing the plugin (default: "plugins_py")
     * @return true if plugin loaded successfully
     */
    bool loadPlugin(const std::string& pluginPath,
                   const std::string& pluginDir = "plugins_py");

    // BasePlugin interface
    bool initialize() override;
    void shutdown() override;
    bool start() override;
    void stop() override;
    PluginInfo getInfo() const override;
    PluginType getType() const override;
    void setParameter(const std::string& key, const std::string& value) override;
    std::string getParameter(const std::string& key) const override;
    PluginState getState() const override;

    // AudioSourcePlugin interface
    bool readAudio(AudioBuffer& buffer) override;
    void setAudioCallback(AudioSourceCallback callback) override;
    int getSampleRate() const override;
    int getChannels() const override;
    void setSampleRate(int sampleRate) override;
    void setChannels(int channels) override;

    // AudioSinkPlugin interface
    bool writeAudio(const AudioBuffer& buffer) override;
    int getBufferSize() const override;
    void setBufferSize(int samples) override;
    int getAvailableSpace() const override;
    
    // AudioProcessorPlugin interface (v2.0)
    bool processAudio(AudioBuffer& buffer) override;
    // Bridge methods for AudioProcessorPlugin (delegates to AudioSourcePlugin methods)
    int getChannelCount() const override { return getChannels(); }
    void setChannelCount(int channels) override { setChannels(channels); }
    double getProcessingLatency() const override;

private:
    /**
     * @brief Create Python AudioBuffer object from C++ AudioBuffer
     */
    PyObject* createPythonAudioBuffer(const AudioBuffer& buffer) const;

    /**
     * @brief Copy data from Python AudioBuffer to C++ AudioBuffer
     */
    void copyFromPythonBuffer(PyObject* pyBuffer, AudioBuffer& buffer) const;

    PyObject* pModule_;           // Python module object
    PyObject* pPluginInstance_;   // Python plugin instance
    PluginState state_;           // Current plugin state

    static bool pythonInitialized_;  // Python interpreter initialized flag
    
    // v2.0 Optimization: Object and method caching (6-30x performance improvement)
    PyObject* cachedBasePluginModule_;     // base_plugin module (reused across calls)
    PyObject* cachedAudioBufferClass_;     // AudioBuffer class object
    PyObject* cachedBufferInstance_;       // Reused buffer object (recreated only on size change)
    PyArrayObject* cachedNumpyArray_;      // Reused NumPy array reference
    
    // Cached method objects (avoid repeated attribute lookup)
    PyObject* cachedReadAudioMethod_;      // plugin.read_audio method
    PyObject* cachedWriteAudioMethod_;     // plugin.write_audio method
    PyObject* cachedProcessAudioMethod_;   // plugin.process_audio method
    
    // Buffer dimension tracking for cache invalidation
    int cachedChannels_;
    int cachedFrames_;
    
    // Cache management methods
    void initializeCache();
    void destroyCache();
    PyObject* getOrCreateCachedBuffer(const AudioBuffer& buffer);
    void updateCachedBufferData(const AudioBuffer& buffer, PyObject* pyBuffer);
};

/**
 * @brief Factory class to create Python plugin bridges
 */
class PythonPluginFactory {
public:
    /**
     * @brief Create a Python plugin bridge instance
     * @param pluginPath Path to the Python plugin file
     * @param pluginDir Directory containing the plugin
     * @return Pointer to new PythonPluginBridge instance (nullptr on failure)
     */
    static PythonPluginBridge* createPlugin(const std::string& pluginPath,
                                           const std::string& pluginDir = "plugins_py") {
        auto* bridge = new PythonPluginBridge();
        if (!bridge->loadPlugin(pluginPath, pluginDir)) {
            delete bridge;
            return nullptr;
        }
        return bridge;
    }
};

} // namespace nda

#endif // NDA_ENABLE_PYTHON

#endif // PYTHONPLUGINBRIDGE_H
