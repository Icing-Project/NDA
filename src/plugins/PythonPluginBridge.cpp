#include "plugins/PythonPluginBridge.h"
#include <iostream>
#include <sstream>

#ifdef NADE_ENABLE_PYTHON

namespace NADE {

// Static initialization
bool PythonPluginBridge::pythonInitialized_ = false;

// Helper to initialize NumPy - must be void* return for import_array macro
static int initializeNumPy() {
    import_array1(-1);  // import_array1 is for functions returning int
    return 0;
}

PythonPluginBridge::PythonPluginBridge()
    : pModule_(nullptr),
      pPluginInstance_(nullptr),
      state_(PluginState::Unloaded)
{
    if (!pythonInitialized_) {
        Py_Initialize();
        // Initialize NumPy C API
        if (initializeNumPy() < 0) {
            std::cerr << "[PythonBridge] Failed to initialize NumPy" << std::endl;
        }
        // Note: PyEval_InitThreads() is no longer needed in Python 3.7+
        // The GIL is automatically initialized by Py_Initialize()
        pythonInitialized_ = true;
        std::cout << "[PythonBridge] Python interpreter initialized" << std::endl;
    }
}

PythonPluginBridge::~PythonPluginBridge() {
    shutdown();
}

bool PythonPluginBridge::loadPlugin(const std::string& pluginPath, const std::string& pluginDir) {
    if (state_ != PluginState::Unloaded) {
        std::cerr << "[PythonBridge] Plugin already loaded" << std::endl;
        return false;
    }

    // Add plugin directory to Python path (if not already there)
    PyObject* sysPath = PySys_GetObject("path");
    PyObject* pluginDirPy = PyUnicode_FromString(pluginDir.c_str());

    // Check if path is already in sys.path
    int pathLen = PyList_Size(sysPath);
    bool pathExists = false;
    for (int i = 0; i < pathLen; ++i) {
        PyObject* item = PyList_GetItem(sysPath, i);
        if (PyUnicode_Check(item)) {
            const char* pathStr = PyUnicode_AsUTF8(item);
            if (pathStr && std::string(pathStr) == pluginDir) {
                pathExists = true;
                break;
            }
        }
    }

    if (!pathExists) {
        PyList_Append(sysPath, pluginDirPy);
        std::cout << "[PythonBridge] Added to sys.path: " << pluginDir << std::endl;
    }
    Py_DECREF(pluginDirPy);

    // Extract module name from path (remove .py extension)
    std::string moduleName = pluginPath;
    if (moduleName.size() > 3 && moduleName.substr(moduleName.size() - 3) == ".py") {
        moduleName = moduleName.substr(0, moduleName.size() - 3);
    }

    // Import the module
    PyObject* pName = PyUnicode_FromString(moduleName.c_str());
    pModule_ = PyImport_Import(pName);
    Py_DECREF(pName);

    if (!pModule_) {
        PyErr_Print();
        std::cerr << "[PythonBridge] Failed to load module: " << moduleName << std::endl;
        return false;
    }

    // Get the create_plugin factory function
    PyObject* pFunc = PyObject_GetAttrString(pModule_, "create_plugin");
    if (!pFunc || !PyCallable_Check(pFunc)) {
        std::cerr << "[PythonBridge] Cannot find function 'create_plugin'" << std::endl;
        Py_XDECREF(pFunc);
        Py_DECREF(pModule_);
        pModule_ = nullptr;
        return false;
    }

    // Call create_plugin()
    pPluginInstance_ = PyObject_CallObject(pFunc, nullptr);
    Py_DECREF(pFunc);

    if (!pPluginInstance_) {
        PyErr_Print();
        std::cerr << "[PythonBridge] Failed to create plugin instance" << std::endl;
        Py_DECREF(pModule_);
        pModule_ = nullptr;
        return false;
    }

    state_ = PluginState::Loaded;
    std::cout << "[PythonBridge] Successfully loaded Python plugin: " << moduleName << std::endl;
    return true;
}

bool PythonPluginBridge::initialize() {
    if (state_ != PluginState::Loaded) {
        return false;
    }

    // Acquire GIL for thread-safe Python API calls
    PyGILState_STATE gstate = PyGILState_Ensure();

    PyObject* result = PyObject_CallMethod(pPluginInstance_, "initialize", nullptr);
    if (!result) {
        PyErr_Print();
        PyGILState_Release(gstate);
        return false;
    }

    bool success = PyObject_IsTrue(result);
    Py_DECREF(result);

    if (success) {
        state_ = PluginState::Initialized;
    }

    // Release GIL
    PyGILState_Release(gstate);
    return success;
}

void PythonPluginBridge::shutdown() {
    if (state_ == PluginState::Unloaded) {
        return;
    }

    // Acquire GIL for thread-safe Python API calls
    PyGILState_STATE gstate = PyGILState_Ensure();

    if (pPluginInstance_) {
        PyObject* result = PyObject_CallMethod(pPluginInstance_, "shutdown", nullptr);
        if (!result) {
            std::cerr << "[PythonBridge] Error calling shutdown()" << std::endl;
            PyErr_Print();
        } else {
            Py_DECREF(result);
        }
        Py_DECREF(pPluginInstance_);
        pPluginInstance_ = nullptr;
    }

    if (pModule_) {
        Py_DECREF(pModule_);
        pModule_ = nullptr;
    }

    // Release GIL
    PyGILState_Release(gstate);

    state_ = PluginState::Unloaded;
}

bool PythonPluginBridge::start() {
    if (state_ != PluginState::Initialized) {
        return false;
    }

    // Acquire GIL for thread-safe Python API calls
    PyGILState_STATE gstate = PyGILState_Ensure();

    PyObject* result = PyObject_CallMethod(pPluginInstance_, "start", nullptr);
    if (!result) {
        PyErr_Print();
        PyGILState_Release(gstate);
        return false;
    }

    bool success = PyObject_IsTrue(result);
    Py_DECREF(result);

    if (success) {
        state_ = PluginState::Running;
    }

    // Release GIL
    PyGILState_Release(gstate);
    return success;
}

void PythonPluginBridge::stop() {
    if (state_ != PluginState::Running) {
        return;
    }

    // Acquire GIL for thread-safe Python API calls
    PyGILState_STATE gstate = PyGILState_Ensure();

    PyObject* result = PyObject_CallMethod(pPluginInstance_, "stop", nullptr);
    if (!result) {
        std::cerr << "[PythonBridge] Error calling stop()" << std::endl;
        PyErr_Print();
    } else {
        Py_DECREF(result);
    }

    PyGILState_Release(gstate);

    state_ = PluginState::Initialized;
}

PluginInfo PythonPluginBridge::getInfo() const {
    if (!pPluginInstance_) {
        return {"", "", "", "", PluginType::AudioSource, 0};
    }

    // Acquire GIL for thread-safe Python API calls
    PyGILState_STATE gstate = PyGILState_Ensure();

    PyObject* pInfo = PyObject_CallMethod(pPluginInstance_, "get_info", nullptr);
    if (!pInfo) {
        PyErr_Print();
        PyGILState_Release(gstate);
        return {"", "", "", "", PluginType::AudioSource, 0};
    }

    PluginInfo info;

    // Extract fields from PluginInfo object
    PyObject* pName = PyObject_GetAttrString(pInfo, "name");
    if (pName) {
        info.name = PyUnicode_AsUTF8(pName);
        Py_DECREF(pName);
    }

    PyObject* pVersion = PyObject_GetAttrString(pInfo, "version");
    if (pVersion) {
        info.version = PyUnicode_AsUTF8(pVersion);
        Py_DECREF(pVersion);
    }

    PyObject* pAuthor = PyObject_GetAttrString(pInfo, "author");
    if (pAuthor) {
        info.author = PyUnicode_AsUTF8(pAuthor);
        Py_DECREF(pAuthor);
    }

    PyObject* pDescription = PyObject_GetAttrString(pInfo, "description");
    if (pDescription) {
        info.description = PyUnicode_AsUTF8(pDescription);
        Py_DECREF(pDescription);
    }

    PyObject* pApiVersion = PyObject_GetAttrString(pInfo, "api_version");
    if (pApiVersion) {
        info.apiVersion = PyLong_AsLong(pApiVersion);
        Py_DECREF(pApiVersion);
    }

    // Map Python PluginType to C++ PluginType
    PyObject* pType = PyObject_GetAttrString(pInfo, "type");
    if (pType) {
        PyObject* pTypeValue = PyObject_GetAttrString(pType, "value");
        if (pTypeValue) {
            std::string typeStr = PyUnicode_AsUTF8(pTypeValue);
            if (typeStr == "AudioSource") info.type = PluginType::AudioSource;
            else if (typeStr == "AudioSink") info.type = PluginType::AudioSink;
            else if (typeStr == "Bearer") info.type = PluginType::Bearer;
            else if (typeStr == "Encryptor") info.type = PluginType::Encryptor;
            else if (typeStr == "Processor") info.type = PluginType::Processor;
            Py_DECREF(pTypeValue);
        }
        Py_DECREF(pType);
    }

    Py_DECREF(pInfo);

    // Release GIL
    PyGILState_Release(gstate);
    return info;
}

PluginType PythonPluginBridge::getType() const {
    return getInfo().type;
}

void PythonPluginBridge::setParameter(const std::string& key, const std::string& value) {
    if (!pPluginInstance_) return;

    // Acquire GIL for thread-safe Python API calls
    PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject_CallMethod(pPluginInstance_, "set_parameter", "ss", key.c_str(), value.c_str());
    PyGILState_Release(gstate);
}

std::string PythonPluginBridge::getParameter(const std::string& key) const {
    if (!pPluginInstance_) return "";

    // Acquire GIL for thread-safe Python API calls
    PyGILState_STATE gstate = PyGILState_Ensure();

    PyObject* result = PyObject_CallMethod(pPluginInstance_, "get_parameter", "s", key.c_str());
    if (!result) {
        PyErr_Print();
        PyGILState_Release(gstate);
        return "";
    }

    std::string value = PyUnicode_AsUTF8(result);
    Py_DECREF(result);

    // Release GIL
    PyGILState_Release(gstate);
    return value;
}

PluginState PythonPluginBridge::getState() const {
    return state_;
}

bool PythonPluginBridge::readAudio(AudioBuffer& buffer) {
    if (state_ != PluginState::Running || !pPluginInstance_) {
        return false;
    }

    // Acquire GIL for thread-safe Python API calls
    PyGILState_STATE gstate = PyGILState_Ensure();

    // Create Python AudioBuffer object
    PyObject* pBuffer = createPythonAudioBuffer(buffer);
    if (!pBuffer) {
        PyGILState_Release(gstate);
        return false;
    }

    PyObject* result = PyObject_CallMethod(pPluginInstance_, "read_audio", "O", pBuffer);
    if (!result) {
        PyErr_Print();
        Py_DECREF(pBuffer);
        PyGILState_Release(gstate);
        return false;
    }

    bool success = PyObject_IsTrue(result);
    Py_DECREF(result);

    if (success) {
        // Copy data back from Python buffer to C++ buffer
        copyFromPythonBuffer(pBuffer, buffer);
    }

    Py_DECREF(pBuffer);

    // Release GIL
    PyGILState_Release(gstate);

    return success;
}

bool PythonPluginBridge::writeAudio(const AudioBuffer& buffer) {
    if (state_ != PluginState::Running || !pPluginInstance_) {
        return false;
    }

    // Acquire GIL for thread-safe Python API calls
    PyGILState_STATE gstate = PyGILState_Ensure();

    // Create Python AudioBuffer object
    PyObject* pBuffer = createPythonAudioBuffer(buffer);
    if (!pBuffer) {
        PyGILState_Release(gstate);
        return false;
    }

    PyObject* result = PyObject_CallMethod(pPluginInstance_, "write_audio", "O", pBuffer);
    if (!result) {
        PyErr_Print();
        Py_DECREF(pBuffer);
        PyGILState_Release(gstate);
        return false;
    }

    bool success = PyObject_IsTrue(result);
    Py_DECREF(result);
    Py_DECREF(pBuffer);

    // Release GIL
    PyGILState_Release(gstate);

    return success;
}

void PythonPluginBridge::setAudioCallback(AudioSourceCallback callback) {
    // Python plugins use pull model, callback not used
}

int PythonPluginBridge::getSampleRate() const {
    if (!pPluginInstance_) return 48000;

    // Acquire GIL for thread-safe Python API calls
    PyGILState_STATE gstate = PyGILState_Ensure();

    PyObject* result = PyObject_CallMethod(pPluginInstance_, "get_sample_rate", nullptr);
    if (!result) {
        PyErr_Print();
        PyGILState_Release(gstate);
        return 48000;
    }

    int sampleRate = PyLong_AsLong(result);
    Py_DECREF(result);

    // Release GIL
    PyGILState_Release(gstate);
    return sampleRate;
}

int PythonPluginBridge::getChannels() const {
    if (!pPluginInstance_) return 2;

    // Acquire GIL for thread-safe Python API calls
    PyGILState_STATE gstate = PyGILState_Ensure();

    PyObject* result = PyObject_CallMethod(pPluginInstance_, "get_channels", nullptr);
    if (!result) {
        PyErr_Print();
        PyGILState_Release(gstate);
        return 2;
    }

    int channels = PyLong_AsLong(result);
    Py_DECREF(result);

    // Release GIL
    PyGILState_Release(gstate);
    return channels;
}

void PythonPluginBridge::setSampleRate(int sampleRate) {
    if (!pPluginInstance_) return;

    // Acquire GIL for thread-safe Python API calls
    PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject_CallMethod(pPluginInstance_, "set_sample_rate", "i", sampleRate);
    PyGILState_Release(gstate);
}

void PythonPluginBridge::setChannels(int channels) {
    if (!pPluginInstance_) return;

    // Acquire GIL for thread-safe Python API calls
    PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject_CallMethod(pPluginInstance_, "set_channels", "i", channels);
    PyGILState_Release(gstate);
}

int PythonPluginBridge::getBufferSize() const {
    if (!pPluginInstance_) return 512;

    // Acquire GIL for thread-safe Python API calls
    PyGILState_STATE gstate = PyGILState_Ensure();

    PyObject* result = PyObject_CallMethod(pPluginInstance_, "get_buffer_size", nullptr);
    if (!result) {
        PyErr_Print();
        PyGILState_Release(gstate);
        return 512;
    }

    int bufferSize = PyLong_AsLong(result);
    Py_DECREF(result);

    // Release GIL
    PyGILState_Release(gstate);
    return bufferSize;
}

void PythonPluginBridge::setBufferSize(int samples) {
    if (!pPluginInstance_) return;

    // Acquire GIL for thread-safe Python API calls
    PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject_CallMethod(pPluginInstance_, "set_buffer_size", "i", samples);
    PyGILState_Release(gstate);
}

int PythonPluginBridge::getAvailableSpace() const {
    if (!pPluginInstance_) return 512;

    // Acquire GIL for thread-safe Python API calls
    PyGILState_STATE gstate = PyGILState_Ensure();

    PyObject* result = PyObject_CallMethod(pPluginInstance_, "get_available_space", nullptr);
    if (!result) {
        PyErr_Print();
        PyGILState_Release(gstate);
        return 512;
    }

    int space = PyLong_AsLong(result);
    Py_DECREF(result);

    // Release GIL
    PyGILState_Release(gstate);
    return space;
}

PyObject* PythonPluginBridge::createPythonAudioBuffer(const AudioBuffer& buffer) const {
    // Import base_plugin module
    PyObject* basePlugin = PyImport_ImportModule("base_plugin");
    if (!basePlugin) {
        std::cerr << "[PythonBridge] Failed to import base_plugin" << std::endl;
        PyErr_Print();
        return nullptr;
    }

    // Get AudioBuffer class
    PyObject* audioBufferClass = PyObject_GetAttrString(basePlugin, "AudioBuffer");
    if (!audioBufferClass) {
        std::cerr << "[PythonBridge] Failed to get AudioBuffer class" << std::endl;
        PyErr_Print();
        Py_DECREF(basePlugin);
        return nullptr;
    }

    // Create AudioBuffer instance
    PyObject* audioBuffer = PyObject_CallFunction(audioBufferClass, "ii",
                                                   buffer.getChannelCount(),
                                                   buffer.getFrameCount());
    if (!audioBuffer) {
        std::cerr << "[PythonBridge] Failed to create AudioBuffer instance" << std::endl;
        PyErr_Print();
        Py_DECREF(audioBufferClass);
        Py_DECREF(basePlugin);
        return nullptr;
    }

    // Get the data attribute (numpy array)
    PyObject* pyData = PyObject_GetAttrString(audioBuffer, "data");
    if (!pyData || !PyArray_Check(pyData)) {
        std::cerr << "[PythonBridge] AudioBuffer.data is not a numpy array" << std::endl;
        Py_XDECREF(pyData);
        Py_DECREF(audioBuffer);
        Py_DECREF(audioBufferClass);
        Py_DECREF(basePlugin);
        return nullptr;
    }

    // Copy data from C++ buffer to numpy array
    float* arrayData = (float*)PyArray_DATA((PyArrayObject*)pyData);
    if (!arrayData) {
        std::cerr << "[PythonBridge] Failed to get array data pointer" << std::endl;
        Py_DECREF(pyData);
        Py_DECREF(audioBuffer);
        Py_DECREF(audioBufferClass);
        Py_DECREF(basePlugin);
        return nullptr;
    }

    for (int ch = 0; ch < buffer.getChannelCount(); ++ch) {
        const float* channelData = buffer.getChannelData(ch);
        if (!channelData) {
            std::cerr << "[PythonBridge] Channel data is null" << std::endl;
            Py_DECREF(pyData);
            Py_DECREF(audioBuffer);
            Py_DECREF(audioBufferClass);
            Py_DECREF(basePlugin);
            return nullptr;
        }
        for (int frame = 0; frame < buffer.getFrameCount(); ++frame) {
            arrayData[ch * buffer.getFrameCount() + frame] = channelData[frame];
        }
    }

    Py_DECREF(pyData);
    Py_DECREF(audioBufferClass);
    Py_DECREF(basePlugin);

    return audioBuffer;
}

void PythonPluginBridge::copyFromPythonBuffer(PyObject* pyBuffer, AudioBuffer& buffer) const {
    if (!pyBuffer) {
        std::cerr << "[PythonBridge] Python buffer is null" << std::endl;
        return;
    }

    // Get the data attribute (numpy array)
    PyObject* pyData = PyObject_GetAttrString(pyBuffer, "data");
    if (!pyData) {
        std::cerr << "[PythonBridge] Failed to get data attribute" << std::endl;
        PyErr_Print();
        return;
    }

    if (!PyArray_Check(pyData)) {
        std::cerr << "[PythonBridge] data is not a numpy array" << std::endl;
        Py_DECREF(pyData);
        return;
    }

    // Copy from numpy array back to C++ buffer
    float* arrayData = (float*)PyArray_DATA((PyArrayObject*)pyData);
    if (!arrayData) {
        std::cerr << "[PythonBridge] Array data pointer is null" << std::endl;
        Py_DECREF(pyData);
        return;
    }

    for (int ch = 0; ch < buffer.getChannelCount(); ++ch) {
        float* channelData = buffer.getChannelData(ch);
        if (!channelData) {
            std::cerr << "[PythonBridge] Channel data is null for channel " << ch << std::endl;
            continue;
        }
        for (int frame = 0; frame < buffer.getFrameCount(); ++frame) {
            channelData[frame] = arrayData[ch * buffer.getFrameCount() + frame];
        }
    }

    Py_DECREF(pyData);
}

} // namespace NADE

#endif // NADE_ENABLE_PYTHON
