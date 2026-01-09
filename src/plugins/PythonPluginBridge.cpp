#include "plugins/PythonPluginBridge.h"
#include <iostream>
#include <sstream>
#include <cstring>  // For std::memcpy
#include <chrono>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <iomanip>
#include <string>
#include <cctype>

#ifdef NDA_ENABLE_PYTHON

namespace nda {

// Static initialization
bool PythonPluginBridge::pythonInitialized_ = false;

// Helper to initialize NumPy - must be void* return for import_array macro
static int initializeNumPy() {
    import_array1(-1);  // import_array1 is for functions returning int
    return 0;
}

namespace {

bool isTruthyEnv(const char* name)
{
    const char* value = std::getenv(name);
    if (!value) return false;

    std::string s(value);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    return s == "1" || s == "true" || s == "yes" || s == "on";
}

int readEnvInt(const char* name, int defaultValue)
{
    const char* value = std::getenv(name);
    if (!value) return defaultValue;

    try {
        return std::stoi(value);
    } catch (...) {
        return defaultValue;
    }
}

uint64_t toMicros(std::chrono::steady_clock::duration d)
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(d).count());
}

} // namespace

struct PythonPluginBridge::ProfilingData
{
    struct Stat
    {
        uint64_t count = 0;
        uint64_t totalUs = 0;
        uint64_t maxUs = 0;

        void add(uint64_t us)
        {
            totalUs += us;
            count++;
            if (us > maxUs) maxUs = us;
        }

        void reset()
        {
            count = 0;
            totalUs = 0;
            maxUs = 0;
        }

        double avgUs() const
        {
            return count ? static_cast<double>(totalUs) / static_cast<double>(count) : 0.0;
        }
    };

    std::chrono::steady_clock::time_point lastLogTime{};
    std::chrono::milliseconds logInterval{1000};

    Stat readTotalUs{};
    Stat readGILUs{};
    Stat readBufferUs{};
    Stat readPyCallUs{};
    Stat readCopyUs{};

    Stat writeTotalUs{};
    Stat writeGILUs{};
    Stat writeBufferUs{};
    Stat writeCopyToPyUs{};
    Stat writePyCallUs{};

    Stat availTotalUs{};
    Stat availGILUs{};
    Stat availPyCallUs{};

    uint64_t readErrors = 0;
    uint64_t writeErrors = 0;
    uint64_t availErrors = 0;

    void reset(std::chrono::steady_clock::time_point now)
    {
        lastLogTime = now;
        readTotalUs.reset();
        readGILUs.reset();
        readBufferUs.reset();
        readPyCallUs.reset();
        readCopyUs.reset();

        writeTotalUs.reset();
        writeGILUs.reset();
        writeBufferUs.reset();
        writeCopyToPyUs.reset();
        writePyCallUs.reset();

        availTotalUs.reset();
        availGILUs.reset();
        availPyCallUs.reset();

        readErrors = 0;
        writeErrors = 0;
        availErrors = 0;
    }

    void maybeLog(std::chrono::steady_clock::time_point now, const std::string& label)
    {
        if (lastLogTime.time_since_epoch().count() == 0) {
            reset(now);
            return;
        }

        auto dt = now - lastLogTime;
        if (dt < logInterval) return;

        const double dtMs = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(dt).count();

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "[PythonBridgeProfile:" << (label.empty() ? "unknown" : label) << "]"
                  << " dt=" << dtMs << "ms"
                  << " calls(r/w/a)=" << readTotalUs.count << "/" << writeTotalUs.count << "/" << availTotalUs.count
                  << " read(avgUs total/gil/buf/py/copy)="
                  << readTotalUs.avgUs() << "/"
                  << readGILUs.avgUs() << "/"
                  << readBufferUs.avgUs() << "/"
                  << readPyCallUs.avgUs() << "/"
                  << readCopyUs.avgUs()
                  << " write(avgUs total/gil/buf/copy/py)="
                  << writeTotalUs.avgUs() << "/"
                  << writeGILUs.avgUs() << "/"
                  << writeBufferUs.avgUs() << "/"
                  << writeCopyToPyUs.avgUs() << "/"
                  << writePyCallUs.avgUs()
                  << " avail(avgUs total/gil/py)="
                  << availTotalUs.avgUs() << "/"
                  << availGILUs.avgUs() << "/"
                  << availPyCallUs.avgUs()
                  << " maxUs(gilRead/gilWrite/gilAvail)="
                  << readGILUs.maxUs << "/"
                  << writeGILUs.maxUs << "/"
                  << availGILUs.maxUs
                  << " maxUs(totalR/pyR/totalW/pyW/totalA/pyA)="
                  << readTotalUs.maxUs << "/"
                  << readPyCallUs.maxUs << "/"
                  << writeTotalUs.maxUs << "/"
                  << writePyCallUs.maxUs << "/"
                  << availTotalUs.maxUs << "/"
                  << availPyCallUs.maxUs
                  << " errors(r/w/a)=" << readErrors << "/" << writeErrors << "/" << availErrors
                  << std::endl;

        reset(now);
    }
};

PythonPluginBridge::PythonPluginBridge()
    : pModule_(nullptr)
    , pPluginInstance_(nullptr)
    , state_(PluginState::Unloaded)
    , moduleName_()
    // v2.0 Optimization: Initialize cache members
    , cachedBasePluginModule_(nullptr)
    , cachedAudioBufferClass_(nullptr)
    , cachedBufferInstance_(nullptr)
    , cachedNumpyArray_(nullptr)
    , cachedReadAudioMethod_(nullptr)
    , cachedWriteAudioMethod_(nullptr)
    , cachedProcessAudioMethod_(nullptr)
    , cachedChannels_(0)
    , cachedFrames_(0)
    , profiling_(nullptr)
{
    if (isTruthyEnv("NDA_PROFILE") || isTruthyEnv("NDA_PROFILE_PYBRIDGE")) {
        profiling_ = std::make_unique<ProfilingData>();
        const int intervalMs = std::max(100, readEnvInt("NDA_PROFILE_PYBRIDGE_INTERVAL_MS", 1000));
        profiling_->logInterval = std::chrono::milliseconds(intervalMs);
        std::cout << "[PythonBridgeProfile] Enabled (interval " << intervalMs << "ms)" << std::endl;
    }

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

        std::cout << "[PythonBridge] Python version: " << Py_GetVersion() << std::endl;
        PyObject* sysModule = PyImport_ImportModule("sys");
        if (sysModule) {
            auto printSysStringAttr = [](PyObject* sysObj, const char* attrName, const char* label) {
                PyObject* value = PyObject_GetAttrString(sysObj, attrName);
                if (!value) {
                    PyErr_Clear();
                    return;
                }

                const char* utf8 = PyUnicode_Check(value) ? PyUnicode_AsUTF8(value) : nullptr;
                if (utf8) {
                    std::cout << "[PythonBridge] " << label << ": " << utf8 << std::endl;
                } else {
                    PyErr_Clear();
                }

                Py_DECREF(value);
            };

            printSysStringAttr(sysModule, "executable", "sys.executable");
            printSysStringAttr(sysModule, "prefix", "sys.prefix");
            printSysStringAttr(sysModule, "base_prefix", "sys.base_prefix");
            Py_DECREF(sysModule);
        } else {
            PyErr_Clear();
        }
    }
}

PythonPluginBridge::~PythonPluginBridge() {
    destroyCache();  // v2.0: Clean up optimization cache
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
    moduleName_ = moduleName;

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
    
    // v2.0: Initialize optimization cache after plugin instance created
    initializeCache();
    
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
    if (success && profiling_) {
        profiling_->reset(std::chrono::steady_clock::now());
    }
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

// v2.0 Optimization: Cache management methods

void PythonPluginBridge::initializeCache() {
    // Must be called with GIL acquired or after plugin instance creation
    PyGILState_STATE gilState = PyGILState_Ensure();
    
    // Step 1: Cache base_plugin module (avoid repeated imports)
    if (!cachedBasePluginModule_) {
        cachedBasePluginModule_ = PyImport_ImportModule("base_plugin");
        if (cachedBasePluginModule_) {
            Py_INCREF(cachedBasePluginModule_);  // Keep alive
            std::cout << "[PythonBridge] Cached base_plugin module" << std::endl;
        } else {
            std::cerr << "[PythonBridge] Failed to cache base_plugin module" << std::endl;
            PyErr_Print();
        }
    }
    
    // Step 2: Cache AudioBuffer class (avoid repeated attribute lookup)
    if (!cachedAudioBufferClass_ && cachedBasePluginModule_) {
        cachedAudioBufferClass_ = PyObject_GetAttrString(
            cachedBasePluginModule_, "AudioBuffer"
        );
        if (cachedAudioBufferClass_) {
            Py_INCREF(cachedAudioBufferClass_);  // Keep alive
            std::cout << "[PythonBridge] Cached AudioBuffer class" << std::endl;
        } else {
            std::cerr << "[PythonBridge] Failed to cache AudioBuffer class" << std::endl;
            PyErr_Print();
        }
    }
    
    // Step 3: Cache method objects (avoid repeated attribute lookups)
    // Only cache methods that exist for this plugin type
    if (pPluginInstance_) {
        cachedReadAudioMethod_ = PyObject_GetAttrString(pPluginInstance_, "read_audio");
        if (PyErr_Occurred()) {
            PyErr_Clear();  // Clear error if method doesn't exist
            cachedReadAudioMethod_ = nullptr;
        }
        
        cachedWriteAudioMethod_ = PyObject_GetAttrString(pPluginInstance_, "write_audio");
        if (PyErr_Occurred()) {
            PyErr_Clear();  // Clear error if method doesn't exist
            cachedWriteAudioMethod_ = nullptr;
        }
        
        cachedProcessAudioMethod_ = PyObject_GetAttrString(pPluginInstance_, "process_audio");
        if (PyErr_Occurred()) {
            PyErr_Clear();  // Clear error if method doesn't exist (sinks don't have this)
            cachedProcessAudioMethod_ = nullptr;
        }
        
        // Note: Don't INCREF these - we own them from GetAttrString
        // They'll be cleaned up in destroyCache()
        
        std::cout << "[PythonBridge] Cached method objects" << std::endl;
    }
    
    PyGILState_Release(gilState);
}

void PythonPluginBridge::destroyCache() {
    PyGILState_STATE gilState = PyGILState_Ensure();
    
    // Clean up method caches
    Py_XDECREF(cachedProcessAudioMethod_);
    Py_XDECREF(cachedWriteAudioMethod_);
    Py_XDECREF(cachedReadAudioMethod_);
    
    // Clean up buffer cache
    Py_XDECREF((PyObject*)cachedNumpyArray_);  // Release NumPy array reference
    Py_XDECREF(cachedBufferInstance_);
    Py_XDECREF(cachedAudioBufferClass_);
    Py_XDECREF(cachedBasePluginModule_);

    // Null out pointers
    cachedProcessAudioMethod_ = nullptr;
    cachedWriteAudioMethod_ = nullptr;
    cachedReadAudioMethod_ = nullptr;
    cachedBufferInstance_ = nullptr;
    cachedAudioBufferClass_ = nullptr;
    cachedBasePluginModule_ = nullptr;
    cachedNumpyArray_ = nullptr;
    cachedDataPtr_ = nullptr;  // v2.1: Clear cached data pointer
    
    // Reset dimension tracking
    cachedChannels_ = 0;
    cachedFrames_ = 0;
    
    PyGILState_Release(gilState);
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
        if (profiling_) {
            profiling_->reset(std::chrono::steady_clock::now());
        }
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
            else if (typeStr == "Processor") info.type = PluginType::Processor;
            // v2.0: Bearer and Encryptor removed
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

    // v2.1 OPTIMIZATION: Only read clocks when profiling is enabled
    std::chrono::steady_clock::time_point totalStart, afterGIL, bufferStart, pyCallStart;
    if (profiling_) {
        totalStart = std::chrono::steady_clock::now();
    }

    // OPTIMIZATION: Batch GIL operations (acquire ONCE per frame)
    PyGILState_STATE gstate = PyGILState_Ensure();
    if (profiling_) {
        afterGIL = std::chrono::steady_clock::now();
        profiling_->readGILUs.add(toMicros(afterGIL - totalStart));
    }

    auto finish = [&](bool ok) {
        PyGILState_Release(gstate);
        if (profiling_) {
            const auto now = std::chrono::steady_clock::now();
            profiling_->readTotalUs.add(toMicros(now - totalStart));
            profiling_->maybeLog(now, moduleName_);
        }
        return ok;
    };

    // OPTIMIZATION: Use cached buffer (avoid repeated allocation)
    if (profiling_) {
        bufferStart = std::chrono::steady_clock::now();
    }
    PyObject* pBuffer = getOrCreateCachedBuffer(buffer);
    if (!pBuffer) {
        // Fallback to legacy method
        pBuffer = createPythonAudioBuffer(buffer);
        if (!pBuffer) {
            if (profiling_) profiling_->readErrors++;
            return finish(false);
        }
    } else {
        // OPTIMIZATION: Fast memcpy data update (not needed for read, but prepare buffer)
        // For readAudio, Python fills the buffer, so we don't need to copy TO Python
    }
    if (profiling_) {
        profiling_->readBufferUs.add(toMicros(std::chrono::steady_clock::now() - bufferStart));
    }

    // OPTIMIZATION: Use cached method object (avoid attribute lookup)
    if (profiling_) {
        pyCallStart = std::chrono::steady_clock::now();
    }
    PyObject* result = nullptr;
    if (cachedReadAudioMethod_) {
        result = PyObject_CallFunctionObjArgs(
            cachedReadAudioMethod_, pBuffer, nullptr
        );
    } else {
        // Fallback
        result = PyObject_CallMethod(pPluginInstance_, "read_audio", "O", pBuffer);
    }
     
    if (!result) {
        PyErr_Print();
        if (!cachedBufferInstance_ || pBuffer != cachedBufferInstance_) {
            Py_DECREF(pBuffer);
        }
        if (profiling_) {
            profiling_->readPyCallUs.add(toMicros(std::chrono::steady_clock::now() - pyCallStart));
            profiling_->readErrors++;
        }
        return finish(false);
    }

    if (profiling_) {
        profiling_->readPyCallUs.add(toMicros(std::chrono::steady_clock::now() - pyCallStart));
    }

    bool success = PyObject_IsTrue(result);
    Py_DECREF(result);

    if (success) {
        // Copy data back from Python buffer to C++ buffer
        std::chrono::steady_clock::time_point copyStart;
        if (profiling_) {
            copyStart = std::chrono::steady_clock::now();
        }
        copyFromPythonBuffer(pBuffer, buffer);
        if (profiling_) {
            profiling_->readCopyUs.add(toMicros(std::chrono::steady_clock::now() - copyStart));
        }
    }

    // Don't DECREF if using cached buffer
    if (!cachedBufferInstance_ || pBuffer != cachedBufferInstance_) {
        Py_DECREF(pBuffer);
    }

    return finish(success);
}

bool PythonPluginBridge::writeAudio(const AudioBuffer& buffer) {
    if (state_ != PluginState::Running || !pPluginInstance_) {
        return false;
    }

    // v2.1 OPTIMIZATION: Only read clocks when profiling is enabled
    std::chrono::steady_clock::time_point totalStart, afterGIL, bufferStart, pyCallStart;
    if (profiling_) {
        totalStart = std::chrono::steady_clock::now();
    }

    // OPTIMIZATION: Batch GIL operations (acquire ONCE per frame)
    PyGILState_STATE gstate = PyGILState_Ensure();
    if (profiling_) {
        afterGIL = std::chrono::steady_clock::now();
        profiling_->writeGILUs.add(toMicros(afterGIL - totalStart));
    }

    auto finish = [&](bool ok) {
        PyGILState_Release(gstate);
        if (profiling_) {
            const auto now = std::chrono::steady_clock::now();
            profiling_->writeTotalUs.add(toMicros(now - totalStart));
            profiling_->maybeLog(now, moduleName_);
        }
        return ok;
    };

    // OPTIMIZATION: Use cached buffer (avoid repeated allocation)
    // Note: Need to cast away const for getOrCreateCachedBuffer
    if (profiling_) {
        bufferStart = std::chrono::steady_clock::now();
    }
    PyObject* pBuffer = getOrCreateCachedBuffer(const_cast<AudioBuffer&>(buffer));
    if (!pBuffer) {
        // Fallback to legacy method
        pBuffer = createPythonAudioBuffer(buffer);
        if (!pBuffer) {
            if (profiling_) profiling_->writeErrors++;
            return finish(false);
        }
    } else {
        // OPTIMIZATION: Fast memcpy data update (copy TO Python for write)
        std::chrono::steady_clock::time_point copyStart;
        if (profiling_) {
            copyStart = std::chrono::steady_clock::now();
        }
        updateCachedBufferData(buffer, pBuffer);
        if (profiling_) {
            profiling_->writeCopyToPyUs.add(toMicros(std::chrono::steady_clock::now() - copyStart));
        }
    }
    if (profiling_) {
        profiling_->writeBufferUs.add(toMicros(std::chrono::steady_clock::now() - bufferStart));
    }

    // OPTIMIZATION: Use cached method object (avoid attribute lookup)
    if (profiling_) {
        pyCallStart = std::chrono::steady_clock::now();
    }
    PyObject* result = nullptr;
    if (cachedWriteAudioMethod_) {
        result = PyObject_CallFunctionObjArgs(
            cachedWriteAudioMethod_, pBuffer, nullptr
        );
    } else {
        // Fallback
        result = PyObject_CallMethod(pPluginInstance_, "write_audio", "O", pBuffer);
    }
     
    if (!result) {
        PyErr_Print();
        if (!cachedBufferInstance_ || pBuffer != cachedBufferInstance_) {
            Py_DECREF(pBuffer);
        }
        if (profiling_) {
            profiling_->writePyCallUs.add(toMicros(std::chrono::steady_clock::now() - pyCallStart));
            profiling_->writeErrors++;
        }
        return finish(false);
    }

    if (profiling_) {
        profiling_->writePyCallUs.add(toMicros(std::chrono::steady_clock::now() - pyCallStart));
    }

    bool success = PyObject_IsTrue(result);
    Py_DECREF(result);
     
    // Don't DECREF if using cached buffer
    if (!cachedBufferInstance_ || pBuffer != cachedBufferInstance_) {
        Py_DECREF(pBuffer);
    }

    return finish(success);
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

    // v2.1 OPTIMIZATION: Only read clocks when profiling is enabled
    std::chrono::steady_clock::time_point totalStart, afterGIL, pyCallStart;
    if (profiling_) {
        totalStart = std::chrono::steady_clock::now();
    }

    // Acquire GIL for thread-safe Python API calls
    PyGILState_STATE gstate = PyGILState_Ensure();
    if (profiling_) {
        afterGIL = std::chrono::steady_clock::now();
        profiling_->availGILUs.add(toMicros(afterGIL - totalStart));
        pyCallStart = std::chrono::steady_clock::now();
    }

    PyObject* result = PyObject_CallMethod(pPluginInstance_, "get_available_space", nullptr);
    if (profiling_) {
        profiling_->availPyCallUs.add(toMicros(std::chrono::steady_clock::now() - pyCallStart));
    }

    int space = 512;
    if (!result) {
        PyErr_Print();
        if (profiling_) profiling_->availErrors++;
    } else {
        space = PyLong_AsLong(result);
        Py_DECREF(result);
    }

    PyGILState_Release(gstate);

    if (profiling_) {
        const auto now = std::chrono::steady_clock::now();
        profiling_->availTotalUs.add(toMicros(now - totalStart));
        profiling_->maybeLog(now, moduleName_);
    }

    return space;
}

// AudioProcessorPlugin interface implementation (v2.0)
bool PythonPluginBridge::processAudio(AudioBuffer& buffer) {
    if (state_ != PluginState::Running || !pPluginInstance_) {
        return false;
    }

    // OPTIMIZATION: Batch GIL operations (acquire ONCE per frame)
    PyGILState_STATE gstate = PyGILState_Ensure();

    try {
        // OPTIMIZATION: Use cached buffer (avoid repeated allocation)
        PyObject* pBuffer = getOrCreateCachedBuffer(buffer);
        if (!pBuffer) {
            // Fallback to legacy method if cache fails
            pBuffer = createPythonAudioBuffer(buffer);
            if (!pBuffer) {
                PyGILState_Release(gstate);
                return false;
            }
        } else {
            // OPTIMIZATION: Fast memcpy data update (not element-by-element)
            updateCachedBufferData(buffer, pBuffer);
        }

        // OPTIMIZATION: Use cached method object (avoid attribute lookup)
        PyObject* result = nullptr;
        if (cachedProcessAudioMethod_) {
            result = PyObject_CallFunctionObjArgs(
                cachedProcessAudioMethod_, pBuffer, nullptr
            );
        } else {
            // Fallback to method call
            result = PyObject_CallMethod(pPluginInstance_, "process_audio", "O", pBuffer);
        }
        
        if (!result) {
            std::cerr << "[PythonBridge] Error calling process_audio()" << std::endl;
            PyErr_Print();
            // Don't DECREF pBuffer if it's cached
            if (!cachedBufferInstance_ || pBuffer != cachedBufferInstance_) {
                Py_DECREF(pBuffer);
            }
            PyGILState_Release(gstate);
            return false;  // Processor failed - pipeline will passthrough
        }

        bool success = PyObject_IsTrue(result);
        Py_DECREF(result);

        if (success) {
            // Copy processed data back from Python to C++ (in-place modification)
            copyFromPythonBuffer(pBuffer, buffer);
        }

        // Don't DECREF if using cached buffer
        if (!cachedBufferInstance_ || pBuffer != cachedBufferInstance_) {
            Py_DECREF(pBuffer);
        }
        
        PyGILState_Release(gstate);
        return success;
        
    } catch (const std::exception& e) {
        std::cerr << "[PythonBridge] Exception in processAudio: " << e.what() << std::endl;
        PyGILState_Release(gstate);
        return false;
    } catch (...) {
        std::cerr << "[PythonBridge] Unknown exception in processAudio" << std::endl;
        PyGILState_Release(gstate);
        return false;
    }
}

double PythonPluginBridge::getProcessingLatency() const {
    if (!pPluginInstance_) return 0.0;

    PyGILState_STATE gstate = PyGILState_Ensure();

    PyObject* result = PyObject_CallMethod(pPluginInstance_, "get_processing_latency", nullptr);
    if (!result) {
        // Method might not exist (default 0.0)
        PyErr_Clear();
        PyGILState_Release(gstate);
        return 0.0;
    }

    double latency = PyFloat_AsDouble(result);
    Py_DECREF(result);
    PyGILState_Release(gstate);
    
    return latency;
}

// v2.0 Optimization: Get or create cached buffer (avoid repeated allocation)
PyObject* PythonPluginBridge::getOrCreateCachedBuffer(const AudioBuffer& buffer) {
    // Check if buffer dimensions changed (need to recreate)
    if (cachedBufferInstance_ &&
        (cachedChannels_ != buffer.getChannelCount() ||
         cachedFrames_ != buffer.getFrameCount())) {
        // Dimensions changed - release old buffer and cached array
        Py_XDECREF((PyObject*)cachedNumpyArray_);
        cachedNumpyArray_ = nullptr;
        cachedDataPtr_ = nullptr;
        Py_DECREF(cachedBufferInstance_);
        cachedBufferInstance_ = nullptr;
        std::cout << "[PythonBridge] Cache invalidated (size changed: "
                  << cachedChannels_ << "x" << cachedFrames_ << " → "
                  << buffer.getChannelCount() << "x" << buffer.getFrameCount() << ")" << std::endl;
    }
    
    // Create new buffer if needed
    if (!cachedBufferInstance_ && cachedAudioBufferClass_) {
        cachedBufferInstance_ = PyObject_CallFunction(
            cachedAudioBufferClass_, "ii",
            buffer.getChannelCount(),
            buffer.getFrameCount()
        );
        
        if (cachedBufferInstance_) {
            Py_INCREF(cachedBufferInstance_);  // Keep alive
            cachedChannels_ = buffer.getChannelCount();
            cachedFrames_ = buffer.getFrameCount();

            // OPTIMIZATION: Cache the NumPy array pointer (avoids per-frame attribute lookup)
            PyObject* dataAttr = PyObject_GetAttrString(cachedBufferInstance_, "data");
            if (dataAttr && PyArray_Check(dataAttr)) {
                cachedNumpyArray_ = reinterpret_cast<PyArrayObject*>(dataAttr);
                cachedDataPtr_ = static_cast<float*>(PyArray_DATA(cachedNumpyArray_));
            } else {
                Py_XDECREF(dataAttr);
                cachedNumpyArray_ = nullptr;
                cachedDataPtr_ = nullptr;
            }

            std::cout << "[PythonBridge] Created cached buffer: "
                      << cachedChannels_ << "x" << cachedFrames_
                      << " (dataPtr=" << (cachedDataPtr_ ? "valid" : "null") << ")" << std::endl;
        } else {
            std::cerr << "[PythonBridge] Failed to create cached buffer" << std::endl;
            PyErr_Print();
        }
    }
    
    return cachedBufferInstance_;
}

// v2.0 Optimization: Update cached buffer data using fast memcpy
// v2.1 Optimization: Use cached data pointer (no per-frame attribute lookup)
void PythonPluginBridge::updateCachedBufferData(const AudioBuffer& buffer, PyObject* pyBuffer) {
    if (!pyBuffer) return;

    float* pyData = nullptr;
    PyObject* dataAttr = nullptr;

    // OPTIMIZATION: Use cached pointer if available (avoids per-frame PyObject_GetAttrString)
    if (cachedDataPtr_ && pyBuffer == cachedBufferInstance_) {
        pyData = cachedDataPtr_;
    } else {
        // Fallback: Get pointer to NumPy array inside Python buffer
        dataAttr = PyObject_GetAttrString(pyBuffer, "data");
        if (!dataAttr || !PyArray_Check(dataAttr)) {
            std::cerr << "[PythonBridge] Failed to get buffer data attribute" << std::endl;
            Py_XDECREF(dataAttr);
            return;
        }
        PyArrayObject* array = reinterpret_cast<PyArrayObject*>(dataAttr);
        pyData = static_cast<float*>(PyArray_DATA(array));
    }

    if (!pyData) {
        std::cerr << "[PythonBridge] NumPy array data pointer is null" << std::endl;
        Py_XDECREF(dataAttr);
        return;
    }

    const int frames = buffer.getFrameCount();
    const int channels = buffer.getChannelCount();

    // OPTIMIZATION: Fast memcpy per channel (NOT element-by-element loop)
    for (int ch = 0; ch < channels; ++ch) {
        const float* cppData = buffer.getChannelData(ch);
        if (!cppData) {
            std::cerr << "[PythonBridge] C++ channel data is null" << std::endl;
            continue;
        }

        std::memcpy(
            pyData + (ch * frames),
            cppData,
            frames * sizeof(float)
        );
    }

    // Only DECREF if we acquired it in this call (not using cached)
    Py_XDECREF(dataAttr);
}

PyObject* PythonPluginBridge::createPythonAudioBuffer(const AudioBuffer& buffer) const {
    // LEGACY METHOD - kept for fallback, but optimized path uses getOrCreateCachedBuffer()
    
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

    // OPTIMIZATION: Fast memcpy per channel (NOT element-by-element loop)
    const int frames = buffer.getFrameCount();
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
        std::memcpy(
            arrayData + (ch * frames),
            channelData,
            frames * sizeof(float)
        );
    }

    Py_DECREF(pyData);
    Py_DECREF(audioBufferClass);
    Py_DECREF(basePlugin);

    return audioBuffer;
}

// v2.1 Optimization: Use cached data pointer when available
void PythonPluginBridge::copyFromPythonBuffer(PyObject* pyBuffer, AudioBuffer& buffer) const {
    if (!pyBuffer) {
        std::cerr << "[PythonBridge] Python buffer is null" << std::endl;
        return;
    }

    float* arrayData = nullptr;
    PyObject* pyData = nullptr;

    // OPTIMIZATION: Use cached pointer if available (avoids per-frame PyObject_GetAttrString)
    if (cachedDataPtr_ && pyBuffer == cachedBufferInstance_) {
        arrayData = cachedDataPtr_;
    } else {
        // Fallback: Get the data attribute (numpy array)
        pyData = PyObject_GetAttrString(pyBuffer, "data");
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

        arrayData = (float*)PyArray_DATA((PyArrayObject*)pyData);
    }

    if (!arrayData) {
        std::cerr << "[PythonBridge] Array data pointer is null" << std::endl;
        Py_XDECREF(pyData);
        return;
    }

    // OPTIMIZATION: Fast memcpy per channel (NOT element-by-element loop)
    const int frames = buffer.getFrameCount();
    for (int ch = 0; ch < buffer.getChannelCount(); ++ch) {
        float* channelData = buffer.getChannelData(ch);
        if (!channelData) {
            std::cerr << "[PythonBridge] Channel data is null for channel " << ch << std::endl;
            continue;
        }
        std::memcpy(
            channelData,
            arrayData + (ch * frames),
            frames * sizeof(float)
        );
    }

    // Only DECREF if we acquired it in this call (not using cached)
    Py_XDECREF(pyData);
}

} // namespace nda

#endif // NDA_ENABLE_PYTHON
