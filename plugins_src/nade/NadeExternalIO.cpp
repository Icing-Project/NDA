/**
 * @file NadeExternalIO.cpp
 * @brief Implementation of NadeExternalIO singleton for Nade-Python integration.
 *
 * @version 3.0
 * @date 2026-01-30
 */

#include "NadeExternalIO.h"

#include <chrono>
#include <iostream>
#include <stdexcept>

// Only include pybind11 in implementation file
#ifdef NDA_ENABLE_PYTHON
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#endif

namespace nade {

// =============================================================================
// AudioRingBuffer Implementation
// =============================================================================

AudioRingBuffer::AudioRingBuffer(size_t capacity)
    : buffer_(capacity), capacity_(capacity) {}

size_t AudioRingBuffer::write(const float* data, size_t count) {
    size_t writePos = writePos_.load(std::memory_order_relaxed);
    size_t readPos = readPos_.load(std::memory_order_acquire);

    // Calculate available space
    size_t used = (writePos >= readPos) ? (writePos - readPos)
                                        : (capacity_ - readPos + writePos);
    size_t free = capacity_ - used - 1;  // -1 to distinguish full from empty

    size_t toWrite = std::min(count, free);
    if (toWrite == 0) return 0;

    // Copy data
    for (size_t i = 0; i < toWrite; ++i) {
        buffer_[writePos] = data[i];
        writePos = (writePos + 1) % capacity_;
    }

    writePos_.store(writePos, std::memory_order_release);
    return toWrite;
}

size_t AudioRingBuffer::read(float* data, size_t count) {
    size_t writePos = writePos_.load(std::memory_order_acquire);
    size_t readPos = readPos_.load(std::memory_order_relaxed);

    // Calculate available data
    size_t avail = (writePos >= readPos) ? (writePos - readPos)
                                         : (capacity_ - readPos + writePos);

    size_t toRead = std::min(count, avail);
    if (toRead == 0) return 0;

    // Copy data
    for (size_t i = 0; i < toRead; ++i) {
        data[i] = buffer_[readPos];
        readPos = (readPos + 1) % capacity_;
    }

    readPos_.store(readPos, std::memory_order_release);
    return toRead;
}

size_t AudioRingBuffer::available() const {
    size_t writePos = writePos_.load(std::memory_order_acquire);
    size_t readPos = readPos_.load(std::memory_order_relaxed);
    return (writePos >= readPos) ? (writePos - readPos)
                                 : (capacity_ - readPos + writePos);
}

size_t AudioRingBuffer::space() const {
    size_t writePos = writePos_.load(std::memory_order_relaxed);
    size_t readPos = readPos_.load(std::memory_order_acquire);
    size_t used = (writePos >= readPos) ? (writePos - readPos)
                                        : (capacity_ - readPos + writePos);
    return capacity_ - used - 1;
}

void AudioRingBuffer::clear() {
    writePos_.store(0, std::memory_order_relaxed);
    readPos_.store(0, std::memory_order_relaxed);
}

// =============================================================================
// NadeExternalIO Static Members
// =============================================================================

std::shared_ptr<NadeExternalIO> NadeExternalIO::instance_;
std::mutex NadeExternalIO::instanceMutex_;

// =============================================================================
// NadeExternalIO Singleton Methods
// =============================================================================

std::shared_ptr<NadeExternalIO> NadeExternalIO::getInstance() {
    std::lock_guard<std::mutex> lock(instanceMutex_);
    return instance_;
}

std::shared_ptr<NadeExternalIO> NadeExternalIO::createInstance(
    const std::vector<uint8_t>& localPrivateKey,
    const std::vector<uint8_t>& localPublicKey,
    const std::vector<uint8_t>& remotePublicKey,
    int ndaSampleRate,
    bool isInitiator,
    bool enableDiscovery)
{
    std::lock_guard<std::mutex> lock(instanceMutex_);

    if (instance_) {
        // Instance already exists - return it
        return instance_;
    }

    // Create new instance
    instance_ = std::shared_ptr<NadeExternalIO>(
        new NadeExternalIO(localPrivateKey, localPublicKey, remotePublicKey,
                          ndaSampleRate, isInitiator, enableDiscovery)
    );

    return instance_;
}

void NadeExternalIO::destroyInstance() {
    std::lock_guard<std::mutex> lock(instanceMutex_);
    if (instance_) {
        instance_->stopWorkerThread();
        instance_.reset();
    }
}

bool NadeExternalIO::hasInstance() {
    std::lock_guard<std::mutex> lock(instanceMutex_);
    return instance_ != nullptr;
}

// =============================================================================
// NadeExternalIO Constructor/Destructor
// =============================================================================

NadeExternalIO::NadeExternalIO(
    const std::vector<uint8_t>& localPriv,
    const std::vector<uint8_t>& localPub,
    const std::vector<uint8_t>& remotePub,
    int sampleRate,
    bool isInitiator,
    bool enableDiscovery)
    : ndaSampleRate_(sampleRate)
    , isInitiator_(isInitiator)
    , enableDiscovery_(enableDiscovery)
    , localPrivateKey_(localPriv)
    , localPublicKey_(localPub)
    , remotePublicKey_(remotePub)
{
    // Pre-buffer: ~1 second of audio at NDA sample rate
    size_t preBufferSize = static_cast<size_t>(sampleRate);
    txPreBuffer_ = std::make_unique<AudioRingBuffer>(preBufferSize);

    // Post-buffer: ~1 second of audio at NDA sample rate
    rxPostBuffer_ = std::make_unique<AudioRingBuffer>(preBufferSize);

    // Start worker thread
    startWorkerThread();
}

NadeExternalIO::~NadeExternalIO() {
    stopWorkerThread();
}

// =============================================================================
// Worker Thread Management
// =============================================================================

void NadeExternalIO::startWorkerThread() {
    running_.store(true);
    workerThread_ = std::thread(&NadeExternalIO::workerThreadFunction, this);
}

void NadeExternalIO::stopWorkerThread() {
    if (running_.load()) {
        running_.store(false);
        workerCv_.notify_all();

        if (workerThread_.joinable()) {
            workerThread_.join();
        }
    }
}

void NadeExternalIO::workerThreadFunction() {
#ifdef NDA_ENABLE_PYTHON
    try {
        std::cout << "[NadeExternalIO] Starting worker thread..." << std::endl;

        // Set Python home to the BASE Python installation (not venv)
        // UV venvs don't contain the stdlib - they reference the base Python installation
        // We'll add venv's site-packages to sys.path separately below
#ifdef _WIN32
        // For UV venvs, we need the base Python installation (where stdlib is)
        // Try NDA_PYTHON_HOME first (should point to base Python with stdlib)
        const char* pythonHomeEnv = std::getenv("NDA_PYTHON_HOME");

        if (pythonHomeEnv) {
            // Convert to wide string for Py_SetPythonHome on Windows
            size_t len = strlen(pythonHomeEnv);
            std::wstring pythonHome(len, L'\0');
            std::mbstowcs(&pythonHome[0], pythonHomeEnv, len);
            Py_SetPythonHome(pythonHome.c_str());
            std::cout << "[NadeExternalIO] Set PYTHONHOME to: " << pythonHomeEnv << std::endl;
        } else {
            std::cout << "[NadeExternalIO] Warning: NDA_PYTHON_HOME not set" << std::endl;
            std::cout << "[NadeExternalIO] Hint: Set NDA_PYTHON_HOME to base Python installation (with stdlib)" << std::endl;
            std::cout << "[NadeExternalIO]       For UV: C:\\Users\\...\\uv\\python\\cpython-3.12.11-windows-x86_64-none" << std::endl;
        }
#endif

        // Initialize Python interpreter
        std::cout << "[NadeExternalIO] Initializing Python interpreter..." << std::endl;
        py::scoped_interpreter guard{};
        pythonInitialized_.store(true);
        std::cout << "[NadeExternalIO] Python interpreter initialized" << std::endl;

        // Add Nade-Python to sys.path (sibling directory to NDA)
        // This allows importing the nade module without installing it
        py::module_ sys = py::module_::import("sys");
        py::module_ os = py::module_::import("os");
        py::list path = sys.attr("path");

        // Print Python version and path
        std::cout << "[NadeExternalIO] Python version: "
                  << sys.attr("version").cast<std::string>() << std::endl;
        std::cout << "[NadeExternalIO] Python sys.path before modification:" << std::endl;
        for (auto item : path) {
            std::cout << "  - " << item.cast<std::string>() << std::endl;
        }

        // Add venv site-packages to sys.path (CRITICAL for finding nade-python installed via UV)
        // This is where UV installs nade-python and its dependencies
        py::object venv_path = os.attr("environ").attr("get")("VIRTUAL_ENV", py::none());
        if (!venv_path.is_none()) {
            std::string venvStr = venv_path.cast<std::string>();
            // Construct site-packages path: {VIRTUAL_ENV}/Lib/site-packages on Windows
            py::object path_join = os.attr("path").attr("join");
            py::object site_packages = path_join(venvStr, "Lib", "site-packages");
            std::string sitePackagesStr = site_packages.cast<std::string>();

            std::cout << "[NadeExternalIO] Adding venv site-packages: " << sitePackagesStr << std::endl;
            path.attr("insert")(0, sitePackagesStr);

#ifdef _WIN32
            // CRITICAL: Add DLL directories for native Python packages like 'cryptography'
            // When Python is embedded, DLL search paths differ from standalone Python.
            // Without this, packages with Rust/C extensions fail with "DLL load failed".
            try {
                // Add site-packages and common DLL locations to DLL search path
                // os.add_dll_directory() was added in Python 3.8
                py::object add_dll_directory = os.attr("add_dll_directory");

                // Add site-packages directory (where cryptography's DLLs might be)
                add_dll_directory(site_packages);
                std::cout << "[NadeExternalIO] Added DLL directory: " << sitePackagesStr << std::endl;

                // Add cryptography's .libs directory (where bundled DLLs are typically stored)
                py::object crypto_libs = path_join(site_packages, "cryptography.libs");
                if (os.attr("path").attr("isdir")(crypto_libs).cast<bool>()) {
                    add_dll_directory(crypto_libs);
                    std::cout << "[NadeExternalIO] Added DLL directory: "
                              << crypto_libs.cast<std::string>() << std::endl;
                }

                // Also try cryptography/hazmat/bindings
                py::object crypto_bindings = path_join(site_packages, "cryptography", "hazmat", "bindings");
                if (os.attr("path").attr("isdir")(crypto_bindings).cast<bool>()) {
                    add_dll_directory(crypto_bindings);
                }

                // Add the venv's Scripts directory (may contain DLLs)
                py::object scripts_dir = path_join(venvStr, "Scripts");
                if (os.attr("path").attr("isdir")(scripts_dir).cast<bool>()) {
                    add_dll_directory(scripts_dir);
                }

                // Add the base Python installation directory (NDA_PYTHON_HOME)
                // This is where OpenSSL and other system DLLs might be located
                const char* pythonHomeEnv = std::getenv("NDA_PYTHON_HOME");
                if (pythonHomeEnv) {
                    py::str pythonHomeStr(pythonHomeEnv);
                    add_dll_directory(pythonHomeStr);
                    std::cout << "[NadeExternalIO] Added DLL directory: " << pythonHomeEnv << std::endl;

                    // Also add DLLs subdirectory if it exists
                    py::object dlls_dir = path_join(pythonHomeStr, "DLLs");
                    if (os.attr("path").attr("isdir")(dlls_dir).cast<bool>()) {
                        add_dll_directory(dlls_dir);
                        std::cout << "[NadeExternalIO] Added DLL directory: "
                                  << dlls_dir.cast<std::string>() << std::endl;
                    }
                }

            } catch (const std::exception& e) {
                std::cout << "[NadeExternalIO] Warning: Could not add DLL directories: " << e.what() << std::endl;
            }
#endif
        } else {
            std::cout << "[NadeExternalIO] Warning: VIRTUAL_ENV not set, may not find nade-python" << std::endl;
        }

        // Check for NADE_PYTHON_PATH environment variable (fallback for direct Nade-Python path)
        py::object env_path = os.attr("environ").attr("get")("NADE_PYTHON_PATH", py::none());
        if (!env_path.is_none()) {
            std::string nadePath = env_path.cast<std::string>();
            std::cout << "[NadeExternalIO] Using NADE_PYTHON_PATH: " << nadePath << std::endl;
            path.attr("insert")(0, nadePath);
        }

        // Try relative paths from NDA executable (../Nade-Python) as last resort
        path.attr("insert")(0, "../Nade-Python");
        path.attr("insert")(0, "../../Nade-Python");
        path.attr("insert")(0, "../../../Nade-Python");

        std::cout << "[NadeExternalIO] Python sys.path after modification:" << std::endl;
        for (auto item : path) {
            std::cout << "  - " << item.cast<std::string>() << std::endl;
        }

        // Import NDA adapter module with defensive error handling
        std::cout << "[NadeExternalIO] Importing nade.adapters.nda_adapter..." << std::endl;
        std::cout << "[NadeExternalIO] (This may take a moment on first import)" << std::endl;

        py::module_ nade_adapters;
        py::object NDAAdapterClass;

        try {
            nade_adapters = py::module_::import("nade.adapters.nda_adapter");
            std::cout << "[NadeExternalIO] Module imported successfully" << std::endl;
        } catch (const py::error_already_set& e) {
            std::cerr << "[NadeExternalIO] FATAL: Failed to import nade.adapters.nda_adapter" << std::endl;

            // Try to safely get error message using e.what() (C++ method, should be safe)
            try {
                std::cerr << "[NadeExternalIO] Error message: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "[NadeExternalIO] (Could not retrieve error message)" << std::endl;
            }

            // Restore the exception for outer handler without accessing Python objects
            throw std::runtime_error(std::string("Failed to import nade.adapters.nda_adapter: ") + e.what());
        } catch (...) {
            std::cerr << "[NadeExternalIO] FATAL: Unknown error importing nade.adapters.nda_adapter" << std::endl;
            throw std::runtime_error("Unknown error importing nade.adapters.nda_adapter");
        }

        try {
            NDAAdapterClass = nade_adapters.attr("NDAAdapter");
            std::cout << "[NadeExternalIO] NDAAdapter class obtained" << std::endl;
        } catch (...) {
            std::cerr << "[NadeExternalIO] FATAL: Failed to get NDAAdapter class from module" << std::endl;
            throw std::runtime_error("Failed to get NDAAdapter class");
        }

        // Convert keys to Python bytes
        std::cout << "[NadeExternalIO] Converting keys (sizes: priv=" << localPrivateKey_.size()
                  << ", pub=" << localPublicKey_.size() << ", remote=" << remotePublicKey_.size()
                  << ")" << std::endl;
        py::bytes privKey(reinterpret_cast<const char*>(localPrivateKey_.data()),
                         localPrivateKey_.size());
        py::bytes pubKey(reinterpret_cast<const char*>(localPublicKey_.data()),
                        localPublicKey_.size());
        py::bytes remoteKey(reinterpret_cast<const char*>(remotePublicKey_.data()),
                           remotePublicKey_.size());

        // Create adapter instance
        std::cout << "[NadeExternalIO] Creating NDAAdapter instance (sampleRate=" << ndaSampleRate_
                  << ", initiator=" << isInitiator_ << ", discovery=" << enableDiscovery_ << ")" << std::endl;

        py::object adapter;
        try {
            adapter = NDAAdapterClass(
                privKey, pubKey, remoteKey, ndaSampleRate_, "4fsk", isInitiator_, enableDiscovery_);
        } catch (py::error_already_set& e) {
            // pybind11 fetched the Python error into the exception object, clearing PyErr_Occurred()
            // We need to restore it before we can print it
            std::cerr << "[NadeExternalIO] NDAAdapter creation failed (Python exception)!" << std::endl;
            try {
                // Restore the Python error state so PyErr_Print can access it
                e.restore();
                PyErr_Print();
                // Don't call PyErr_Clear() - PyErr_Print already clears it
            } catch (...) {
                std::cerr << "[NadeExternalIO] (could not print Python error)" << std::endl;
            }
            pythonInitialized_.store(false);
            return;  // Exit worker thread gracefully
        } catch (const std::exception& e) {
            std::cerr << "[NadeExternalIO] NDAAdapter creation failed (C++ exception)!" << std::endl;
            std::cerr << "[NadeExternalIO] Error: " << e.what() << std::endl;
            pythonInitialized_.store(false);
            return;
        } catch (...) {
            std::cerr << "[NadeExternalIO] NDAAdapter creation failed (unknown exception)!" << std::endl;
            if (PyErr_Occurred()) {
                PyErr_Print();
            }
            pythonInitialized_.store(false);
            return;
        }

        if (!adapter || adapter.is_none()) {
            std::cerr << "[NadeExternalIO] NDAAdapter creation returned None/null" << std::endl;
            pythonInitialized_.store(false);
            return;
        }

        std::cout << "[NadeExternalIO] NDAAdapter created successfully" << std::endl;

        // Store for later access
        ndaAdapter_ = new py::object(adapter);

        sessionEstablished_.store(true);
        std::cout << "[NadeExternalIO] Session established, entering main loop" << std::endl;

        // Worker loop
        while (running_.load()) {
            try {
                // Poll handshake phase for UI
                int phase = adapter.attr("get_handshake_phase")().cast<int>();
                handshakePhase_.store(phase, std::memory_order_relaxed);
                // Process TX: Generate FSK audio and fill pre-buffer
            if (txPreBuffer_->space() >= static_cast<size_t>(ndaSampleRate_ / 100)) {
                // Generate ~10ms of audio
                float durationMs = 10.67f;
                py::array_t<float> fskAudio = adapter.attr("get_tx_audio")(durationMs);

                auto info = fskAudio.request();
                const float* data = static_cast<const float*>(info.ptr);
                size_t count = info.size;

                if (count > 0) {
                    txPreBuffer_->write(data, count);
                }

                // Check if still transmitting
                bool transmitting = adapter.attr("is_transmitting_active")().cast<bool>();
                bool wasTransmitting = isTransmitting_.load();
                isTransmitting_.store(transmitting);

                // Notify if transmission completed
                if (wasTransmitting && !transmitting) {
                    std::lock_guard<std::mutex> lock(callbackMutex_);
                    if (transmitCompleteCallback_) {
                        transmitCompleteCallback_();
                    }
                }
            }

            // Process RX: Read from post-buffer and demodulate
            if (rxPostBuffer_->available() >= static_cast<size_t>(ndaSampleRate_ / 100)) {
                float durationMs = 10.67f;
                size_t samplesToRead = static_cast<size_t>(ndaSampleRate_ * durationMs / 1000.0f);

                std::vector<float> rxAudio(samplesToRead);
                size_t read = rxPostBuffer_->read(rxAudio.data(), samplesToRead);

                if (read > 0) {
                    // Create numpy array
                    py::array_t<float> rxArray({static_cast<py::ssize_t>(read)});
                    auto buf = rxArray.request();
                    std::memcpy(buf.ptr, rxAudio.data(), read * sizeof(float));

                    // Process through adapter
                    adapter.attr("process_rx_audio")(rxArray, durationMs);

                    // Get decoded messages
                    py::list messages = adapter.attr("get_received_messages")();
                    for (auto& msg : messages) {
                        std::string text = msg.cast<std::string>();

                        {
                            std::lock_guard<std::mutex> lock(messagesMutex_);
                            rxMessageQueue_.push(text);
                        }

                        {
                            std::lock_guard<std::mutex> lock(callbackMutex_);
                            if (messageReceivedCallback_) {
                                messageReceivedCallback_(text);
                            }
                        }
                    }
                }
            }

            // Process TX message queue
            {
                std::lock_guard<std::mutex> lock(messagesMutex_);
                while (!txMessageQueue_.empty()) {
                    std::string msg = txMessageQueue_.front();
                    txMessageQueue_.pop();

                    py::dict result = adapter.attr("send_text_message")(msg);
                    bool success = result["success"].cast<bool>();
                    if (success) {
                        isTransmitting_.store(true);
                    }
                }
            }

                // Sleep to avoid busy-waiting
                std::this_thread::sleep_for(std::chrono::milliseconds(5));

            } catch (py::error_already_set& e) {
                std::cerr << "[NadeExternalIO] Python error in worker loop: " << e.what() << std::endl;
                if (e.trace()) {
                    std::cerr << "  Traceback: " << py::str(e.trace()).cast<std::string>() << std::endl;
                }
                // Continue running, don't crash the thread
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            } catch (const std::exception& e) {
                std::cerr << "[NadeExternalIO] Error in worker loop: " << e.what() << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }

        // Cleanup
        std::cout << "[NadeExternalIO] Worker loop exited, cleaning up..." << std::endl;
        delete static_cast<py::object*>(ndaAdapter_);
        ndaAdapter_ = nullptr;
        std::cout << "[NadeExternalIO] Cleanup complete" << std::endl;

    } catch (const std::exception& e) {
        // Catch C++ exceptions (including pybind11's error_already_set which inherits from std::runtime_error)
        std::cerr << "\n========================================" << std::endl;
        std::cerr << "[NadeExternalIO] Exception in worker thread" << std::endl;
        std::cerr.flush();

        // Try to get the error message safely
        try {
            std::cerr << "[NadeExternalIO] Error: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "[NadeExternalIO] (could not retrieve error message)" << std::endl;
        }

        // Check if there's also a Python error set
        if (PyErr_Occurred()) {
            std::cerr << "[NadeExternalIO] Python error details:" << std::endl;
            PyErr_Print();
            PyErr_Clear();
        }

        std::cerr << "[NadeExternalIO] Run manual test: python -c \"from nade.adapters.nda_adapter import NDAAdapter\"" << std::endl;
        std::cerr << "========================================\n" << std::endl;
        std::cerr.flush();

        pythonInitialized_.store(false);
    } catch (...) {
        std::cerr << "\n========================================" << std::endl;
        std::cerr << "[NadeExternalIO] Unknown exception in worker thread" << std::endl;
        std::cerr.flush();

        // Check if there's a Python error set
        if (PyErr_Occurred()) {
            std::cerr << "[NadeExternalIO] Python error details:" << std::endl;
            PyErr_Print();
            PyErr_Clear();
        }

        std::cerr << "========================================\n" << std::endl;
        std::cerr.flush();

        pythonInitialized_.store(false);
    }
#else
    // No Python support - just sleep until stopped
    while (running_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
#endif
}

// =============================================================================
// Text Messaging API
// =============================================================================

NadeResult NadeExternalIO::sendTextMessage(const std::string& text) {
    if (text.length() > 256) {
        return NadeResult::error("Message too long (max 256 characters)");
    }

    std::lock_guard<std::mutex> lock(messagesMutex_);
    txMessageQueue_.push(text);
    return NadeResult::ok();
}

std::vector<std::string> NadeExternalIO::getReceivedMessages() {
    std::vector<std::string> messages;
    std::lock_guard<std::mutex> lock(messagesMutex_);

    while (!rxMessageQueue_.empty()) {
        messages.push_back(rxMessageQueue_.front());
        rxMessageQueue_.pop();
    }

    return messages;
}

bool NadeExternalIO::isTransmitting() const {
    return isTransmitting_.load();
}

// =============================================================================
// Audio Processing (Non-blocking)
// =============================================================================

std::vector<float> NadeExternalIO::getTxAudio(float durationMs) {
    size_t numSamples = static_cast<size_t>(ndaSampleRate_ * durationMs / 1000.0f);
    std::vector<float> audio(numSamples, 0.0f);

    size_t read = txPreBuffer_->read(audio.data(), numSamples);

    // If we didn't get enough samples, the rest is already zeros (silence)
    (void)read;

    return audio;
}

void NadeExternalIO::processRxAudio(const float* audio, size_t numSamples, float durationMs) {
    (void)durationMs;  // Not used, sample count is sufficient

    // Write to post-buffer (may drop samples if buffer is full)
    size_t written = rxPostBuffer_->write(audio, numSamples);
    if (written < numSamples) {
        // Buffer overflow - samples dropped
        // This shouldn't happen if worker thread is keeping up
    }
}

// =============================================================================
// Diagnostics
// =============================================================================

float NadeExternalIO::getRxSignalQuality() const {
    // TODO: Get from Python adapter
    return 0.0f;
}

bool NadeExternalIO::isRxSynchronized() const {
    // TODO: Get from Python adapter
    return false;
}

std::string NadeExternalIO::getMode() const {
    return isTransmitting_.load() ? "tx" : "rx";
}

bool NadeExternalIO::isSessionEstablished() const {
    return sessionEstablished_.load();
}

int NadeExternalIO::getHandshakePhase() const {
    return handshakePhase_.load(std::memory_order_relaxed);
}

void NadeExternalIO::forceHandshake(bool isInitiator) {
#ifdef NDA_ENABLE_PYTHON
    if (!ndaAdapter_ || !pythonInitialized_.load()) {
        return;
    }

    try {
        py::gil_scoped_acquire gil;
        py::object* adapter = static_cast<py::object*>(ndaAdapter_);

        // Get ForceHandshake event class
        py::module_ events = py::module_::import("nade.protocol.events");
        py::object ForceHandshakeClass = events.attr("ForceHandshake");

        // Create event
        std::string role = isInitiator ? "initiator" : "responder";
        py::object event = ForceHandshakeClass(py::arg("role") = role);

        // Feed to engine
        py::object engine = (*adapter).attr("engine");
        engine.attr("feed_event")(event);

        std::cout << "[NadeExternalIO] Force handshake: " << role << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[NadeExternalIO] Force handshake error: " << e.what() << std::endl;
    }
#endif
}

void NadeExternalIO::restartDiscovery() {
#ifdef NDA_ENABLE_PYTHON
    if (!ndaAdapter_ || !pythonInitialized_.load()) {
        return;
    }

    try {
        py::gil_scoped_acquire gil;
        py::object* adapter = static_cast<py::object*>(ndaAdapter_);

        // Call public restart_discovery method on adapter
        (*adapter).attr("restart_discovery")();

        std::cout << "[NadeExternalIO] Restarted discovery" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[NadeExternalIO] Restart discovery error: " << e.what() << std::endl;
    }
#endif
}

// =============================================================================
// Callbacks
// =============================================================================

void NadeExternalIO::setTransmitCompleteCallback(TransmitCompleteCallback callback) {
    std::lock_guard<std::mutex> lock(callbackMutex_);
    transmitCompleteCallback_ = std::move(callback);
}

void NadeExternalIO::setMessageReceivedCallback(MessageReceivedCallback callback) {
    std::lock_guard<std::mutex> lock(callbackMutex_);
    messageReceivedCallback_ = std::move(callback);
}

}  // namespace nade
