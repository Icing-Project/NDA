#include "plugins/AudioSourcePlugin.h"
#include "audio/RingBuffer.h"
#include <algorithm>
#include <iostream>
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>

#ifdef _WIN32
#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <functiondiscoverykeys_devpkey.h>

// Link required COM libraries
#pragma comment(lib, "ole32.lib")

// Define KSDATAFORMAT_SUBTYPE_IEEE_FLOAT if not already defined
#ifndef KSDATAFORMAT_SUBTYPE_IEEE_FLOAT
static const GUID KSDATAFORMAT_SUBTYPE_IEEE_FLOAT = {
    0x00000003, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71}
};
#endif
#endif

namespace nda {

class WindowsMicrophoneSourcePlugin : public AudioSourcePlugin {
public:
    WindowsMicrophoneSourcePlugin()
        : state_(PluginState::Unloaded),
          sampleRate_(48000),
          channels_(2),
          bufferSize_(512),
          volume_(1.0f),
          mute_(false),
          framesCaptured_(0),
          underruns_(0),
          readCalls_(0),
#ifdef _WIN32
          deviceEnum_(nullptr),
          device_(nullptr),
          audioClient_(nullptr),
          captureClient_(nullptr),
#endif
          comInitialized_(false),
          comOwned_(false),
          // v2.2: Event-driven support
          dataReadyThreshold_(512)  // Default: notify when 512 frames available
    {
    }

    ~WindowsMicrophoneSourcePlugin() override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ != PluginState::Unloaded) {
            shutdown();
        }
    }

    bool initialize() override {
        std::lock_guard<std::mutex> lock(mutex_);

        if (state_ != PluginState::Unloaded) {
            std::cerr << "[WindowsMicrophone] Initialize failed: already initialized\n";
            return false;
        }

#ifdef _WIN32
        // 1. COM initialization (MULTITHREADED for plugin compatibility)
        HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        if (hr == RPC_E_CHANGED_MODE) {
            // Already initialized by host - that's OK
            comInitialized_ = true;
            comOwned_ = false;
            std::cerr << "[WindowsMicrophone] COM already initialized by host\n";
        } else if (SUCCEEDED(hr)) {
            comInitialized_ = true;
            comOwned_ = true;
            std::cerr << "[WindowsMicrophone] COM initialized successfully\n";
        } else {
            std::cerr << "[WindowsMicrophone] COM init failed: 0x" << std::hex << hr << std::dec << "\n";
            return false;
        }

        // 2. Create device enumerator
        hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
                              __uuidof(IMMDeviceEnumerator), (void**)&deviceEnum_);
        if (FAILED(hr)) {
            std::cerr << "[WindowsMicrophone] Device enumerator failed: 0x" << std::hex << hr << std::dec << "\n";
            return false;
        }

        // 3. Get default capture device (eCapture for input, eConsole for general use)
        hr = deviceEnum_->GetDefaultAudioEndpoint(eCapture, eConsole, &device_);
        if (FAILED(hr)) {
            std::cerr << "[WindowsMicrophone] No default capture device found: 0x" << std::hex << hr << std::dec << "\n";
            return false;
        }

        // 4. Activate audio client on the device
        hr = device_->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, (void**)&audioClient_);
        if (FAILED(hr)) {
            std::cerr << "[WindowsMicrophone] Audio client activation failed: 0x" << std::hex << hr << std::dec << "\n";
            return false;
        }

        // 5. Query device's preferred mix format (CRITICAL: always check this first)
        WAVEFORMATEX* mixFormat = nullptr;
        hr = audioClient_->GetMixFormat(&mixFormat);
        if (FAILED(hr)) {
            std::cerr << "[WindowsMicrophone] GetMixFormat failed: 0x" << std::hex << hr << std::dec << "\n";
            return false;
        }

        // 6. Log device's native format for debugging
        std::cerr << "[WindowsMicrophone] Device mix format: " << mixFormat->nSamplesPerSec << "Hz, "
                  << mixFormat->nChannels << "ch, " << mixFormat->wBitsPerSample << "bit\n";

        // 7. Prepare format for WASAPI
        //    CRITICAL: In shared mode, we MUST use the device's native sample rate
        //    Windows audio engine handles resampling, but we need to match the device
        bool useExtended = (mixFormat->wFormatTag == WAVE_FORMAT_EXTENSIBLE);

        // In shared mode, Windows requires using the device's mix format sample rate
        int deviceSampleRate = mixFormat->nSamplesPerSec;

        if (useExtended) {
            // Modify the extended format in place for float32 output
            WAVEFORMATEXTENSIBLE* formatEx = reinterpret_cast<WAVEFORMATEXTENSIBLE*>(mixFormat);
            formatEx->SubFormat = KSDATAFORMAT_SUBTYPE_IEEE_FLOAT;
            formatEx->Format.wBitsPerSample = 32;
            formatEx->Format.nBlockAlign = formatEx->Format.nChannels * 4;
            formatEx->Format.nAvgBytesPerSec = formatEx->Format.nSamplesPerSec * formatEx->Format.nBlockAlign;
            formatEx->Samples.wValidBitsPerSample = 32;
        }

        // Update our internal state to match device's actual sample rate
        // This is CRITICAL - pipeline must know the true rate for correct pacing
        if (deviceSampleRate != sampleRate_) {
            std::cerr << "[WindowsMicrophone] Adapting to device sample rate: "
                      << sampleRate_ << "Hz -> " << deviceSampleRate << "Hz\n";
            sampleRate_ = deviceSampleRate;
        }
        channels_ = mixFormat->nChannels;

        // 8. Calculate buffer duration at device's actual sample rate
        int bufferFrames = (sampleRate_ * 200) / 1000;  // 200ms at device rate
        REFERENCE_TIME bufferDuration = (REFERENCE_TIME)((10000000.0 * bufferFrames) / sampleRate_);

        // 9. Initialize audio client in SHARED mode
        //    In shared mode, we MUST use the device's mix format (Windows requirement)
        hr = audioClient_->Initialize(
            AUDCLNT_SHAREMODE_SHARED,           // Shared mode (not exclusive)
            0,                                   // No special flags - polling mode
            bufferDuration,                      // Buffer size (200ms)
            0,                                   // Must be 0 in shared mode
            mixFormat,                           // MUST use device format in shared mode
            nullptr                              // No session GUID
        );

        if (FAILED(hr)) {
            // Fallback: retry with unmodified format
            std::cerr << "[WindowsMicrophone] Initialize failed: 0x" << std::hex << hr << std::dec
                      << ", retrying with device defaults\n";

            // Re-query device format (ours might have been modified)
            CoTaskMemFree(mixFormat);
            hr = audioClient_->GetMixFormat(&mixFormat);
            if (FAILED(hr)) {
                std::cerr << "[WindowsMicrophone] GetMixFormat retry failed\n";
                return false;
            }

            sampleRate_ = mixFormat->nSamplesPerSec;
            channels_ = mixFormat->nChannels;
            bufferFrames = (sampleRate_ * 200) / 1000;
            bufferDuration = (REFERENCE_TIME)((10000000.0 * bufferFrames) / sampleRate_);

            hr = audioClient_->Initialize(
                AUDCLNT_SHAREMODE_SHARED,
                0,
                bufferDuration,
                0,
                mixFormat,
                nullptr
            );
        }

        CoTaskMemFree(mixFormat);

        if (FAILED(hr)) {
            std::cerr << "[WindowsMicrophone] Initialize failed: 0x" << std::hex << hr << std::dec << "\n";
            return false;
        }

        // 9. Query the actual allocated buffer size (may differ from request)
        UINT32 actualBufferFrames = 0;
        hr = audioClient_->GetBufferSize(&actualBufferFrames);
        if (SUCCEEDED(hr)) {
            std::cerr << "[WindowsMicrophone] WASAPI allocated buffer: " << actualBufferFrames
                      << " frames (" << (actualBufferFrames * 1000 / sampleRate_) << "ms)\n";
        } else {
            std::cerr << "[WindowsMicrophone] GetBufferSize failed: 0x" << std::hex << hr << std::dec << "\n";
        }

        // 10. Get capture client service interface (used to read audio data)
        hr = audioClient_->GetService(__uuidof(IAudioCaptureClient), (void**)&captureClient_);
        if (FAILED(hr)) {
            std::cerr << "[WindowsMicrophone] Capture client service failed: 0x" << std::hex << hr << std::dec << "\n";
            return false;
        }

        // Reset counters
        framesCaptured_ = 0;
        underruns_ = 0;
        readCalls_ = 0;

        // ============================================================
        // Ring Buffer Initialization (v2.1)
        // ============================================================

        // Calculate ring buffer capacity: 200ms for maximum stability
        int ringBufferCapacity = (sampleRate_ * 200) / 1000;  // 200ms

        if (!ringBuffer_.initialize(channels_, ringBufferCapacity)) {
            std::cerr << "[WindowsMicrophone] Ring buffer initialization failed\n";
            return false;
        }

        std::cerr << "[WindowsMicrophone] Ring buffer initialized: "
                  << ringBufferCapacity << " frames ("
                  << (ringBufferCapacity * 1000 / sampleRate_) << "ms, "
                  << channels_ << " channels)\n";

        // Initialize capture thread flag (thread started in start(), not here)
        captureThreadRunning_.store(false, std::memory_order_relaxed);

        // Allocate temporary conversion buffer (reused in capture thread)
        tempBuffer_.resize(channels_);
        const int maxPacketFrames = (sampleRate_ * 100) / 1000;  // 100ms
        for (int ch = 0; ch < channels_; ++ch) {
            tempBuffer_[ch].resize(maxPacketFrames);
        }

        // ============================================================

        state_ = PluginState::Initialized;
        std::cerr << "[WindowsMicrophone] Initialized: " << sampleRate_ << "Hz, "
                  << channels_ << "ch\n";
        return true;
#else
        std::cerr << "[WindowsMicrophone] WASAPI only supported on Windows\n";
        return false;
#endif
    }

    void shutdown() override {
        std::lock_guard<std::mutex> lock(mutex_);

        if (state_ == PluginState::Running) {
            stop();
        }

#ifdef _WIN32
        // Release WASAPI resources
        if (captureClient_) {
            captureClient_->Release();
            captureClient_ = nullptr;
        }

        if (audioClient_) {
            audioClient_->Release();
            audioClient_ = nullptr;
        }

        if (device_) {
            device_->Release();
            device_ = nullptr;
        }

        if (deviceEnum_) {
            deviceEnum_->Release();
            deviceEnum_ = nullptr;
        }

        // Uninitialize COM if we own it
        if (comInitialized_ && comOwned_) {
            CoUninitialize();
        }
        comInitialized_ = false;
        comOwned_ = false;
#endif

        // ============================================================
        // Log Ring Buffer Statistics (v2.1)
        // ============================================================

        std::cerr << "[WindowsMicrophone] Ring buffer final stats:\n";
        std::cerr << "  Overruns (buffer full): " << ringBuffer_.getOverruns() << "\n";
        std::cerr << "  Underruns (buffer empty): " << ringBuffer_.getUnderruns() << "\n";

        // ============================================================

        std::cerr << "[WindowsMicrophone] Shutdown: Captured " << framesCaptured_
                  << " frames, " << underruns_ << " underruns across "
                  << readCalls_ << " calls\n";

        state_ = PluginState::Unloaded;
    }

    bool start() override {
        std::lock_guard<std::mutex> lock(mutex_);

        if (state_ != PluginState::Initialized) {
            std::cerr << "[WindowsMicrophone] Start failed: not initialized (state="
                      << static_cast<int>(state_) << ")\n";
            return false;
        }

#ifdef _WIN32
        if (audioClient_) {
            HRESULT hr = audioClient_->Start();
            if (FAILED(hr)) {
                std::cerr << "[WindowsMicrophone] Start failed: 0x" << std::hex << hr << std::dec << "\n";
                return false;
            }
        }

        // ============================================================
        // Start Capture Thread (v2.1 Ring Buffer)
        // ============================================================

        captureThreadRunning_.store(true, std::memory_order_release);

        captureThread_ = std::thread([this]() {
            this->captureThreadFunc();
        });

        std::cerr << "[WindowsMicrophone] Capture thread started\n";

        // ============================================================
#endif

        state_ = PluginState::Running;
        std::cerr << "[WindowsMicrophone] Started\n";
        return true;
    }

    void stop() override {
        std::unique_lock<std::mutex> lock(mutex_);

        if (state_ != PluginState::Running) {
            return;
        }

#ifdef _WIN32
        // ============================================================
        // Stop Capture Thread (v2.1 Ring Buffer)
        // ============================================================

        if (captureThreadRunning_.load(std::memory_order_acquire)) {
            std::cerr << "[WindowsMicrophone] Stopping capture thread...\n";

            // Signal thread to stop
            captureThreadRunning_.store(false, std::memory_order_release);
        }

        // Temporarily release mutex for thread join to avoid deadlock
        lock.unlock();
        if (captureThread_.joinable()) {
            captureThread_.join();
            std::cerr << "[WindowsMicrophone] Capture thread joined\n";
        }
        lock.lock();

        // Clear ring buffer
        ringBuffer_.clear();

        // ============================================================

        if (audioClient_) {
            audioClient_->Stop();
        }
#endif

        state_ = PluginState::Initialized;
        std::cerr << "[WindowsMicrophone] Stopped\n";
    }

    PluginInfo getInfo() const override {
        return {
            "Windows Microphone (WASAPI)",
            "1.0.0",
            "Icing Project",
            "WASAPI capture from default Windows microphone input",
            PluginType::AudioSource,
            NDA_PLUGIN_API_VERSION
        };
    }

    PluginState getState() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return state_;
    }

    void setParameter(const std::string& key, const std::string& value) override {
        std::lock_guard<std::mutex> lock(mutex_);

        if (key == "volume") {
            float newVol = std::stof(value);
            volume_ = std::clamp(newVol, 0.0f, 2.0f);
            std::cerr << "[WindowsMicrophone] Volume set to " << volume_ << "\n";
        } else if (key == "mute") {
            mute_ = (value == "true" || value == "1");
            std::cerr << "[WindowsMicrophone] Mute set to " << (mute_ ? "true" : "false") << "\n";
        }
        // sampleRate, channels can only be set before initialize()
    }

    std::string getParameter(const std::string& key) const override {
        std::lock_guard<std::mutex> lock(mutex_);

        if (key == "volume") return std::to_string(volume_);
        if (key == "mute") return mute_ ? "true" : "false";
        if (key == "sampleRate") return std::to_string(sampleRate_);
        if (key == "channels") return std::to_string(channels_);
        if (key == "bufferSize") return std::to_string(bufferSize_);
        if (key == "framesCaptured") return std::to_string(framesCaptured_);
        if (key == "underruns") return std::to_string(underruns_);
        if (key == "readCalls") return std::to_string(readCalls_);
        return "";
    }

    void setAudioCallback(AudioSourceCallback callback) override {
        std::lock_guard<std::mutex> lock(mutex_);
        callback_ = callback;
    }

    bool readAudio(AudioBuffer& buffer) override {
        std::lock_guard<std::mutex> lock(mutex_);

        if (state_ != PluginState::Running) {
            buffer.clear();
            return false;
        }

#ifdef _WIN32
        readCalls_++;
        int requestedFrames = buffer.getFrameCount();
        int bufferChannels = buffer.getChannelCount();

        // Validate channel count
        if (bufferChannels != channels_) {
            std::cerr << "[WindowsMicrophone] Channel mismatch: expected "
                      << channels_ << ", got " << bufferChannels << "\n";
            buffer.clear();
            return false;
        }

        // ============================================================
        // Read from ring buffer (not WASAPI directly) - v2.1
        // ============================================================

        std::vector<float*> channelPtrs(channels_);
        for (int ch = 0; ch < channels_; ++ch) {
            channelPtrs[ch] = buffer.getChannelData(ch);
        }

        int framesRead = ringBuffer_.read(channelPtrs.data(), requestedFrames);

        // ============================================================
        // Handle underrun: pad with silence if not enough data
        // ============================================================

        if (framesRead < requestedFrames) {
            // Ring buffer underrun - pad remainder with silence
            for (int ch = 0; ch < bufferChannels; ++ch) {
                float* channelData = buffer.getChannelData(ch);
                for (int frame = framesRead; frame < requestedFrames; ++frame) {
                    channelData[frame] = 0.0f;
                }
            }
            underruns_++;
        }

        // ============================================================
        // Apply volume and mute (pipeline-side processing)
        // ============================================================

        for (int ch = 0; ch < bufferChannels; ++ch) {
            float* channelData = buffer.getChannelData(ch);
            for (int frame = 0; frame < requestedFrames; ++frame) {
                float sample = channelData[frame];

                if (mute_) {
                    sample = 0.0f;
                } else {
                    sample *= volume_;
                }

                channelData[frame] = sample;
            }
        }

        // ============================================================
        // Update metrics
        // ============================================================

        framesCaptured_ += framesRead;

        // Log progress every 100 calls (include ring buffer diagnostics)
        if (readCalls_ % 100 == 0) {
            double secondsCaptured = static_cast<double>(framesCaptured_) / sampleRate_;
            int availableFrames = ringBuffer_.getAvailableRead();
            double fillMs = (availableFrames * 1000.0) / sampleRate_;

            std::cerr << "[WindowsMicrophone] Stats: " << framesCaptured_
                      << " frames (" << secondsCaptured << "s), "
                      << underruns_ << " underruns, "
                      << "ring buffer: " << availableFrames << " frames ("
                      << fillMs << "ms fill)\n";
        }

        // ============================================================
        // Always return true (silence on underrun, not failure)
        // ============================================================

        return true;
#else
        buffer.clear();
        return false;
#endif
    }

    int getSampleRate() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return sampleRate_;
    }

    int getChannels() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return channels_;
    }

    int getBufferSize() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return bufferSize_;
    }

    void setBufferSize(int frames) override {
        std::lock_guard<std::mutex> lock(mutex_);

        if (state_ == PluginState::Unloaded) {
            if (frames > 0 && frames <= 8192) {
                bufferSize_ = frames;
            }
        }
    }

    void setSampleRate(int rate) override {
        std::lock_guard<std::mutex> lock(mutex_);

        if (state_ == PluginState::Unloaded) {
            if (rate > 0 && rate <= 192000) {
                sampleRate_ = rate;
            }
        }
    }

    void setChannels(int ch) override {
        std::lock_guard<std::mutex> lock(mutex_);

        if (state_ == PluginState::Unloaded) {
            if (ch > 0 && ch <= 8) {
                channels_ = ch;
            }
        }
    }

    // ===== v2.2: Event-Driven Pipeline Support =====

    bool supportsAsyncMode() const override {
        // This plugin uses ring buffer + background thread, supports async mode
        return true;
    }

    void setDataReadyCallback(DataReadyCallback callback) override {
        std::lock_guard<std::mutex> lock(mutex_);
        dataReadyCallback_ = callback;

        if (callback) {
            std::cerr << "[WindowsMicrophone] v2.2: Data-ready callback registered "
                      << "(threshold: " << dataReadyThreshold_ << " frames)\n";
        } else {
            std::cerr << "[WindowsMicrophone] v2.2: Data-ready callback cleared\n";
        }
    }

    int getDataReadyThreshold() const override {
        return dataReadyThreshold_;
    }

    // ===== End Event-Driven Pipeline Support =====

private:
    mutable std::mutex mutex_;
    PluginState state_;

    // Configuration
    int sampleRate_;
    int channels_;
    int bufferSize_;
    float volume_;
    bool mute_;

    // Metrics
    uint64_t framesCaptured_;
    uint64_t underruns_;
    uint64_t readCalls_;

#ifdef _WIN32
    // WASAPI handles
    IMMDeviceEnumerator* deviceEnum_;
    IMMDevice* device_;
    IAudioClient* audioClient_;
    IAudioCaptureClient* captureClient_;
#endif

    // COM state
    bool comInitialized_;
    bool comOwned_;

    // Callback (not used in pull model, but required by interface)
    AudioSourceCallback callback_;

    // ===== Ring Buffer Integration (v2.1) =====

    /// Ring buffer for decoupling WASAPI async delivery from pipeline sync polling
    RingBuffer ringBuffer_;

    /// Background capture thread (polls WASAPI continuously)
    std::thread captureThread_;

    /// Capture thread lifecycle flag
    std::atomic<bool> captureThreadRunning_;

    /// Temporary conversion buffer (reused in capture thread, planar format)
    std::vector<std::vector<float>> tempBuffer_;

    // ===== v2.2: Event-Driven Pipeline Support =====

    /// Callback to notify pipeline when data is ready
    DataReadyCallback dataReadyCallback_;

    /// Threshold for triggering data-ready callback (frames)
    int dataReadyThreshold_;

    // ===== End Ring Buffer / Event-Driven Integration =====

    /**
     * @brief Capture thread function - polls WASAPI and feeds ring buffer.
     */
    void captureThreadFunc() {
#ifdef _WIN32
        // Set thread priority to time-critical for low latency
        if (!SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL)) {
            std::cerr << "[WindowsMicrophone] Capture thread: SetThreadPriority failed (error "
                      << GetLastError() << ")\n";
        } else {
            std::cerr << "[WindowsMicrophone] Capture thread: priority set to TIME_CRITICAL\n";
        }

        // Initialize COM for this thread (WASAPI requires COM)
        HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        bool comInitInThread = SUCCEEDED(hr) || (hr == RPC_E_CHANGED_MODE);
        bool comOwnedByThread = SUCCEEDED(hr) && (hr != RPC_E_CHANGED_MODE);

        if (!comInitInThread) {
            std::cerr << "[WindowsMicrophone] Capture thread: COM init failed (HRESULT 0x"
                      << std::hex << hr << std::dec << ")\n";
            return;
        }

        std::cerr << "[WindowsMicrophone] Capture thread: COM initialized "
                  << (comOwnedByThread ? "(owned)" : "(inherited)") << "\n";

        uint64_t packetsProcessed = 0;
        uint64_t totalFramesCaptured = 0;

        while (captureThreadRunning_.load(std::memory_order_acquire)) {
            // Query WASAPI for next packet
            UINT32 packetFrames = 0;
            hr = captureClient_->GetNextPacketSize(&packetFrames);

            if (FAILED(hr)) {
                std::cerr << "[WindowsMicrophone] Capture thread: GetNextPacketSize failed (HRESULT 0x"
                          << std::hex << hr << std::dec << ")\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            if (packetFrames == 0) {
                // No data available yet - sleep briefly and retry
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            // Get packet data from WASAPI
            BYTE* captureData = nullptr;
            DWORD flags = 0;
            UINT64 devicePosition = 0;
            UINT64 qpcPosition = 0;

            hr = captureClient_->GetBuffer(
                &captureData,
                &packetFrames,
                &flags,
                &devicePosition,
                &qpcPosition
            );

            if (FAILED(hr)) {
                std::cerr << "[WindowsMicrophone] Capture thread: GetBuffer failed (HRESULT 0x"
                          << std::hex << hr << std::dec << ")\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            // Ensure temp buffer is large enough
            for (int ch = 0; ch < channels_; ++ch) {
                if (tempBuffer_[ch].size() < packetFrames) {
                    tempBuffer_[ch].resize(packetFrames);
                }
            }

            // Convert interleaved → planar
            if (flags & AUDCLNT_BUFFERFLAGS_SILENT) {
                // Silence packet - write zeros
                for (int ch = 0; ch < channels_; ++ch) {
                    std::fill(tempBuffer_[ch].begin(),
                              tempBuffer_[ch].begin() + packetFrames,
                              0.0f);
                }
            } else {
                // Normal audio data - convert format
                float* src = reinterpret_cast<float*>(captureData);
                for (UINT32 frame = 0; frame < packetFrames; ++frame) {
                    for (int ch = 0; ch < channels_; ++ch) {
                        tempBuffer_[ch][frame] = *src++;
                    }
                }
            }

            // Write to ring buffer
            std::vector<float*> channelPtrs(channels_);
            for (int ch = 0; ch < channels_; ++ch) {
                channelPtrs[ch] = tempBuffer_[ch].data();
            }

            int framesWritten = ringBuffer_.write(
                const_cast<const float**>(channelPtrs.data()),
                packetFrames
            );

            // Release WASAPI buffer
            hr = captureClient_->ReleaseBuffer(packetFrames);
            if (FAILED(hr)) {
                std::cerr << "[WindowsMicrophone] Capture thread: ReleaseBuffer failed (HRESULT 0x"
                          << std::hex << hr << std::dec << ")\n";
            }

            packetsProcessed++;
            totalFramesCaptured += framesWritten;

            // v2.2: Notify pipeline when enough data is available
            int availableFrames = ringBuffer_.getAvailableRead();
            if (availableFrames >= dataReadyThreshold_ && dataReadyCallback_) {
                // Call the callback to wake the pipeline thread
                // Note: This is thread-safe - the callback uses condition_variable
                dataReadyCallback_();
            }

            // Log progress every 100 packets (~1 second)
            if (packetsProcessed % 100 == 0) {
                int bufferFill = ringBuffer_.getAvailableRead();
                double fillMs = (bufferFill * 1000.0) / sampleRate_;

                std::cerr << "[WindowsMicrophone] Capture thread: "
                          << packetsProcessed << " packets, "
                          << totalFramesCaptured << " frames, "
                          << "ring buffer fill: " << bufferFill << " frames ("
                          << fillMs << "ms)\n";
            }
        }

        std::cerr << "[WindowsMicrophone] Capture thread exiting (processed "
                  << packetsProcessed << " packets, "
                  << totalFramesCaptured << " frames)\n";

        // Cleanup COM if we own it
        if (comOwnedByThread) {
            CoUninitialize();
        }
#endif
    }
};

} // namespace nda

// Export the plugin
NDA_DECLARE_PLUGIN(nda::WindowsMicrophoneSourcePlugin)
