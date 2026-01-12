#include "plugins/AudioSinkPlugin.h"
#include "audio/RingBuffer.h"
#include <algorithm>
#include <iostream>
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>

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

class WindowsSpeakerSinkPlugin : public AudioSinkPlugin {
public:
    WindowsSpeakerSinkPlugin()
        : state_(PluginState::Unloaded),
          sampleRate_(48000),
          channels_(2),
          bufferFrames_(4800),  // 100ms at 48kHz
          volume_(1.0f),
          mute_(false),
          framesWritten_(0),
          underruns_(0),
          overruns_(0),
          writeCalls_(0),
#ifdef _WIN32
          deviceEnum_(nullptr),
          device_(nullptr),
          audioClient_(nullptr),
          renderClient_(nullptr),
#endif
          comInitialized_(false),
          comOwned_(false),
          // v2.2: Event-driven / ring buffer support
          renderThreadRunning_(false),
          spaceAvailableThreshold_(512)
    {
    }

    ~WindowsSpeakerSinkPlugin() override {
        // v2.2: Don't lock here - shutdown() handles its own locking
        // Locking here + shutdown() locking = undefined behavior (recursive lock)
        if (state_ != PluginState::Unloaded) {
            shutdown();
        }
    }

    bool initialize() override {
        std::lock_guard<std::mutex> lock(mutex_);

        if (state_ != PluginState::Unloaded) {
            std::cerr << "[WindowsSpeaker] Initialize failed: already initialized\n";
            return false;
        }

#ifdef _WIN32
        // 1. COM initialization (MULTITHREADED for plugin compatibility)
        HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        if (hr == RPC_E_CHANGED_MODE) {
            // Already initialized by host - that's OK
            comInitialized_ = true;
            comOwned_ = false;
            std::cerr << "[WindowsSpeaker] COM already initialized by host\n";
        } else if (SUCCEEDED(hr)) {
            comInitialized_ = true;
            comOwned_ = true;
            std::cerr << "[WindowsSpeaker] COM initialized successfully\n";
        } else {
            std::cerr << "[WindowsSpeaker] COM init failed: 0x" << std::hex << hr << std::dec << "\n";
            return false;
        }

        // 2. Create device enumerator
        hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
                              __uuidof(IMMDeviceEnumerator), (void**)&deviceEnum_);
        if (FAILED(hr)) {
            std::cerr << "[WindowsSpeaker] Device enumerator failed: 0x" << std::hex << hr << std::dec << "\n";
            return false;
        }

        // 3. Get default playback device (eRender for output, eConsole for general use)
        hr = deviceEnum_->GetDefaultAudioEndpoint(eRender, eConsole, &device_);
        if (FAILED(hr)) {
            std::cerr << "[WindowsSpeaker] No default playback device found: 0x" << std::hex << hr << std::dec << "\n";
            return false;
        }

        // 4. Activate audio client on the device
        hr = device_->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, (void**)&audioClient_);
        if (FAILED(hr)) {
            std::cerr << "[WindowsSpeaker] Audio client activation failed: 0x" << std::hex << hr << std::dec << "\n";
            return false;
        }

        // 5. Query device's preferred mix format (CRITICAL: always check this first)
        WAVEFORMATEX* mixFormat = nullptr;
        hr = audioClient_->GetMixFormat(&mixFormat);
        if (FAILED(hr)) {
            std::cerr << "[WindowsSpeaker] GetMixFormat failed: 0x" << std::hex << hr << std::dec << "\n";
            return false;
        }

        // 6. Log device's native format for debugging
        std::cerr << "[WindowsSpeaker] Device mix format: " << mixFormat->nSamplesPerSec << "Hz, "
                  << mixFormat->nChannels << "ch, " << mixFormat->wBitsPerSample << "bit\n";

        // 7. Prepare our requested format (48kHz, stereo, float32)
        //    CRITICAL: In shared mode, we MUST use the device's native sample rate
        //    Windows audio engine handles resampling, but we need to match the device
        bool useExtended = (mixFormat->wFormatTag == WAVE_FORMAT_EXTENSIBLE);

        // In shared mode, Windows requires using the device's mix format sample rate
        // We can change channels and bit depth, but sample rate is fixed by the device
        int deviceSampleRate = mixFormat->nSamplesPerSec;

        if (useExtended) {
            // Modify the extended format in place
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
            std::cerr << "[WindowsSpeaker] Adapting to device sample rate: "
                      << sampleRate_ << "Hz -> " << deviceSampleRate << "Hz\n";
            sampleRate_ = deviceSampleRate;
        }
        channels_ = mixFormat->nChannels;

        // 8. Calculate buffer duration at device's actual sample rate
        bufferFrames_ = (sampleRate_ * 200) / 1000;  // 200ms at device rate
        REFERENCE_TIME bufferDuration = (REFERENCE_TIME)((10000000.0 * bufferFrames_) / sampleRate_);

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
            // Fallback: Use device's original mix format (always works)
            std::cerr << "[WindowsSpeaker] Custom format rejected, using device format\n";

            // Update our config to match device
            sampleRate_ = mixFormat->nSamplesPerSec;
            channels_ = mixFormat->nChannels;
            bufferFrames_ = (int)((mixFormat->nSamplesPerSec * 200) / 1000);  // 200ms at device rate

            // Recalculate buffer duration for device rate
            bufferDuration = (REFERENCE_TIME)((10000000.0 * bufferFrames_) / sampleRate_);

            hr = audioClient_->Initialize(
                AUDCLNT_SHAREMODE_SHARED,
                0,  // No event callback
                bufferDuration,
                0,
                mixFormat,
                nullptr
            );
        }

        CoTaskMemFree(mixFormat);

        if (FAILED(hr)) {
            std::cerr << "[WindowsSpeaker] Initialize failed: 0x" << std::hex << hr << std::dec << "\n";
            return false;
        }

        // 9. Query the actual allocated buffer size (may differ from request)
        UINT32 actualBufferFrames = 0;
        hr = audioClient_->GetBufferSize(&actualBufferFrames);
        if (SUCCEEDED(hr)) {
            // CRITICAL: Use the actual allocated size, not our request
            bufferFrames_ = actualBufferFrames;
            std::cerr << "[WindowsSpeaker] WASAPI allocated buffer: " << actualBufferFrames
                      << " frames (" << (actualBufferFrames * 1000 / sampleRate_) << "ms)\n";
        } else {
            std::cerr << "[WindowsSpeaker] GetBufferSize failed: 0x" << std::hex << hr << std::dec << "\n";
        }

        // 10. Get render client service interface (used to write audio data)
        hr = audioClient_->GetService(__uuidof(IAudioRenderClient), (void**)&renderClient_);
        if (FAILED(hr)) {
            std::cerr << "[WindowsSpeaker] Render client service failed: 0x" << std::hex << hr << std::dec << "\n";
            return false;
        }

        // Reset counters
        framesWritten_ = 0;
        underruns_ = 0;
        overruns_ = 0;
        writeCalls_ = 0;

        // ============================================================
        // v2.2: Ring Buffer Initialization (Event-Driven Pipeline)
        // ============================================================

        // Calculate ring buffer capacity: 200ms for maximum stability
        int ringBufferCapacity = (sampleRate_ * 200) / 1000;  // 200ms

        if (!ringBuffer_.initialize(channels_, ringBufferCapacity)) {
            std::cerr << "[WindowsSpeaker] Ring buffer initialization failed\n";
            return false;
        }

        std::cerr << "[WindowsSpeaker] Ring buffer initialized: "
                  << ringBufferCapacity << " frames ("
                  << (ringBufferCapacity * 1000 / sampleRate_) << "ms, "
                  << channels_ << " channels)\n";

        // Initialize render thread flag (thread started in start(), not here)
        renderThreadRunning_.store(false, std::memory_order_relaxed);

        // Allocate temporary conversion buffer (reused in render thread)
        tempBuffer_.resize(channels_);
        const int maxFramesPerWrite = 2048;  // Conservative max per WASAPI write
        for (int ch = 0; ch < channels_; ++ch) {
            tempBuffer_[ch].resize(maxFramesPerWrite);
        }

        // ============================================================

        state_ = PluginState::Initialized;
        std::cerr << "[WindowsSpeaker] Initialized: " << sampleRate_ << "Hz, "
                  << channels_ << "ch, " << bufferFrames_ << " frames\n";
        return true;
#else
        std::cerr << "[WindowsSpeaker] WASAPI only supported on Windows\n";
        return false;
#endif
    }

    void shutdown() override {
        // v2.2: Always call stop() BEFORE locking to avoid recursive mutex deadlock
        // stop() has its own locking and state check - safe to call unconditionally
        // This ensures threads are properly stopped before we release resources
        stop();

        std::lock_guard<std::mutex> lock(mutex_);

#ifdef _WIN32
        // Release WASAPI resources
        if (renderClient_) {
            renderClient_->Release();
            renderClient_ = nullptr;
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
        // v2.2: Log Ring Buffer Statistics
        // ============================================================

        std::cerr << "[WindowsSpeaker] Ring buffer final stats:\n";
        std::cerr << "  Overruns (buffer full): " << ringBuffer_.getOverruns() << "\n";
        std::cerr << "  Underruns (buffer empty): " << ringBuffer_.getUnderruns() << "\n";

        // ============================================================

        std::cerr << "[WindowsSpeaker] Shutdown: Wrote " << framesWritten_
                  << " frames, " << underruns_ << " underruns, " << overruns_
                  << " overruns across " << writeCalls_ << " calls\n";

        state_ = PluginState::Unloaded;
    }

    bool start() override {
        std::lock_guard<std::mutex> lock(mutex_);

        if (state_ != PluginState::Initialized) {
            std::cerr << "[WindowsSpeaker] Start failed: not initialized (state="
                      << static_cast<int>(state_) << ")\n";
            return false;
        }

#ifdef _WIN32
        // ============================================================
        // v2.2: Start Render Thread FIRST (Event-Driven Pipeline)
        // ============================================================
        // With ring buffer architecture, the render thread handles feeding WASAPI.
        // We do NOT prime WASAPI buffer - that causes backpressure when the ring
        // buffer fills up before WASAPI can drain the primed silence.

        renderThreadRunning_.store(true, std::memory_order_release);

        renderThread_ = std::thread([this]() {
            this->renderThreadFunc();
        });

        std::cerr << "[WindowsSpeaker] Render thread started\n";

        // ============================================================

        if (audioClient_) {
            // v2.2: NO buffer priming - the render thread feeds WASAPI from ring buffer
            // Small initial latency (~10-20ms) as ring buffer fills, but no backpressure
            HRESULT hr = audioClient_->Start();
            if (FAILED(hr)) {
                std::cerr << "[WindowsSpeaker] Start failed: 0x" << std::hex << hr << std::dec << "\n";
                renderThreadRunning_.store(false, std::memory_order_release);
                if (renderThread_.joinable()) {
                    renderThread_.join();
                }
                return false;
            }
            std::cerr << "[WindowsSpeaker] WASAPI audio client started (no buffer priming)\n";
        }
#endif

        state_ = PluginState::Running;
        std::cerr << "[WindowsSpeaker] Started\n";
        return true;
    }

    void stop() override {
        std::unique_lock<std::mutex> lock(mutex_);

        if (state_ != PluginState::Running) {
            return;
        }

#ifdef _WIN32
        // ============================================================
        // v2.2: Stop Render Thread (Event-Driven Pipeline)
        // ============================================================

        if (renderThreadRunning_.load(std::memory_order_acquire)) {
            std::cerr << "[WindowsSpeaker] Stopping render thread...\n";

            // Signal thread to stop
            renderThreadRunning_.store(false, std::memory_order_release);

            // Wake the thread if it's waiting on the condition variable
            renderCV_.notify_all();
        }

        // Temporarily release mutex for thread join to avoid deadlock
        lock.unlock();
        if (renderThread_.joinable()) {
            // v2.2: Use timed join to prevent hanging on unresponsive threads
            auto joinFuture = std::async(std::launch::async, [this]() {
                renderThread_.join();
            });

            if (joinFuture.wait_for(std::chrono::milliseconds(500)) == std::future_status::timeout) {
                std::cerr << "[WindowsSpeaker] Render thread join timeout (500ms) - detaching\n";
                renderThread_.detach();
            } else {
                std::cerr << "[WindowsSpeaker] Render thread joined\n";
            }
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
        std::cerr << "[WindowsSpeaker] Stopped\n";
    }

    PluginInfo getInfo() const override {
        return {
            "Windows Speaker (WASAPI)",
            "1.0.0",
            "Icing Project",
            "WASAPI playback to default Windows speaker output",
            PluginType::AudioSink,
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
            std::cerr << "[WindowsSpeaker] Volume set to " << volume_ << "\n";
        } else if (key == "mute") {
            mute_ = (value == "true" || value == "1");
            std::cerr << "[WindowsSpeaker] Mute set to " << (mute_ ? "true" : "false") << "\n";
        }
        // sampleRate, channels, bufferFrames can only be set before initialize()
    }

    std::string getParameter(const std::string& key) const override {
        std::lock_guard<std::mutex> lock(mutex_);

        if (key == "volume") return std::to_string(volume_);
        if (key == "mute") return mute_ ? "true" : "false";
        if (key == "sampleRate") return std::to_string(sampleRate_);
        if (key == "channels") return std::to_string(channels_);
        if (key == "bufferSize") return std::to_string(bufferFrames_);
        if (key == "framesWritten") return std::to_string(framesWritten_);
        if (key == "underruns") return std::to_string(underruns_);
        if (key == "overruns") return std::to_string(overruns_);
        if (key == "writeCalls") return std::to_string(writeCalls_);
        return "";
    }

    bool writeAudio(const AudioBuffer& buffer) override {
        // v2.2: Non-blocking write to ring buffer (NOT directly to WASAPI)
        // The background render thread handles WASAPI writes
        // This allows the pipeline to never block on sink backpressure

        std::lock_guard<std::mutex> lock(mutex_);

        if (state_ != PluginState::Running) {
            return false;
        }

        writeCalls_++;
        int frames = buffer.getFrameCount();
        int bufferChannels = buffer.getChannelCount();

        // Validate channel count
        if (bufferChannels != channels_) {
            std::cerr << "[WindowsSpeaker] Channel mismatch: expected "
                      << channels_ << ", got " << bufferChannels << "\n";
            return false;
        }

        // ============================================================
        // v2.2: Write to ring buffer (NON-BLOCKING)
        // ============================================================

        // Check available space in ring buffer
        int availableSpace = ringBuffer_.getAvailableWrite();

        if (availableSpace < frames) {
            // Ring buffer full - signal overrun but still try partial write
            overruns_++;

            // Log if this is becoming frequent
            if (overruns_ % 100 == 1) {
                std::cerr << "[WindowsSpeaker] Ring buffer backpressure: "
                          << availableSpace << " available, need " << frames << "\n";
            }

            // Return false to signal backpressure to pipeline
            // But pipeline won't block - it will retry next frame
            return false;
        }

        // Prepare channel pointers for planar format
        std::vector<const float*> channelPtrs(channels_);
        for (int ch = 0; ch < channels_; ++ch) {
            channelPtrs[ch] = buffer.getChannelData(ch);
        }

        // Write to ring buffer (lock-free, fast)
        int framesWritten = ringBuffer_.write(channelPtrs.data(), frames);

        if (framesWritten < frames) {
            // Partial write (shouldn't happen given check above, but be safe)
            overruns_++;
        }

        // Wake render thread if it might be sleeping
        renderCV_.notify_one();

        // Update metrics
        framesWritten_ += framesWritten;

        // ============================================================
        // Log progress every 100 calls (include ring buffer diagnostics)
        // ============================================================

        if (writeCalls_ % 100 == 0) {
            double secondsWritten = static_cast<double>(framesWritten_) / sampleRate_;
            int bufferFill = ringBuffer_.getAvailableRead();
            double fillMs = (bufferFill * 1000.0) / sampleRate_;

            std::cerr << "[WindowsSpeaker] Stats: " << framesWritten_
                      << " frames (" << secondsWritten << "s), "
                      << underruns_ << " underruns, " << overruns_ << " overruns, "
                      << "ring buffer: " << bufferFill << " frames ("
                      << fillMs << "ms fill)\n";
        }

        return framesWritten > 0;
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
        return bufferFrames_;
    }

    int getAvailableSpace() const override {
        // v2.2: Return ring buffer available space (not WASAPI buffer)
        // This allows pipeline to know if write will succeed without blocking

        std::lock_guard<std::mutex> lock(mutex_);

        if (state_ != PluginState::Running) {
            return 0;
        }

        return ringBuffer_.getAvailableWrite();
    }

    // ===== v2.2: Event-Driven Pipeline Support =====

    bool supportsAsyncMode() const override {
        // This plugin uses ring buffer + background thread, supports async mode
        return true;
    }

    void setSpaceAvailableCallback(SpaceAvailableCallback callback) override {
        std::lock_guard<std::mutex> lock(mutex_);
        spaceAvailableCallback_ = callback;

        if (callback) {
            std::cerr << "[WindowsSpeaker] v2.2: Space-available callback registered "
                      << "(threshold: " << spaceAvailableThreshold_ << " frames)\n";
        } else {
            std::cerr << "[WindowsSpeaker] v2.2: Space-available callback cleared\n";
        }
    }

    bool isNonBlocking() const override {
        // writeAudio() writes to ring buffer, never blocks
        return true;
    }

    // ===== End Event-Driven Pipeline Support =====

    void setSampleRate(int rate) override {
        std::lock_guard<std::mutex> lock(mutex_);

        if (state_ == PluginState::Unloaded) {
            if (rate > 0 && rate <= 192000) {
                sampleRate_ = rate;
                bufferFrames_ = (rate * 100) / 1000;  // Adjust buffer to maintain 100ms
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

    void setBufferSize(int frames) override {
        std::lock_guard<std::mutex> lock(mutex_);

        if (state_ == PluginState::Unloaded) {
            if (frames > 0 && frames <= 16384) {
                bufferFrames_ = frames;
            }
        }
    }

private:
    mutable std::mutex mutex_;
    PluginState state_;

    // Configuration
    int sampleRate_;
    int channels_;
    int bufferFrames_;
    float volume_;
    bool mute_;

    // Metrics
    uint64_t framesWritten_;
    uint64_t underruns_;
    uint64_t overruns_;
    uint64_t writeCalls_;

#ifdef _WIN32
    // WASAPI handles
    IMMDeviceEnumerator* deviceEnum_;
    IMMDevice* device_;
    IAudioClient* audioClient_;
    IAudioRenderClient* renderClient_;
#endif

    // COM state
    bool comInitialized_;
    bool comOwned_;

    // ===== v2.2: Event-Driven Pipeline / Ring Buffer =====

    // Lock-free ring buffer for async writes (pipeline → render thread)
    RingBuffer ringBuffer_;

    // Background render thread (reads ring buffer → writes WASAPI)
    std::thread renderThread_;
    std::atomic<bool> renderThreadRunning_;

    // Condition variable to wake render thread when data arrives
    std::condition_variable renderCV_;
    std::mutex renderMutex_;

    // Temporary buffer for WASAPI interleave conversion
    std::vector<std::vector<float>> tempBuffer_;

    // Callback for event-driven pipeline (space available notification)
    SpaceAvailableCallback spaceAvailableCallback_;
    int spaceAvailableThreshold_;

    // Background render thread function
    void renderThreadFunc() {
#ifdef _WIN32
        // Set thread priority to HIGH for reliable audio rendering
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);

        // Initialize COM on this thread (required for WASAPI)
        HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        bool comOwned = SUCCEEDED(hr);

        std::cerr << "[WindowsSpeaker] Render thread started with TIME_CRITICAL priority\n";

        std::vector<float*> readPtrs(channels_);
        for (int ch = 0; ch < channels_; ++ch) {
            readPtrs[ch] = tempBuffer_[ch].data();
        }

        uint64_t totalFramesRendered = 0;
        uint64_t loopCount = 0;

        // Target WASAPI buffer level: ~50ms (keeps WASAPI fed without over-filling)
        // This prevents the render thread from filling WASAPI completely and then
        // waiting for it to drain, which would cause ring buffer to fill up.
        const int targetBufferFrames = (sampleRate_ * 50) / 1000;  // 50ms = 2400 frames at 48kHz
        const int minBufferFrames = (sampleRate_ * 20) / 1000;     // 20ms minimum before writing

        std::cerr << "[WindowsSpeaker] Render thread target buffer: " << targetBufferFrames
                  << " frames (" << (targetBufferFrames * 1000 / sampleRate_) << "ms)\n";

        while (renderThreadRunning_.load(std::memory_order_acquire)) {
            loopCount++;

            // Query WASAPI for current buffer level
            UINT32 padding = 0;
            hr = audioClient_->GetCurrentPadding(&padding);
            if (FAILED(hr)) {
                std::cerr << "[WindowsSpeaker] GetCurrentPadding failed: 0x"
                          << std::hex << hr << std::dec << "\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            int currentPadding = static_cast<int>(padding);

            // Only write if WASAPI buffer is below target level
            if (currentPadding >= targetBufferFrames) {
                // WASAPI has enough data, sleep until it drains below target
                // Sleep for half the excess time to stay responsive
                int excessFrames = currentPadding - minBufferFrames;
                int sleepMs = std::max(1, (excessFrames * 1000) / (sampleRate_ * 2));
                sleepMs = std::min(sleepMs, 10);  // Cap at 10ms for responsiveness
                std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
                continue;
            }

            // Calculate how many frames to write to reach target level
            int framesToTarget = targetBufferFrames - currentPadding;

            // Check ring buffer availability
            int availableInRing = ringBuffer_.getAvailableRead();

            if (availableInRing == 0) {
                // Ring buffer empty - wait for data from pipeline
                std::unique_lock<std::mutex> lock(renderMutex_);
                renderCV_.wait_for(lock, std::chrono::milliseconds(5),
                    [this]() {
                        return ringBuffer_.getAvailableRead() > 0 ||
                               !renderThreadRunning_.load(std::memory_order_relaxed);
                    });
                continue;
            }

            // Write up to target level, limited by available data and temp buffer
            int framesToWrite = std::min({framesToTarget, availableInRing,
                                          static_cast<int>(tempBuffer_[0].size())});

            if (framesToWrite <= 0) {
                continue;
            }

            // Read from ring buffer (lock-free)
            int framesRead = ringBuffer_.read(readPtrs.data(), framesToWrite);
            if (framesRead == 0) {
                continue;
            }

            // Get WASAPI buffer and write interleaved data
            BYTE* pData = nullptr;
            hr = renderClient_->GetBuffer(framesRead, &pData);
            if (FAILED(hr)) {
                std::cerr << "[WindowsSpeaker] GetBuffer failed: 0x"
                          << std::hex << hr << std::dec << "\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            // Convert planar → interleaved float32
            float* output = reinterpret_cast<float*>(pData);
            for (int f = 0; f < framesRead; ++f) {
                for (int ch = 0; ch < channels_; ++ch) {
                    // Apply volume and mute
                    float sample = readPtrs[ch][f];
                    if (!mute_) {
                        sample *= volume_;
                    } else {
                        sample = 0.0f;
                    }
                    output[f * channels_ + ch] = sample;
                }
            }

            hr = renderClient_->ReleaseBuffer(framesRead, 0);
            if (FAILED(hr)) {
                std::cerr << "[WindowsSpeaker] ReleaseBuffer failed: 0x"
                          << std::hex << hr << std::dec << "\n";
            }

            totalFramesRendered += framesRead;

            // Notify pipeline that space is available (event-driven mode)
            int newAvailableWrite = ringBuffer_.getAvailableWrite();
            if (newAvailableWrite >= spaceAvailableThreshold_ && spaceAvailableCallback_) {
                spaceAvailableCallback_();
            }

            // Log progress periodically
            if (loopCount % 1000 == 0) {
                int currentFill = ringBuffer_.getAvailableRead();
                double fillMs = (currentFill * 1000.0) / sampleRate_;
                std::cerr << "[WindowsSpeaker] Render: " << totalFramesRendered
                          << " frames, WASAPI padding: " << currentPadding
                          << ", ring fill: " << currentFill << " ("
                          << fillMs << "ms)\n";
            }
        }

        // Cleanup COM on this thread
        if (comOwned) {
            CoUninitialize();
        }

        std::cerr << "[WindowsSpeaker] Render thread exiting (rendered "
                  << totalFramesRendered << " frames)\n";
#endif
    }

    // ===== End Event-Driven Pipeline =====
};

} // namespace nda

// Export the plugin
NDA_DECLARE_PLUGIN(nda::WindowsSpeakerSinkPlugin)
