#include "plugins/AudioSinkPlugin.h"
#include <algorithm>
#include <iostream>
#include <mutex>

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
          comOwned_(false)
    {
    }

    ~WindowsSpeakerSinkPlugin() override {
        std::lock_guard<std::mutex> lock(mutex_);
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

        // 6. Try to request our preferred format (48kHz, stereo, float32)
        //    But keep original format as backup
        WAVEFORMATEX requestedFormat = *mixFormat;
        requestedFormat.nSamplesPerSec = sampleRate_;
        requestedFormat.nChannels = channels_;
        requestedFormat.wBitsPerSample = 32;  // float32
        requestedFormat.nBlockAlign = channels_ * 4;
        requestedFormat.nAvgBytesPerSec = sampleRate_ * channels_ * 4;

        // For WAVEFORMATEXTENSIBLE (most modern devices)
        bool useExtended = (mixFormat->wFormatTag == WAVE_FORMAT_EXTENSIBLE);
        if (useExtended) {
            WAVEFORMATEXTENSIBLE* formatEx = reinterpret_cast<WAVEFORMATEXTENSIBLE*>(mixFormat);
            formatEx->SubFormat = KSDATAFORMAT_SUBTYPE_IEEE_FLOAT;
        } else {
            requestedFormat.wFormatTag = WAVE_FORMAT_IEEE_FLOAT;
        }

        // 7. Calculate buffer duration
        //    200ms = more stable buffer for polling mode (reduces backpressure)
        //    In shared mode, larger buffers prevent constant overruns
        bufferFrames_ = (sampleRate_ * 200) / 1000;  // 200ms at sample rate
        REFERENCE_TIME bufferDuration = (REFERENCE_TIME)((10000000.0 * bufferFrames_) / sampleRate_);

        // 8. Initialize audio client in SHARED mode with POLLING (not event-driven)
        //    Shared mode: multiple apps can use device, Windows handles mixing
        //    No event callback: simpler polling mode, more reliable for our sync API
        hr = audioClient_->Initialize(
            AUDCLNT_SHAREMODE_SHARED,           // Shared mode (not exclusive)
            0,                                   // No special flags - polling mode
            bufferDuration,                      // Buffer size (200ms)
            0,                                   // Must be 0 in shared mode
            useExtended ? mixFormat : &requestedFormat,
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
        std::lock_guard<std::mutex> lock(mutex_);

        if (state_ == PluginState::Running) {
            stop();
        }

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
        if (audioClient_ && renderClient_) {
            // CRITICAL: Prime the buffer with silence before Start()
            // Microsoft docs: "ensures there is always data in the pipeline so the engine doesn't glitch"
            // Reference: https://github.com/microsoft/Windows-classic-samples/.../RenderSharedTimerDriven
            UINT32 bufferFrameCount = 0;
            HRESULT hr = audioClient_->GetBufferSize(&bufferFrameCount);
            if (SUCCEEDED(hr)) {
                // Pre-roll the entire buffer with silence
                BYTE* pData = nullptr;
                hr = renderClient_->GetBuffer(bufferFrameCount, &pData);
                if (SUCCEEDED(hr)) {
                    // Release with SILENT flag - fills buffer with zeros
                    hr = renderClient_->ReleaseBuffer(bufferFrameCount, AUDCLNT_BUFFERFLAGS_SILENT);
                    if (SUCCEEDED(hr)) {
                        std::cerr << "[WindowsSpeaker] Buffer primed with " << bufferFrameCount
                                  << " frames of silence\n";
                    } else {
                        std::cerr << "[WindowsSpeaker] ReleaseBuffer (prime) failed: 0x"
                                  << std::hex << hr << std::dec << "\n";
                    }
                } else {
                    std::cerr << "[WindowsSpeaker] GetBuffer (prime) failed: 0x"
                              << std::hex << hr << std::dec << "\n";
                }
            }

            // Now start the audio engine
            hr = audioClient_->Start();
            if (FAILED(hr)) {
                std::cerr << "[WindowsSpeaker] Start failed: 0x" << std::hex << hr << std::dec << "\n";
                return false;
            }
        }
#endif

        state_ = PluginState::Running;
        std::cerr << "[WindowsSpeaker] Started\n";
        return true;
    }

    void stop() override {
        std::lock_guard<std::mutex> lock(mutex_);

        if (state_ != PluginState::Running) {
            return;
        }

#ifdef _WIN32
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
        std::lock_guard<std::mutex> lock(mutex_);

        if (state_ != PluginState::Running) {
            return false;
        }

#ifdef _WIN32
        writeCalls_++;
        int frames = buffer.getFrameCount();
        int bufferChannels = buffer.getChannelCount();

        // Validate channel count
        if (bufferChannels != channels_) {
            std::cerr << "[WindowsSpeaker] Channel mismatch: expected "
                      << channels_ << ", got " << bufferChannels << "\n";
            return false;
        }

        // Check available space
        UINT32 padding = 0;
        UINT32 deviceBufferSize = 0;

        if (FAILED(audioClient_->GetCurrentPadding(&padding)) ||
            FAILED(audioClient_->GetBufferSize(&deviceBufferSize))) {
            std::cerr << "[WindowsSpeaker] Buffer query failed\n";
            overruns_++;
            return false;
        }

        UINT32 availableFrames = deviceBufferSize - padding;

        if (availableFrames < (UINT32)frames) {
            // Buffer full - backpressure signal
            overruns_++;
            return false;
        }

        // Get buffer from WASAPI
        BYTE* deviceBuffer = nullptr;
        HRESULT hr = renderClient_->GetBuffer(frames, &deviceBuffer);
        if (FAILED(hr)) {
            std::cerr << "[WindowsSpeaker] GetBuffer failed: 0x" << std::hex << hr << std::dec << "\n";
            underruns_++;
            return false;
        }

        // Convert planar → interleaved with volume/mute
        float* dest = reinterpret_cast<float*>(deviceBuffer);

        for (int frame = 0; frame < frames; ++frame) {
            for (int ch = 0; ch < channels_; ++ch) {
                float sample = buffer.getChannelData(ch)[frame];

                // Apply volume and mute
                if (mute_) {
                    sample = 0.0f;
                } else {
                    sample *= volume_;
                }

                // Clamp to prevent distortion
                sample = std::clamp(sample, -1.0f, 1.0f);

                *dest++ = sample;
            }
        }

        // Release buffer back to WASAPI
        hr = renderClient_->ReleaseBuffer(frames, 0);
        if (FAILED(hr)) {
            std::cerr << "[WindowsSpeaker] ReleaseBuffer failed: 0x" << std::hex << hr << std::dec << "\n";
            return false;
        }

        framesWritten_ += frames;

        // Log progress every 100 calls
        if (writeCalls_ % 100 == 0) {
            double secondsWritten = static_cast<double>(framesWritten_) / sampleRate_;
            std::cerr << "[WindowsSpeaker] Stats: " << framesWritten_
                      << " frames (" << secondsWritten << "s), "
                      << underruns_ << " underruns, " << overruns_ << " overruns\n";
        }

        return true;
#else
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
        return bufferFrames_;
    }

    int getAvailableSpace() const override {
        std::lock_guard<std::mutex> lock(mutex_);

#ifdef _WIN32
        if (state_ != PluginState::Running || !audioClient_) {
            return 0;
        }

        UINT32 padding = 0;
        UINT32 deviceBufferSize = 0;

        if (FAILED(audioClient_->GetCurrentPadding(&padding)) ||
            FAILED(audioClient_->GetBufferSize(&deviceBufferSize))) {
            return 0;
        }

        return static_cast<int>(deviceBufferSize - padding);
#else
        return 0;
#endif
    }

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
};

} // namespace nda

// Export the plugin
NDA_DECLARE_PLUGIN(nda::WindowsSpeakerSinkPlugin)
