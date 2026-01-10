#include "plugins/AudioSourcePlugin.h"
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
          captureEvent_(nullptr),
#endif
          comInitialized_(false),
          comOwned_(false)
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
        //    100ms = stable for most systems (balance between latency and stability)
        int bufferFrames = (sampleRate_ * 100) / 1000;  // 100ms
        REFERENCE_TIME bufferDuration = (REFERENCE_TIME)((10000000.0 * bufferFrames) / sampleRate_);

        // 8. Initialize audio client in SHARED mode with EVENT-DRIVEN callback
        //    Shared mode: multiple apps can use device, Windows handles mixing
        //    Event-driven: more efficient than polling
        hr = audioClient_->Initialize(
            AUDCLNT_SHAREMODE_SHARED,           // Shared mode (not exclusive)
            AUDCLNT_STREAMFLAGS_EVENTCALLBACK,  // Event-driven (not polling)
            bufferDuration,                      // Buffer size
            0,                                   // Must be 0 in shared mode
            useExtended ? mixFormat : &requestedFormat,
            nullptr                              // No session GUID
        );

        if (FAILED(hr)) {
            // Fallback: Use device's original mix format (always works)
            std::cerr << "[WindowsMicrophone] Custom format rejected, using device format\n";

            // Update our config to match device
            sampleRate_ = mixFormat->nSamplesPerSec;
            channels_ = mixFormat->nChannels;

            // Recalculate buffer duration for device rate
            bufferFrames = (sampleRate_ * 100) / 1000;  // 100ms at device rate
            bufferDuration = (REFERENCE_TIME)((10000000.0 * bufferFrames) / sampleRate_);

            hr = audioClient_->Initialize(
                AUDCLNT_SHAREMODE_SHARED,
                AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
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

        // 9. Create event for buffer readiness notification
        captureEvent_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (!captureEvent_) {
            std::cerr << "[WindowsMicrophone] Event creation failed\n";
            return false;
        }

        hr = audioClient_->SetEventHandle(captureEvent_);
        if (FAILED(hr)) {
            std::cerr << "[WindowsMicrophone] SetEventHandle failed: 0x" << std::hex << hr << std::dec << "\n";
            CloseHandle(captureEvent_);
            captureEvent_ = nullptr;
            return false;
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

        if (captureEvent_) {
            CloseHandle(captureEvent_);
            captureEvent_ = nullptr;
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
#endif

        state_ = PluginState::Running;
        std::cerr << "[WindowsMicrophone] Started\n";
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

        // Get next packet from WASAPI
        UINT32 packetFrames = 0;
        HRESULT hr = captureClient_->GetNextPacketSize(&packetFrames);
        if (FAILED(hr)) {
            std::cerr << "[WindowsMicrophone] GetNextPacketSize failed: 0x" << std::hex << hr << std::dec << "\n";
            buffer.clear();
            underruns_++;
            return false;
        }

        if (packetFrames == 0) {
            // No data available - return silence
            buffer.clear();
            underruns_++;
            return false;
        }

        // Get buffer from WASAPI
        BYTE* captureData = nullptr;
        DWORD flags = 0;
        hr = captureClient_->GetBuffer(&captureData, &packetFrames, &flags, nullptr, nullptr);
        if (FAILED(hr)) {
            std::cerr << "[WindowsMicrophone] GetBuffer failed: 0x" << std::hex << hr << std::dec << "\n";
            buffer.clear();
            underruns_++;
            return false;
        }

        // Handle flags (silence/discontinuity)
        if (flags & AUDCLNT_BUFFERFLAGS_SILENT) {
            buffer.clear();
        } else {
            // Convert interleaved → planar with volume/mute
            float* src = reinterpret_cast<float*>(captureData);
            int framesToCopy = std::min((int)packetFrames, requestedFrames);

            for (int frame = 0; frame < framesToCopy; ++frame) {
                for (int ch = 0; ch < channels_; ++ch) {
                    float sample = *src++;

                    // Apply volume and mute
                    if (mute_) {
                        sample = 0.0f;
                    } else {
                        sample *= volume_;
                    }

                    buffer.getChannelData(ch)[frame] = sample;
                }
            }

            // Pad with silence if needed
            if (framesToCopy < requestedFrames) {
                for (int ch = 0; ch < bufferChannels; ++ch) {
                    float* channelData = buffer.getChannelData(ch);
                    for (int frame = framesToCopy; frame < requestedFrames; ++frame) {
                        channelData[frame] = 0.0f;
                    }
                }
            }
        }

        // Release buffer back to WASAPI
        captureClient_->ReleaseBuffer(packetFrames);

        framesCaptured_ += packetFrames;

        // Log progress every 100 calls
        if (readCalls_ % 100 == 0) {
            double secondsCaptured = static_cast<double>(framesCaptured_) / sampleRate_;
            std::cerr << "[WindowsMicrophone] Stats: " << framesCaptured_
                      << " frames (" << secondsCaptured << "s), "
                      << underruns_ << " underruns\n";
        }

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
    HANDLE captureEvent_;
#endif

    // COM state
    bool comInitialized_;
    bool comOwned_;

    // Callback (not used in pull model, but required by interface)
    AudioSourceCallback callback_;
};

} // namespace nda

// Export the plugin
NDA_DECLARE_PLUGIN(nda::WindowsMicrophoneSourcePlugin)
