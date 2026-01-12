#include "plugins/AudioSourcePlugin.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <mutex>

namespace nda {

class SineWaveSourcePlugin : public AudioSourcePlugin {
public:
    SineWaveSourcePlugin()
        : state_(PluginState::Unloaded),
          sampleRate_(48000),
          channels_(2),
          bufferSize_(512),
          frequency_(440.0),        // A4 note
          amplitude_(0.5f),         // Default amplitude (prevents clipping)
          phase_(0.0),
          framesGenerated_(0),
          readCalls_(0)
    {
    }

    ~SineWaveSourcePlugin() override {
        // v2.2: Don't lock here - shutdown() handles its own locking
        // Locking here + shutdown() locking = undefined behavior (recursive lock)
        if (state_ != PluginState::Unloaded) {
            shutdown();
        }
    }

    bool initialize() override {
        std::lock_guard<std::mutex> lock(mutex_);

        if (state_ != PluginState::Unloaded) {
            std::cerr << "[SineWaveSource] Initialize failed: already initialized\n";
            return false;
        }

        // Reset counters
        framesGenerated_ = 0;
        readCalls_ = 0;
        phase_ = 0.0;

        state_ = PluginState::Initialized;
        std::cerr << "[SineWaveSource] Initialized: " << sampleRate_ << "Hz, "
                  << channels_ << "ch, " << bufferSize_ << " frames, "
                  << frequency_ << "Hz tone, amplitude=" << amplitude_ << "\n";
        return true;
    }

    void shutdown() override {
        // v2.2: Always call stop() BEFORE locking to avoid recursive mutex deadlock
        // stop() has its own locking and state check - safe to call unconditionally
        stop();

        std::lock_guard<std::mutex> lock(mutex_);

        std::cerr << "[SineWaveSource] Shutdown: Generated " << framesGenerated_
                  << " frames across " << readCalls_ << " calls\n";

        state_ = PluginState::Unloaded;
    }

    bool start() override {
        std::lock_guard<std::mutex> lock(mutex_);

        if (state_ != PluginState::Initialized) {
            std::cerr << "[SineWaveSource] Start failed: not initialized (state="
                      << static_cast<int>(state_) << ")\n";
            return false;
        }

        phase_ = 0.0;  // Reset phase on start for clean tone
        state_ = PluginState::Running;
        std::cerr << "[SineWaveSource] Started\n";
        return true;
    }

    void stop() override {
        std::lock_guard<std::mutex> lock(mutex_);

        if (state_ == PluginState::Running) {
            state_ = PluginState::Initialized;
            std::cerr << "[SineWaveSource] Stopped\n";
        }
    }

    PluginInfo getInfo() const override {
        return {
            "Sine Wave Generator (Enhanced)",
            "2.0.0",
            "Icing Project",
            "Generates configurable sine wave test tones (440Hz default)",
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

        if (key == "frequency") {
            double newFreq = std::stod(value);
            if (newFreq > 0.0 && newFreq <= 20000.0) {
                frequency_ = newFreq;
                std::cerr << "[SineWaveSource] Frequency set to " << frequency_ << "Hz\n";
            } else {
                std::cerr << "[SineWaveSource] Invalid frequency: " << newFreq
                          << "Hz (must be 0-20000Hz)\n";
            }
        } else if (key == "amplitude") {
            float newAmp = std::stof(value);
            amplitude_ = std::clamp(newAmp, 0.0f, 1.0f);
            std::cerr << "[SineWaveSource] Amplitude set to " << amplitude_ << "\n";
        }
    }

    std::string getParameter(const std::string& key) const override {
        std::lock_guard<std::mutex> lock(mutex_);

        if (key == "frequency") return std::to_string(frequency_);
        if (key == "amplitude") return std::to_string(amplitude_);
        if (key == "sampleRate") return std::to_string(sampleRate_);
        if (key == "channels") return std::to_string(channels_);
        if (key == "bufferSize") return std::to_string(bufferSize_);
        if (key == "framesGenerated") return std::to_string(framesGenerated_);
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

        readCalls_++;
        int frameCount = buffer.getFrameCount();
        int bufferChannels = buffer.getChannelCount();

        // Verify buffer has correct channel count
        if (bufferChannels != channels_) {
            std::cerr << "[SineWaveSource] Channel mismatch: expected "
                      << channels_ << ", got " << bufferChannels << "\n";
            buffer.clear();
            return false;
        }

        double phaseIncrement = 2.0 * M_PI * frequency_ / sampleRate_;

        // Generate sine wave with configurable amplitude
        for (int frame = 0; frame < frameCount; ++frame) {
            float sample = amplitude_ * static_cast<float>(std::sin(phase_));

            // Write to all channels (mono test tone across all channels)
            for (int ch = 0; ch < channels_; ++ch) {
                float* channelData = buffer.getChannelData(ch);
                channelData[frame] = sample;
            }

            // Advance phase and wrap to prevent overflow
            phase_ += phaseIncrement;
            if (phase_ >= 2.0 * M_PI) {
                phase_ -= 2.0 * M_PI;
            }
        }

        framesGenerated_ += frameCount;

        // Log progress every 100 calls for diagnostics (not too spammy)
        if (readCalls_ % 100 == 0) {
            double secondsGenerated = static_cast<double>(framesGenerated_) / sampleRate_;
            std::cerr << "[SineWaveSource] Stats: " << framesGenerated_
                      << " frames (" << std::fixed << std::setprecision(1) << secondsGenerated
                      << "s), " << readCalls_ << " calls\n";
        }

        return true;
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

        if (state_ == PluginState::Unloaded || state_ == PluginState::Initialized) {
            if (frames > 0 && frames <= 8192) {
                bufferSize_ = frames;
                std::cerr << "[SineWaveSource] Buffer size set to " << bufferSize_ << " frames\n";
            } else {
                std::cerr << "[SineWaveSource] Invalid buffer size: " << frames
                          << " (must be 1-8192)\n";
            }
        } else {
            std::cerr << "[SineWaveSource] Cannot change buffer size while running\n";
        }
    }

    void setSampleRate(int sampleRate) override {
        std::lock_guard<std::mutex> lock(mutex_);

        if (state_ == PluginState::Unloaded || state_ == PluginState::Initialized) {
            if (sampleRate > 0 && sampleRate <= 192000) {
                sampleRate_ = sampleRate;
                std::cerr << "[SineWaveSource] Sample rate set to " << sampleRate_ << "Hz\n";
            } else {
                std::cerr << "[SineWaveSource] Invalid sample rate: " << sampleRate
                          << "Hz (must be 1-192000Hz)\n";
            }
        } else {
            std::cerr << "[SineWaveSource] Cannot change sample rate while running\n";
        }
    }

    void setChannels(int channels) override {
        std::lock_guard<std::mutex> lock(mutex_);

        if (state_ == PluginState::Unloaded || state_ == PluginState::Initialized) {
            if (channels > 0 && channels <= 8) {
                channels_ = channels;
                std::cerr << "[SineWaveSource] Channels set to " << channels_ << "\n";
            } else {
                std::cerr << "[SineWaveSource] Invalid channel count: " << channels
                          << " (must be 1-8)\n";
            }
        } else {
            std::cerr << "[SineWaveSource] Cannot change channels while running\n";
        }
    }

private:
    mutable std::mutex mutex_;      // Protects all member variables
    PluginState state_;
    int sampleRate_;
    int channels_;
    int bufferSize_;
    double frequency_;
    float amplitude_;
    double phase_;
    uint64_t framesGenerated_;      // Total frames generated (diagnostics)
    uint64_t readCalls_;             // Total readAudio() calls (diagnostics)
    AudioSourceCallback callback_;
};

} // namespace nda

// Export the plugin
NDA_DECLARE_PLUGIN(nda::SineWaveSourcePlugin)
