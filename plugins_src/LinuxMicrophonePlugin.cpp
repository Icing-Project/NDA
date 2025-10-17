#include "plugins/AudioSourcePlugin.h"
#include <pulse/simple.h>
#include <pulse/error.h>
#include <cstring>

namespace NADE {

class LinuxMicrophonePlugin : public AudioSourcePlugin {
public:
    LinuxMicrophonePlugin()
        : state_(PluginState::Unloaded),
          sampleRate_(48000),
          channels_(2),
          pa_s_(nullptr)
    {
    }

    ~LinuxMicrophonePlugin() override {
        if (state_ != PluginState::Unloaded) {
            shutdown();
        }
    }

    bool initialize() override {
        state_ = PluginState::Initialized;
        return true;
    }

    void shutdown() override {
        stop();
        state_ = PluginState::Unloaded;
    }

    bool start() override {
        if (state_ != PluginState::Initialized) return false;

        // PulseAudio sample specification
        pa_sample_spec ss;
        ss.format = PA_SAMPLE_FLOAT32LE;
        ss.rate = sampleRate_;
        ss.channels = channels_;

        // Create PulseAudio stream
        int error;
        pa_s_ = pa_simple_new(
            nullptr,                    // Use default server
            "NADE",                     // Application name
            PA_STREAM_RECORD,           // Direction: record
            nullptr,                    // Use default device
            "Audio Source",             // Stream description
            &ss,                        // Sample format
            nullptr,                    // Use default channel map
            nullptr,                    // Use default buffering
            &error
        );

        if (!pa_s_) {
            return false;
        }

        state_ = PluginState::Running;
        return true;
    }

    void stop() override {
        if (pa_s_) {
            pa_simple_free(pa_s_);
            pa_s_ = nullptr;
        }
        if (state_ == PluginState::Running) {
            state_ = PluginState::Initialized;
        }
    }

    PluginInfo getInfo() const override {
        return {
            "Linux Microphone (PulseAudio)",
            "1.0.0",
            "NADE Team",
            "Captures audio from default microphone using PulseAudio",
            PluginType::AudioSource,
            NADE_PLUGIN_API_VERSION
        };
    }

    PluginState getState() const override {
        return state_;
    }

    void setParameter(const std::string& key, const std::string& value) override {
        // Could handle device selection here
    }

    std::string getParameter(const std::string& key) const override {
        if (key == "sampleRate") return std::to_string(sampleRate_);
        if (key == "channels") return std::to_string(channels_);
        return "";
    }

    void setAudioCallback(AudioSourceCallback callback) override {
        callback_ = callback;
    }

    bool readAudio(AudioBuffer& buffer) override {
        if (!pa_s_ || state_ != PluginState::Running) {
            buffer.clear();
            return false;
        }

        // Read audio from PulseAudio
        int frameCount = buffer.getFrameCount();
        int totalSamples = frameCount * channels_;
        std::vector<float> tempBuffer(totalSamples);

        int error;
        if (pa_simple_read(pa_s_, tempBuffer.data(),
                          totalSamples * sizeof(float), &error) < 0) {
            return false;
        }

        // Deinterleave into AudioBuffer
        for (int ch = 0; ch < channels_; ++ch) {
            float* channelData = buffer.getChannelData(ch);
            for (int frame = 0; frame < frameCount; ++frame) {
                channelData[frame] = tempBuffer[frame * channels_ + ch];
            }
        }

        return true;
    }

    int getSampleRate() const override { return sampleRate_; }
    int getChannels() const override { return channels_; }

    void setSampleRate(int sampleRate) override {
        if (state_ == PluginState::Unloaded || state_ == PluginState::Initialized) {
            sampleRate_ = sampleRate;
        }
    }

    void setChannels(int channels) override {
        if (state_ == PluginState::Unloaded || state_ == PluginState::Initialized) {
            channels_ = channels;
        }
    }

private:
    PluginState state_;
    int sampleRate_;
    int channels_;
    AudioSourceCallback callback_;
    pa_simple* pa_s_;
};

} // namespace NADE

// Export the plugin
NADE_DECLARE_PLUGIN(NADE::LinuxMicrophonePlugin)
