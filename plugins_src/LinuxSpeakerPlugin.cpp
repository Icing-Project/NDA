#include "plugins/AudioSinkPlugin.h"
#include <pulse/simple.h>
#include <pulse/error.h>
#include <vector>

namespace nda {

class LinuxSpeakerPlugin : public AudioSinkPlugin {
public:
    LinuxSpeakerPlugin()
        : state_(PluginState::Unloaded),
          sampleRate_(48000),
          channels_(2),
          bufferSize_(512),
          pa_s_(nullptr)
    {
    }

    ~LinuxSpeakerPlugin() override {
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
            "NDA",                     // Application name
            PA_STREAM_PLAYBACK,         // Direction: playback
            nullptr,                    // Use default device
            "Audio Sink",               // Stream description
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
            pa_simple_drain(pa_s_, nullptr);
            pa_simple_free(pa_s_);
            pa_s_ = nullptr;
        }
        if (state_ == PluginState::Running) {
            state_ = PluginState::Initialized;
        }
    }

    PluginInfo getInfo() const override {
        return {
            "Linux Speakers (PulseAudio)",
            "1.0.0",
            "Icing Project",
            "Outputs audio to default speakers using PulseAudio",
            PluginType::AudioSink,
            NDA_PLUGIN_API_VERSION
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
        if (key == "bufferSize") return std::to_string(bufferSize_);
        return "";
    }

    bool writeAudio(const AudioBuffer& buffer) override {
        if (!pa_s_ || state_ != PluginState::Running) {
            return false;
        }

        // Interleave buffer for PulseAudio
        int frameCount = buffer.getFrameCount();
        int totalSamples = frameCount * channels_;
        std::vector<float> tempBuffer(totalSamples);

        for (int frame = 0; frame < frameCount; ++frame) {
            for (int ch = 0; ch < channels_; ++ch) {
                const float* channelData = buffer.getChannelData(ch);
                tempBuffer[frame * channels_ + ch] = channelData[frame];
            }
        }

        // Write to PulseAudio
        int error;
        if (pa_simple_write(pa_s_, tempBuffer.data(),
                           totalSamples * sizeof(float), &error) < 0) {
            return false;
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

    int getBufferSize() const override { return bufferSize_; }
    void setBufferSize(int samples) override { bufferSize_ = samples; }
    int getAvailableSpace() const override { return bufferSize_; }

private:
    PluginState state_;
    int sampleRate_;
    int channels_;
    int bufferSize_;
    pa_simple* pa_s_;
};

} // namespace nda

// Export the plugin
NDA_DECLARE_PLUGIN(nda::LinuxSpeakerPlugin)
