#include "plugins/AudioSourcePlugin.h"
#include <cmath>

namespace NADE {

class SineWaveSourcePlugin : public AudioSourcePlugin {
public:
    SineWaveSourcePlugin()
        : state_(PluginState::Unloaded),
          sampleRate_(48000),
          channels_(2),
          frequency_(440.0), // A4 note
          phase_(0.0)
    {
    }

    ~SineWaveSourcePlugin() override {
        if (state_ != PluginState::Unloaded) {
            shutdown();
        }
    }

    bool initialize() override {
        state_ = PluginState::Initialized;
        return true;
    }

    void shutdown() override {
        state_ = PluginState::Unloaded;
    }

    bool start() override {
        if (state_ != PluginState::Initialized) return false;
        state_ = PluginState::Running;
        phase_ = 0.0;
        return true;
    }

    void stop() override {
        if (state_ == PluginState::Running) {
            state_ = PluginState::Initialized;
        }
    }

    PluginInfo getInfo() const override {
        return {
            "Sine Wave Generator",
            "1.0.0",
            "NADE Team",
            "Generates a 440Hz sine wave (A4 note) for testing",
            PluginType::AudioSource,
            NADE_PLUGIN_API_VERSION
        };
    }

    PluginState getState() const override {
        return state_;
    }

    void setParameter(const std::string& key, const std::string& value) override {
        if (key == "frequency") {
            frequency_ = std::stod(value);
        }
    }

    std::string getParameter(const std::string& key) const override {
        if (key == "frequency") return std::to_string(frequency_);
        if (key == "sampleRate") return std::to_string(sampleRate_);
        if (key == "channels") return std::to_string(channels_);
        return "";
    }

    void setAudioCallback(AudioSourceCallback callback) override {
        callback_ = callback;
    }

    bool readAudio(AudioBuffer& buffer) override {
        if (state_ != PluginState::Running) {
            buffer.clear();
            return false;
        }

        int frameCount = buffer.getFrameCount();
        double phaseIncrement = 2.0 * M_PI * frequency_ / sampleRate_;

        // Generate sine wave
        for (int frame = 0; frame < frameCount; ++frame) {
            float sample = 0.5f * std::sin(phase_);

            // Write to all channels
            for (int ch = 0; ch < channels_; ++ch) {
                float* channelData = buffer.getChannelData(ch);
                channelData[frame] = sample;
            }

            phase_ += phaseIncrement;
            if (phase_ >= 2.0 * M_PI) {
                phase_ -= 2.0 * M_PI;
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
    double frequency_;
    double phase_;
    AudioSourceCallback callback_;
};

} // namespace NADE

// Export the plugin
NADE_DECLARE_PLUGIN(NADE::SineWaveSourcePlugin)
