#include "plugins/AudioSinkPlugin.h"
#include <iostream>
#include <iomanip>
#include <cmath>

namespace NADE {

class NullSinkPlugin : public AudioSinkPlugin {
public:
    NullSinkPlugin()
        : state_(PluginState::Unloaded),
          sampleRate_(48000),
          channels_(2),
          bufferSize_(512),
          framesProcessed_(0),
          showMetrics_(true)
    {
    }

    ~NullSinkPlugin() override {
        if (state_ != PluginState::Unloaded) {
            shutdown();
        }
    }

    bool initialize() override {
        state_ = PluginState::Initialized;
        framesProcessed_ = 0;
        return true;
    }

    void shutdown() override {
        state_ = PluginState::Unloaded;
    }

    bool start() override {
        if (state_ != PluginState::Initialized) return false;
        state_ = PluginState::Running;
        framesProcessed_ = 0;
        std::cout << "[NullSink] Started - consuming audio data" << std::endl;
        return true;
    }

    void stop() override {
        if (state_ == PluginState::Running) {
            std::cout << "[NullSink] Stopped - processed " << framesProcessed_
                     << " frames (" << (double)framesProcessed_ / sampleRate_
                     << " seconds)" << std::endl;
            state_ = PluginState::Initialized;
        }
    }

    PluginInfo getInfo() const override {
        return {
            "Null Sink (Console Monitor)",
            "1.0.0",
            "NADE Team",
            "Discards audio but shows metrics in console",
            PluginType::AudioSink,
            NADE_PLUGIN_API_VERSION
        };
    }

    PluginState getState() const override {
        return state_;
    }

    void setParameter(const std::string& key, const std::string& value) override {
        if (key == "showMetrics") {
            showMetrics_ = (value == "true" || value == "1");
        }
    }

    std::string getParameter(const std::string& key) const override {
        if (key == "sampleRate") return std::to_string(sampleRate_);
        if (key == "channels") return std::to_string(channels_);
        if (key == "bufferSize") return std::to_string(bufferSize_);
        if (key == "showMetrics") return showMetrics_ ? "true" : "false";
        return "";
    }

    bool writeAudio(const AudioBuffer& buffer) override {
        if (state_ != PluginState::Running) {
            return false;
        }

        int frameCount = buffer.getFrameCount();
        framesProcessed_ += frameCount;

        // Calculate RMS level for monitoring
        if (showMetrics_ && framesProcessed_ % (sampleRate_ / 10) == 0) {
            float rmsL = 0.0f;
            float rmsR = 0.0f;

            const float* leftChannel = buffer.getChannelData(0);
            const float* rightChannel = channels_ > 1 ? buffer.getChannelData(1) : leftChannel;

            for (int i = 0; i < frameCount; ++i) {
                rmsL += leftChannel[i] * leftChannel[i];
                rmsR += rightChannel[i] * rightChannel[i];
            }

            rmsL = std::sqrt(rmsL / frameCount);
            rmsR = std::sqrt(rmsR / frameCount);

            float dbL = 20.0f * std::log10(rmsL + 1e-10f);
            float dbR = 20.0f * std::log10(rmsR + 1e-10f);

            std::cout << "[NullSink] " << std::fixed << std::setprecision(1)
                     << "L: " << std::setw(6) << dbL << " dB  "
                     << "R: " << std::setw(6) << dbR << " dB  "
                     << "(" << framesProcessed_ / sampleRate_ << "s)"
                     << std::endl;
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
    uint64_t framesProcessed_;
    bool showMetrics_;
};

} // namespace NADE

// Export the plugin
NADE_DECLARE_PLUGIN(NADE::NullSinkPlugin)
