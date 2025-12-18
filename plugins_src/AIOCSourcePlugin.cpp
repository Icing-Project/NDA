#include "plugins/AudioSourcePlugin.h"
#include "AIOCPluginCommon.h"
#include <iostream>

namespace nda {

class AIOCSourcePlugin : public AudioSourcePlugin {
public:
    AIOCSourcePlugin()
        : state_(PluginState::Unloaded),
          sampleRate_(48000),
          channels_(1),
          bufferFrames_(512),
          loopbackTest_(false)
    {
        // Lock to AIOC hardware endpoints by default (per-device IDs provided).
        session_.setDeviceIds(
            "{0D94B72A-8A15-4C85-B8F7-5AC442A88BFB}", // Microphone (AIOC Audio)
            session_.deviceOutId());
        session_.setSampleRate(sampleRate_);
        session_.setChannels(channels_);
        session_.setBufferFrames(bufferFrames_);
    }

    ~AIOCSourcePlugin() override {
        if (state_ != PluginState::Unloaded) {
            shutdown();
        }
    }

    bool initialize() override {
        if (state_ != PluginState::Unloaded && state_ != PluginState::Initialized) {
            return false;
        }
        session_.enableLoopback(loopbackTest_);
        session_.setSampleRate(sampleRate_);
        session_.setChannels(channels_);
        session_.setBufferFrames(bufferFrames_);
        state_ = PluginState::Initialized;
        return true;
    }

    void shutdown() override {
        stop();
        session_.disconnect();
        state_ = PluginState::Unloaded;
    }

    bool start() override {
        if (state_ != PluginState::Initialized) return false;
        if (!session_.isConnected() && !session_.connect()) {
            state_ = PluginState::Error;
            std::cerr << "[AIOCSource] Failed to connect to AIOC device: "
                      << session_.getTelemetry().lastMessage << std::endl;
            return false;
        }
        if (!session_.start()) {
            state_ = PluginState::Error;
            return false;
        }
        state_ = PluginState::Running;
        return true;
    }

    void stop() override {
        if (state_ == PluginState::Running) {
            session_.stop();
            state_ = PluginState::Initialized;
        }
    }

    PluginInfo getInfo() const override {
        return {
            "AIOC Source",
            "0.1.0",
            "Icing Project",
            "Reads audio from AIOC (USB mic) and exposes COS/VOX telemetry",
            PluginType::AudioSource,
            NDA_PLUGIN_API_VERSION
        };
    }

    PluginState getState() const override {
        return state_;
    }

    void setParameter(const std::string& key, const std::string& value) override {
        if (key == "sampleRate") {
            sampleRate_ = std::stoi(value);
            session_.setSampleRate(sampleRate_);
        } else if (key == "channels") {
            channels_ = std::stoi(value);
            session_.setChannels(channels_);
        } else if (key == "bufferFrames") {
            bufferFrames_ = std::stoi(value);
            session_.setBufferFrames(bufferFrames_);
        } else if (key == "volume_in") {
            session_.setVolumeIn(std::stof(value));
        } else if (key == "mute_in") {
            session_.setMuteIn(value == "true" || value == "1");
        } else if (key == "device_id") {
            session_.setDeviceIds(value, session_.deviceOutId());
        } else if (key == "loopback_test") {
            loopbackTest_ = (value == "true" || value == "1");
            session_.enableLoopback(loopbackTest_);
        } else if (key == "vcos_threshold") {
            session_.setVcosThreshold(static_cast<uint32_t>(std::stoul(value)));
        } else if (key == "vcos_hang_ms") {
            session_.setVcosHangMs(static_cast<uint32_t>(std::stoul(value)));
        }
    }

    std::string getParameter(const std::string& key) const override {
        if (key == "sampleRate") return std::to_string(sampleRate_);
        if (key == "channels") return std::to_string(channels_);
        if (key == "bufferFrames") return std::to_string(bufferFrames_);
        if (key == "volume_in") return std::to_string(session_.volumeIn());
        if (key == "mute_in") return session_.muteIn() ? "true" : "false";
        if (key == "device_id") return session_.deviceInId();
        if (key == "loopback_test") return loopbackTest_ ? "true" : "false";
        if (key == "vcos_threshold") return std::to_string(session_.vcosThreshold());
        if (key == "vcos_hang_ms") return std::to_string(session_.vcosHangMs());
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

        // Ensure buffer reflects configured channels/frames.
        if (buffer.getChannelCount() != channels_ || buffer.getFrameCount() != bufferFrames_) {
            buffer.resize(channels_, bufferFrames_);
        }

        if (!session_.readCapture(buffer)) {
            buffer.clear();
            return false;
        }

        // Apply mute/volume.
        if (session_.muteIn() || session_.volumeIn() != 1.0f) {
            float gain = session_.muteIn() ? 0.0f : session_.volumeIn();
            int frames = buffer.getFrameCount();
            int chans = buffer.getChannelCount();
            for (int ch = 0; ch < chans; ++ch) {
                float* channelData = buffer.getChannelData(ch);
                for (int i = 0; i < frames; ++i) {
                    channelData[i] *= gain;
                }
            }
        }

        if (callback_) {
            callback_(buffer);
        }
        return true;
    }

    int getSampleRate() const override { return sampleRate_; }
    int getChannels() const override { return channels_; }
    void setSampleRate(int sampleRate) override {
        if (state_ == PluginState::Unloaded || state_ == PluginState::Initialized) {
            sampleRate_ = sampleRate;
            session_.setSampleRate(sampleRate_);
        }
    }
    void setChannels(int channels) override {
        if (state_ == PluginState::Unloaded || state_ == PluginState::Initialized) {
            channels_ = channels;
            session_.setChannels(channels_);
        }
    }

private:
    PluginState state_;
    int sampleRate_;
    int channels_;
    int bufferFrames_;
    bool loopbackTest_;
    AudioSourceCallback callback_;
    AIOCSession session_;
};

} // namespace nda

NDA_DECLARE_PLUGIN(nda::AIOCSourcePlugin)
