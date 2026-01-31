#include "plugins/AudioSourcePlugin.h"
#include "AIOCPluginCommon.h"
#include <cstring>
#include <iostream>

namespace nda {

class AIOCSourcePlugin : public AudioSourcePlugin {
public:
    AIOCSourcePlugin()
        : state_(PluginState::Unloaded),
          sampleRate_(48000),
          channels_(2),  // v2.2: Output stereo (AIOC device is mono, we duplicate)
          deviceChannels_(1),  // AIOC device is always mono
          bufferFrames_(512),
          loopbackTest_(false),
          dataReadyThreshold_(512)
    {
        // v2.2: Use default WASAPI capture device initially
        // User can select specific AIOC device via PluginSidebar UI
        session_.setSampleRate(sampleRate_);
        session_.setChannels(deviceChannels_);  // Session uses device channels (mono)
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
        session_.setChannels(deviceChannels_);  // Session uses device channels (mono)
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

        // CRITICAL: Query actual format from session after connect (may differ from requested)
        int actualSampleRate = session_.sampleRate();
        int actualDeviceChannels = session_.channels();
        if (actualSampleRate != sampleRate_ || actualDeviceChannels != deviceChannels_) {
            std::cerr << "[AIOCSource] WARNING: Format mismatch after connect!" << std::endl;
            std::cerr << "[AIOCSource] Requested: " << sampleRate_ << "Hz, " << deviceChannels_ << "ch device" << std::endl;
            std::cerr << "[AIOCSource] Actual: " << actualSampleRate << "Hz, " << actualDeviceChannels << "ch device" << std::endl;
            sampleRate_ = actualSampleRate;
            deviceChannels_ = actualDeviceChannels;
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
            // Note: Session always uses deviceChannels_ (mono) - we duplicate to stereo in readAudio()
        } else if (key == "bufferFrames") {
            bufferFrames_ = std::stoi(value);
            session_.setBufferFrames(bufferFrames_);
        } else if (key == "volume_in") {
            session_.setVolumeIn(std::stof(value));
        } else if (key == "mute_in") {
            session_.setMuteIn(value == "true" || value == "1");
        } else if (key == "device_id") {
            std::string oldId = session_.deviceInId();
            session_.setDeviceIds(value, session_.deviceOutId());
            // v2.2: Reconnect to new device if device ID changed
            if (value != oldId) {
                if (state_ == PluginState::Running) {
                    std::cerr << "[AIOCSource] Device changed while running, reconnecting to: " << value << std::endl;
                    session_.stop();
                    session_.disconnect();
                    if (session_.connect() && session_.start()) {
                        std::cerr << "[AIOCSource] Successfully switched to new device" << std::endl;
                    } else {
                        std::cerr << "[AIOCSource] Failed to switch device" << std::endl;
                        state_ = PluginState::Error;
                    }
                } else if (state_ == PluginState::Initialized && session_.isConnected()) {
                    // Disconnect so next start() will connect to the new device
                    std::cerr << "[AIOCSource] Device changed while initialized, disconnecting..." << std::endl;
                    session_.disconnect();
                }
            }
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

        // v2.2: AIOC device is mono - read into mono buffer first, then duplicate to stereo
        // Resize mono buffer if needed
        if (monoBuffer_.getChannelCount() != deviceChannels_ || monoBuffer_.getFrameCount() != bufferFrames_) {
            monoBuffer_.resize(deviceChannels_, bufferFrames_);
        }

        if (!session_.readCapture(monoBuffer_)) {
            buffer.clear();
            return false;
        }

        // Ensure output buffer is stereo
        if (buffer.getChannelCount() != channels_ || buffer.getFrameCount() != bufferFrames_) {
            buffer.resize(channels_, bufferFrames_);
        }

        // Duplicate mono to stereo (copy mono channel to all output channels)
        const float* monoData = monoBuffer_.getChannelData(0);
        for (int ch = 0; ch < channels_; ++ch) {
            float* outData = buffer.getChannelData(ch);
            std::memcpy(outData, monoData, bufferFrames_ * sizeof(float));
        }

        // Apply mute/volume to all channels
        if (session_.muteIn() || session_.volumeIn() != 1.0f) {
            float gain = session_.muteIn() ? 0.0f : session_.volumeIn();
            for (int ch = 0; ch < channels_; ++ch) {
                float* channelData = buffer.getChannelData(ch);
                for (int i = 0; i < bufferFrames_; ++i) {
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
            // Note: Session always uses deviceChannels_ (mono) - we duplicate to stereo in readAudio()
        }
    }

    // v2.2: Event-driven async mode support
    bool supportsAsyncMode() const override {
        return true;  // We use ring buffer + background thread
    }

    void setDataReadyCallback(DataReadyCallback callback) override {
        dataReadyCallback_ = callback;
        session_.setDataReadyCallback(callback);  // Propagate to AIOCSession
    }

    int getDataReadyThreshold() const override {
        return dataReadyThreshold_;  // Default: 512 frames
    }

private:
    PluginState state_;
    int sampleRate_;
    int channels_;          // Output channels (stereo)
    int deviceChannels_;    // Device channels (mono for AIOC)
    int bufferFrames_;
    bool loopbackTest_;
    AudioSourceCallback callback_;
    AIOCSession session_;
    AudioBuffer monoBuffer_;  // v2.2: Temporary buffer for mono capture

    // v2.2: Event-driven members
    DataReadyCallback dataReadyCallback_;
    int dataReadyThreshold_;
};

} // namespace nda

NDA_DECLARE_PLUGIN(nda::AIOCSourcePlugin)
