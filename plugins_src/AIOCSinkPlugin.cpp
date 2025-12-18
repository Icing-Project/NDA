#include "plugins/AudioSinkPlugin.h"
#include "AIOCPluginCommon.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

namespace nda {

class AIOCSinkPlugin : public AudioSinkPlugin {
public:
    AIOCSinkPlugin()
        : state_(PluginState::Unloaded),
          sampleRate_(48000),
          channels_(1),
          bufferFrames_(512),
          pttArmed_(false),
          pttMode_(AIOCPttMode::HidManual),
          loopbackTest_(false)
    {
        // Lock to AIOC hardware endpoints by default (per-device IDs provided).
        session_.setDeviceIds(
            session_.deviceInId(),
            "{DF6E2579-254F-44A3-AFD9-301BDD499759}"); // Speakers (AIOC Audio)
        session_.setCdcPort("COM8"); // AIOC CDC port (COM8)
        session_.setSampleRate(sampleRate_);
        session_.setChannels(channels_);
        session_.setBufferFrames(bufferFrames_);
        session_.setPttMode(pttMode_);
    }

    ~AIOCSinkPlugin() override {
        if (state_ != PluginState::Unloaded) {
            shutdown();
        }
    }

    bool initialize() override {
        if (state_ != PluginState::Unloaded && state_ != PluginState::Initialized) {
            return false;
        }
        session_.setSampleRate(sampleRate_);
        session_.setChannels(channels_);
        session_.setBufferFrames(bufferFrames_);
        session_.setPttMode(pttMode_);
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
            std::cerr << "[AIOCSink] Failed to connect to AIOC device: "
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
            session_.setPttState(false);
            session_.stop();
            state_ = PluginState::Initialized;
        }
    }

    PluginInfo getInfo() const override {
        return {
            "AIOC Sink",
            "0.1.0",
            "Icing Project",
            "Writes audio to AIOC (USB speaker) and manages PTT/VOX",
            PluginType::AudioSink,
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
        } else if (key == "volume_out") {
            session_.setVolumeOut(std::stof(value));
        } else if (key == "mute_out") {
            session_.setMuteOut(value == "true" || value == "1");
        } else if (key == "ptt_state") {
            pttArmed_ = (value == "true" || value == "1");
            if (pttMode_ != AIOCPttMode::VpttAuto) {
                session_.setPttState(pttArmed_);
            }
        } else if (key == "ptt_mode") {
            if (value == "hid_manual") {
                pttMode_ = AIOCPttMode::HidManual;
            } else if (value == "cdc_manual") {
                pttMode_ = AIOCPttMode::CdcManual;
            } else if (value == "vptt_auto") {
                pttMode_ = AIOCPttMode::VpttAuto;
            }
            session_.setPttMode(pttMode_);
        } else if (key == "device_id") {
            session_.setDeviceIds(session_.deviceInId(), value);
        } else if (key == "cdc_port") {
            session_.setCdcPort(value);
        } else if (key == "vptt_threshold") {
            session_.setVpttThreshold(static_cast<uint32_t>(std::stoul(value)));
        } else if (key == "vptt_hang_ms") {
            session_.setVpttHangMs(static_cast<uint32_t>(std::stoul(value)));
        } else if (key == "loopback_test") {
            loopbackTest_ = (value == "true" || value == "1");
            session_.enableLoopback(loopbackTest_);
        }
    }

    std::string getParameter(const std::string& key) const override {
        if (key == "sampleRate") return std::to_string(sampleRate_);
        if (key == "channels") return std::to_string(channels_);
        if (key == "bufferFrames") return std::to_string(bufferFrames_);
        if (key == "volume_out") return std::to_string(session_.volumeOut());
        if (key == "mute_out") return session_.muteOut() ? "true" : "false";
        if (key == "ptt_state") return pttArmed_ ? "true" : "false";
        if (key == "ptt_mode") return modeToString(pttMode_);
        if (key == "device_id") return session_.deviceOutId();
        if (key == "cdc_port") return session_.cdcPort();
        if (key == "vptt_threshold") return std::to_string(session_.vpttThreshold());
        if (key == "vptt_hang_ms") return std::to_string(session_.vpttHangMs());
        if (key == "loopback_test") return loopbackTest_ ? "true" : "false";
        return "";
    }

    bool writeAudio(const AudioBuffer& buffer) override {
        if (state_ != PluginState::Running) {
            return false;
        }

        // Prepare a mutable copy to apply gain/mute if needed.
        AudioBuffer work(buffer.getChannelCount(), buffer.getFrameCount());
        work.copyFrom(buffer);

        if (session_.muteOut() || session_.volumeOut() != 1.0f) {
            float gain = session_.muteOut() ? 0.0f : session_.volumeOut();
            int frames = work.getFrameCount();
            int chans = work.getChannelCount();
            for (int ch = 0; ch < chans; ++ch) {
                float* channelData = work.getChannelData(ch);
                for (int i = 0; i < frames; ++i) {
                    channelData[i] *= gain;
                }
            }
        }

        handlePtt(work);
        return session_.writePlayback(work);
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
    int getBufferSize() const override { return bufferFrames_; }
    void setBufferSize(int samples) override {
        if (state_ == PluginState::Unloaded || state_ == PluginState::Initialized) {
            bufferFrames_ = samples;
            session_.setBufferFrames(bufferFrames_);
        }
    }
    int getAvailableSpace() const override { return bufferFrames_; }

private:
    void handlePtt(const AudioBuffer& buffer) {
        if (pttMode_ == AIOCPttMode::VpttAuto) {
            auto now = std::chrono::steady_clock::now();
            float peak = measurePeak(buffer);
            float threshold = static_cast<float>(session_.vpttThreshold()) / 32768.0f;

            if (peak >= threshold) {
                session_.setPttState(true);
                lastVoice_ = now;
                return;
            }

            if (session_.vpttHangMs() == 0) {
                session_.setPttState(false);
                return;
            }

            if (session_.isPttAsserted()) {
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastVoice_).count();
                if (elapsed > session_.vpttHangMs()) {
                    session_.setPttState(false);
                }
            }
        } else {
            session_.setPttState(pttArmed_);
        }
    }

    float measurePeak(const AudioBuffer& buffer) const {
        int frames = buffer.getFrameCount();
        int chans = buffer.getChannelCount();
        float peak = 0.0f;
        for (int ch = 0; ch < chans; ++ch) {
            const float* channelData = buffer.getChannelData(ch);
            for (int i = 0; i < frames; ++i) {
                peak = std::max(peak, std::abs(channelData[i]));
            }
        }
        return peak;
    }

    std::string modeToString(AIOCPttMode mode) const {
        switch (mode) {
            case AIOCPttMode::HidManual: return "hid_manual";
            case AIOCPttMode::CdcManual: return "cdc_manual";
            case AIOCPttMode::VpttAuto: return "vptt_auto";
        }
        return "hid_manual";
    }

    PluginState state_;
    int sampleRate_;
    int channels_;
    int bufferFrames_;
    bool pttArmed_;
    AIOCPttMode pttMode_;
    bool loopbackTest_;
    std::chrono::steady_clock::time_point lastVoice_;
    AIOCSession session_;
};

} // namespace nda

NDA_DECLARE_PLUGIN(nda::AIOCSinkPlugin)
