#include "plugins/AudioSourcePlugin.h"
#include <alsa/asoundlib.h>
#include <cstring>
#include <iostream>

namespace NADE {

class AlsaMicrophonePlugin : public AudioSourcePlugin {
public:
    AlsaMicrophonePlugin()
        : state_(PluginState::Unloaded),
          sampleRate_(48000),
          channels_(2),
          handle_(nullptr)
    {
    }

    ~AlsaMicrophonePlugin() override {
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

        int err;

        // Open PCM device for recording (non-blocking mode)
        if ((err = snd_pcm_open(&handle_, "default", SND_PCM_STREAM_CAPTURE, SND_PCM_NONBLOCK)) < 0) {
            std::cerr << "[AlsaMicrophone] Cannot open audio device: "
                      << snd_strerror(err) << std::endl;
            return false;
        }

        // Allocate hardware parameters object
        snd_pcm_hw_params_t* params;
        snd_pcm_hw_params_alloca(&params);

        // Fill with default values
        snd_pcm_hw_params_any(handle_, params);

        // Set hardware parameters
        snd_pcm_hw_params_set_access(handle_, params, SND_PCM_ACCESS_RW_INTERLEAVED);
        snd_pcm_hw_params_set_format(handle_, params, SND_PCM_FORMAT_FLOAT_LE);
        snd_pcm_hw_params_set_channels(handle_, params, channels_);

        unsigned int rate = sampleRate_;
        snd_pcm_hw_params_set_rate_near(handle_, params, &rate, nullptr);

        // Set buffer size
        snd_pcm_uframes_t frames = 512;
        snd_pcm_hw_params_set_period_size_near(handle_, params, &frames, nullptr);

        // Write parameters to device
        if ((err = snd_pcm_hw_params(handle_, params)) < 0) {
            std::cerr << "[AlsaMicrophone] Cannot set parameters: "
                      << snd_strerror(err) << std::endl;
            snd_pcm_close(handle_);
            handle_ = nullptr;
            return false;
        }

        // Prepare device
        if ((err = snd_pcm_prepare(handle_)) < 0) {
            std::cerr << "[AlsaMicrophone] Cannot prepare device: "
                      << snd_strerror(err) << std::endl;
            snd_pcm_close(handle_);
            handle_ = nullptr;
            return false;
        }

        state_ = PluginState::Running;
        std::cout << "[AlsaMicrophone] Started recording at " << sampleRate_
                  << " Hz, " << channels_ << " channels" << std::endl;

        return true;
    }

    void stop() override {
        if (handle_) {
            // Don't drain in capture mode - just drop and close immediately
            snd_pcm_drop(handle_);  // Drop pending frames instead of draining
            snd_pcm_close(handle_);
            handle_ = nullptr;
            std::cout << "[AlsaMicrophone] Stopped recording" << std::endl;
        }
        if (state_ == PluginState::Running) {
            state_ = PluginState::Initialized;
        }
    }

    PluginInfo getInfo() const override {
        return {
            "ALSA Microphone",
            "1.0.0",
            "NADE Team",
            "Captures audio from default microphone using ALSA",
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
        if (!handle_ || state_ != PluginState::Running) {
            buffer.clear();
            return false;
        }

        // Read interleaved audio from ALSA
        int frameCount = buffer.getFrameCount();
        int totalSamples = frameCount * channels_;
        std::vector<float> tempBuffer(totalSamples);

        int err = snd_pcm_readi(handle_, tempBuffer.data(), frameCount);

        if (err == -EAGAIN || err == -EWOULDBLOCK) {
            // Non-blocking mode: data not available yet
            buffer.clear();
            return true; // Not an error, just no data yet
        } else if (err == -EPIPE) {
            // Buffer overrun
            std::cerr << "[AlsaMicrophone] Buffer overrun" << std::endl;
            snd_pcm_prepare(handle_);
            buffer.clear();
            return false;
        } else if (err < 0) {
            std::cerr << "[AlsaMicrophone] Read error: " << snd_strerror(err) << std::endl;
            buffer.clear();
            return false;
        } else if (err != frameCount) {
            // Short read in non-blocking mode is normal
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
    snd_pcm_t* handle_;
};

} // namespace NADE

// Export the plugin
NADE_DECLARE_PLUGIN(NADE::AlsaMicrophonePlugin)
