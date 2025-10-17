#ifndef MICROPHONESOURCEPLUGIN_H
#define MICROPHONESOURCEPLUGIN_H

#include "plugins/AudioSourcePlugin.h"

namespace NADE {

class MicrophoneSourcePlugin : public AudioSourcePlugin {
public:
    MicrophoneSourcePlugin();
    ~MicrophoneSourcePlugin() override;

    // BasePlugin interface
    bool initialize() override;
    void shutdown() override;
    bool start() override;
    void stop() override;
    PluginInfo getInfo() const override;
    PluginState getState() const override;
    void setParameter(const std::string& key, const std::string& value) override;
    std::string getParameter(const std::string& key) const override;

    // AudioSourcePlugin interface
    void setAudioCallback(AudioSourceCallback callback) override;
    bool readAudio(AudioBuffer& buffer) override;
    int getSampleRate() const override { return sampleRate_; }
    int getChannels() const override { return channels_; }
    void setSampleRate(int sampleRate) override { sampleRate_ = sampleRate; }
    void setChannels(int channels) override { channels_ = channels; }

private:
    PluginState state_;
    int sampleRate_;
    int channels_;
    AudioSourceCallback callback_;
    std::string deviceName_;
    void* audioDevice_; // Platform-specific device handle
};

} // namespace NADE

// Export plugin
NADE_DECLARE_PLUGIN(NADE::MicrophoneSourcePlugin)

#endif // MICROPHONESOURCEPLUGIN_H
