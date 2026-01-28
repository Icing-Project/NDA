#ifndef PIPEBRIDGESOURCE_H
#define PIPEBRIDGESOURCE_H

#include "plugins/AudioSourcePlugin.h"
#include "PipeBridgeSession.h"

namespace nda {

/**
 * @brief Audio source plugin that receives from PipeBridgeSink via shared session.
 *
 * This plugin acts as the RX endpoint of a pipeline bridge. Audio is received
 * directly from PipeBridgeSink with zero buffering/latency.
 *
 * The source inherits sample rate and channel configuration from the connected sink.
 */
class PipeBridgeSource : public AudioSourcePlugin {
public:
    PipeBridgeSource();
    ~PipeBridgeSource() override;

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
    int getBufferSize() const override;
    void setBufferSize(int samples) override;
    int getSampleRate() const override;
    int getChannels() const override;
    void setSampleRate(int sampleRate) override;
    void setChannels(int channels) override;

private:
    PluginState state_;
    int bufferSize_;
    AudioSourceCallback callback_;
};

} // namespace nda

#endif // PIPEBRIDGESOURCE_H
