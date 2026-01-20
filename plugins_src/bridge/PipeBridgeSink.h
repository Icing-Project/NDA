#ifndef PIPEBRIDGESINK_H
#define PIPEBRIDGESINK_H

#include "plugins/AudioSinkPlugin.h"
#include "PipeBridgeSession.h"

namespace nda {

/**
 * @brief Audio sink plugin that bridges to PipeBridgeSource via shared session.
 *
 * This plugin acts as the TX endpoint of a pipeline bridge. Audio written here
 * is directly handed off to the PipeBridgeSource with zero buffering.
 *
 * When no PipeBridgeSource is connected, this behaves as a null sink (discards frames).
 */
class PipeBridgeSink : public AudioSinkPlugin {
public:
    PipeBridgeSink();
    ~PipeBridgeSink() override;

    // BasePlugin interface
    bool initialize() override;
    void shutdown() override;
    bool start() override;
    void stop() override;
    PluginInfo getInfo() const override;
    PluginState getState() const override;
    void setParameter(const std::string& key, const std::string& value) override;
    std::string getParameter(const std::string& key) const override;

    // AudioSinkPlugin interface
    bool writeAudio(const AudioBuffer& buffer) override;
    int getSampleRate() const override;
    int getChannels() const override;
    void setSampleRate(int sampleRate) override;
    void setChannels(int channels) override;
    int getBufferSize() const override;
    void setBufferSize(int samples) override;
    int getAvailableSpace() const override;

private:
    PluginState state_;
    int sampleRate_;
    int channels_;
    int bufferSize_;
};

} // namespace nda

#endif // PIPEBRIDGESINK_H
