#ifndef AUDIOSINKPLUGIN_H
#define AUDIOSINKPLUGIN_H

#include "BasePlugin.h"
#include "audio/AudioBuffer.h"

namespace NADE {

class AudioSinkPlugin : public virtual BasePlugin {
public:
    virtual ~AudioSinkPlugin() = default;

    PluginType getType() const override { return PluginType::AudioSink; }

    // Audio sink specific methods
    virtual bool writeAudio(const AudioBuffer& buffer) = 0;

    // Audio configuration
    virtual int getSampleRate() const = 0;
    virtual int getChannels() const = 0;
    virtual void setSampleRate(int sampleRate) = 0;
    virtual void setChannels(int channels) = 0;

    // Buffer management
    virtual int getBufferSize() const = 0;
    virtual void setBufferSize(int samples) = 0;
    virtual int getAvailableSpace() const = 0;
};

} // namespace NADE

#endif // AUDIOSINKPLUGIN_H
