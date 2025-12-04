#ifndef AUDIOSOURCEPLUGIN_H
#define AUDIOSOURCEPLUGIN_H

#include "BasePlugin.h"
#include "audio/AudioBuffer.h"
#include <functional>

namespace nda {

// Callback for audio data from source
using AudioSourceCallback = std::function<void(const AudioBuffer& buffer)>;

class AudioSourcePlugin : public virtual BasePlugin {
public:
    virtual ~AudioSourcePlugin() = default;

    PluginType getType() const override { return PluginType::AudioSource; }

    // Audio source specific methods
    virtual void setAudioCallback(AudioSourceCallback callback) = 0;

    // Get audio data (pull model)
    virtual bool readAudio(AudioBuffer& buffer) = 0;

    // Audio configuration
    virtual int getSampleRate() const = 0;
    virtual int getChannels() const = 0;
    virtual void setSampleRate(int sampleRate) = 0;
    virtual void setChannels(int channels) = 0;
};

} // namespace nda

#endif // AUDIOSOURCEPLUGIN_H
