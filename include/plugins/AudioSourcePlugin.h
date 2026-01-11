#ifndef AUDIOSOURCEPLUGIN_H
#define AUDIOSOURCEPLUGIN_H

#include "BasePlugin.h"
#include "audio/AudioBuffer.h"
#include <functional>

namespace nda {

// Callback for audio data from source
using AudioSourceCallback = std::function<void(const AudioBuffer& buffer)>;

// v2.2: Callback for event-driven pipeline - notifies when data is ready
using DataReadyCallback = std::function<void()>;

class AudioSourcePlugin : public virtual BasePlugin {
public:
    virtual ~AudioSourcePlugin() = default;

    PluginType getType() const override { return PluginType::AudioSource; }

    // Audio source specific methods
    virtual void setAudioCallback(AudioSourceCallback callback) = 0;

    // Get audio data (pull model)
    virtual bool readAudio(AudioBuffer& buffer) = 0;

    // Buffer sizing (frames per buffer). Default to 512 if not overridden.
    virtual int getBufferSize() const { return 512; }
    virtual void setBufferSize(int /*samples*/) {}

    // Audio configuration
    virtual int getSampleRate() const = 0;
    virtual int getChannels() const = 0;
    virtual void setSampleRate(int sampleRate) = 0;
    virtual void setChannels(int channels) = 0;

    // ===== v2.2: Event-Driven Pipeline Support =====

    /**
     * @brief Check if this plugin supports async/event-driven mode.
     *
     * Plugins that support async mode use internal ring buffers and background
     * threads, calling the data ready callback when audio is available instead
     * of requiring the pipeline to poll.
     *
     * @return true if plugin supports setDataReadyCallback(), false for legacy polling
     */
    virtual bool supportsAsyncMode() const { return false; }

    /**
     * @brief Register callback for data-ready events (event-driven mode).
     *
     * When the source has enough audio data available (typically >= one frame),
     * it should call this callback to wake the pipeline thread.
     *
     * Only called if supportsAsyncMode() returns true.
     *
     * @param callback Function to call when data is ready (thread-safe)
     */
    virtual void setDataReadyCallback(DataReadyCallback callback) {
        (void)callback;  // Default: no-op for legacy plugins
    }

    /**
     * @brief Get minimum frames needed before signaling data ready.
     *
     * For event-driven mode, this is the threshold at which the plugin
     * should call the data ready callback.
     *
     * @return Number of frames that trigger a data-ready notification
     */
    virtual int getDataReadyThreshold() const { return getBufferSize(); }
};

} // namespace nda

#endif // AUDIOSOURCEPLUGIN_H
