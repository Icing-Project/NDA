#ifndef AUDIOSINKPLUGIN_H
#define AUDIOSINKPLUGIN_H

#include "BasePlugin.h"
#include "audio/AudioBuffer.h"
#include <functional>

namespace nda {

// v2.2: Callback for event-driven pipeline - notifies when space is available
using SpaceAvailableCallback = std::function<void()>;

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

    // ===== v2.2: Event-Driven Pipeline Support =====

    /**
     * @brief Check if this plugin supports async/event-driven mode.
     *
     * Plugins that support async mode use internal ring buffers and background
     * threads, providing non-blocking writeAudio() calls and signaling when
     * space becomes available instead of requiring backpressure retry loops.
     *
     * @return true if plugin supports setSpaceAvailableCallback(), false for legacy polling
     */
    virtual bool supportsAsyncMode() const { return false; }

    /**
     * @brief Register callback for space-available events (event-driven mode).
     *
     * When the sink has space for more audio data (typically >= one frame),
     * it should call this callback to wake the pipeline thread.
     *
     * Only called if supportsAsyncMode() returns true.
     *
     * @param callback Function to call when space is available (thread-safe)
     */
    virtual void setSpaceAvailableCallback(SpaceAvailableCallback callback) {
        (void)callback;  // Default: no-op for legacy plugins
    }

    /**
     * @brief Check if writeAudio() is non-blocking in async mode.
     *
     * Async sinks should return true, meaning writeAudio() writes to an
     * internal ring buffer and returns immediately without waiting for
     * the audio device.
     *
     * @return true if writeAudio() never blocks, false if it may block
     */
    virtual bool isNonBlocking() const { return false; }
};

} // namespace nda

#endif // AUDIOSINKPLUGIN_H
