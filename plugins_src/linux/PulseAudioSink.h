#ifndef PULSEAUDIOSINK_H
#define PULSEAUDIOSINK_H

#include "plugins/AudioSinkPlugin.h"
#include "audio/RingBuffer.h"
#include <pulse/pulseaudio.h>
#include <string>
#include <atomic>
#include <mutex>
#include <vector>

namespace nda {

/**
 * @brief PulseAudio speaker playback plugin for Linux.
 *
 * Plays audio through system speakers using PulseAudio async API.
 * Works with both PulseAudio and PipeWire (via pipewire-pulse).
 *
 * Threading model:
 * - Dedicated pa_threaded_mainloop for this plugin instance
 * - writeAudio() pushes data to lock-free ring buffer
 * - PulseAudio callbacks pull data from ring buffer
 */
class PulseAudioSink : public AudioSinkPlugin {
public:
    PulseAudioSink();
    ~PulseAudioSink() override;

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
    // PulseAudio resources
    pa_threaded_mainloop* mainloop_;
    pa_context* context_;
    pa_stream* stream_;

    // Audio configuration
    int sampleRate_;
    int channels_;
    int bufferSize_;
    std::string deviceName_;  // Empty = default device

    // State
    std::atomic<PluginState> state_;

    // Ring buffer for thread-safe audio transfer
    // Capacity: ~170ms of audio (8 buffers * 512 frames / 48000 Hz)
    static constexpr int RING_BUFFER_FRAMES = 512 * 8;
    RingBuffer ringBuffer_;

    // Temporary buffer for planar→interleaved conversion
    std::vector<float> interleavedBuffer_;
    std::vector<const float*> planarPtrs_;

    // Statistics
    std::atomic<uint64_t> underrunCount_;
    std::atomic<uint64_t> overrunCount_;

    // Mutex for parameter changes during operation
    mutable std::mutex paramMutex_;

    // PulseAudio callbacks (static, dispatch to instance methods)
    static void contextStateCallback(pa_context* c, void* userdata);
    static void streamStateCallback(pa_stream* s, void* userdata);
    static void streamWriteCallback(pa_stream* s, size_t nbytes, void* userdata);
    static void streamUnderflowCallback(pa_stream* s, void* userdata);

    // Instance methods called by callbacks
    void onContextState(pa_context* c);
    void onStreamState(pa_stream* s);
    void onStreamWrite(pa_stream* s, size_t nbytes);
    void onStreamUnderflow(pa_stream* s);

    // Helper methods
    bool createContext();
    bool createStream();
    void destroyStream();
    void destroyContext();

    // Wait for context/stream to reach ready state
    bool waitForContextReady();
    bool waitForStreamReady();

    // Pre-fill stream buffer with silence to avoid initial underruns
    void prefillStream();
};

} // namespace nda

// Plugin factory functions (C interface for dynamic loading)
extern "C" {
    nda::BasePlugin* createPlugin();
    void destroyPlugin(nda::BasePlugin* plugin);
}

#endif // PULSEAUDIOSINK_H
