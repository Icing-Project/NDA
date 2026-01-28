#ifndef PULSEAUDIOSOURCE_H
#define PULSEAUDIOSOURCE_H

#include "plugins/AudioSourcePlugin.h"
#include "audio/RingBuffer.h"
#include <pulse/pulseaudio.h>
#include <string>
#include <atomic>
#include <mutex>
#include <vector>

namespace nda {

/**
 * @brief PulseAudio microphone capture plugin for Linux.
 *
 * Captures audio from system microphone using PulseAudio async API.
 * Works with both PulseAudio and PipeWire (via pipewire-pulse).
 *
 * Threading model:
 * - Dedicated pa_threaded_mainloop for this plugin instance
 * - PulseAudio callbacks push data to lock-free ring buffer
 * - readAudio() pulls from ring buffer (non-blocking)
 */
class PulseAudioSource : public AudioSourcePlugin {
public:
    PulseAudioSource();
    ~PulseAudioSource() override;

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

    int getSampleRate() const override;
    int getChannels() const override;
    void setSampleRate(int sampleRate) override;
    void setChannels(int channels) override;

    int getBufferSize() const override;
    void setBufferSize(int samples) override;

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
    AudioSourceCallback callback_;

    // Ring buffer for thread-safe audio transfer
    // Capacity: ~170ms of audio (8 buffers * 512 frames / 48000 Hz)
    static constexpr int RING_BUFFER_FRAMES = 512 * 8;
    RingBuffer ringBuffer_;

    // Temporary buffer for interleaved→planar conversion
    std::vector<float> interleavedBuffer_;
    std::vector<float*> planarPtrs_;

    // Statistics
    std::atomic<uint64_t> underrunCount_;
    std::atomic<uint64_t> overrunCount_;

    // Mutex for parameter changes during operation
    mutable std::mutex paramMutex_;

    // PulseAudio callbacks (static, dispatch to instance methods)
    static void contextStateCallback(pa_context* c, void* userdata);
    static void streamStateCallback(pa_stream* s, void* userdata);
    static void streamReadCallback(pa_stream* s, size_t nbytes, void* userdata);

    // Instance methods called by callbacks
    void onContextState(pa_context* c);
    void onStreamState(pa_stream* s);
    void onStreamRead(pa_stream* s, size_t nbytes);

    // Helper methods
    bool createContext();
    bool createStream();
    void destroyStream();
    void destroyContext();

    // Wait for context/stream to reach ready state
    bool waitForContextReady();
    bool waitForStreamReady();
};

} // namespace nda

// Plugin factory functions (C interface for dynamic loading)
extern "C" {
    nda::BasePlugin* createPlugin();
    void destroyPlugin(nda::BasePlugin* plugin);
}

#endif // PULSEAUDIOSOURCE_H
