#ifndef LINUXAIOCSOURCEPLUGIN_H
#define LINUXAIOCSOURCEPLUGIN_H

#include "plugins/AudioSourcePlugin.h"
#include "audio/RingBuffer.h"
#include "LinuxAIOCSession.h"
#include <pulse/pulseaudio.h>
#include <memory>
#include <string>
#include <atomic>
#include <mutex>
#include <vector>

namespace nda {

/**
 * @brief Linux AIOC audio source plugin for microphone capture.
 *
 * Captures audio from AIOC device microphone input via PulseAudio.
 * Features:
 * - Auto-detection of AIOC audio devices (by name matching)
 * - Optional integration with LinuxAIOCSession for PTT state monitoring
 * - Ring buffer for thread-safe audio transfer
 * - Works with both PulseAudio and PipeWire (via pipewire-pulse)
 *
 * Parameters:
 * - device: PulseAudio source name ("auto" for AIOC auto-detection)
 * - sampleRate: Sample rate in Hz (default: 48000)
 * - channels: Channel count (default: 1 for mono radio)
 * - bufferSize: Frames per buffer (default: 512)
 *
 * Threading model:
 * - Dedicated pa_threaded_mainloop for this plugin instance
 * - PulseAudio callbacks push data to ring buffer
 * - readAudio() pulls from ring buffer (non-blocking with timeout)
 */
class LinuxAIOCSourcePlugin : public AudioSourcePlugin {
public:
    LinuxAIOCSourcePlugin();
    ~LinuxAIOCSourcePlugin() override;

    // ==================== BasePlugin Interface ====================

    bool initialize() override;
    void shutdown() override;
    bool start() override;
    void stop() override;

    PluginInfo getInfo() const override;
    PluginState getState() const override;

    void setParameter(const std::string& key, const std::string& value) override;
    std::string getParameter(const std::string& key) const override;

    // ==================== AudioSourcePlugin Interface ====================

    void setAudioCallback(AudioSourceCallback callback) override;
    bool readAudio(AudioBuffer& buffer) override;

    int getSampleRate() const override;
    int getChannels() const override;
    void setSampleRate(int sampleRate) override;
    void setChannels(int channels) override;

    int getBufferSize() const override;
    void setBufferSize(int samples) override;

    // ==================== AIOC-Specific ====================

    /**
     * @brief Set shared AIOC session for PTT coordination.
     * @param session Shared session (can be shared with sink plugin)
     *
     * This is optional - the source plugin can work without a session.
     * When set, the session can be used to monitor PTT state.
     */
    void setAIOCSession(std::shared_ptr<LinuxAIOCSession> session);

    /**
     * @brief Get the detected/configured device name.
     */
    std::string getDeviceName() const;

    /**
     * @brief Check if an AIOC device was detected.
     */
    bool isAIOCDeviceDetected() const;

private:
    // AIOC device detection
    std::string findAIOCSourceDevice();

    // PulseAudio resources
    pa_threaded_mainloop* mainloop_;
    pa_context* context_;
    pa_stream* stream_;

    // Configuration
    int sampleRate_;
    int channels_;
    int bufferSize_;
    std::string deviceName_;       ///< Configured device name ("auto" or explicit)
    std::string resolvedDevice_;   ///< Actual device name after detection
    bool autoDetectDevice_;        ///< True if deviceName_ is "auto"

    // State
    std::atomic<PluginState> state_;
    AudioSourceCallback callback_;

    // Ring buffer for thread-safe audio transfer
    // Capacity: ~170ms of audio (8 buffers * 512 frames / 48000 Hz)
    static constexpr int RING_BUFFER_FRAMES = 512 * 8;
    RingBuffer ringBuffer_;

    // Temporary buffers for interleaved→planar conversion
    std::vector<float> interleavedBuffer_;
    std::vector<float*> planarPtrs_;

    // AIOC session (optional, shared with sink)
    std::shared_ptr<LinuxAIOCSession> aiocSession_;

    // Statistics
    std::atomic<uint64_t> underrunCount_;
    std::atomic<uint64_t> overrunCount_;
    std::atomic<bool> aiocDetected_;

    // Mutex for parameter changes
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
    bool waitForContextReady();
    bool waitForStreamReady();
};

} // namespace nda

// Plugin factory functions (C interface for dynamic loading)
extern "C" {
    nda::BasePlugin* createPlugin();
    void destroyPlugin(nda::BasePlugin* plugin);
}

#endif // LINUXAIOCSOURCEPLUGIN_H
