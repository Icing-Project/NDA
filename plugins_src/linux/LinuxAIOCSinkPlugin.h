#ifndef LINUXAIOCSINKPLUGIN_H
#define LINUXAIOCSINKPLUGIN_H

#include "plugins/AudioSinkPlugin.h"
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
 * @brief Linux AIOC audio sink plugin for speaker playback with PTT control.
 *
 * Plays audio to AIOC device speaker output via PulseAudio and controls PTT.
 * Features:
 * - Auto-detection of AIOC audio devices (by name matching)
 * - PTT control via HID or CDC (through LinuxAIOCSession)
 * - Ring buffer for thread-safe audio transfer
 * - Works with both PulseAudio and PipeWire (via pipewire-pulse)
 *
 * Parameters:
 * - device: PulseAudio sink name ("auto" for AIOC auto-detection)
 * - sampleRate: Sample rate in Hz (default: 48000)
 * - channels: Channel count (default: 1 for mono radio)
 * - bufferSize: Frames per buffer (default: 512)
 * - ptt_mode: PTT control mode ("auto", "hid", "cdc")
 * - cdc_port: Serial port for CDC PTT ("auto" or "/dev/ttyACM0")
 *
 * Threading model:
 * - Dedicated pa_threaded_mainloop for this plugin instance
 * - writeAudio() pushes data to ring buffer (non-blocking)
 * - PulseAudio callbacks pull data from ring buffer
 */
class LinuxAIOCSinkPlugin : public AudioSinkPlugin {
public:
    LinuxAIOCSinkPlugin();
    ~LinuxAIOCSinkPlugin() override;

    // ==================== BasePlugin Interface ====================

    bool initialize() override;
    void shutdown() override;
    bool start() override;
    void stop() override;

    PluginInfo getInfo() const override;
    PluginState getState() const override;

    void setParameter(const std::string& key, const std::string& value) override;
    std::string getParameter(const std::string& key) const override;

    // ==================== AudioSinkPlugin Interface ====================

    bool writeAudio(const AudioBuffer& buffer) override;

    int getSampleRate() const override;
    int getChannels() const override;
    void setSampleRate(int sampleRate) override;
    void setChannels(int channels) override;

    int getBufferSize() const override;
    void setBufferSize(int samples) override;
    int getAvailableSpace() const override;

    // ==================== AIOC-Specific: PTT Control ====================

    /**
     * @brief Set shared AIOC session for PTT control.
     * @param session Shared session (can be shared with source plugin)
     *
     * If not set, the plugin will create its own session.
     */
    void setAIOCSession(std::shared_ptr<LinuxAIOCSession> session);

    /**
     * @brief Get the AIOC session (creates one if needed).
     */
    std::shared_ptr<LinuxAIOCSession> getAIOCSession();

    /**
     * @brief Set PTT state (assert or release).
     * @param asserted true = PTT on (transmitting), false = PTT off
     * @return true if PTT state was set successfully
     */
    bool setPttState(bool asserted);

    /**
     * @brief Get current PTT state.
     */
    bool isPttAsserted() const;

    /**
     * @brief Get AIOC telemetry data.
     */
    LinuxAIOCTelemetry getAIOCTelemetry() const;

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
    std::string findAIOCSinkDevice();

    // Parse PTT mode string to enum
    LinuxPttMode parsePttMode(const std::string& mode) const;
    std::string pttModeToString(LinuxPttMode mode) const;

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
    LinuxPttMode pttMode_;         ///< PTT control mode
    std::string cdcPort_;          ///< CDC serial port path

    // State
    std::atomic<PluginState> state_;

    // Ring buffer for thread-safe audio transfer
    // Capacity: ~170ms of audio (8 buffers * 512 frames / 48000 Hz)
    static constexpr int RING_BUFFER_FRAMES = 512 * 8;
    RingBuffer ringBuffer_;

    // Temporary buffers for planar→interleaved conversion
    std::vector<float> interleavedBuffer_;
    std::vector<const float*> planarPtrs_;

    // AIOC session (PTT control)
    std::shared_ptr<LinuxAIOCSession> aiocSession_;
    bool ownsSession_;  ///< True if this plugin created the session

    // Statistics
    std::atomic<uint64_t> underrunCount_;
    std::atomic<uint64_t> overrunCount_;
    std::atomic<bool> aiocDetected_;

    // Mutex for parameter changes
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
    bool waitForContextReady();
    bool waitForStreamReady();
    void prefillStream();
};

} // namespace nda

// Plugin factory functions (C interface for dynamic loading)
extern "C" {
    nda::BasePlugin* createPlugin();
    void destroyPlugin(nda::BasePlugin* plugin);
}

#endif // LINUXAIOCSINKPLUGIN_H
