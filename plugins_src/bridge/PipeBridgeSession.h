#ifndef PIPEBRIDGESESSION_H
#define PIPEBRIDGESESSION_H

#include "audio/AudioBuffer.h"
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <cstdint>

// Export/import macros for shared library
#ifdef _WIN32
    #ifdef PIPEBRIDGE_SESSION_EXPORTS
        #define PIPEBRIDGE_API __declspec(dllexport)
    #else
        #define PIPEBRIDGE_API __declspec(dllimport)
    #endif
#else
    #define PIPEBRIDGE_API __attribute__((visibility("default")))
#endif

namespace nda {

/**
 * @brief Singleton session for direct frame handoff between PipeBridgeSink and PipeBridgeSource.
 *
 * This class provides zero-latency, zero-buffering audio transfer between two pipelines.
 * The sink blocks until the source consumes the frame (with timeout fallback to null sink behavior).
 * The source blocks until the sink produces a frame.
 *
 * Thread Safety:
 * - All public methods are thread-safe
 * - Designed for exactly one sink thread and one source thread
 *
 * Behavior:
 * - When source is connected: sink blocks until frame is consumed (with timeout)
 * - When source is disconnected: sink discards frames immediately (null sink behavior)
 * - When sink is disconnected: source's read() returns false
 */
class PIPEBRIDGE_API PipeBridgeSession {
public:
    /**
     * @brief Get the singleton instance.
     */
    static PipeBridgeSession& getInstance();

    // Non-copyable, non-movable
    PipeBridgeSession(const PipeBridgeSession&) = delete;
    PipeBridgeSession& operator=(const PipeBridgeSession&) = delete;
    PipeBridgeSession(PipeBridgeSession&&) = delete;
    PipeBridgeSession& operator=(PipeBridgeSession&&) = delete;

    // ===== Sink Interface =====

    /**
     * @brief Write a frame to the bridge (called by PipeBridgeSink).
     *
     * Behavior:
     * - If source is not connected: discards frame immediately (null sink behavior)
     * - If source is connected: blocks until source consumes or timeout
     * - On timeout: discards frame and continues (prevents pipeline stall)
     *
     * @param buffer The audio frame to pass through
     * @return true always (never blocks the pipeline permanently)
     */
    bool write(const AudioBuffer& buffer);

    /**
     * @brief Notify that sink has connected.
     * @param sampleRate Sample rate configured on the sink
     * @param channels Channel count configured on the sink
     */
    void connectSink(int sampleRate, int channels);

    /**
     * @brief Notify that sink has disconnected.
     */
    void disconnectSink();

    // ===== Source Interface =====

    /**
     * @brief Read a frame from the bridge (called by PipeBridgeSource).
     *
     * Blocks until a frame is available from the sink.
     *
     * @param buffer Output buffer to fill with audio data
     * @return true if frame received, false if sink disconnected
     */
    bool read(AudioBuffer& buffer);

    /**
     * @brief Notify that source has connected.
     */
    void connectSource();

    /**
     * @brief Notify that source has disconnected.
     */
    void disconnectSource();

    // ===== Configuration =====

    int getSampleRate() const { return sampleRate_.load(); }
    int getChannels() const { return channels_.load(); }
    bool isSinkConnected() const { return sinkConnected_.load(); }
    bool isSourceConnected() const { return sourceConnected_.load(); }

    // ===== Metrics =====

    uint64_t getFramesPassed() const { return framesPassed_.load(); }
    uint64_t getFramesDropped() const { return framesDropped_.load(); }
    void resetMetrics();

private:
    PipeBridgeSession();
    ~PipeBridgeSession() = default;

    // Handoff state
    AudioBuffer frame_;
    std::mutex mutex_;
    std::condition_variable frameReady_;
    std::condition_variable frameConsumed_;
    std::atomic<bool> hasFrame_{false};

    // Connection state
    std::atomic<bool> sinkConnected_{false};
    std::atomic<bool> sourceConnected_{false};

    // Format (set by sink on connect)
    std::atomic<int> sampleRate_{48000};
    std::atomic<int> channels_{2};

    // Metrics
    std::atomic<uint64_t> framesPassed_{0};
    std::atomic<uint64_t> framesDropped_{0};
};

} // namespace nda

#endif // PIPEBRIDGESESSION_H
