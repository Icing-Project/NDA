/**
 * @file NadeExternalIO.h
 * @brief Singleton interface for Nade-Python text messaging modem.
 *
 * This class provides a thread-safe singleton that manages communication between
 * NDA C++ plugins and the Nade-Python modem. It uses a dedicated Python worker
 * thread to avoid blocking the audio thread on Python's GIL.
 *
 * Architecture:
 *   - Audio thread calls getTxAudio() and processRxAudio() (non-blocking)
 *   - Python worker thread handles modem operations asynchronously
 *   - Lock-free ring buffers for audio data transfer
 *
 * @version 3.0
 * @date 2026-01-30
 */

#pragma once

#include <memory>
#include <mutex>
#include <thread>
#include <atomic>
#include <vector>
#include <string>
#include <condition_variable>
#include <queue>
#include <functional>

namespace nade {

/**
 * @brief Result structure for operations across C++/Python boundary.
 *
 * Uses error codes instead of exceptions for audio-safe error handling.
 */
struct NadeResult {
    bool success = false;
    std::string error_message;

    static NadeResult ok() { return {true, ""}; }
    static NadeResult error(const std::string& msg) { return {false, msg}; }
};

/**
 * @brief Thread-safe lock-free ring buffer for audio samples.
 *
 * Single Producer Single Consumer (SPSC) pattern.
 * Producer calls write(), Consumer calls read().
 */
class AudioRingBuffer {
public:
    explicit AudioRingBuffer(size_t capacity);

    /// Write samples to buffer. Returns number of samples written.
    size_t write(const float* data, size_t count);

    /// Read samples from buffer. Returns number of samples read.
    size_t read(float* data, size_t count);

    /// Get available samples for reading.
    size_t available() const;

    /// Get available space for writing.
    size_t space() const;

    /// Clear buffer.
    void clear();

private:
    std::vector<float> buffer_;
    std::atomic<size_t> writePos_{0};
    std::atomic<size_t> readPos_{0};
    size_t capacity_;
};

/**
 * @brief Singleton manager for Nade-Python integration.
 *
 * This class is shared between NadeEncryptorPlugin and NadeDecryptorPlugin.
 * It manages a Python interpreter, worker thread, and audio buffers.
 *
 * Thread Safety:
 *   - Singleton creation is thread-safe (mutex protected)
 *   - Audio thread calls are non-blocking (lock-free buffers)
 *   - Python calls happen only on worker thread
 *
 * Usage:
 *   // Create singleton with keys
 *   auto nade = NadeExternalIO::createInstance(priv, pub, remote);
 *
 *   // Use from plugins
 *   nade->sendTextMessage("Hello");
 *   auto fsk = nade->getTxAudio(10.67f);
 *   nade->processRxAudio(audio, 512, 10.67f);
 *   auto messages = nade->getReceivedMessages();
 *
 *   // Cleanup
 *   NadeExternalIO::destroyInstance();
 */
class NadeExternalIO {
public:
    // =========================================================================
    // SINGLETON MANAGEMENT
    // =========================================================================

    /**
     * @brief Get existing singleton instance.
     * @return Shared pointer to instance, or nullptr if not created.
     */
    static std::shared_ptr<NadeExternalIO> getInstance();

    /**
     * @brief Create singleton instance with crypto keys.
     *
     * If instance already exists, returns existing instance (ignores keys).
     *
     * @param localPrivateKey 32-byte X25519 private key
     * @param localPublicKey 32-byte X25519 public key
     * @param remotePublicKey 32-byte peer's X25519 public key
     * @param ndaSampleRate NDA's audio sample rate (default 48000)
     * @param isInitiator true if this side initiates handshake (default true)
     * @return Shared pointer to instance
     */
    static std::shared_ptr<NadeExternalIO> createInstance(
        const std::vector<uint8_t>& localPrivateKey,
        const std::vector<uint8_t>& localPublicKey,
        const std::vector<uint8_t>& remotePublicKey,
        int ndaSampleRate = 48000,
        bool isInitiator = true,  // Deprecated (kept for compatibility)
        bool enableDiscovery = true  // NEW: Auto-start discovery
    );

    /**
     * @brief Destroy singleton instance.
     *
     * Stops worker thread and releases Python resources.
     * Safe to call even if instance doesn't exist.
     */
    static void destroyInstance();

    /**
     * @brief Check if singleton instance exists.
     */
    static bool hasInstance();

    // =========================================================================
    // TEXT MESSAGING API (Thread-safe)
    // =========================================================================

    /**
     * @brief Queue text message for transmission.
     *
     * Called from GUI thread. Message is queued for worker thread to process.
     *
     * @param text UTF-8 text message (max 256 characters)
     * @return Result indicating success or error
     */
    NadeResult sendTextMessage(const std::string& text);

    /**
     * @brief Retrieve all decoded messages.
     *
     * Called from any thread. Returns and clears pending messages.
     *
     * @return Vector of decoded UTF-8 messages
     */
    std::vector<std::string> getReceivedMessages();

    /**
     * @brief Check if currently transmitting.
     * @return true if TX buffer has data
     */
    bool isTransmitting() const;

    // =========================================================================
    // AUDIO PROCESSING (Called from Audio Thread - Non-blocking)
    // =========================================================================

    /**
     * @brief Get FSK audio for transmission.
     *
     * Reads from pre-buffer filled by worker thread.
     * Returns silence if no data available.
     *
     * @param durationMs Buffer duration in milliseconds
     * @return FSK audio samples at NDA sample rate (float32)
     */
    std::vector<float> getTxAudio(float durationMs);

    /**
     * @brief Process received FSK audio.
     *
     * Writes to post-buffer for worker thread to demodulate.
     * Non-blocking - excess samples are dropped if buffer full.
     *
     * @param audio Audio samples at NDA sample rate (float32)
     * @param numSamples Number of samples
     * @param durationMs Buffer duration in milliseconds
     */
    void processRxAudio(const float* audio, size_t numSamples, float durationMs);

    // =========================================================================
    // DIAGNOSTICS
    // =========================================================================

    /**
     * @brief Get RX signal quality [0.0, 1.0].
     */
    float getRxSignalQuality() const;

    /**
     * @brief Check if demodulator is synchronized.
     */
    bool isRxSynchronized() const;

    /**
     * @brief Get current mode ("tx" or "rx").
     */
    std::string getMode() const;

    /**
     * @brief Check if Noise handshake is complete.
     */
    bool isSessionEstablished() const;

    /**
     * @brief Get handshake phase for UI display.
     * @return 0=Idle, 1=Discovering, 2=Handshaking, 3=Established
     */
    int getHandshakePhase() const;

    /**
     * @brief Force handshake with manual role selection (skip discovery).
     * @param isInitiator true for initiator role, false for responder
     */
    void forceHandshake(bool isInitiator);

    /**
     * @brief Restart automatic discovery mode.
     */
    void restartDiscovery();

    /**
     * @brief Get NDA sample rate.
     */
    int getSampleRate() const { return ndaSampleRate_; }

    // =========================================================================
    // CALLBACKS (Optional)
    // =========================================================================

    using TransmitCompleteCallback = std::function<void()>;
    using MessageReceivedCallback = std::function<void(const std::string&)>;

    /**
     * @brief Set callback for when transmission completes.
     */
    void setTransmitCompleteCallback(TransmitCompleteCallback callback);

    /**
     * @brief Set callback for when message is received.
     */
    void setMessageReceivedCallback(MessageReceivedCallback callback);

    // Destructor must be public for shared_ptr
    ~NadeExternalIO();

private:
    // Private constructor - use createInstance()
    NadeExternalIO(
        const std::vector<uint8_t>& localPriv,
        const std::vector<uint8_t>& localPub,
        const std::vector<uint8_t>& remotePub,
        int sampleRate,
        bool isInitiator,
        bool enableDiscovery
    );

    // Non-copyable, non-movable
    NadeExternalIO(const NadeExternalIO&) = delete;
    NadeExternalIO& operator=(const NadeExternalIO&) = delete;
    NadeExternalIO(NadeExternalIO&&) = delete;
    NadeExternalIO& operator=(NadeExternalIO&&) = delete;

    // Worker thread
    void startWorkerThread();
    void stopWorkerThread();
    void workerThreadFunction();

    // Singleton
    static std::shared_ptr<NadeExternalIO> instance_;
    static std::mutex instanceMutex_;

    // Configuration
    int ndaSampleRate_;
    bool isInitiator_;
    bool enableDiscovery_;
    std::vector<uint8_t> localPrivateKey_;
    std::vector<uint8_t> localPublicKey_;
    std::vector<uint8_t> remotePublicKey_;

    // Worker thread
    std::thread workerThread_;
    std::atomic<bool> running_{false};
    std::condition_variable workerCv_;
    std::mutex workerMutex_;

    // Lock-free audio buffers
    std::unique_ptr<AudioRingBuffer> txPreBuffer_;   // Python writes, audio reads
    std::unique_ptr<AudioRingBuffer> rxPostBuffer_;  // Audio writes, Python reads

    // Message queues (mutex protected)
    std::queue<std::string> txMessageQueue_;
    std::queue<std::string> rxMessageQueue_;
    mutable std::mutex messagesMutex_;

    // State
    std::atomic<bool> isTransmitting_{false};
    std::atomic<bool> sessionEstablished_{false};
    std::atomic<int> handshakePhase_{0};

    // Callbacks
    TransmitCompleteCallback transmitCompleteCallback_;
    MessageReceivedCallback messageReceivedCallback_;
    std::mutex callbackMutex_;

    // Python state (accessed from worker thread and potentially UI thread via forceHandshake/restartDiscovery)
    std::atomic<bool> pythonInitialized_{false};
    void* ndaAdapter_ = nullptr;  // py::object* (opaque to avoid pybind11 header in public API)
};

}  // namespace nade
