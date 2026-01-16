#pragma once

#include "audio/AudioBuffer.h"
#include "audio/RingBuffer.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace nda {

enum class AIOCPttMode
{
    HidManual,
    CdcManual,
    VpttAuto
};

struct AIOCTelemetry
{
    bool connected{false};
    bool running{false};
    bool pttAsserted{false};
    bool vpttActive{false};
    bool vcosActive{false};
    uint64_t framesCaptured{0};
    uint64_t framesPlayed{0};
    uint64_t underruns{0};
    uint64_t overruns{0};
    std::string lastMessage;
};

// v2.2: WASAPI device enumeration support
struct WASAPIDeviceInfo
{
    std::string id;           // GUID string for setParameter("device_id")
    std::string friendlyName; // Display name for UI
};

// Enumerate available WASAPI audio devices
// direction: eCapture (microphones) or eRender (speakers)
std::vector<WASAPIDeviceInfo> enumerateWASAPIDevices(int direction);

class AIOCSession
{
public:
    AIOCSession();
    ~AIOCSession();

    // Configuration
    void setSampleRate(int rate);
    void setChannels(int channels);
    void setBufferFrames(int frames);
    void setVolumeIn(float volume);
    void setVolumeOut(float volume);
    void setMuteIn(bool mute);
    void setMuteOut(bool mute);
    void setPttMode(AIOCPttMode mode);
    void setDeviceIds(const std::string& inId, const std::string& outId);
    void setCdcPort(const std::string& port);
    void setVpttThreshold(uint32_t threshold);
    void setVpttHangMs(uint32_t hangMs);
    void setVcosThreshold(uint32_t threshold);
    void setVcosHangMs(uint32_t hangMs);
    void enableLoopback(bool enable);

    // Lifecycle
    bool connect();
    void disconnect();
    bool isConnected() const;
    bool start();
    void stop();

    // PTT
    bool setPttState(bool asserted);
    bool isPttAsserted() const;

    // Audio IO
    bool writePlayback(const AudioBuffer& buffer);
    bool readCapture(AudioBuffer& buffer);
    void pushIncoming(const AudioBuffer& buffer);

    // Telemetry
    AIOCTelemetry getTelemetry() const;

    // v2.2: Event-driven callbacks and ring buffer stats
    void setDataReadyCallback(std::function<void()> callback);
    void setSpaceAvailableCallback(std::function<void()> callback);
    int getCaptureRingBufferAvailable() const;
    int getPlaybackRingBufferAvailable() const;

    // Accessors
    int sampleRate() const;
    int channels() const;
    int bufferFrames() const;
    float volumeIn() const;
    float volumeOut() const;
    bool muteIn() const;
    bool muteOut() const;
    AIOCPttMode pttMode() const;
    uint32_t vpttThreshold() const;
    uint32_t vpttHangMs() const;
    uint32_t vcosThreshold() const;
    uint32_t vcosHangMs() const;
    std::string deviceInId() const;
    std::string deviceOutId() const;
    std::string cdcPort() const;

private:
    void copyToInterleaved(const AudioBuffer& src, std::vector<float>& dest) const;
    void copyFromInterleaved(const std::vector<float>& src, AudioBuffer& dest) const;

    // Internal helpers
    bool openAudioDevices();
    void closeAudioDevices();
    bool initRenderClient();
    bool initCaptureClient();
    bool ensureComInitialized();
    void teardownCom();
    void closeHid();
    void closeCdc();

    // v2.2: Background thread functions
    void captureThreadFunc();
    void playbackThreadFunc();

    // State
    mutable std::mutex mutex_;
    int sampleRate_;
    int channels_;
    int bufferFrames_;
    float volumeIn_;
    float volumeOut_;
    bool muteIn_;
    bool muteOut_;
    AIOCPttMode pttMode_;
    bool connected_;
    bool running_;
    bool pttAsserted_;
    bool loopbackEnabled_;
    std::deque<std::vector<float>> captureQueue_;
    uint64_t framesCaptured_;
    uint64_t framesPlayed_;
    uint64_t underruns_;
    uint64_t overruns_;
    uint32_t vpttThreshold_;
    uint32_t vpttHangMs_;
    uint32_t vcosThreshold_;
    uint32_t vcosHangMs_;
    std::string deviceInId_;
    std::string deviceOutId_;
    std::string cdcPort_;
    std::string lastMessage_;

    // v2.2: Ring buffers for async operation (200ms capacity at 48kHz = 9600 frames)
    RingBuffer captureRingBuffer_;    // AIOC mic → host
    RingBuffer playbackRingBuffer_;   // Host → AIOC speaker

    // v2.2: Background threads (decouple WASAPI from plugin)
    std::thread captureThread_;
    std::thread playbackThread_;
    std::atomic<bool> captureThreadRunning_;
    std::atomic<bool> playbackThreadRunning_;

    // v2.2: Event-driven callbacks
    std::function<void()> dataReadyCallback_;
    std::function<void()> spaceAvailableCallback_;
    int dataReadyThreshold_;
    int spaceAvailableThreshold_;

    // v2.2: Condition variable for playback thread signaling
    std::condition_variable playbackCV_;
    std::mutex playbackMutex_;

    // v2.2: Temporary conversion buffers (planar ↔ interleaved)
    std::vector<std::vector<float>> captureTempBuffer_;
    std::vector<std::vector<float>> playbackTempBuffer_;

    // Platform handles
    void* hidDevice_; // hidapi device handle
    void* cdcHandle_; // HANDLE on Windows, nullptr elsewhere

    // WASAPI handles (Windows)
    void* renderClient_;       // IAudioClient*
    void* captureClient_;      // IAudioClient*
    void* audioRender_;        // IAudioRenderClient*
    void* audioCapture_;       // IAudioCaptureClient*
    void* renderEvent_;        // HANDLE
    void* captureEvent_;       // HANDLE
    bool comInitialized_;
    bool comOwnsCom_;
};

} // namespace nda
