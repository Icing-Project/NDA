#include "AIOCPluginCommon.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <functiondiscoverykeys_devpkey.h>
#include <avrt.h>
#include <setupapi.h>
#include <hidsdi.h>
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "avrt.lib")
#pragma comment(lib, "setupapi.lib")
#pragma comment(lib, "hid.lib")
#endif

#ifdef _WIN32
#include <initguid.h>
#endif

namespace nda {

#ifdef _WIN32
// Helper to convert wstring to UTF-8 string (avoids wchar_t to char truncation warnings)
static std::string wstringToUtf8(const std::wstring& wstr)
{
    if (wstr.empty()) return std::string();
    int sizeNeeded = WideCharToMultiByte(CP_UTF8, 0, wstr.data(),
                                          static_cast<int>(wstr.size()),
                                          nullptr, 0, nullptr, nullptr);
    if (sizeNeeded <= 0) return std::string();
    std::string result(sizeNeeded, 0);
    WideCharToMultiByte(CP_UTF8, 0, wstr.data(),
                        static_cast<int>(wstr.size()),
                        &result[0], sizeNeeded, nullptr, nullptr);
    return result;
}
#endif

static constexpr unsigned int kDefaultVid = 0x1209;
static constexpr unsigned int kDefaultPid = 0x7388;
static constexpr size_t kMaxQueuedBuffers = 32;

AIOCSession::AIOCSession()
    : sampleRate_(48000),
      channels_(1),
      bufferFrames_(512),
      volumeIn_(1.0f),
      volumeOut_(1.0f),
      muteIn_(false),
      muteOut_(false),
      pttMode_(AIOCPttMode::HidManual),
      connected_(false),
      running_(false),
      pttAsserted_(false),
      loopbackEnabled_(false),
      framesCaptured_(0),
      framesPlayed_(0),
      underruns_(0),
      overruns_(0),
      vpttThreshold_(0x00000040), // conservative defaults
      vpttHangMs_(200),
      vcosThreshold_(0x00000020),
      vcosHangMs_(200),
      captureThreadRunning_(false),
      playbackThreadRunning_(false),
      dataReadyThreshold_(512),
      spaceAvailableThreshold_(512),
      hidDevice_(nullptr),
      cdcHandle_(nullptr),
      renderClient_(nullptr),
      captureClient_(nullptr),
      audioRender_(nullptr),
      audioCapture_(nullptr),
      renderEvent_(nullptr),
      captureEvent_(nullptr),
      comInitialized_(false),
      comOwnsCom_(false)
{
}

AIOCSession::~AIOCSession()
{
    disconnect();
}

void AIOCSession::setSampleRate(int rate) { std::lock_guard<std::mutex> lock(mutex_); sampleRate_ = rate; }
void AIOCSession::setChannels(int channels) { std::lock_guard<std::mutex> lock(mutex_); channels_ = channels; }
void AIOCSession::setBufferFrames(int frames) { std::lock_guard<std::mutex> lock(mutex_); bufferFrames_ = frames; }
void AIOCSession::setVolumeIn(float volume) { std::lock_guard<std::mutex> lock(mutex_); volumeIn_ = volume; }
void AIOCSession::setVolumeOut(float volume) { std::lock_guard<std::mutex> lock(mutex_); volumeOut_ = volume; }
void AIOCSession::setMuteIn(bool mute) { std::lock_guard<std::mutex> lock(mutex_); muteIn_ = mute; }
void AIOCSession::setMuteOut(bool mute) { std::lock_guard<std::mutex> lock(mutex_); muteOut_ = mute; }
void AIOCSession::setPttMode(AIOCPttMode mode) { std::lock_guard<std::mutex> lock(mutex_); pttMode_ = mode; }
void AIOCSession::setDeviceIds(const std::string& inId, const std::string& outId) { std::lock_guard<std::mutex> lock(mutex_); deviceInId_ = inId; deviceOutId_ = outId; }
void AIOCSession::setCdcPort(const std::string& port) { std::lock_guard<std::mutex> lock(mutex_); cdcPort_ = port; }
void AIOCSession::setVpttThreshold(uint32_t threshold) { std::lock_guard<std::mutex> lock(mutex_); vpttThreshold_ = threshold; }
void AIOCSession::setVpttHangMs(uint32_t hangMs) { std::lock_guard<std::mutex> lock(mutex_); vpttHangMs_ = hangMs; }
void AIOCSession::setVcosThreshold(uint32_t threshold) { std::lock_guard<std::mutex> lock(mutex_); vcosThreshold_ = threshold; }
void AIOCSession::setVcosHangMs(uint32_t hangMs) { std::lock_guard<std::mutex> lock(mutex_); vcosHangMs_ = hangMs; }
void AIOCSession::enableLoopback(bool enable) { std::lock_guard<std::mutex> lock(mutex_); loopbackEnabled_ = enable; }

// v2.2: Callback setters
void AIOCSession::setDataReadyCallback(std::function<void()> callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    dataReadyCallback_ = callback;
}

void AIOCSession::setSpaceAvailableCallback(std::function<void()> callback) {
    std::lock_guard<std::mutex> lock(playbackMutex_);
    spaceAvailableCallback_ = callback;
}

// v2.2: Ring buffer stats
int AIOCSession::getCaptureRingBufferAvailable() const {
    return captureRingBuffer_.getAvailableRead();
}

int AIOCSession::getPlaybackRingBufferAvailable() const {
    return playbackRingBuffer_.getAvailableWrite();
}

bool AIOCSession::ensureComInitialized()
{
#ifdef _WIN32
    if (!comInitialized_) {
        HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        if (hr == RPC_E_CHANGED_MODE) {
            // Already initialized in different apartment; treat as success but do not uninitialize.
            comInitialized_ = true;
            comOwnsCom_ = false;
        } else {
            comInitialized_ = SUCCEEDED(hr);
            comOwnsCom_ = comInitialized_;
        }
    }
    return comInitialized_;
#else
    return true;
#endif
}

void AIOCSession::teardownCom()
{
#ifdef _WIN32
    if (comInitialized_ && comOwnsCom_) {
        CoUninitialize();
    }
    comInitialized_ = false;
    comOwnsCom_ = false;
#endif
}

bool AIOCSession::connect()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (connected_) return true;

    if (!ensureComInitialized()) {
        lastMessage_ = "COM init failed";
        return false;
    }

#ifdef _WIN32
    // HID open (VID/PID match)
    GUID hidGuid;
    HidD_GetHidGuid(&hidGuid);
    HDEVINFO deviceInfo = SetupDiGetClassDevs(&hidGuid, nullptr, nullptr, DIGCF_PRESENT | DIGCF_DEVICEINTERFACE);
    if (deviceInfo != INVALID_HANDLE_VALUE) {
        SP_DEVICE_INTERFACE_DATA interfaceData;
        interfaceData.cbSize = sizeof(SP_DEVICE_INTERFACE_DATA);
        for (DWORD idx = 0; SetupDiEnumDeviceInterfaces(deviceInfo, nullptr, &hidGuid, idx, &interfaceData); ++idx) {
            DWORD required = 0;
            SetupDiGetDeviceInterfaceDetail(deviceInfo, &interfaceData, nullptr, 0, &required, nullptr);
            std::vector<char> buffer(required);
            auto detail = reinterpret_cast<PSP_DEVICE_INTERFACE_DETAIL_DATA>(buffer.data());
            detail->cbSize = sizeof(SP_DEVICE_INTERFACE_DETAIL_DATA);
            if (!SetupDiGetDeviceInterfaceDetail(deviceInfo, &interfaceData, detail, required, nullptr, nullptr)) {
                continue;
            }

            HANDLE h = CreateFile(detail->DevicePath, GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                                  OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (h == INVALID_HANDLE_VALUE) {
                continue;
            }

            HIDD_ATTRIBUTES attrs;
            attrs.Size = sizeof(attrs);
            if (HidD_GetAttributes(h, &attrs) && attrs.VendorID == kDefaultVid && attrs.ProductID == kDefaultPid) {
                hidDevice_ = h;
                break;
            }

            CloseHandle(h);
        }
        SetupDiDestroyDeviceInfoList(deviceInfo);
    }
#endif

    // CDC open (optional, requires port set)
#ifdef _WIN32
    if (!cdcPort_.empty()) {
        std::wstring path = L"\\\\.\\" + std::wstring(cdcPort_.begin(), cdcPort_.end());
        HANDLE h = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
        if (h != INVALID_HANDLE_VALUE) {
            cdcHandle_ = h;
        } else {
            lastMessage_ = "CDC open failed";
        }
    }
#endif

    if (!openAudioDevices()) {
        lastMessage_ = "Audio device open failed";
        closeHid();
        closeCdc();
        return false;
    }

    connected_ = true;
    lastMessage_ = "Connected";
    return true;
}

void AIOCSession::disconnect()
{
    // v2.2: Call stop() BEFORE locking to avoid recursive mutex deadlock
    // stop() has its own locking - safe to call first
    stop();

    std::lock_guard<std::mutex> lock(mutex_);
    closeAudioDevices();
    closeHid();
    closeCdc();
    teardownCom();
    connected_ = false;
    running_ = false;
    pttAsserted_ = false;
    captureQueue_.clear();
    lastMessage_ = "Disconnected";
}

bool AIOCSession::isConnected() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return connected_;
}

bool AIOCSession::start()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!connected_) {
        lastMessage_ = "Start failed: not connected";
        return false;
    }

    // v2.2: Initialize ring buffers (200ms capacity at 48kHz = 9600 frames)
    int ringBufferCapacity = (sampleRate_ * 200) / 1000;
    captureRingBuffer_.initialize(channels_, ringBufferCapacity);
    playbackRingBuffer_.initialize(channels_, ringBufferCapacity);

#ifdef _WIN32
    if (captureClient_) {
        static_cast<IAudioClient*>(captureClient_)->Start();
    }
    if (renderClient_) {
        static_cast<IAudioClient*>(renderClient_)->Start();
    }
#endif

    running_ = true;

    // v2.2: Start background threads AFTER WASAPI is running
    captureThreadRunning_.store(true, std::memory_order_release);
    playbackThreadRunning_.store(true, std::memory_order_release);

    captureThread_ = std::thread([this]() { this->captureThreadFunc(); });
    playbackThread_ = std::thread([this]() { this->playbackThreadFunc(); });

    lastMessage_ = "Streaming started with ring buffers";
    return true;
}

void AIOCSession::stop()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!running_) return;
        running_ = false;  // Set this early so threads can see it
    }

    // v2.2: Signal background threads to stop
    captureThreadRunning_.store(false, std::memory_order_release);
    playbackThreadRunning_.store(false, std::memory_order_release);
    playbackCV_.notify_all();  // Wake playback thread if waiting

    // Join threads WITHOUT holding mutex (avoid deadlock)
    if (captureThread_.joinable()) {
        captureThread_.join();
    }
    if (playbackThread_.joinable()) {
        playbackThread_.join();
    }

    // Now stop WASAPI and clean up (reacquire mutex)
    {
        std::lock_guard<std::mutex> lock(mutex_);
#ifdef _WIN32
        if (captureClient_) {
            static_cast<IAudioClient*>(captureClient_)->Stop();
        }
        if (renderClient_) {
            static_cast<IAudioClient*>(renderClient_)->Stop();
        }
#endif
        // Clear ring buffers
        captureRingBuffer_.clear();
        playbackRingBuffer_.clear();

        pttAsserted_ = false;
        lastMessage_ = "Streaming stopped";
    }
}

bool AIOCSession::setPttState(bool asserted)
{
    std::lock_guard<std::mutex> lock(mutex_);
    pttAsserted_ = asserted;

#ifdef _WIN32
    if (pttMode_ == AIOCPttMode::HidManual && hidDevice_) {
        BYTE report[4] = {0};
        report[0] = asserted ? 0x01 : 0x00;
        DWORD written = 0;
        WriteFile(static_cast<HANDLE>(hidDevice_), report, sizeof(report), &written, nullptr);
    }

    if (pttMode_ == AIOCPttMode::CdcManual && cdcHandle_) {
        if (asserted) {
            EscapeCommFunction(static_cast<HANDLE>(cdcHandle_), SETDTR);
            EscapeCommFunction(static_cast<HANDLE>(cdcHandle_), CLRRTS);
        } else {
            EscapeCommFunction(static_cast<HANDLE>(cdcHandle_), CLRDTR);
            EscapeCommFunction(static_cast<HANDLE>(cdcHandle_), CLRRTS);
        }
    }
#endif

    lastMessage_ = asserted ? "PTT asserted" : "PTT released";
    return true;
}

bool AIOCSession::isPttAsserted() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return pttAsserted_;
}

bool AIOCSession::writePlayback(const AudioBuffer& buffer)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_) {
        lastMessage_ = "Playback rejected: not running";
        return false;
    }

    // v2.2: Non-blocking write to ring buffer (background playback thread drains to WASAPI)
    int availableSpace = playbackRingBuffer_.getAvailableWrite();
    int frames = buffer.getFrameCount();
    int channels = buffer.getChannelCount();

    if (availableSpace < frames) {
        ++overruns_;
        lastMessage_ = "Playback ring buffer full";
        return false;  // Signal backpressure to plugin
    }

    // Prepare channel pointers (apply volume/mute first)
    std::vector<std::vector<float>> processedBuffer(channels);
    for (int ch = 0; ch < channels; ++ch) {
        processedBuffer[ch].resize(frames);
        const float* srcData = buffer.getChannelData(ch);
        float gain = muteOut_ ? 0.0f : volumeOut_;

        for (int f = 0; f < frames; ++f) {
            processedBuffer[ch][f] = srcData[f] * gain;
        }
    }

    // Write to ring buffer
    std::vector<const float*> channelPtrs(channels);
    for (int ch = 0; ch < channels; ++ch) {
        channelPtrs[ch] = processedBuffer[ch].data();
    }

    int written = playbackRingBuffer_.write(channelPtrs.data(), frames);

    // Wake playback thread
    playbackCV_.notify_one();

    if (written < frames) {
        ++overruns_;
        lastMessage_ = "Partial playback write";
        return false;
    }

    return true;
}

bool AIOCSession::readCapture(AudioBuffer& buffer)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_) {
        buffer.clear();
        lastMessage_ = "Capture rejected: not running";
        return false;
    }

    // v2.2: Non-blocking read from ring buffer (background capture thread fills from WASAPI)
    int requestedFrames = buffer.getFrameCount();
    int channels = buffer.getChannelCount();

    // Prepare channel pointers for reading
    std::vector<float*> channelPtrs(channels);
    for (int ch = 0; ch < channels; ++ch) {
        channelPtrs[ch] = buffer.getChannelData(ch);
    }

    int framesRead = captureRingBuffer_.read(channelPtrs.data(), requestedFrames);

    // Pad with silence on underrun (pipeline keeps moving)
    if (framesRead < requestedFrames) {
        for (int ch = 0; ch < channels; ++ch) {
            float* data = buffer.getChannelData(ch);
            std::fill(data + framesRead, data + requestedFrames, 0.0f);
        }
        ++underruns_;
    }

    // Apply volume/mute
    if (muteIn_ || volumeIn_ != 1.0f) {
        float gain = muteIn_ ? 0.0f : volumeIn_;
        for (int ch = 0; ch < channels; ++ch) {
            float* channelData = buffer.getChannelData(ch);
            for (int i = 0; i < requestedFrames; ++i) {
                channelData[i] *= gain;
            }
        }
    }

    return true;
}

void AIOCSession::pushIncoming(const AudioBuffer& buffer)
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<float> interleaved;
    copyToInterleaved(buffer, interleaved);
    if (captureQueue_.size() >= kMaxQueuedBuffers) {
        captureQueue_.pop_front();
        ++overruns_;
    }
    captureQueue_.push_back(std::move(interleaved));
}

AIOCTelemetry AIOCSession::getTelemetry() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    AIOCTelemetry t;
    t.connected = connected_;
    t.running = running_;
    t.pttAsserted = pttAsserted_;
    t.vpttActive = (pttMode_ == AIOCPttMode::VpttAuto);
    t.vcosActive = vcosThreshold_ > 0;
    t.framesCaptured = framesCaptured_;
    t.framesPlayed = framesPlayed_;
    t.underruns = underruns_;
    t.overruns = overruns_;
    t.lastMessage = lastMessage_;
    return t;
}

// v2.2: Background capture thread (AIOC microphone → ring buffer)
void AIOCSession::captureThreadFunc()
{
#ifdef _WIN32
    // Initialize COM for this thread
    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    bool threadOwnsCom = SUCCEEDED(hr);

    // Set thread priority to time-critical for low-latency audio
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);

    std::cerr << "[AIOCSession] Capture thread started\n";

    while (captureThreadRunning_.load(std::memory_order_acquire)) {
        // v2.2 fix: Check state and get pointers with minimal mutex hold time
        bool shouldSleep = false;
        int sleepMs = 1;
        IAudioCaptureClient* pCapture = nullptr;
        int channels = 0;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!running_ || !captureClient_) {
                shouldSleep = true;
                sleepMs = 10;
            } else {
                pCapture = static_cast<IAudioCaptureClient*>(audioCapture_);
                channels = channels_;
                if (!pCapture) {
                    shouldSleep = true;
                    sleepMs = 10;
                }
            }
        }

        // Sleep OUTSIDE mutex to avoid blocking readCapture()
        if (shouldSleep) {
            std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
            continue;
        }

        // WASAPI operations (these don't need our mutex - WASAPI is thread-safe)
        UINT32 packetLength = 0;
        hr = pCapture->GetNextPacketSize(&packetLength);
        if (FAILED(hr) || packetLength == 0) {
            // No data available - sleep outside mutex
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        BYTE* pData = nullptr;
        UINT32 numFramesAvailable = 0;
        DWORD flags = 0;

        hr = pCapture->GetBuffer(&pData, &numFramesAvailable, &flags, nullptr, nullptr);
        if (FAILED(hr)) {
            continue;
        }

        if (numFramesAvailable > 0 && pData) {
            // Allocate temp buffer if needed (thread-local, no mutex needed)
            if (captureTempBuffer_.size() != static_cast<size_t>(channels) ||
                captureTempBuffer_[0].size() < numFramesAvailable) {
                captureTempBuffer_.resize(channels);
                for (auto& ch : captureTempBuffer_) {
                    ch.resize(numFramesAvailable);
                }
            }

            // Convert interleaved float32 from WASAPI to planar
            float* pFloat = reinterpret_cast<float*>(pData);
            for (UINT32 frame = 0; frame < numFramesAvailable; ++frame) {
                for (int ch = 0; ch < channels; ++ch) {
                    captureTempBuffer_[ch][frame] = pFloat[frame * channels + ch];
                }
            }

            // Write to ring buffer (lock-free, no mutex needed)
            std::vector<const float*> channelPtrs(channels);
            for (int ch = 0; ch < channels; ++ch) {
                channelPtrs[ch] = captureTempBuffer_[ch].data();
            }

            int framesWritten = captureRingBuffer_.write(channelPtrs.data(), numFramesAvailable);
            if (framesWritten < static_cast<int>(numFramesAvailable)) {
                // Ring buffer full - this is an overrun
            }

            // Update stats with mutex
            {
                std::lock_guard<std::mutex> lock(mutex_);
                framesCaptured_ += numFramesAvailable;
            }
        }

        pCapture->ReleaseBuffer(numFramesAvailable);

        // Check if we should trigger data-ready callback
        int availableFrames = captureRingBuffer_.getAvailableRead();
        if (availableFrames >= dataReadyThreshold_ && dataReadyCallback_) {
            dataReadyCallback_();
        }

        std::this_thread::yield();
    }

    std::cerr << "[AIOCSession] Capture thread stopped\n";

    if (threadOwnsCom) {
        CoUninitialize();
    }
#endif
}

// v2.2: Background playback thread (ring buffer → AIOC speaker)
void AIOCSession::playbackThreadFunc()
{
#ifdef _WIN32
    // Initialize COM for this thread
    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    bool threadOwnsCom = SUCCEEDED(hr);

    // Set thread priority to time-critical
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);

    std::cerr << "[AIOCSession] Playback thread started\n";

    while (playbackThreadRunning_.load(std::memory_order_acquire)) {
        // v2.2 fix: Check state and get pointers with minimal mutex hold time
        bool shouldSleep = false;
        int sleepMs = 1;
        IAudioClient* pClient = nullptr;
        IAudioRenderClient* pRender = nullptr;
        int channels = 0;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!running_ || !renderClient_) {
                shouldSleep = true;
                sleepMs = 10;
            } else {
                pClient = static_cast<IAudioClient*>(renderClient_);
                pRender = static_cast<IAudioRenderClient*>(audioRender_);
                channels = channels_;
                if (!pClient || !pRender) {
                    shouldSleep = true;
                    sleepMs = 10;
                }
            }
        }

        // Sleep OUTSIDE mutex to avoid blocking writePlayback()
        if (shouldSleep) {
            std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
            continue;
        }

        // WASAPI operations (these don't need our mutex - WASAPI is thread-safe)
        UINT32 padding = 0;
        hr = pClient->GetCurrentPadding(&padding);
        if (FAILED(hr)) {
            continue;
        }

        UINT32 bufferSize = 0;
        hr = pClient->GetBufferSize(&bufferSize);
        if (FAILED(hr)) {
            continue;
        }

        // Target: keep buffer ~50% full, minimum write threshold 20%
        UINT32 targetPadding = bufferSize / 2;
        UINT32 minWriteThreshold = bufferSize / 5;

        if (padding >= targetPadding) {
            // Buffer is full enough - sleep outside mutex
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        UINT32 availableFrames = bufferSize - padding;
        if (availableFrames < minWriteThreshold) {
            // Not enough space - sleep outside mutex
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }

        // Check ring buffer (lock-free)
        int ringBufferFrames = playbackRingBuffer_.getAvailableRead();
        if (ringBufferFrames == 0) {
            // No data - wait on CV (uses separate playbackMutex_)
            std::unique_lock<std::mutex> cvLock(playbackMutex_);
            playbackCV_.wait_for(cvLock, std::chrono::milliseconds(10));
            continue;
        }

        // Write as much as we can
        UINT32 framesToWrite = std::min(availableFrames, static_cast<UINT32>(ringBufferFrames));

        // Allocate temp buffer if needed (thread-local)
        if (playbackTempBuffer_.size() != static_cast<size_t>(channels) ||
            playbackTempBuffer_[0].size() < framesToWrite) {
            playbackTempBuffer_.resize(channels);
            for (auto& ch : playbackTempBuffer_) {
                ch.resize(framesToWrite);
            }
        }

        // Read from ring buffer (lock-free)
        std::vector<float*> channelPtrs(channels);
        for (int ch = 0; ch < channels; ++ch) {
            channelPtrs[ch] = playbackTempBuffer_[ch].data();
        }

        int framesRead = playbackRingBuffer_.read(channelPtrs.data(), framesToWrite);
        if (framesRead > 0) {
            BYTE* pData = nullptr;
            hr = pRender->GetBuffer(framesRead, &pData);
            if (SUCCEEDED(hr) && pData) {
                float* pFloat = reinterpret_cast<float*>(pData);
                for (int frame = 0; frame < framesRead; ++frame) {
                    for (int ch = 0; ch < channels; ++ch) {
                        pFloat[frame * channels + ch] = playbackTempBuffer_[ch][frame];
                    }
                }
                pRender->ReleaseBuffer(framesRead, 0);

                // Update stats with mutex
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    framesPlayed_ += framesRead;
                }
            }
        }

        // Check if we should trigger space-available callback
        int availableSpace = playbackRingBuffer_.getAvailableWrite();
        if (availableSpace >= spaceAvailableThreshold_ && spaceAvailableCallback_) {
            spaceAvailableCallback_();
        }

        std::this_thread::yield();
    }

    std::cerr << "[AIOCSession] Playback thread stopped\n";

    if (threadOwnsCom) {
        CoUninitialize();
    }
#endif
}

int AIOCSession::sampleRate() const { std::lock_guard<std::mutex> lock(mutex_); return sampleRate_; }
int AIOCSession::channels() const { std::lock_guard<std::mutex> lock(mutex_); return channels_; }
int AIOCSession::bufferFrames() const { std::lock_guard<std::mutex> lock(mutex_); return bufferFrames_; }
float AIOCSession::volumeIn() const { std::lock_guard<std::mutex> lock(mutex_); return volumeIn_; }
float AIOCSession::volumeOut() const { std::lock_guard<std::mutex> lock(mutex_); return volumeOut_; }
bool AIOCSession::muteIn() const { std::lock_guard<std::mutex> lock(mutex_); return muteIn_; }
bool AIOCSession::muteOut() const { std::lock_guard<std::mutex> lock(mutex_); return muteOut_; }
AIOCPttMode AIOCSession::pttMode() const { std::lock_guard<std::mutex> lock(mutex_); return pttMode_; }
uint32_t AIOCSession::vpttThreshold() const { std::lock_guard<std::mutex> lock(mutex_); return vpttThreshold_; }
uint32_t AIOCSession::vpttHangMs() const { std::lock_guard<std::mutex> lock(mutex_); return vpttHangMs_; }
uint32_t AIOCSession::vcosThreshold() const { std::lock_guard<std::mutex> lock(mutex_); return vcosThreshold_; }
uint32_t AIOCSession::vcosHangMs() const { std::lock_guard<std::mutex> lock(mutex_); return vcosHangMs_; }
std::string AIOCSession::deviceInId() const { std::lock_guard<std::mutex> lock(mutex_); return deviceInId_; }
std::string AIOCSession::deviceOutId() const { std::lock_guard<std::mutex> lock(mutex_); return deviceOutId_; }
std::string AIOCSession::cdcPort() const { std::lock_guard<std::mutex> lock(mutex_); return cdcPort_; }

void AIOCSession::copyToInterleaved(const AudioBuffer& src, std::vector<float>& dest) const
{
    int frames = src.getFrameCount();
    int chans = src.getChannelCount();
    dest.resize(frames * chans);
    for (int frame = 0; frame < frames; ++frame) {
        for (int ch = 0; ch < chans; ++ch) {
            const float* channelData = src.getChannelData(ch);
            dest[frame * chans + ch] = channelData ? channelData[frame] : 0.0f;
        }
    }
}

void AIOCSession::copyFromInterleaved(const std::vector<float>& src, AudioBuffer& dest) const
{
    int chans = dest.getChannelCount();
    int frames = dest.getFrameCount();
    int stride = chans;

    if (static_cast<int>(src.size()) < frames * stride) {
        dest.clear();
        return;
    }

    for (int frame = 0; frame < frames; ++frame) {
        for (int ch = 0; ch < chans; ++ch) {
            float* channelData = dest.getChannelData(ch);
            channelData[frame] = src[frame * stride + ch];
        }
    }
}

bool AIOCSession::openAudioDevices()
{
#ifdef _WIN32
    if (!ensureComInitialized()) return false;

    closeAudioDevices();

    if (!initRenderClient()) return false;
    if (!initCaptureClient()) return false;
    return true;
#else
    return false;
#endif
}

void AIOCSession::closeAudioDevices()
{
#ifdef _WIN32
    if (audioRender_) { static_cast<IAudioRenderClient*>(audioRender_)->Release(); audioRender_ = nullptr; }
    if (audioCapture_) { static_cast<IAudioCaptureClient*>(audioCapture_)->Release(); audioCapture_ = nullptr; }
    if (renderClient_) { static_cast<IAudioClient*>(renderClient_)->Release(); renderClient_ = nullptr; }
    if (captureClient_) { static_cast<IAudioClient*>(captureClient_)->Release(); captureClient_ = nullptr; }
    if (renderEvent_) { CloseHandle(static_cast<HANDLE>(renderEvent_)); renderEvent_ = nullptr; }
    if (captureEvent_) { CloseHandle(static_cast<HANDLE>(captureEvent_)); captureEvent_ = nullptr; }
#endif
}

bool AIOCSession::initRenderClient()
{
#ifdef _WIN32
    IMMDeviceEnumerator* enumerator = nullptr;
    IMMDevice* device = nullptr;
    HRESULT hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, __uuidof(IMMDeviceEnumerator), (void**)&enumerator);
    if (FAILED(hr) || !enumerator) return false;

    if (!deviceOutId_.empty()) {
        std::wstring wid(deviceOutId_.begin(), deviceOutId_.end());
        hr = enumerator->GetDevice(wid.c_str(), &device);
    } else {
        hr = enumerator->GetDefaultAudioEndpoint(eRender, eConsole, &device);
    }
    enumerator->Release();
    if (FAILED(hr) || !device) return false;

    hr = device->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, &renderClient_);
    device->Release();
    if (FAILED(hr) || !renderClient_) return false;

    IAudioClient* client = static_cast<IAudioClient*>(renderClient_);
    WAVEFORMATEX* mix = nullptr;
    client->GetMixFormat(&mix);
    if (!mix) return false;

    WAVEFORMATEX original = *mix;
    mix->nSamplesPerSec = static_cast<DWORD>(sampleRate_);
    mix->nChannels = static_cast<WORD>(channels_);
    mix->nBlockAlign = (mix->wBitsPerSample / 8) * mix->nChannels;
    mix->nAvgBytesPerSec = mix->nBlockAlign * mix->nSamplesPerSec;

    REFERENCE_TIME hns = static_cast<REFERENCE_TIME>((10000000.0 * bufferFrames_) / sampleRate_);
    hr = client->Initialize(AUDCLNT_SHAREMODE_SHARED, 0, hns, 0, mix, nullptr);
    if (FAILED(hr)) {
        // Fallback to device mix format
        *mix = original;
        hr = client->Initialize(AUDCLNT_SHAREMODE_SHARED, 0, hns, 0, mix, nullptr);
    }
    CoTaskMemFree(mix);
    if (FAILED(hr)) return false;

    hr = client->GetService(__uuidof(IAudioRenderClient), &audioRender_);
    if (FAILED(hr) || !audioRender_) return false;

    return true;
#else
    return false;
#endif
}

bool AIOCSession::initCaptureClient()
{
#ifdef _WIN32
    IMMDeviceEnumerator* enumerator = nullptr;
    IMMDevice* device = nullptr;
    HRESULT hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, __uuidof(IMMDeviceEnumerator), (void**)&enumerator);
    if (FAILED(hr) || !enumerator) return false;

    if (!deviceInId_.empty()) {
        std::wstring wid(deviceInId_.begin(), deviceInId_.end());
        hr = enumerator->GetDevice(wid.c_str(), &device);
    } else {
        hr = enumerator->GetDefaultAudioEndpoint(eCapture, eConsole, &device);
    }
    enumerator->Release();
    if (FAILED(hr) || !device) return false;

    hr = device->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, &captureClient_);
    device->Release();
    if (FAILED(hr) || !captureClient_) return false;

    IAudioClient* client = static_cast<IAudioClient*>(captureClient_);
    WAVEFORMATEX* mix = nullptr;
    client->GetMixFormat(&mix);
    if (!mix) return false;

    WAVEFORMATEX original = *mix;
    mix->nSamplesPerSec = static_cast<DWORD>(sampleRate_);
    mix->nChannels = static_cast<WORD>(channels_);
    mix->nBlockAlign = (mix->wBitsPerSample / 8) * mix->nChannels;
    mix->nAvgBytesPerSec = mix->nBlockAlign * mix->nSamplesPerSec;

    REFERENCE_TIME hns = static_cast<REFERENCE_TIME>((10000000.0 * bufferFrames_) / sampleRate_);
    hr = client->Initialize(AUDCLNT_SHAREMODE_SHARED, 0, hns, 0, mix, nullptr);
    if (FAILED(hr)) {
        *mix = original;
        hr = client->Initialize(AUDCLNT_SHAREMODE_SHARED, 0, hns, 0, mix, nullptr);
    }
    CoTaskMemFree(mix);
    if (FAILED(hr)) return false;

    hr = client->GetService(__uuidof(IAudioCaptureClient), &audioCapture_);
    if (FAILED(hr) || !audioCapture_) return false;

    return true;
#else
    return false;
#endif
}

void AIOCSession::closeHid()
{
#ifdef _WIN32
    if (hidDevice_) {
        CloseHandle(static_cast<HANDLE>(hidDevice_));
        hidDevice_ = nullptr;
    }
#endif
}

void AIOCSession::closeCdc()
{
#ifdef _WIN32
    if (cdcHandle_) {
        CloseHandle(static_cast<HANDLE>(cdcHandle_));
        cdcHandle_ = nullptr;
    }
#endif
}

// v2.2: WASAPI device enumeration implementation
std::vector<WASAPIDeviceInfo> enumerateWASAPIDevices(int direction)
{
    std::vector<WASAPIDeviceInfo> devices;

#ifdef _WIN32
    // direction: 0 = eCapture (microphones), 1 = eRender (speakers)
    EDataFlow dataFlow = static_cast<EDataFlow>((direction == 0) ? eCapture : eRender);

    // Initialize COM (may already be initialized by Qt, that's OK)
    HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    bool comInitialized = SUCCEEDED(hr);

    // Create device enumerator
    IMMDeviceEnumerator* enumerator = nullptr;
    hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr,
                          CLSCTX_ALL, __uuidof(IMMDeviceEnumerator),
                          (void**)&enumerator);
    if (FAILED(hr)) {
        if (comInitialized) CoUninitialize();
        return devices;
    }

    // Enumerate active audio endpoints
    IMMDeviceCollection* collection = nullptr;
    hr = enumerator->EnumAudioEndpoints(dataFlow, DEVICE_STATE_ACTIVE, &collection);
    if (FAILED(hr)) {
        enumerator->Release();
        if (comInitialized) CoUninitialize();
        return devices;
    }

    UINT count = 0;
    collection->GetCount(&count);

    for (UINT i = 0; i < count; ++i) {
        IMMDevice* device = nullptr;
        if (FAILED(collection->Item(i, &device))) continue;

        // Get device ID (GUID string)
        LPWSTR pwszID = nullptr;
        device->GetId(&pwszID);
        if (pwszID) {
            std::wstring wid(pwszID);
            std::string id = wstringToUtf8(wid);
            CoTaskMemFree(pwszID);

            // Get friendly name from property store
            IPropertyStore* props = nullptr;
            std::string friendlyName = "Unknown Device";
            if (SUCCEEDED(device->OpenPropertyStore(STGM_READ, &props))) {
                PROPVARIANT varName;
                PropVariantInit(&varName);
                if (SUCCEEDED(props->GetValue(PKEY_Device_FriendlyName, &varName))) {
                    if (varName.pwszVal) {
                        std::wstring wname(varName.pwszVal);
                        friendlyName = wstringToUtf8(wname);
                    }
                    PropVariantClear(&varName);
                }
                props->Release();
            }

            WASAPIDeviceInfo info;
            info.id = id;
            info.friendlyName = friendlyName;
            devices.push_back(info);
        }

        device->Release();
    }

    collection->Release();
    enumerator->Release();
    if (comInitialized) CoUninitialize();
#endif

    return devices;
}

} // namespace nda
