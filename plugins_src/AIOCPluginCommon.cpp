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
    std::lock_guard<std::mutex> lock(mutex_);
    stop();
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
#ifdef _WIN32
    if (captureClient_) {
        static_cast<IAudioClient*>(captureClient_)->Start();
    }
    if (renderClient_) {
        static_cast<IAudioClient*>(renderClient_)->Start();
    }
#endif
    running_ = true;
    lastMessage_ = "Streaming started";
    return true;
}

void AIOCSession::stop()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_) return;
#ifdef _WIN32
    if (captureClient_) {
        static_cast<IAudioClient*>(captureClient_)->Stop();
    }
    if (renderClient_) {
        static_cast<IAudioClient*>(renderClient_)->Stop();
    }
#endif
    pttAsserted_ = false;
    running_ = false;
    lastMessage_ = "Streaming stopped";
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

#ifdef _WIN32
    if (renderClient_ && audioRender_) {
        IAudioClient* client = static_cast<IAudioClient*>(renderClient_);
        IAudioRenderClient* render = static_cast<IAudioRenderClient*>(audioRender_);

        UINT32 padding = 0;
        UINT32 bufferSize = 0;
        if (FAILED(client->GetCurrentPadding(&padding)) ||
            FAILED(client->GetBufferSize(&bufferSize))) {
            lastMessage_ = "Render padding failed";
            return false;
        }

        UINT32 frames = buffer.getFrameCount();
        if (frames > bufferSize - padding) {
            overruns_++;
            lastMessage_ = "Render overrun";
            return false;
        }

        BYTE* data = nullptr;
        if (FAILED(render->GetBuffer(frames, &data))) {
            lastMessage_ = "Render GetBuffer failed";
            return false;
        }

        // Assume float format; if not, we'll convert to 16-bit.
        WAVEFORMATEX* mixFmt = nullptr;
        client->GetMixFormat(&mixFmt);
        bool isFloat = mixFmt && mixFmt->wFormatTag == WAVE_FORMAT_IEEE_FLOAT;
        UINT32 channels = mixFmt ? mixFmt->nChannels : static_cast<UINT32>(buffer.getChannelCount());

        for (UINT32 f = 0; f < frames; ++f) {
            for (UINT32 c = 0; c < channels; ++c) {
                const float* srcCh = buffer.getChannelData(static_cast<int>(c < static_cast<UINT32>(buffer.getChannelCount()) ? c : 0));
                float sample = srcCh ? srcCh[f] * volumeOut_ : 0.0f;
                if (muteOut_) sample = 0.0f;
                if (isFloat) {
                    reinterpret_cast<float*>(data)[f * channels + c] = sample;
                } else {
                    int16_t s = static_cast<int16_t>(std::max(-1.0f, std::min(1.0f, sample)) * 32767.0f);
                    reinterpret_cast<int16_t*>(data)[f * channels + c] = s;
                }
            }
        }

        if (mixFmt) CoTaskMemFree(mixFmt);
        render->ReleaseBuffer(frames, 0);
        framesPlayed_ += frames;
        return true;
    }
#endif

    // Loopback-only path
    if (loopbackEnabled_) {
        std::vector<float> interleaved;
        copyToInterleaved(buffer, interleaved);
        if (captureQueue_.size() >= kMaxQueuedBuffers) {
            captureQueue_.pop_front();
            ++overruns_;
        }
        captureQueue_.push_back(std::move(interleaved));
        framesPlayed_ += static_cast<uint64_t>(buffer.getFrameCount());
        return true;
    }

    return false;
}

bool AIOCSession::readCapture(AudioBuffer& buffer)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_) {
        buffer.clear();
        lastMessage_ = "Capture rejected: not running";
        return false;
    }

#ifdef _WIN32
    if (captureClient_ && audioCapture_) {
        IAudioCaptureClient* cap = static_cast<IAudioCaptureClient*>(audioCapture_);
        IAudioClient* client = static_cast<IAudioClient*>(captureClient_);

        UINT32 packetFrames = 0;
        BYTE* data = nullptr;
        DWORD flags = 0;

        HRESULT hr = cap->GetBuffer(&data, &packetFrames, &flags, nullptr, nullptr);
        if (hr == AUDCLNT_S_BUFFER_EMPTY) {
            buffer.clear();
            ++underruns_;
            return true; // silence to keep pipeline moving
        }
        if (FAILED(hr)) {
            lastMessage_ = "Capture GetBuffer failed";
            buffer.clear();
            return false;
        }

        WAVEFORMATEX* mixFmt = nullptr;
        client->GetMixFormat(&mixFmt);
        bool isFloat = mixFmt && mixFmt->wFormatTag == WAVE_FORMAT_IEEE_FLOAT;
        UINT32 channels = mixFmt ? mixFmt->nChannels : static_cast<UINT32>(buffer.getChannelCount());

        // Resize buffer if needed
        if (buffer.getChannelCount() != static_cast<int>(channels) || buffer.getFrameCount() != static_cast<int>(packetFrames)) {
            buffer.resize(static_cast<int>(channels), static_cast<int>(packetFrames));
        }

        for (UINT32 f = 0; f < packetFrames; ++f) {
            for (UINT32 c = 0; c < channels; ++c) {
                float* dst = buffer.getChannelData(static_cast<int>(c));
                if (isFloat) {
                    dst[f] = reinterpret_cast<float*>(data)[f * channels + c];
                } else {
                    int16_t s = reinterpret_cast<int16_t*>(data)[f * channels + c];
                    dst[f] = static_cast<float>(s) / 32768.0f;
                }
            }
        }

        cap->ReleaseBuffer(packetFrames);
        if (mixFmt) CoTaskMemFree(mixFmt);

        // Apply volume/mute
        if (muteIn_ || volumeIn_ != 1.0f) {
            float gain = muteIn_ ? 0.0f : volumeIn_;
            int frames = buffer.getFrameCount();
            int chans = buffer.getChannelCount();
            for (int ch = 0; ch < chans; ++ch) {
                float* channelData = buffer.getChannelData(ch);
                for (int i = 0; i < frames; ++i) {
                    channelData[i] *= gain;
                }
            }
        }

        framesCaptured_ += packetFrames;
        return true;
    }
#endif

    // Loopback path
    if (!captureQueue_.empty()) {
        auto interleaved = std::move(captureQueue_.front());
        captureQueue_.pop_front();
        copyFromInterleaved(interleaved, buffer);
        framesCaptured_ += static_cast<uint64_t>(buffer.getFrameCount());
        return true;
    }

    buffer.clear();
    ++underruns_;
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

} // namespace nda
