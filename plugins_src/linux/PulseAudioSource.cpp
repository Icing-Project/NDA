#include "PulseAudioSource.h"
#include <pulse/error.h>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <thread>
#include <chrono>

namespace nda {

PulseAudioSource::PulseAudioSource()
    : mainloop_(nullptr)
    , context_(nullptr)
    , stream_(nullptr)
    , sampleRate_(48000)
    , channels_(1)
    , bufferSize_(512)
    , state_(PluginState::Unloaded)
    , callback_(nullptr)
    , underrunCount_(0)
    , overrunCount_(0)
{
}

PulseAudioSource::~PulseAudioSource() {
    shutdown();
}

bool PulseAudioSource::initialize() {
    if (state_ != PluginState::Unloaded) {
        return state_ == PluginState::Initialized;
    }

    // Initialize ring buffer
    if (!ringBuffer_.initialize(channels_, RING_BUFFER_FRAMES)) {
        std::cerr << "[PulseAudioSource] Failed to initialize ring buffer\n";
        state_ = PluginState::Error;
        return false;
    }

    // Allocate conversion buffers
    interleavedBuffer_.resize(bufferSize_ * channels_);
    planarPtrs_.resize(channels_);

    // Create threaded mainloop
    mainloop_ = pa_threaded_mainloop_new();
    if (!mainloop_) {
        std::cerr << "[PulseAudioSource] Failed to create threaded mainloop\n";
        state_ = PluginState::Error;
        return false;
    }

    // Create context
    if (!createContext()) {
        pa_threaded_mainloop_free(mainloop_);
        mainloop_ = nullptr;
        state_ = PluginState::Error;
        return false;
    }

    // Start mainloop thread
    if (pa_threaded_mainloop_start(mainloop_) < 0) {
        std::cerr << "[PulseAudioSource] Failed to start mainloop thread\n";
        destroyContext();
        pa_threaded_mainloop_free(mainloop_);
        mainloop_ = nullptr;
        state_ = PluginState::Error;
        return false;
    }

    // Wait for context to be ready
    if (!waitForContextReady()) {
        std::cerr << "[PulseAudioSource] Context failed to connect\n";
        pa_threaded_mainloop_stop(mainloop_);
        destroyContext();
        pa_threaded_mainloop_free(mainloop_);
        mainloop_ = nullptr;
        state_ = PluginState::Error;
        return false;
    }

    state_ = PluginState::Initialized;
    std::cout << "[PulseAudioSource] Initialized - " << sampleRate_ << "Hz, "
              << channels_ << " channels\n";
    return true;
}

void PulseAudioSource::shutdown() {
    if (state_ == PluginState::Unloaded) {
        return;
    }

    stop();

    if (mainloop_) {
        pa_threaded_mainloop_stop(mainloop_);
        destroyContext();
        pa_threaded_mainloop_free(mainloop_);
        mainloop_ = nullptr;
    }

    state_ = PluginState::Unloaded;
    std::cout << "[PulseAudioSource] Shutdown complete\n";
}

bool PulseAudioSource::start() {
    if (state_ != PluginState::Initialized) {
        return false;
    }

    pa_threaded_mainloop_lock(mainloop_);

    // Clear ring buffer
    ringBuffer_.clear();
    underrunCount_ = 0;
    overrunCount_ = 0;

    // Reinitialize ring buffer if channels changed
    if (ringBuffer_.getChannels() != channels_) {
        ringBuffer_.initialize(channels_, RING_BUFFER_FRAMES);
        interleavedBuffer_.resize(bufferSize_ * channels_);
        planarPtrs_.resize(channels_);
    }

    // Create and connect stream
    if (!createStream()) {
        pa_threaded_mainloop_unlock(mainloop_);
        return false;
    }

    pa_threaded_mainloop_unlock(mainloop_);

    // Wait for stream to be ready
    if (!waitForStreamReady()) {
        pa_threaded_mainloop_lock(mainloop_);
        destroyStream();
        pa_threaded_mainloop_unlock(mainloop_);
        return false;
    }

    state_ = PluginState::Running;

    std::string devName = deviceName_.empty() ? "default" : deviceName_;
    std::cout << "[PulseAudioSource] Started - device: " << devName << "\n";
    return true;
}

void PulseAudioSource::stop() {
    if (state_ != PluginState::Running) {
        return;
    }

    pa_threaded_mainloop_lock(mainloop_);
    destroyStream();
    pa_threaded_mainloop_unlock(mainloop_);

    state_ = PluginState::Initialized;

    std::cout << "[PulseAudioSource] Stopped - underruns: " << underrunCount_
              << ", overruns: " << overrunCount_ << "\n";
}

PluginInfo PulseAudioSource::getInfo() const {
    PluginInfo info;
    info.name = "PulseAudio Microphone";
    info.version = "1.0.0";
    info.author = "NDA Project";
    info.description = "Captures audio from microphone via PulseAudio (Linux)";
    info.type = PluginType::AudioSource;
    info.apiVersion = 1;
    return info;
}

PluginState PulseAudioSource::getState() const {
    return state_;
}

void PulseAudioSource::setParameter(const std::string& key, const std::string& value) {
    std::lock_guard<std::mutex> lock(paramMutex_);

    if (key == "device") {
        if (state_ == PluginState::Running) {
            std::cerr << "[PulseAudioSource] Cannot change device while running\n";
            return;
        }
        deviceName_ = value;
    }
    else if (key == "sampleRate") {
        if (state_ == PluginState::Running) {
            std::cerr << "[PulseAudioSource] Cannot change sample rate while running\n";
            return;
        }
        try {
            sampleRate_ = std::stoi(value);
        } catch (...) {
            // Invalid value, ignore
        }
    }
    else if (key == "channels") {
        if (state_ == PluginState::Running) {
            std::cerr << "[PulseAudioSource] Cannot change channels while running\n";
            return;
        }
        try {
            channels_ = std::max(1, std::min(8, std::stoi(value)));
        } catch (...) {
            // Invalid value, ignore
        }
    }
    else if (key == "bufferSize") {
        if (state_ == PluginState::Running) {
            std::cerr << "[PulseAudioSource] Cannot change buffer size while running\n";
            return;
        }
        try {
            bufferSize_ = std::max(64, std::stoi(value));
            interleavedBuffer_.resize(bufferSize_ * channels_);
        } catch (...) {
            // Invalid value, ignore
        }
    }
}

std::string PulseAudioSource::getParameter(const std::string& key) const {
    std::lock_guard<std::mutex> lock(paramMutex_);

    if (key == "device") {
        return deviceName_;
    }
    else if (key == "sampleRate") {
        return std::to_string(sampleRate_);
    }
    else if (key == "channels") {
        return std::to_string(channels_);
    }
    else if (key == "bufferSize") {
        return std::to_string(bufferSize_);
    }
    else if (key == "underruns") {
        return std::to_string(underrunCount_.load());
    }
    else if (key == "overruns") {
        return std::to_string(overrunCount_.load());
    }

    return "";
}

void PulseAudioSource::setAudioCallback(AudioSourceCallback callback) {
    callback_ = callback;
}

bool PulseAudioSource::readAudio(AudioBuffer& buffer) {
    if (state_ != PluginState::Running) {
        buffer.clear();
        return false;
    }

    int framesToRead = buffer.getFrameCount();

    // Wait for enough data with timeout (matching Python plugin behavior)
    // Timeout: 50ms to match latency budget
    static constexpr int TIMEOUT_MS = 50;
    static constexpr int POLL_INTERVAL_MS = 1;
    int waitedMs = 0;

    while (ringBuffer_.getAvailableRead() < framesToRead && waitedMs < TIMEOUT_MS) {
        if (state_ != PluginState::Running) {
            buffer.clear();
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(POLL_INTERVAL_MS));
        waitedMs += POLL_INTERVAL_MS;
    }

    int available = ringBuffer_.getAvailableRead();
    if (available < framesToRead) {
        // Timeout - not enough data
        underrunCount_++;
        buffer.clear();
        return false;
    }

    // Set up planar pointers for AudioBuffer channels
    for (int ch = 0; ch < buffer.getChannelCount(); ++ch) {
        planarPtrs_[ch] = buffer.getChannelData(ch);
    }

    // Read from ring buffer (already in planar format)
    int framesRead = ringBuffer_.read(planarPtrs_.data(), framesToRead);

    if (framesRead < framesToRead) {
        // Partial read - clear remaining frames
        for (int ch = 0; ch < buffer.getChannelCount(); ++ch) {
            std::fill(planarPtrs_[ch] + framesRead,
                      planarPtrs_[ch] + framesToRead, 0.0f);
        }
    }

    return framesRead > 0;
}

int PulseAudioSource::getSampleRate() const {
    return sampleRate_;
}

int PulseAudioSource::getChannels() const {
    return channels_;
}

void PulseAudioSource::setSampleRate(int sampleRate) {
    if (state_ == PluginState::Running) return;
    sampleRate_ = sampleRate;
}

void PulseAudioSource::setChannels(int channels) {
    if (state_ == PluginState::Running) return;
    channels_ = std::max(1, std::min(8, channels));
}

int PulseAudioSource::getBufferSize() const {
    return bufferSize_;
}

void PulseAudioSource::setBufferSize(int samples) {
    if (state_ == PluginState::Running) return;
    bufferSize_ = std::max(64, samples);
    interleavedBuffer_.resize(bufferSize_ * channels_);
}

// ==================== PulseAudio Callbacks ====================

void PulseAudioSource::contextStateCallback(pa_context* c, void* userdata) {
    auto* self = static_cast<PulseAudioSource*>(userdata);
    self->onContextState(c);
}

void PulseAudioSource::streamStateCallback(pa_stream* s, void* userdata) {
    auto* self = static_cast<PulseAudioSource*>(userdata);
    self->onStreamState(s);
}

void PulseAudioSource::streamReadCallback(pa_stream* s, size_t nbytes, void* userdata) {
    auto* self = static_cast<PulseAudioSource*>(userdata);
    self->onStreamRead(s, nbytes);
}

void PulseAudioSource::onContextState(pa_context* c) {
    pa_context_state_t state = pa_context_get_state(c);

    switch (state) {
        case PA_CONTEXT_READY:
        case PA_CONTEXT_FAILED:
        case PA_CONTEXT_TERMINATED:
            pa_threaded_mainloop_signal(mainloop_, 0);
            break;
        default:
            break;
    }
}

void PulseAudioSource::onStreamState(pa_stream* s) {
    pa_stream_state_t state = pa_stream_get_state(s);

    switch (state) {
        case PA_STREAM_READY:
        case PA_STREAM_FAILED:
        case PA_STREAM_TERMINATED:
            pa_threaded_mainloop_signal(mainloop_, 0);
            break;
        default:
            break;
    }
}

void PulseAudioSource::onStreamRead(pa_stream* s, size_t /*nbytes*/) {
    const void* data;
    size_t actualBytes;

    while (pa_stream_peek(s, &data, &actualBytes) >= 0 && actualBytes > 0) {
        if (data == nullptr) {
            // Hole in stream - skip
            pa_stream_drop(s);
            continue;
        }

        // Data is interleaved float32
        const float* interleavedData = static_cast<const float*>(data);
        size_t frames = actualBytes / (sizeof(float) * channels_);

        // Convert interleaved to planar and write to ring buffer
        // We need temporary planar buffers
        std::vector<std::vector<float>> planarData(channels_);
        std::vector<float*> planarPtrs(channels_);

        for (int ch = 0; ch < channels_; ++ch) {
            planarData[ch].resize(frames);
            planarPtrs[ch] = planarData[ch].data();
        }

        // Deinterleave
        for (size_t f = 0; f < frames; ++f) {
            for (int ch = 0; ch < channels_; ++ch) {
                planarPtrs[ch][f] = interleavedData[f * channels_ + ch];
            }
        }

        // Write to ring buffer
        int written = ringBuffer_.write(const_cast<const float**>(planarPtrs.data()),
                                         static_cast<int>(frames));
        if (written < static_cast<int>(frames)) {
            overrunCount_++;
        }

        pa_stream_drop(s);
    }
}

// ==================== Helper Methods ====================

bool PulseAudioSource::createContext() {
    pa_mainloop_api* api = pa_threaded_mainloop_get_api(mainloop_);

    context_ = pa_context_new(api, "NDA PulseAudio Source");
    if (!context_) {
        std::cerr << "[PulseAudioSource] Failed to create context\n";
        return false;
    }

    pa_context_set_state_callback(context_, contextStateCallback, this);

    if (pa_context_connect(context_, nullptr, PA_CONTEXT_NOFLAGS, nullptr) < 0) {
        std::cerr << "[PulseAudioSource] Failed to connect context: "
                  << pa_strerror(pa_context_errno(context_)) << "\n";
        pa_context_unref(context_);
        context_ = nullptr;
        return false;
    }

    return true;
}

bool PulseAudioSource::createStream() {
    // Sample specification
    pa_sample_spec spec;
    spec.format = PA_SAMPLE_FLOAT32LE;
    spec.rate = static_cast<uint32_t>(sampleRate_);
    spec.channels = static_cast<uint8_t>(channels_);

    // Buffer attributes for low latency
    pa_buffer_attr attr;
    attr.maxlength = static_cast<uint32_t>(-1);  // Server decides
    attr.fragsize = static_cast<uint32_t>(bufferSize_ * sizeof(float) * channels_);

    // Create stream
    stream_ = pa_stream_new(context_, "NDA Capture", &spec, nullptr);
    if (!stream_) {
        std::cerr << "[PulseAudioSource] Failed to create stream: "
                  << pa_strerror(pa_context_errno(context_)) << "\n";
        return false;
    }

    pa_stream_set_state_callback(stream_, streamStateCallback, this);
    pa_stream_set_read_callback(stream_, streamReadCallback, this);

    // Connect for recording
    const char* device = deviceName_.empty() ? nullptr : deviceName_.c_str();
    pa_stream_flags_t flags = static_cast<pa_stream_flags_t>(
        PA_STREAM_INTERPOLATE_TIMING |
        PA_STREAM_ADJUST_LATENCY |
        PA_STREAM_AUTO_TIMING_UPDATE
    );

    if (pa_stream_connect_record(stream_, device, &attr, flags) < 0) {
        std::cerr << "[PulseAudioSource] Failed to connect stream: "
                  << pa_strerror(pa_context_errno(context_)) << "\n";
        pa_stream_unref(stream_);
        stream_ = nullptr;
        return false;
    }

    return true;
}

void PulseAudioSource::destroyStream() {
    if (stream_) {
        pa_stream_disconnect(stream_);
        pa_stream_unref(stream_);
        stream_ = nullptr;
    }
}

void PulseAudioSource::destroyContext() {
    if (context_) {
        pa_context_disconnect(context_);
        pa_context_unref(context_);
        context_ = nullptr;
    }
}

bool PulseAudioSource::waitForContextReady() {
    pa_threaded_mainloop_lock(mainloop_);

    while (true) {
        pa_context_state_t state = pa_context_get_state(context_);

        if (state == PA_CONTEXT_READY) {
            pa_threaded_mainloop_unlock(mainloop_);
            return true;
        }

        if (state == PA_CONTEXT_FAILED || state == PA_CONTEXT_TERMINATED) {
            std::cerr << "[PulseAudioSource] Context connection failed: "
                      << pa_strerror(pa_context_errno(context_)) << "\n";
            pa_threaded_mainloop_unlock(mainloop_);
            return false;
        }

        pa_threaded_mainloop_wait(mainloop_);
    }
}

bool PulseAudioSource::waitForStreamReady() {
    pa_threaded_mainloop_lock(mainloop_);

    while (true) {
        pa_stream_state_t state = pa_stream_get_state(stream_);

        if (state == PA_STREAM_READY) {
            pa_threaded_mainloop_unlock(mainloop_);
            return true;
        }

        if (state == PA_STREAM_FAILED || state == PA_STREAM_TERMINATED) {
            std::cerr << "[PulseAudioSource] Stream connection failed: "
                      << pa_strerror(pa_context_errno(context_)) << "\n";
            pa_threaded_mainloop_unlock(mainloop_);
            return false;
        }

        pa_threaded_mainloop_wait(mainloop_);
    }
}

} // namespace nda

// ==================== Plugin Factory ====================

extern "C" {

nda::BasePlugin* createPlugin() {
    return new nda::PulseAudioSource();
}

void destroyPlugin(nda::BasePlugin* plugin) {
    delete plugin;
}

}
