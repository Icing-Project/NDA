#include "PulseAudioSink.h"
#include <pulse/error.h>
#include <iostream>
#include <cstring>
#include <algorithm>

namespace nda {

PulseAudioSink::PulseAudioSink()
    : mainloop_(nullptr)
    , context_(nullptr)
    , stream_(nullptr)
    , sampleRate_(48000)
    , channels_(1)
    , bufferSize_(512)
    , state_(PluginState::Unloaded)
    , underrunCount_(0)
    , overrunCount_(0)
{
}

PulseAudioSink::~PulseAudioSink() {
    shutdown();
}

bool PulseAudioSink::initialize() {
    if (state_ != PluginState::Unloaded) {
        return state_ == PluginState::Initialized;
    }

    // Initialize ring buffer
    if (!ringBuffer_.initialize(channels_, RING_BUFFER_FRAMES)) {
        std::cerr << "[PulseAudioSink] Failed to initialize ring buffer\n";
        state_ = PluginState::Error;
        return false;
    }

    // Allocate conversion buffers
    interleavedBuffer_.resize(bufferSize_ * channels_);
    planarPtrs_.resize(channels_);

    // Create threaded mainloop
    mainloop_ = pa_threaded_mainloop_new();
    if (!mainloop_) {
        std::cerr << "[PulseAudioSink] Failed to create threaded mainloop\n";
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
        std::cerr << "[PulseAudioSink] Failed to start mainloop thread\n";
        destroyContext();
        pa_threaded_mainloop_free(mainloop_);
        mainloop_ = nullptr;
        state_ = PluginState::Error;
        return false;
    }

    // Wait for context to be ready
    if (!waitForContextReady()) {
        std::cerr << "[PulseAudioSink] Context failed to connect\n";
        pa_threaded_mainloop_stop(mainloop_);
        destroyContext();
        pa_threaded_mainloop_free(mainloop_);
        mainloop_ = nullptr;
        state_ = PluginState::Error;
        return false;
    }

    state_ = PluginState::Initialized;
    std::cout << "[PulseAudioSink] Initialized - " << sampleRate_ << "Hz, "
              << channels_ << " channels\n";
    return true;
}

void PulseAudioSink::shutdown() {
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
    std::cout << "[PulseAudioSink] Shutdown complete\n";
}

bool PulseAudioSink::start() {
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

    // Pre-fill with silence to avoid initial underruns
    prefillStream();

    state_ = PluginState::Running;

    std::string devName = deviceName_.empty() ? "default" : deviceName_;
    std::cout << "[PulseAudioSink] Started - device: " << devName << "\n";
    return true;
}

void PulseAudioSink::stop() {
    if (state_ != PluginState::Running) {
        return;
    }

    pa_threaded_mainloop_lock(mainloop_);

    // Drain stream before stopping (play remaining audio)
    if (stream_) {
        pa_operation* op = pa_stream_drain(stream_, nullptr, nullptr);
        if (op) {
            pa_operation_unref(op);
        }
    }

    destroyStream();
    pa_threaded_mainloop_unlock(mainloop_);

    state_ = PluginState::Initialized;

    std::cout << "[PulseAudioSink] Stopped - underruns: " << underrunCount_
              << ", overruns: " << overrunCount_ << "\n";
}

PluginInfo PulseAudioSink::getInfo() const {
    PluginInfo info;
    info.name = "PulseAudio Speaker";
    info.version = "1.0.0";
    info.author = "NDA Project";
    info.description = "Plays audio through speakers via PulseAudio (Linux)";
    info.type = PluginType::AudioSink;
    info.apiVersion = 1;
    return info;
}

PluginState PulseAudioSink::getState() const {
    return state_;
}

void PulseAudioSink::setParameter(const std::string& key, const std::string& value) {
    std::lock_guard<std::mutex> lock(paramMutex_);

    if (key == "device") {
        if (state_ == PluginState::Running) {
            std::cerr << "[PulseAudioSink] Cannot change device while running\n";
            return;
        }
        deviceName_ = value;
    }
    else if (key == "sampleRate") {
        if (state_ == PluginState::Running) {
            std::cerr << "[PulseAudioSink] Cannot change sample rate while running\n";
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
            std::cerr << "[PulseAudioSink] Cannot change channels while running\n";
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
            std::cerr << "[PulseAudioSink] Cannot change buffer size while running\n";
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

std::string PulseAudioSink::getParameter(const std::string& key) const {
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

bool PulseAudioSink::writeAudio(const AudioBuffer& buffer) {
    if (state_ != PluginState::Running) {
        return false;
    }

    int frames = buffer.getFrameCount();
    int bufferChannels = buffer.getChannelCount();

    // Check if we have space in the ring buffer
    int availableSpace = ringBuffer_.getAvailableWrite();
    if (availableSpace < frames) {
        overrunCount_++;
        return false;  // Would overflow
    }

    // Set up planar pointers from AudioBuffer
    for (int ch = 0; ch < std::min(channels_, bufferChannels); ++ch) {
        planarPtrs_[ch] = buffer.getChannelData(ch);
    }

    // If buffer has fewer channels than we need, duplicate first channel
    for (int ch = bufferChannels; ch < channels_; ++ch) {
        planarPtrs_[ch] = buffer.getChannelData(0);
    }

    // Write to ring buffer (planar format)
    int written = ringBuffer_.write(planarPtrs_.data(), frames);

    if (written < frames) {
        overrunCount_++;
        return false;
    }

    return true;
}

int PulseAudioSink::getSampleRate() const {
    return sampleRate_;
}

int PulseAudioSink::getChannels() const {
    return channels_;
}

void PulseAudioSink::setSampleRate(int sampleRate) {
    if (state_ == PluginState::Running) return;
    sampleRate_ = sampleRate;
}

void PulseAudioSink::setChannels(int channels) {
    if (state_ == PluginState::Running) return;
    channels_ = std::max(1, std::min(8, channels));
}

int PulseAudioSink::getBufferSize() const {
    return bufferSize_;
}

void PulseAudioSink::setBufferSize(int samples) {
    if (state_ == PluginState::Running) return;
    bufferSize_ = std::max(64, samples);
    interleavedBuffer_.resize(bufferSize_ * channels_);
}

int PulseAudioSink::getAvailableSpace() const {
    return ringBuffer_.getAvailableWrite();
}

// ==================== PulseAudio Callbacks ====================

void PulseAudioSink::contextStateCallback(pa_context* c, void* userdata) {
    auto* self = static_cast<PulseAudioSink*>(userdata);
    self->onContextState(c);
}

void PulseAudioSink::streamStateCallback(pa_stream* s, void* userdata) {
    auto* self = static_cast<PulseAudioSink*>(userdata);
    self->onStreamState(s);
}

void PulseAudioSink::streamWriteCallback(pa_stream* s, size_t nbytes, void* userdata) {
    auto* self = static_cast<PulseAudioSink*>(userdata);
    self->onStreamWrite(s, nbytes);
}

void PulseAudioSink::streamUnderflowCallback(pa_stream* s, void* userdata) {
    auto* self = static_cast<PulseAudioSink*>(userdata);
    self->onStreamUnderflow(s);
}

void PulseAudioSink::onContextState(pa_context* c) {
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

void PulseAudioSink::onStreamState(pa_stream* s) {
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

void PulseAudioSink::onStreamWrite(pa_stream* s, size_t nbytes) {
    // Calculate how many frames PulseAudio wants
    size_t bytesPerFrame = sizeof(float) * static_cast<size_t>(channels_);
    size_t framesRequested = nbytes / bytesPerFrame;

    // Check how many frames we have available
    int framesAvailable = ringBuffer_.getAvailableRead();

    if (framesAvailable <= 0) {
        // No data available - write silence
        std::vector<float> silence(framesRequested * static_cast<size_t>(channels_), 0.0f);
        pa_stream_write(s, silence.data(), nbytes, nullptr, 0, PA_SEEK_RELATIVE);
        return;
    }

    size_t framesToWrite = std::min(framesRequested, static_cast<size_t>(framesAvailable));
    size_t bytesToWrite = framesToWrite * bytesPerFrame;

    // Read from ring buffer (planar format)
    std::vector<std::vector<float>> planarData(static_cast<size_t>(channels_));
    std::vector<float*> planarPtrs(static_cast<size_t>(channels_));

    for (int ch = 0; ch < channels_; ++ch) {
        planarData[static_cast<size_t>(ch)].resize(framesToWrite);
        planarPtrs[static_cast<size_t>(ch)] = planarData[static_cast<size_t>(ch)].data();
    }

    int framesRead = ringBuffer_.read(planarPtrs.data(), static_cast<int>(framesToWrite));

    // Convert planar to interleaved
    std::vector<float> interleaved(static_cast<size_t>(framesRead) * static_cast<size_t>(channels_));
    for (int f = 0; f < framesRead; ++f) {
        for (int ch = 0; ch < channels_; ++ch) {
            interleaved[static_cast<size_t>(f * channels_ + ch)] =
                planarPtrs[static_cast<size_t>(ch)][f];
        }
    }

    // Write to PulseAudio
    pa_stream_write(s, interleaved.data(),
                    static_cast<size_t>(framesRead) * bytesPerFrame,
                    nullptr, 0, PA_SEEK_RELATIVE);

    // If we couldn't provide all requested data, write silence for the rest
    if (static_cast<size_t>(framesRead) < framesRequested) {
        size_t silenceFrames = framesRequested - static_cast<size_t>(framesRead);
        size_t silenceBytes = silenceFrames * bytesPerFrame;
        std::vector<float> silence(silenceFrames * static_cast<size_t>(channels_), 0.0f);
        pa_stream_write(s, silence.data(), silenceBytes, nullptr, 0, PA_SEEK_RELATIVE);
    }
}

void PulseAudioSink::onStreamUnderflow(pa_stream* /*s*/) {
    underrunCount_++;
}

// ==================== Helper Methods ====================

bool PulseAudioSink::createContext() {
    pa_mainloop_api* api = pa_threaded_mainloop_get_api(mainloop_);

    context_ = pa_context_new(api, "NDA PulseAudio Sink");
    if (!context_) {
        std::cerr << "[PulseAudioSink] Failed to create context\n";
        return false;
    }

    pa_context_set_state_callback(context_, contextStateCallback, this);

    if (pa_context_connect(context_, nullptr, PA_CONTEXT_NOFLAGS, nullptr) < 0) {
        std::cerr << "[PulseAudioSink] Failed to connect context: "
                  << pa_strerror(pa_context_errno(context_)) << "\n";
        pa_context_unref(context_);
        context_ = nullptr;
        return false;
    }

    return true;
}

bool PulseAudioSink::createStream() {
    // Sample specification
    pa_sample_spec spec;
    spec.format = PA_SAMPLE_FLOAT32LE;
    spec.rate = static_cast<uint32_t>(sampleRate_);
    spec.channels = static_cast<uint8_t>(channels_);

    // Buffer attributes for reasonable latency
    // tlength: target buffer length (server tries to keep this much data)
    // prebuf: how much data before playback starts
    // minreq: minimum request size from server
    pa_buffer_attr attr;
    attr.maxlength = static_cast<uint32_t>(-1);  // Server decides
    attr.tlength = static_cast<uint32_t>(bufferSize_ * sizeof(float) * channels_ * 4);  // ~40ms
    attr.prebuf = static_cast<uint32_t>(bufferSize_ * sizeof(float) * channels_ * 2);   // ~20ms
    attr.minreq = static_cast<uint32_t>(bufferSize_ * sizeof(float) * channels_);       // ~10ms

    // Create stream
    stream_ = pa_stream_new(context_, "NDA Playback", &spec, nullptr);
    if (!stream_) {
        std::cerr << "[PulseAudioSink] Failed to create stream: "
                  << pa_strerror(pa_context_errno(context_)) << "\n";
        return false;
    }

    pa_stream_set_state_callback(stream_, streamStateCallback, this);
    pa_stream_set_write_callback(stream_, streamWriteCallback, this);
    pa_stream_set_underflow_callback(stream_, streamUnderflowCallback, this);

    // Connect for playback
    const char* device = deviceName_.empty() ? nullptr : deviceName_.c_str();
    pa_stream_flags_t flags = static_cast<pa_stream_flags_t>(
        PA_STREAM_INTERPOLATE_TIMING |
        PA_STREAM_ADJUST_LATENCY |
        PA_STREAM_AUTO_TIMING_UPDATE
    );

    if (pa_stream_connect_playback(stream_, device, &attr, flags, nullptr, nullptr) < 0) {
        std::cerr << "[PulseAudioSink] Failed to connect stream: "
                  << pa_strerror(pa_context_errno(context_)) << "\n";
        pa_stream_unref(stream_);
        stream_ = nullptr;
        return false;
    }

    return true;
}

void PulseAudioSink::destroyStream() {
    if (stream_) {
        pa_stream_disconnect(stream_);
        pa_stream_unref(stream_);
        stream_ = nullptr;
    }
}

void PulseAudioSink::destroyContext() {
    if (context_) {
        pa_context_disconnect(context_);
        pa_context_unref(context_);
        context_ = nullptr;
    }
}

bool PulseAudioSink::waitForContextReady() {
    pa_threaded_mainloop_lock(mainloop_);

    while (true) {
        pa_context_state_t state = pa_context_get_state(context_);

        if (state == PA_CONTEXT_READY) {
            pa_threaded_mainloop_unlock(mainloop_);
            return true;
        }

        if (state == PA_CONTEXT_FAILED || state == PA_CONTEXT_TERMINATED) {
            std::cerr << "[PulseAudioSink] Context connection failed: "
                      << pa_strerror(pa_context_errno(context_)) << "\n";
            pa_threaded_mainloop_unlock(mainloop_);
            return false;
        }

        pa_threaded_mainloop_wait(mainloop_);
    }
}

bool PulseAudioSink::waitForStreamReady() {
    pa_threaded_mainloop_lock(mainloop_);

    while (true) {
        pa_stream_state_t state = pa_stream_get_state(stream_);

        if (state == PA_STREAM_READY) {
            pa_threaded_mainloop_unlock(mainloop_);
            return true;
        }

        if (state == PA_STREAM_FAILED || state == PA_STREAM_TERMINATED) {
            std::cerr << "[PulseAudioSink] Stream connection failed: "
                      << pa_strerror(pa_context_errno(context_)) << "\n";
            pa_threaded_mainloop_unlock(mainloop_);
            return false;
        }

        pa_threaded_mainloop_wait(mainloop_);
    }
}

void PulseAudioSink::prefillStream() {
    // Pre-fill ring buffer with silence to avoid initial underruns
    int silenceFrames = bufferSize_ * 2;  // 2 buffers worth

    std::vector<std::vector<float>> silenceData(static_cast<size_t>(channels_));
    std::vector<const float*> silencePtrs(static_cast<size_t>(channels_));

    for (int ch = 0; ch < channels_; ++ch) {
        silenceData[static_cast<size_t>(ch)].resize(static_cast<size_t>(silenceFrames), 0.0f);
        silencePtrs[static_cast<size_t>(ch)] = silenceData[static_cast<size_t>(ch)].data();
    }

    ringBuffer_.write(silencePtrs.data(), silenceFrames);
}

} // namespace nda

// ==================== Plugin Factory ====================

extern "C" {

nda::BasePlugin* createPlugin() {
    return new nda::PulseAudioSink();
}

void destroyPlugin(nda::BasePlugin* plugin) {
    delete plugin;
}

}
