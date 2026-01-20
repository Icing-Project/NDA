#include "PipeBridgeSource.h"
#include <iostream>

namespace nda {

PipeBridgeSource::PipeBridgeSource()
    : state_(PluginState::Unloaded)
    , bufferSize_(512)
    , callback_(nullptr)
{
}

PipeBridgeSource::~PipeBridgeSource() {
    if (state_ != PluginState::Unloaded) {
        shutdown();
    }
}

bool PipeBridgeSource::initialize() {
    if (state_ != PluginState::Unloaded) {
        std::cerr << "[PipeBridgeSource] Initialize failed: already initialized\n";
        return false;
    }

    state_ = PluginState::Initialized;
    std::cerr << "[PipeBridgeSource] Initialized\n";
    return true;
}

void PipeBridgeSource::shutdown() {
    stop();
    state_ = PluginState::Unloaded;
    std::cerr << "[PipeBridgeSource] Shutdown complete\n";
}

bool PipeBridgeSource::start() {
    if (state_ != PluginState::Initialized) {
        std::cerr << "[PipeBridgeSource] Start failed: not initialized (state="
                  << static_cast<int>(state_) << ")\n";
        return false;
    }

    // Connect to the bridge session
    PipeBridgeSession::getInstance().connectSource();

    state_ = PluginState::Running;
    std::cerr << "[PipeBridgeSource] Started - waiting for sink\n";
    return true;
}

void PipeBridgeSource::stop() {
    if (state_ == PluginState::Running) {
        PipeBridgeSession::getInstance().disconnectSource();
        state_ = PluginState::Initialized;
        std::cerr << "[PipeBridgeSource] Stopped\n";
    }
}

PluginInfo PipeBridgeSource::getInfo() const {
    return {
        "Pipe Bridge Source",
        "1.0.0",
        "Icing Project",
        "Zero-latency bridge source for inter-pipeline audio transfer",
        PluginType::AudioSource,
        NDA_PLUGIN_API_VERSION
    };
}

PluginState PipeBridgeSource::getState() const {
    return state_;
}

void PipeBridgeSource::setParameter(const std::string& key, const std::string& value) {
    // No configurable parameters currently
    (void)key;
    (void)value;
}

std::string PipeBridgeSource::getParameter(const std::string& key) const {
    auto& session = PipeBridgeSession::getInstance();

    if (key == "sampleRate") return std::to_string(session.getSampleRate());
    if (key == "channels") return std::to_string(session.getChannels());
    if (key == "bufferSize") return std::to_string(bufferSize_);
    if (key == "framesPassed") return std::to_string(session.getFramesPassed());
    if (key == "framesDropped") return std::to_string(session.getFramesDropped());
    if (key == "sinkConnected") return session.isSinkConnected() ? "true" : "false";
    return "";
}

void PipeBridgeSource::setAudioCallback(AudioSourceCallback callback) {
    callback_ = callback;
}

bool PipeBridgeSource::readAudio(AudioBuffer& buffer) {
    if (state_ != PluginState::Running) {
        buffer.clear();
        return false;
    }

    // Block until frame available from sink
    bool success = PipeBridgeSession::getInstance().read(buffer);

    if (!success) {
        // Sink disconnected - clear buffer and signal error
        buffer.clear();
        return false;
    }

    return true;
}

int PipeBridgeSource::getBufferSize() const {
    return bufferSize_;
}

void PipeBridgeSource::setBufferSize(int samples) {
    if (state_ == PluginState::Unloaded || state_ == PluginState::Initialized) {
        bufferSize_ = samples;
    }
}

int PipeBridgeSource::getSampleRate() const {
    // Inherit from session (set by sink)
    return PipeBridgeSession::getInstance().getSampleRate();
}

int PipeBridgeSource::getChannels() const {
    // Inherit from session (set by sink)
    return PipeBridgeSession::getInstance().getChannels();
}

void PipeBridgeSource::setSampleRate(int sampleRate) {
    // Source inherits from sink - ignore attempts to set directly
    (void)sampleRate;
    std::cerr << "[PipeBridgeSource] Note: sample rate is inherited from PipeBridgeSink\n";
}

void PipeBridgeSource::setChannels(int channels) {
    // Source inherits from sink - ignore attempts to set directly
    (void)channels;
    std::cerr << "[PipeBridgeSource] Note: channel count is inherited from PipeBridgeSink\n";
}

} // namespace nda

// Export the plugin
NDA_DECLARE_PLUGIN(nda::PipeBridgeSource)
