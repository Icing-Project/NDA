#include "PipeBridgeSink.h"
#include <iostream>

namespace nda {

PipeBridgeSink::PipeBridgeSink()
    : state_(PluginState::Unloaded)
    , sampleRate_(48000)
    , channels_(2)
    , bufferSize_(512)
{
}

PipeBridgeSink::~PipeBridgeSink() {
    if (state_ != PluginState::Unloaded) {
        shutdown();
    }
}

bool PipeBridgeSink::initialize() {
    if (state_ != PluginState::Unloaded) {
        std::cerr << "[PipeBridgeSink] Initialize failed: already initialized\n";
        return false;
    }

    state_ = PluginState::Initialized;
    std::cerr << "[PipeBridgeSink] Initialized: " << sampleRate_ << "Hz, "
              << channels_ << "ch, " << bufferSize_ << " frames\n";
    return true;
}

void PipeBridgeSink::shutdown() {
    stop();
    state_ = PluginState::Unloaded;
    std::cerr << "[PipeBridgeSink] Shutdown complete\n";
}

bool PipeBridgeSink::start() {
    if (state_ != PluginState::Initialized) {
        std::cerr << "[PipeBridgeSink] Start failed: not initialized (state="
                  << static_cast<int>(state_) << ")\n";
        return false;
    }

    // Connect to the bridge session
    PipeBridgeSession::getInstance().connectSink(sampleRate_, channels_);

    state_ = PluginState::Running;
    std::cerr << "[PipeBridgeSink] Started\n";
    return true;
}

void PipeBridgeSink::stop() {
    if (state_ == PluginState::Running) {
        PipeBridgeSession::getInstance().disconnectSink();
        state_ = PluginState::Initialized;
        std::cerr << "[PipeBridgeSink] Stopped\n";
    }
}

PluginInfo PipeBridgeSink::getInfo() const {
    return {
        "Pipe Bridge Sink",
        "1.0.0",
        "Icing Project",
        "Zero-latency bridge sink for inter-pipeline audio transfer",
        PluginType::AudioSink,
        NDA_PLUGIN_API_VERSION
    };
}

PluginState PipeBridgeSink::getState() const {
    return state_;
}

void PipeBridgeSink::setParameter(const std::string& key, const std::string& value) {
    // No configurable parameters currently
    (void)key;
    (void)value;
}

std::string PipeBridgeSink::getParameter(const std::string& key) const {
    auto& session = PipeBridgeSession::getInstance();

    if (key == "sampleRate") return std::to_string(sampleRate_);
    if (key == "channels") return std::to_string(channels_);
    if (key == "bufferSize") return std::to_string(bufferSize_);
    if (key == "framesPassed") return std::to_string(session.getFramesPassed());
    if (key == "framesDropped") return std::to_string(session.getFramesDropped());
    if (key == "sourceConnected") return session.isSourceConnected() ? "true" : "false";
    return "";
}

bool PipeBridgeSink::writeAudio(const AudioBuffer& buffer) {
    if (state_ != PluginState::Running) {
        return false;
    }

    return PipeBridgeSession::getInstance().write(buffer);
}

int PipeBridgeSink::getSampleRate() const {
    return sampleRate_;
}

int PipeBridgeSink::getChannels() const {
    return channels_;
}

void PipeBridgeSink::setSampleRate(int sampleRate) {
    if (state_ == PluginState::Unloaded || state_ == PluginState::Initialized) {
        sampleRate_ = sampleRate;
        std::cerr << "[PipeBridgeSink] Sample rate set to " << sampleRate_ << "Hz\n";
    }
}

void PipeBridgeSink::setChannels(int channels) {
    if (state_ == PluginState::Unloaded || state_ == PluginState::Initialized) {
        channels_ = channels;
        std::cerr << "[PipeBridgeSink] Channels set to " << channels_ << "\n";
    }
}

int PipeBridgeSink::getBufferSize() const {
    return bufferSize_;
}

void PipeBridgeSink::setBufferSize(int samples) {
    if (state_ == PluginState::Unloaded || state_ == PluginState::Initialized) {
        bufferSize_ = samples;
    }
}

int PipeBridgeSink::getAvailableSpace() const {
    // Always report space available - we either pass through or discard
    return bufferSize_;
}

} // namespace nda

// Export the plugin
NDA_DECLARE_PLUGIN(nda::PipeBridgeSink)
