#include "PipeBridgeSession.h"
#include <iostream>

namespace nda {

PipeBridgeSession& PipeBridgeSession::getInstance() {
    static PipeBridgeSession instance;
    return instance;
}

PipeBridgeSession::PipeBridgeSession()
    : frame_(2, 512)  // Default allocation, will be resized on first write
{
}

bool PipeBridgeSession::write(const AudioBuffer& buffer) {
    std::unique_lock<std::mutex> lock(mutex_);

    // If no source connected, behave as null sink (discard immediately)
    if (!sourceConnected_.load()) {
        framesDropped_++;
        return true;  // "Success" - frame handled (discarded)
    }

    // Source is connected - block until previous frame is consumed
    // This synchronizes TX and RX pipelines for zero-latency transfer
    frameConsumed_.wait(lock, [this] {
        return !hasFrame_.load() || !sourceConnected_.load();
    });

    // Source disconnected while waiting - switch to null sink mode
    if (!sourceConnected_.load()) {
        framesDropped_++;
        return true;
    }

    // Handoff: copy frame and signal source
    frame_.copyFrom(buffer);
    hasFrame_ = true;
    framesPassed_++;
    frameReady_.notify_one();

    return true;
}

void PipeBridgeSession::connectSink(int sampleRate, int channels) {
    std::lock_guard<std::mutex> lock(mutex_);

    sampleRate_ = sampleRate;
    channels_ = channels;
    sinkConnected_ = true;

    // Pre-allocate frame buffer to match expected format
    frame_.resize(channels, 512);

    std::cerr << "[PipeBridge] Sink connected: " << sampleRate << "Hz, "
              << channels << "ch\n";
}

void PipeBridgeSession::disconnectSink() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        sinkConnected_ = false;
        hasFrame_ = false;
    }

    // Wake up any waiting source so it can detect disconnection
    frameReady_.notify_all();

    std::cerr << "[PipeBridge] Sink disconnected. Passed: " << framesPassed_.load()
              << ", Dropped: " << framesDropped_.load() << "\n";
}

bool PipeBridgeSession::read(AudioBuffer& buffer) {
    std::unique_lock<std::mutex> lock(mutex_);

    // Wait for frame to arrive
    frameReady_.wait(lock, [this] {
        return hasFrame_.load() || !sinkConnected_.load();
    });

    // Sink disconnected and no pending frame
    if (!hasFrame_.load()) {
        return false;  // Signal error to pipeline
    }

    // Handoff: copy frame out and signal sink
    buffer.copyFrom(frame_);
    hasFrame_ = false;
    frameConsumed_.notify_one();

    return true;
}

void PipeBridgeSession::connectSource() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        sourceConnected_ = true;
    }

    // Wake up sink in case it was discarding frames
    frameConsumed_.notify_all();

    std::cerr << "[PipeBridge] Source connected\n";
}

void PipeBridgeSession::disconnectSource() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        sourceConnected_ = false;
    }

    // Wake up sink so it can switch to null sink mode
    frameConsumed_.notify_all();

    std::cerr << "[PipeBridge] Source disconnected\n";
}

void PipeBridgeSession::resetMetrics() {
    framesPassed_ = 0;
    framesDropped_ = 0;
}

} // namespace nda
