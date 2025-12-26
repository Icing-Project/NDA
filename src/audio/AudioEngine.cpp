#include "audio/AudioEngine.h"
// v2.0: Crypto removed from core - encryption is now handled by processor plugins

AudioEngine::AudioEngine()
    : sampleRate_(48000), bufferSize_(512),
      isRunning_(false), underruns_(0), threadHandle_(nullptr)
{
    inputDevice_ = std::make_unique<AudioDevice>();
    outputDevice_ = std::make_unique<AudioDevice>();
}

AudioEngine::~AudioEngine()
{
    if (isRunning_) {
        stop();
    }
    shutdown();
}

bool AudioEngine::initialize(int sampleRate, int bufferSize)
{
    sampleRate_ = sampleRate;
    bufferSize_ = bufferSize;

    // Open default devices
    auto defaultInput = AudioDevice::getDefaultInputDevice();
    auto defaultOutput = AudioDevice::getDefaultOutputDevice();

    if (!inputDevice_->open(defaultInput.id, sampleRate, bufferSize)) {
        return false;
    }

    if (!outputDevice_->open(defaultOutput.id, sampleRate, bufferSize)) {
        inputDevice_->close();
        return false;
    }

    return true;
}

void AudioEngine::shutdown()
{
    if (inputDevice_->isOpen()) {
        inputDevice_->close();
    }
    if (outputDevice_->isOpen()) {
        outputDevice_->close();
    }
}

bool AudioEngine::start()
{
    if (isRunning_) return false;

    isRunning_ = true;
    // Start audio thread (placeholder - will use std::thread or platform-specific)
    return true;
}

void AudioEngine::stop()
{
    if (!isRunning_) return;

    isRunning_ = false;
    // Stop audio thread
}

void AudioEngine::setInputDevice(const std::string& deviceId)
{
    bool wasRunning = isRunning_;
    if (wasRunning) stop();

    inputDevice_->close();
    inputDevice_->open(deviceId, sampleRate_, bufferSize_);

    if (wasRunning) start();
}

void AudioEngine::setOutputDevice(const std::string& deviceId)
{
    bool wasRunning = isRunning_;
    if (wasRunning) stop();

    outputDevice_->close();
    outputDevice_->open(deviceId, sampleRate_, bufferSize_);

    if (wasRunning) start();
}

// v2.0: setEncryptor() removed - use processor plugins instead

void AudioEngine::setAudioCallback(AudioCallback callback)
{
    callback_ = callback;
}

double AudioEngine::getLatency() const
{
    double inputLatency = inputDevice_->getLatency();
    double outputLatency = outputDevice_->getLatency();
    return inputLatency + outputLatency;
}

float AudioEngine::getCPULoad() const
{
    // Placeholder - will measure actual CPU usage
    return 8.5f;
}

void AudioEngine::audioThread()
{
    // High-priority audio processing thread
    AudioBuffer inputBuffer(2, bufferSize_);
    AudioBuffer outputBuffer(2, bufferSize_);

    while (isRunning_) {
        // Read from input device
        // Process audio
        processAudio(inputBuffer, outputBuffer);
        // Write to output device
    }
}

void AudioEngine::processAudio(const AudioBuffer& input, AudioBuffer& output)
{
    // Copy input to output (bypass for now)
    output.copyFrom(input);

    // v2.0: Encryption removed - use processor plugins instead
    // Processors are managed by ProcessingPipeline, not AudioEngine

    // Call user callback if set
    if (callback_) {
        callback_(input, output);
    }
}
