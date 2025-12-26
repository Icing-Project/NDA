#ifndef AUDIOENGINE_H
#define AUDIOENGINE_H

#include "AudioDevice.h"
#include "AudioBuffer.h"
#include <memory>
#include <functional>

// v2.0: Encryptor class removed - encryption is now handled by processor plugins

using AudioCallback = std::function<void(const AudioBuffer& input, AudioBuffer& output)>;

class AudioEngine
{
public:
    AudioEngine();
    ~AudioEngine();

    bool initialize(int sampleRate = 48000, int bufferSize = 512);
    void shutdown();

    bool start();
    void stop();

    bool isRunning() const { return isRunning_; }

    void setInputDevice(const std::string& deviceId);
    void setOutputDevice(const std::string& deviceId);

    // v2.0: setEncryptor() removed - use processor plugins instead
    void setAudioCallback(AudioCallback callback);

    double getLatency() const;
    float getCPULoad() const;
    int getBufferUnderruns() const { return underruns_; }

private:
    void audioThread();
    void processAudio(const AudioBuffer& input, AudioBuffer& output);

    std::unique_ptr<AudioDevice> inputDevice_;
    std::unique_ptr<AudioDevice> outputDevice_;

    // v2.0: encryptor_ removed - use processor plugins instead
    AudioCallback callback_;

    int sampleRate_;
    int bufferSize_;
    bool isRunning_;
    int underruns_;

    void* threadHandle_;
};

#endif // AUDIOENGINE_H
