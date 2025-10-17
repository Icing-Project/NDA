#ifndef AUDIOENGINE_H
#define AUDIOENGINE_H

#include "AudioDevice.h"
#include "AudioBuffer.h"
#include <memory>
#include <functional>

class Encryptor;

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

    void setEncryptor(Encryptor* encryptor);
    void setAudioCallback(AudioCallback callback);

    double getLatency() const;
    float getCPULoad() const;
    int getBufferUnderruns() const { return underruns_; }

private:
    void audioThread();
    void processAudio(const AudioBuffer& input, AudioBuffer& output);

    std::unique_ptr<AudioDevice> inputDevice_;
    std::unique_ptr<AudioDevice> outputDevice_;

    Encryptor* encryptor_;
    AudioCallback callback_;

    int sampleRate_;
    int bufferSize_;
    bool isRunning_;
    int underruns_;

    void* threadHandle_;
};

#endif // AUDIOENGINE_H
