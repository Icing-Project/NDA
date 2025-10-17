#ifndef AUDIODEVICE_H
#define AUDIODEVICE_H

#include <string>
#include <vector>

enum class AudioAPI {
    WASAPI,
    ASIO,
    WDM_KS
};

struct AudioDeviceInfo {
    std::string id;
    std::string name;
    AudioAPI api;
    int maxInputChannels;
    int maxOutputChannels;
    std::vector<int> supportedSampleRates;
    int defaultSampleRate;
    bool isDefaultInput;
    bool isDefaultOutput;
};

class AudioDevice
{
public:
    AudioDevice();
    ~AudioDevice();

    static std::vector<AudioDeviceInfo> getAvailableDevices();
    static AudioDeviceInfo getDefaultInputDevice();
    static AudioDeviceInfo getDefaultOutputDevice();

    bool open(const std::string& deviceId, int sampleRate, int bufferSize);
    void close();

    bool isOpen() const { return isOpen_; }
    int getSampleRate() const { return sampleRate_; }
    int getBufferSize() const { return bufferSize_; }
    double getLatency() const;

private:
    bool isOpen_;
    int sampleRate_;
    int bufferSize_;
    void* nativeHandle_;
};

#endif // AUDIODEVICE_H
