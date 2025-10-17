#include "audio/AudioDevice.h"

AudioDevice::AudioDevice()
    : isOpen_(false), sampleRate_(48000), bufferSize_(512), nativeHandle_(nullptr)
{
}

AudioDevice::~AudioDevice()
{
    if (isOpen_) {
        close();
    }
}

std::vector<AudioDeviceInfo> AudioDevice::getAvailableDevices()
{
    // Placeholder - will be implemented with WASAPI/ASIO enumeration
    std::vector<AudioDeviceInfo> devices;

    AudioDeviceInfo device1;
    device1.id = "wasapi_default_in";
    device1.name = "Default Input (WASAPI)";
    device1.api = AudioAPI::WASAPI;
    device1.maxInputChannels = 2;
    device1.maxOutputChannels = 0;
    device1.supportedSampleRates = {44100, 48000, 96000};
    device1.defaultSampleRate = 48000;
    device1.isDefaultInput = true;
    device1.isDefaultOutput = false;
    devices.push_back(device1);

    AudioDeviceInfo device2;
    device2.id = "wasapi_default_out";
    device2.name = "Default Output (WASAPI)";
    device2.api = AudioAPI::WASAPI;
    device2.maxInputChannels = 0;
    device2.maxOutputChannels = 2;
    device2.supportedSampleRates = {44100, 48000, 96000};
    device2.defaultSampleRate = 48000;
    device2.isDefaultInput = false;
    device2.isDefaultOutput = true;
    devices.push_back(device2);

    return devices;
}

AudioDeviceInfo AudioDevice::getDefaultInputDevice()
{
    auto devices = getAvailableDevices();
    for (const auto& device : devices) {
        if (device.isDefaultInput) {
            return device;
        }
    }
    return AudioDeviceInfo();
}

AudioDeviceInfo AudioDevice::getDefaultOutputDevice()
{
    auto devices = getAvailableDevices();
    for (const auto& device : devices) {
        if (device.isDefaultOutput) {
            return device;
        }
    }
    return AudioDeviceInfo();
}

bool AudioDevice::open(const std::string& deviceId, int sampleRate, int bufferSize)
{
    // Placeholder - will be implemented with WASAPI/ASIO initialization
    sampleRate_ = sampleRate;
    bufferSize_ = bufferSize;
    isOpen_ = true;
    return true;
}

void AudioDevice::close()
{
    // Placeholder - will cleanup WASAPI/ASIO resources
    isOpen_ = false;
    nativeHandle_ = nullptr;
}

double AudioDevice::getLatency() const
{
    if (!isOpen_) return 0.0;
    // Calculate latency based on buffer size and sample rate
    return (static_cast<double>(bufferSize_) / sampleRate_) * 1000.0;
}
