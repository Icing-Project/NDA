#include "audio/PulseDeviceEnum.h"

namespace nda {

std::vector<PulseDeviceInfo> enumeratePulseSources() {
    return {};
}

std::vector<PulseDeviceInfo> enumeratePulseSinks() {
    return {};
}

std::string getDefaultPulseSource() {
    return "";
}

std::string getDefaultPulseSink() {
    return "";
}

bool isPulseAudioAvailable() {
    return false;
}

std::vector<PulseDeviceInfo> findAIOCPulseSources() {
    return {};
}

std::vector<PulseDeviceInfo> findAIOCPulseSinks() {
    return {};
}

std::string getFirstAIOCSource() {
    return "";
}

std::string getFirstAIOCSink() {
    return "";
}

bool isAIOCAudioPresent() {
    return false;
}

} // namespace nda
