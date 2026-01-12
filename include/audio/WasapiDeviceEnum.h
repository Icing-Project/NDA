#pragma once

#include <string>
#include <vector>

namespace nda {

// WASAPI device enumeration for UI device selectors
struct WASAPIDeviceInfo
{
    std::string id;           // Device ID string for setParameter("device_id")
    std::string friendlyName; // Display name for UI
};

// Enumerate available WASAPI audio devices
// direction: 0 = eCapture (microphones), 1 = eRender (speakers)
std::vector<WASAPIDeviceInfo> enumerateWASAPIDevices(int direction);

} // namespace nda
