#ifndef PULSEDEVICEENUM_H
#define PULSEDEVICEENUM_H

#include <string>
#include <vector>
#include <cstdint>

namespace nda {

/**
 * @brief Information about a PulseAudio audio device.
 *
 * Works with both PulseAudio and PipeWire (via pipewire-pulse compatibility).
 */
struct PulseDeviceInfo {
    std::string name;           ///< Internal PulseAudio name (use for API calls)
    std::string description;    ///< Human-readable name (display in UI)
    uint32_t index;             ///< PulseAudio device index
    bool isDefault;             ///< Is this the default device?
    uint32_t sampleRate;        ///< Native sample rate
    uint8_t channels;           ///< Channel count
};

/**
 * @brief Enumerate available audio capture devices (microphones).
 *
 * Uses PulseAudio API, works with both PulseAudio and PipeWire systems.
 *
 * @return Vector of source devices, empty if PulseAudio unavailable
 */
std::vector<PulseDeviceInfo> enumeratePulseSources();

/**
 * @brief Enumerate available audio playback devices (speakers).
 *
 * Uses PulseAudio API, works with both PulseAudio and PipeWire systems.
 *
 * @return Vector of sink devices, empty if PulseAudio unavailable
 */
std::vector<PulseDeviceInfo> enumeratePulseSinks();

/**
 * @brief Get the default source (microphone) name.
 *
 * @return Default source name, or empty string if unavailable
 */
std::string getDefaultPulseSource();

/**
 * @brief Get the default sink (speaker) name.
 *
 * @return Default sink name, or empty string if unavailable
 */
std::string getDefaultPulseSink();

/**
 * @brief Check if PulseAudio server is available.
 *
 * @return true if PulseAudio/PipeWire server is running and accessible
 */
bool isPulseAudioAvailable();

// ==================== AIOC-Specific Functions ====================

/**
 * @brief Find AIOC audio source devices.
 *
 * Searches for PulseAudio sources that contain "AIOC" or "aioc" in their
 * name or description.
 *
 * @return Vector of AIOC source devices (may be empty)
 */
std::vector<PulseDeviceInfo> findAIOCPulseSources();

/**
 * @brief Find AIOC audio sink devices.
 *
 * Searches for PulseAudio sinks that contain "AIOC" or "aioc" in their
 * name or description.
 *
 * @return Vector of AIOC sink devices (may be empty)
 */
std::vector<PulseDeviceInfo> findAIOCPulseSinks();

/**
 * @brief Get the first detected AIOC source device name.
 *
 * @return AIOC source name, or empty string if not found
 */
std::string getFirstAIOCSource();

/**
 * @brief Get the first detected AIOC sink device name.
 *
 * @return AIOC sink name, or empty string if not found
 */
std::string getFirstAIOCSink();

/**
 * @brief Check if any AIOC audio device is present.
 *
 * @return true if at least one AIOC source or sink is found
 */
bool isAIOCAudioPresent();

} // namespace nda

#endif // PULSEDEVICEENUM_H
