#ifndef LINUXAIOCSESSION_H
#define LINUXAIOCSESSION_H

#include <atomic>
#include <mutex>
#include <string>
#include <cstdint>

namespace nda {

/// PTT control mode for Linux AIOC
enum class LinuxPttMode {
    Auto,       ///< Try HID first, fallback to CDC
    HidManual,  ///< HID only
    CdcManual   ///< CDC/Serial only
};

/// Telemetry data for Linux AIOC session
struct LinuxAIOCTelemetry {
    bool connected{false};
    bool pttAsserted{false};
    std::string pttMode;        ///< "hid", "cdc", or "none"
    std::string hidStatus;      ///< "connected", "error: ...", or "not found"
    std::string cdcStatus;      ///< "connected", "error: ...", or "not found"
    std::string cdcPort;        ///< e.g., "/dev/ttyACM0"
    std::string lastError;
};

/**
 * @brief Centralized PTT control and device detection for AIOC hardware on Linux.
 *
 * This class handles Push-To-Talk control via two methods:
 * - HID: USB HID reports using hidapi (VID 0x1209, PID 0x7388)
 * - CDC: Serial port DTR/RTS control via /dev/ttyACM*
 *
 * The session can be shared between source and sink plugins.
 *
 * Usage:
 * @code
 *   auto session = std::make_shared<LinuxAIOCSession>();
 *   session->setPttMode(LinuxPttMode::Auto);
 *   session->setCdcPort("auto");  // or "/dev/ttyACM0"
 *   if (session->connect()) {
 *       session->setPttState(true);   // PTT on
 *       // ... transmit audio ...
 *       session->setPttState(false);  // PTT off
 *   }
 *   session->disconnect();
 * @endcode
 */
class LinuxAIOCSession {
public:
    /// AIOC USB identifiers
    static constexpr uint16_t AIOC_VID = 0x1209;
    static constexpr uint16_t AIOC_PID = 0x7388;

    LinuxAIOCSession();
    ~LinuxAIOCSession();

    // Non-copyable, movable
    LinuxAIOCSession(const LinuxAIOCSession&) = delete;
    LinuxAIOCSession& operator=(const LinuxAIOCSession&) = delete;
    LinuxAIOCSession(LinuxAIOCSession&&) = default;
    LinuxAIOCSession& operator=(LinuxAIOCSession&&) = default;

    // ==================== Configuration ====================

    /**
     * @brief Set PTT control mode.
     * @param mode Auto, HidManual, or CdcManual
     * Must be called before connect().
     */
    void setPttMode(LinuxPttMode mode);

    /**
     * @brief Set CDC serial port path.
     * @param port Path like "/dev/ttyACM0", or "auto" for auto-detection
     * Must be called before connect().
     */
    void setCdcPort(const std::string& port);

    /**
     * @brief Get currently configured PTT mode.
     */
    LinuxPttMode getPttMode() const;

    /**
     * @brief Get currently configured CDC port.
     */
    std::string getCdcPort() const;

    // ==================== Lifecycle ====================

    /**
     * @brief Connect to AIOC device (open HID and/or serial).
     * @return true if at least one PTT method is available
     */
    bool connect();

    /**
     * @brief Disconnect from AIOC device.
     */
    void disconnect();

    /**
     * @brief Check if connected to AIOC.
     */
    bool isConnected() const;

    // ==================== PTT Control ====================

    /**
     * @brief Set PTT state (assert or release).
     * @param asserted true = PTT on (transmitting), false = PTT off
     * @return true if PTT state was set successfully
     */
    bool setPttState(bool asserted);

    /**
     * @brief Get current PTT state.
     */
    bool isPttAsserted() const;

    // ==================== Telemetry ====================

    /**
     * @brief Get telemetry data for diagnostics.
     */
    LinuxAIOCTelemetry getTelemetry() const;

    // ==================== Static Detection Helpers ====================

    /**
     * @brief Find AIOC serial port by scanning /sys/class/tty.
     * @return Path like "/dev/ttyACM0", or empty string if not found
     */
    static std::string findAIOCSerialPort();

    /**
     * @brief Check if AIOC HID device is present.
     * @return true if device with VID/PID 1209:7388 is found
     */
    static bool isHidDevicePresent();

private:
    // HID handling
    bool openHidDevice();
    void closeHidDevice();
    bool sendHidPttReport(bool asserted);

    // CDC/Serial handling
    bool openSerialPort();
    void closeSerialPort();
    bool setSerialPttState(bool asserted);

    // State
    mutable std::mutex mutex_;
    LinuxPttMode pttMode_;
    std::string cdcPort_;
    std::string resolvedCdcPort_;  // Actual port after auto-detection
    bool connected_;
    std::atomic<bool> pttAsserted_;

    // HID device (hidapi)
    void* hidDevice_;  // hid_device*

    // Serial port
    int serialFd_;

    // Active PTT mode after connection
    std::string activePttMode_;  // "hid", "cdc", or "none"

    // Status strings for telemetry
    std::string hidStatus_;
    std::string cdcStatus_;
    std::string lastError_;
};

} // namespace nda

#endif // LINUXAIOCSESSION_H
