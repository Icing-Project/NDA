#include "LinuxAIOCSession.h"

#include <hidapi/hidapi.h>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <vector>

namespace nda {

namespace {

/// HID report structure for AIOC PTT control
struct AIOCHidReport {
    uint8_t reportId;     ///< Report ID (0x00)
    uint8_t pttState;     ///< PTT state: 0x01 = asserted, 0x00 = released
    uint8_t reserved[2];  ///< Reserved bytes
};

/// Helper to convert string to lowercase
std::string toLower(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

} // anonymous namespace

// ==================== Constructor / Destructor ====================

LinuxAIOCSession::LinuxAIOCSession()
    : pttMode_(LinuxPttMode::CdcManual)  // CDC is more reliable on Linux
    , cdcPort_("auto")
    , connected_(false)
    , pttAsserted_(false)
    , hidDevice_(nullptr)
    , serialFd_(-1)
    , activePttMode_("none")
    , hidStatus_("not initialized")
    , cdcStatus_("not initialized")
{
}

LinuxAIOCSession::~LinuxAIOCSession() {
    disconnect();
}

// ==================== Configuration ====================

void LinuxAIOCSession::setPttMode(LinuxPttMode mode) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!connected_) {
        pttMode_ = mode;
    }
}

void LinuxAIOCSession::setCdcPort(const std::string& port) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!connected_) {
        cdcPort_ = port;
    }
}

LinuxPttMode LinuxAIOCSession::getPttMode() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pttMode_;
}

std::string LinuxAIOCSession::getCdcPort() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cdcPort_;
}

// ==================== Lifecycle ====================

bool LinuxAIOCSession::connect() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (connected_) {
        return true;
    }

    bool hidOk = false;
    bool cdcOk = false;

    // Initialize hidapi library
    if (hid_init() < 0) {
        hidStatus_ = "error: hidapi init failed";
        std::cerr << "[LinuxAIOCSession] Failed to initialize hidapi\n";
    }

    // Try HID if mode allows
    if (pttMode_ == LinuxPttMode::Auto || pttMode_ == LinuxPttMode::HidManual) {
        hidOk = openHidDevice();
    }

    // Try CDC if mode allows
    if (pttMode_ == LinuxPttMode::Auto || pttMode_ == LinuxPttMode::CdcManual) {
        cdcOk = openSerialPort();
    }

    // Determine active PTT mode
    if (pttMode_ == LinuxPttMode::Auto) {
        // Prefer HID if both available
        if (hidOk) {
            activePttMode_ = "hid";
        } else if (cdcOk) {
            activePttMode_ = "cdc";
        } else {
            activePttMode_ = "none";
        }
    } else if (pttMode_ == LinuxPttMode::HidManual) {
        activePttMode_ = hidOk ? "hid" : "none";
    } else {
        activePttMode_ = cdcOk ? "cdc" : "none";
    }

    connected_ = (activePttMode_ != "none");

    if (connected_) {
        std::cout << "[LinuxAIOCSession] Connected - PTT mode: " << activePttMode_ << "\n";
    } else {
        lastError_ = "No PTT control method available";
        std::cerr << "[LinuxAIOCSession] Failed to connect - no PTT method available\n";
        std::cerr << "  HID: " << hidStatus_ << "\n";
        std::cerr << "  CDC: " << cdcStatus_ << "\n";
    }

    return connected_;
}

void LinuxAIOCSession::disconnect() {
    std::lock_guard<std::mutex> lock(mutex_);

    std::cout << "[LinuxAIOCSession] disconnect() called - connected=" << connected_
              << ", pttAsserted=" << pttAsserted_.load()
              << ", activePttMode=" << activePttMode_
              << ", serialFd=" << serialFd_ << "\n";

    if (!connected_) {
        std::cout << "[LinuxAIOCSession] Already disconnected, returning\n";
        return;
    }

    // Release PTT before disconnecting
    if (pttAsserted_) {
        std::cout << "[LinuxAIOCSession] PTT was asserted, releasing...\n";
        pttAsserted_ = false;
        if (activePttMode_ == "hid" && hidDevice_) {
            sendHidPttReport(false);
        } else if (activePttMode_ == "cdc" && serialFd_ >= 0) {
            setSerialPttState(false);
        }
    } else {
        std::cout << "[LinuxAIOCSession] PTT was not asserted (pttAsserted_=false)\n";
    }

    // Always try to release PTT via CDC if port is open, regardless of pttAsserted_ state
    // This handles cases where PTT was set but pttAsserted_ tracking got out of sync
    if (serialFd_ >= 0) {
        std::cout << "[LinuxAIOCSession] Force-releasing PTT via CDC before close\n";
        setSerialPttState(false);
    }

    closeHidDevice();
    closeSerialPort();

    // Cleanup hidapi
    hid_exit();

    connected_ = false;
    activePttMode_ = "none";

    std::cout << "[LinuxAIOCSession] Disconnected\n";
}

bool LinuxAIOCSession::isConnected() const {
    return connected_;
}

// ==================== PTT Control ====================

bool LinuxAIOCSession::setPttState(bool asserted) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!connected_) {
        return false;
    }

    bool success = false;

    if (activePttMode_ == "hid") {
        success = sendHidPttReport(asserted);
    } else if (activePttMode_ == "cdc") {
        success = setSerialPttState(asserted);
    }

    if (success) {
        pttAsserted_ = asserted;
    }

    return success;
}

bool LinuxAIOCSession::isPttAsserted() const {
    return pttAsserted_;
}

// ==================== Telemetry ====================

LinuxAIOCTelemetry LinuxAIOCSession::getTelemetry() const {
    std::lock_guard<std::mutex> lock(mutex_);

    LinuxAIOCTelemetry telemetry;
    telemetry.connected = connected_;
    telemetry.pttAsserted = pttAsserted_;
    telemetry.pttMode = activePttMode_;
    telemetry.hidStatus = hidStatus_;
    telemetry.cdcStatus = cdcStatus_;
    telemetry.cdcPort = resolvedCdcPort_;
    telemetry.lastError = lastError_;

    return telemetry;
}

// ==================== Static Detection Helpers ====================

std::string LinuxAIOCSession::findAIOCSerialPort() {
    namespace fs = std::filesystem;

    try {
        // Scan /sys/class/tty for USB serial devices
        for (const auto& entry : fs::directory_iterator("/sys/class/tty")) {
            std::string name = entry.path().filename().string();

            // Only check ttyACM* and ttyUSB* devices
            if (name.find("ttyACM") != 0 && name.find("ttyUSB") != 0) {
                continue;
            }

            // Build path to device's parent USB device info
            // /sys/class/tty/ttyACM0/device/../uevent
            std::string devicePath = entry.path().string() + "/device";
            if (!fs::exists(devicePath)) {
                continue;
            }

            // Try to find uevent file with USB product info
            // The path varies depending on USB topology, try multiple levels
            std::vector<std::string> ueventPaths = {
                devicePath + "/../uevent",
                devicePath + "/../../uevent",
                devicePath + "/../../../uevent"
            };

            for (const auto& ueventPath : ueventPaths) {
                if (!fs::exists(ueventPath)) {
                    continue;
                }

                std::ifstream uevent(ueventPath);
                std::string line;

                while (std::getline(uevent, line)) {
                    // Look for PRODUCT=1209/7388/... (VID/PID in hex, lowercase)
                    std::string lineLower = toLower(line);
                    if (lineLower.find("product=1209/7388") != std::string::npos) {
                        std::string devPath = "/dev/" + name;
                        std::cout << "[LinuxAIOCSession] Found AIOC serial port: " << devPath << "\n";
                        return devPath;
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[LinuxAIOCSession] Error scanning for serial ports: " << e.what() << "\n";
    }

    return "";  // Not found
}

bool LinuxAIOCSession::isHidDevicePresent() {
    // Initialize hidapi temporarily
    if (hid_init() < 0) {
        return false;
    }

    // Try to enumerate AIOC device
    struct hid_device_info* devs = hid_enumerate(AIOC_VID, AIOC_PID);
    bool found = (devs != nullptr);

    if (devs) {
        hid_free_enumeration(devs);
    }

    hid_exit();

    return found;
}

// ==================== HID Handling ====================

bool LinuxAIOCSession::openHidDevice() {
    // Try to open AIOC HID device
    hidDevice_ = hid_open(AIOC_VID, AIOC_PID, nullptr);

    if (!hidDevice_) {
        hidStatus_ = "not found (VID=1209, PID=7388)";
        return false;
    }

    // Set non-blocking mode for reads (we only write)
    hid_set_nonblocking(static_cast<hid_device*>(hidDevice_), 1);

    hidStatus_ = "connected";
    std::cout << "[LinuxAIOCSession] HID device opened successfully\n";

    return true;
}

void LinuxAIOCSession::closeHidDevice() {
    if (hidDevice_) {
        hid_close(static_cast<hid_device*>(hidDevice_));
        hidDevice_ = nullptr;
        hidStatus_ = "closed";
    }
}

bool LinuxAIOCSession::sendHidPttReport(bool asserted) {
    if (!hidDevice_) {
        return false;
    }

    AIOCHidReport report;
    report.reportId = 0x00;
    report.pttState = asserted ? 0x01 : 0x00;
    report.reserved[0] = 0x00;
    report.reserved[1] = 0x00;

    // hid_write: send report to device
    // For devices that use numbered reports, the first byte is the report number
    int result = hid_write(static_cast<hid_device*>(hidDevice_),
                           reinterpret_cast<uint8_t*>(&report),
                           sizeof(report));

    if (result < 0) {
        const wchar_t* err = hid_error(static_cast<hid_device*>(hidDevice_));
        if (err) {
            std::wcerr << L"[LinuxAIOCSession] HID write error: " << err << L"\n";
        }
        return false;
    }

    return true;
}

// ==================== CDC/Serial Handling ====================

bool LinuxAIOCSession::openSerialPort() {
    // Resolve port if set to "auto"
    if (cdcPort_ == "auto" || cdcPort_.empty()) {
        resolvedCdcPort_ = findAIOCSerialPort();
    } else {
        resolvedCdcPort_ = cdcPort_;
    }

    if (resolvedCdcPort_.empty()) {
        cdcStatus_ = "not found (auto-detection failed)";
        return false;
    }

    // Open serial port
    serialFd_ = open(resolvedCdcPort_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);

    if (serialFd_ < 0) {
        cdcStatus_ = "error: " + std::string(strerror(errno));
        if (errno == EACCES) {
            cdcStatus_ += " (permission denied - check udev rules)";
        } else if (errno == ENOENT) {
            cdcStatus_ += " (device not found)";
        }
        std::cerr << "[LinuxAIOCSession] Failed to open " << resolvedCdcPort_
                  << ": " << strerror(errno) << "\n";
        return false;
    }

    // Configure serial port (we only use control lines, not data transfer)
    struct termios tty;
    if (tcgetattr(serialFd_, &tty) < 0) {
        cdcStatus_ = "error: tcgetattr failed";
        close(serialFd_);
        serialFd_ = -1;
        return false;
    }

    // Raw mode - no processing
    cfmakeraw(&tty);

    // Enable receiver, ignore modem status lines for data,
    // but HUPCL ensures DTR is dropped when port closes (critical for PTT release)
    tty.c_cflag |= (CLOCAL | CREAD | HUPCL);

    // 9600 baud (doesn't matter since we only use control lines)
    cfsetispeed(&tty, B9600);
    cfsetospeed(&tty, B9600);

    if (tcsetattr(serialFd_, TCSANOW, &tty) < 0) {
        cdcStatus_ = "error: tcsetattr failed";
        close(serialFd_);
        serialFd_ = -1;
        return false;
    }

    // Initialize with PTT off
    setSerialPttState(false);

    cdcStatus_ = "connected";
    std::cout << "[LinuxAIOCSession] Serial port opened: " << resolvedCdcPort_ << "\n";

    return true;
}

void LinuxAIOCSession::closeSerialPort() {
    std::cout << "[LinuxAIOCSession] closeSerialPort() called - serialFd_=" << serialFd_ << "\n";

    if (serialFd_ >= 0) {
        // Ensure PTT is released before closing
        // This is critical - without proper release, DTR may stay high
        std::cout << "[LinuxAIOCSession] Releasing PTT (attempt 1)...\n";
        if (!setSerialPttState(false)) {
            std::cerr << "[LinuxAIOCSession] Warning: Failed to clear PTT before close\n";
        }

        // Wait for any pending output to complete
        std::cout << "[LinuxAIOCSession] Calling tcdrain()...\n";
        tcdrain(serialFd_);

        // Small delay to ensure the modem control line change takes effect
        // USB CDC ACM devices may need time to process the DTR change
        std::cout << "[LinuxAIOCSession] Waiting 50ms for DTR change to take effect...\n";
        usleep(50000);  // 50ms (increased from 20ms)

        // Double-check PTT is released (belt and suspenders)
        std::cout << "[LinuxAIOCSession] Releasing PTT (attempt 2)...\n";
        setSerialPttState(false);

        // Another delay after second release
        usleep(50000);  // 50ms

        // Verify DTR is actually low before closing
        int finalStatus;
        if (ioctl(serialFd_, TIOCMGET, &finalStatus) == 0) {
            bool finalDtr = (finalStatus & TIOCM_DTR) != 0;
            bool finalRts = (finalStatus & TIOCM_RTS) != 0;
            std::cout << "[LinuxAIOCSession] Final state before close: DTR=" << finalDtr
                      << ", RTS=" << finalRts << "\n";
            if (finalDtr) {
                std::cerr << "[LinuxAIOCSession] ERROR: DTR still HIGH before close!\n";
            }
        }

        // Now safe to close - HUPCL flag will also drop DTR
        std::cout << "[LinuxAIOCSession] Calling close() on fd " << serialFd_ << "\n";
        close(serialFd_);
        serialFd_ = -1;
        cdcStatus_ = "closed";
        std::cout << "[LinuxAIOCSession] Serial port closed, PTT should be released\n";
    } else {
        std::cout << "[LinuxAIOCSession] Serial port already closed (fd=-1)\n";
    }
}

bool LinuxAIOCSession::setSerialPttState(bool asserted) {
    if (serialFd_ < 0) {
        std::cerr << "[LinuxAIOCSession] setSerialPttState(" << asserted << ") - fd invalid!\n";
        return false;
    }

    int status;

    // Get current modem control line status
    if (ioctl(serialFd_, TIOCMGET, &status) < 0) {
        std::cerr << "[LinuxAIOCSession] ioctl TIOCMGET failed: " << strerror(errno) << "\n";
        return false;
    }

    bool oldDtr = (status & TIOCM_DTR) != 0;
    bool oldRts = (status & TIOCM_RTS) != 0;
    std::cout << "[LinuxAIOCSession] setSerialPttState(" << (asserted ? "ON" : "OFF")
              << ") - before: DTR=" << oldDtr << ", RTS=" << oldRts << "\n";

    if (asserted) {
        // PTT ON: Set DTR high, clear RTS (matches AIOC default: SERIALDTRNRTS)
        status |= TIOCM_DTR;
        status &= ~TIOCM_RTS;
    } else {
        // PTT OFF: Clear DTR (and RTS for good measure)
        status &= ~TIOCM_DTR;
        status &= ~TIOCM_RTS;
    }

    // Set modem control lines
    if (ioctl(serialFd_, TIOCMSET, &status) < 0) {
        std::cerr << "[LinuxAIOCSession] ioctl TIOCMSET failed: " << strerror(errno) << "\n";
        return false;
    }

    // Verify the change took effect
    int verifyStatus;
    if (ioctl(serialFd_, TIOCMGET, &verifyStatus) == 0) {
        bool newDtr = (verifyStatus & TIOCM_DTR) != 0;
        bool newRts = (verifyStatus & TIOCM_RTS) != 0;
        std::cout << "[LinuxAIOCSession] setSerialPttState - after: DTR=" << newDtr
                  << ", RTS=" << newRts << "\n";

        if (newDtr != asserted) {
            std::cerr << "[LinuxAIOCSession] ERROR: DTR state mismatch! "
                      << "Expected " << (asserted ? "1" : "0")
                      << ", got " << newDtr << "\n";
        }
    }

    return true;
}

} // namespace nda
