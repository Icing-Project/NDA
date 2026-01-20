# Linux AIOC Plugins Implementation Plan

**Status:** Planning
**Target:** Full AIOC support on Linux (audio + PTT)
**Date:** January 2026
**Branch:** Linux_Compat

---

## Executive Summary

This document specifies the implementation of Linux AIOC (Audio Input/Output Controller) plugins, providing feature parity with the existing Windows AIOC implementation. The goal is to enable NDA to communicate with AIOC radio devices on Linux systems.

### Goals

1. **AIOC device auto-detection** - Automatically find devices containing "AIOC" in name
2. **HID PTT control** - Push-To-Talk via USB HID reports (same protocol as Windows)
3. **CDC PTT control** - Push-To-Talk via serial port DTR/RTS signals
4. **Audio I/O** - Capture and playback via PulseAudio to AIOC audio interfaces
5. **User device selection** - Allow manual override of auto-detected devices

### Non-Goals

- VOX (voice-activated PTT) - Not implementing in this iteration
- COS (Carrier Operated Squelch) - Future enhancement
- macOS support - Out of scope

---

## Architecture Overview

### Component Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                    LinuxAIOCSession                             │
│  (Shared between Source and Sink plugins)                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────┐ │
│  │  HID PTT     │  │  CDC PTT      │  │  Device Detection    │ │
│  │  (hidapi)    │  │  (/dev/ttyACM)│  │  (PulseAudio + udev) │ │
│  └──────────────┘  └───────────────┘  └──────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  setPttState(bool asserted)                                     │
│  isPttAsserted() → bool                                         │
│  getTelemetry() → AIOCTelemetry                                 │
└────────────────────────────────────────────────────────────────┘
         ▲                                          ▲
         │ shared_ptr                               │ shared_ptr
┌────────┴───────────┐                    ┌────────┴───────────┐
│ LinuxAIOCSource    │                    │ LinuxAIOCSink      │
│ Plugin             │                    │ Plugin             │
├────────────────────┤                    ├────────────────────┤
│ - PulseAudio       │                    │ - PulseAudio       │
│   capture stream   │                    │   playback stream  │
│ - AIOC auto-select │                    │ - AIOC auto-select │
│ - Reads from AIOC  │                    │ - Writes to AIOC   │
│   microphone       │                    │   speaker          │
│                    │                    │ - Drives PTT       │
└────────────────────┘                    └────────────────────┘
```

### Comparison with Windows Implementation

| Feature | Windows (AIOCSession) | Linux (LinuxAIOCSession) |
|---------|----------------------|--------------------------|
| Audio Backend | WASAPI (IAudioClient) | PulseAudio (pa_stream) |
| HID PTT | hidapi (same) | hidapi (same) |
| CDC PTT | Windows COM port (DCB, SetCommState) | Linux serial (/dev/ttyACM*, termios, ioctl) |
| Device Enum | IMMDeviceEnumerator | pa_context_get_source/sink_info_list |
| Ring Buffer | Custom RingBuffer | Custom RingBuffer (same) |

---

## File Structure

```
NDA/
├── plugins_src/
│   └── linux/
│       ├── PulseAudioSource.cpp/h        # Existing generic source
│       ├── PulseAudioSink.cpp/h          # Existing generic sink
│       ├── LinuxAIOCSession.h            # NEW: PTT + device detection
│       ├── LinuxAIOCSession.cpp          # NEW
│       ├── LinuxAIOCSourcePlugin.h       # NEW: AIOC source plugin
│       ├── LinuxAIOCSourcePlugin.cpp     # NEW
│       ├── LinuxAIOCSinkPlugin.h         # NEW: AIOC sink plugin
│       └── LinuxAIOCSinkPlugin.cpp       # NEW
│
├── include/
│   └── audio/
│       └── PulseDeviceEnum.h             # Add: findAIOCDevices()
│
├── src/
│   └── audio/
│       └── PulseDeviceEnum.cpp           # Add: AIOC detection logic
│
└── CMakeLists.txt                         # Add: hidapi dependency
```

---

## Detailed Component Specifications

### 1. LinuxAIOCSession

**Purpose:** Centralized PTT control and device detection for AIOC hardware.

**Header: `plugins_src/linux/LinuxAIOCSession.h`**

```cpp
#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <functional>

namespace nda {

enum class LinuxPttMode {
    Auto,       // Try HID first, fallback to CDC
    HidManual,  // HID only
    CdcManual   // CDC only
};

struct LinuxAIOCTelemetry {
    bool connected{false};
    bool pttAsserted{false};
    std::string pttMode;        // "hid", "cdc", or "none"
    std::string hidStatus;      // "connected", "error: ...", or "not found"
    std::string cdcStatus;      // "connected", "error: ...", or "not found"
    std::string cdcPort;        // e.g., "/dev/ttyACM0"
    std::string lastError;
};

class LinuxAIOCSession {
public:
    LinuxAIOCSession();
    ~LinuxAIOCSession();

    // Configuration (call before connect())
    void setPttMode(LinuxPttMode mode);
    void setCdcPort(const std::string& port);  // e.g., "/dev/ttyACM0" or "auto"

    // Lifecycle
    bool connect();
    void disconnect();
    bool isConnected() const;

    // PTT Control
    bool setPttState(bool asserted);
    bool isPttAsserted() const;

    // Telemetry
    LinuxAIOCTelemetry getTelemetry() const;

    // Device detection helpers (static)
    static std::string findAIOCSerialPort();  // Returns /dev/ttyACM* path or ""
    static bool isHidDevicePresent();         // Check for AIOC HID device

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
    bool connected_;
    std::atomic<bool> pttAsserted_;

    // HID device (hidapi)
    void* hidDevice_;  // hid_device*

    // Serial port
    int serialFd_;

    // Telemetry
    std::string lastError_;
    std::string activePttMode_;
};

} // namespace nda
```

**Implementation Notes:**

#### HID PTT Protocol

```cpp
// AIOC USB identifiers
constexpr uint16_t AIOC_VID = 0x1209;
constexpr uint16_t AIOC_PID = 0x7388;

// HID report structure (4 bytes)
struct AIOCHidReport {
    uint8_t reportId;   // 0x00
    uint8_t pttState;   // 0x01 = asserted, 0x00 = released
    uint8_t reserved[2];
};

bool LinuxAIOCSession::sendHidPttReport(bool asserted) {
    if (!hidDevice_) return false;

    AIOCHidReport report = {0x00, asserted ? 0x01 : 0x00, {0, 0}};

    // hid_write expects report data starting after report ID for some devices
    // For AIOC, include the full report
    int result = hid_write(static_cast<hid_device*>(hidDevice_),
                           reinterpret_cast<uint8_t*>(&report),
                           sizeof(report));

    return result >= 0;
}
```

#### CDC/Serial PTT Protocol

```cpp
#include <termios.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

bool LinuxAIOCSession::openSerialPort() {
    std::string port = cdcPort_;
    if (port == "auto" || port.empty()) {
        port = findAIOCSerialPort();
        if (port.empty()) {
            lastError_ = "No AIOC serial port found";
            return false;
        }
    }

    serialFd_ = open(port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (serialFd_ < 0) {
        lastError_ = "Failed to open " + port + ": " + strerror(errno);
        return false;
    }

    // Configure serial port (no actual data transfer, just control lines)
    struct termios tty;
    tcgetattr(serialFd_, &tty);
    cfmakeraw(&tty);
    tty.c_cflag |= CLOCAL;  // Ignore modem control lines
    tcsetattr(serialFd_, TCSANOW, &tty);

    return true;
}

bool LinuxAIOCSession::setSerialPttState(bool asserted) {
    if (serialFd_ < 0) return false;

    int status;
    if (ioctl(serialFd_, TIOCMGET, &status) < 0) {
        return false;
    }

    if (asserted) {
        status |= TIOCM_DTR;   // DTR high = PTT on
        status &= ~TIOCM_RTS;  // RTS low
    } else {
        status &= ~TIOCM_DTR;  // DTR low = PTT off
    }

    return ioctl(serialFd_, TIOCMSET, &status) >= 0;
}
```

#### AIOC Serial Port Detection

```cpp
#include <filesystem>
#include <fstream>

std::string LinuxAIOCSession::findAIOCSerialPort() {
    namespace fs = std::filesystem;

    // Check /sys/class/tty for USB serial devices
    for (const auto& entry : fs::directory_iterator("/sys/class/tty")) {
        std::string name = entry.path().filename().string();

        // Only check ttyACM* and ttyUSB* devices
        if (name.find("ttyACM") != 0 && name.find("ttyUSB") != 0) {
            continue;
        }

        // Check if this device belongs to AIOC (by USB VID/PID)
        std::string ueventPath = entry.path().string() + "/device/../uevent";
        std::ifstream uevent(ueventPath);
        std::string line;

        bool isAIOC = false;
        while (std::getline(uevent, line)) {
            // Look for PRODUCT=1209/7388/... (VID/PID)
            if (line.find("PRODUCT=1209/7388") != std::string::npos) {
                isAIOC = true;
                break;
            }
        }

        if (isAIOC) {
            return "/dev/" + name;
        }
    }

    return "";  // Not found
}
```

---

### 2. LinuxAIOCSourcePlugin

**Purpose:** Audio capture from AIOC microphone input with auto-detection.

**Header: `plugins_src/linux/LinuxAIOCSourcePlugin.h`**

```cpp
#pragma once

#include "plugins/AudioSourcePlugin.h"
#include "audio/RingBuffer.h"
#include "LinuxAIOCSession.h"
#include <pulse/pulseaudio.h>
#include <memory>
#include <string>
#include <atomic>
#include <mutex>

namespace nda {

class LinuxAIOCSourcePlugin : public AudioSourcePlugin {
public:
    LinuxAIOCSourcePlugin();
    ~LinuxAIOCSourcePlugin() override;

    // BasePlugin interface
    bool initialize() override;
    void shutdown() override;
    bool start() override;
    void stop() override;

    PluginInfo getInfo() const override;
    PluginState getState() const override;

    void setParameter(const std::string& key, const std::string& value) override;
    std::string getParameter(const std::string& key) const override;

    // AudioSourcePlugin interface
    void setAudioCallback(AudioSourceCallback callback) override;
    bool readAudio(AudioBuffer& buffer) override;

    int getSampleRate() const override;
    int getChannels() const override;
    void setSampleRate(int rate) override;
    void setChannels(int channels) override;
    int getBufferSize() const override;
    void setBufferSize(int samples) override;

    // AIOC-specific
    void setAIOCSession(std::shared_ptr<LinuxAIOCSession> session);

private:
    // AIOC device detection
    std::string findAIOCSourceDevice();

    // PulseAudio resources (similar to PulseAudioSource)
    pa_threaded_mainloop* mainloop_;
    pa_context* context_;
    pa_stream* stream_;

    // Configuration
    int sampleRate_;
    int channels_;
    int bufferSize_;
    std::string deviceName_;
    bool autoDetectDevice_;

    // State
    std::atomic<PluginState> state_;
    AudioSourceCallback callback_;

    // Ring buffer
    static constexpr int RING_BUFFER_FRAMES = 512 * 8;
    RingBuffer ringBuffer_;

    // AIOC session (shared with sink)
    std::shared_ptr<LinuxAIOCSession> alocSession_;

    // Statistics
    std::atomic<uint64_t> underrunCount_;
    std::atomic<uint64_t> overrunCount_;

    mutable std::mutex paramMutex_;

    // PulseAudio callbacks
    static void contextStateCallback(pa_context* c, void* userdata);
    static void streamStateCallback(pa_stream* s, void* userdata);
    static void streamReadCallback(pa_stream* s, size_t nbytes, void* userdata);

    void onContextState(pa_context* c);
    void onStreamState(pa_stream* s);
    void onStreamRead(pa_stream* s, size_t nbytes);

    // Helpers
    bool createContext();
    bool createStream();
    void destroyStream();
    void destroyContext();
    bool waitForContextReady();
    bool waitForStreamReady();
};

} // namespace nda
```

**Auto-Detection Logic:**

```cpp
std::string LinuxAIOCSourcePlugin::findAIOCSourceDevice() {
    auto sources = enumeratePulseSources();

    for (const auto& source : sources) {
        // Case-insensitive search for "AIOC" in name or description
        std::string nameLower = source.name;
        std::string descLower = source.description;
        std::transform(nameLower.begin(), nameLower.end(), nameLower.begin(), ::tolower);
        std::transform(descLower.begin(), descLower.end(), descLower.begin(), ::tolower);

        if (nameLower.find("aioc") != std::string::npos ||
            descLower.find("aioc") != std::string::npos) {
            std::cout << "[LinuxAIOCSource] Auto-detected AIOC device: "
                      << source.description << "\n";
            return source.name;
        }
    }

    return "";  // Not found, will use default
}
```

---

### 3. LinuxAIOCSinkPlugin

**Purpose:** Audio playback to AIOC speaker output with PTT control.

**Header: `plugins_src/linux/LinuxAIOCSinkPlugin.h`**

```cpp
#pragma once

#include "plugins/AudioSinkPlugin.h"
#include "audio/RingBuffer.h"
#include "LinuxAIOCSession.h"
#include <pulse/pulseaudio.h>
#include <memory>
#include <string>
#include <atomic>
#include <mutex>

namespace nda {

class LinuxAIOCSinkPlugin : public AudioSinkPlugin {
public:
    LinuxAIOCSinkPlugin();
    ~LinuxAIOCSinkPlugin() override;

    // BasePlugin interface
    bool initialize() override;
    void shutdown() override;
    bool start() override;
    void stop() override;

    PluginInfo getInfo() const override;
    PluginState getState() const override;

    void setParameter(const std::string& key, const std::string& value) override;
    std::string getParameter(const std::string& key) const override;

    // AudioSinkPlugin interface
    bool writeAudio(const AudioBuffer& buffer) override;

    int getSampleRate() const override;
    int getChannels() const override;
    void setSampleRate(int rate) override;
    void setChannels(int channels) override;
    int getBufferSize() const override;
    void setBufferSize(int samples) override;
    int getAvailableSpace() const override;

    // AIOC-specific: PTT control
    void setAIOCSession(std::shared_ptr<LinuxAIOCSession> session);
    bool setPttState(bool asserted);
    bool isPttAsserted() const;
    LinuxAIOCTelemetry getAIOCTelemetry() const;

private:
    // AIOC device detection
    std::string findAIOCSinkDevice();

    // PulseAudio resources
    pa_threaded_mainloop* mainloop_;
    pa_context* context_;
    pa_stream* stream_;

    // Configuration
    int sampleRate_;
    int channels_;
    int bufferSize_;
    std::string deviceName_;
    bool autoDetectDevice_;
    LinuxPttMode pttMode_;
    std::string cdcPort_;

    // State
    std::atomic<PluginState> state_;

    // Ring buffer
    static constexpr int RING_BUFFER_FRAMES = 512 * 8;
    RingBuffer ringBuffer_;

    // AIOC session (owns or shares with source)
    std::shared_ptr<LinuxAIOCSession> aiocSession_;
    bool ownsSession_;

    // Statistics
    std::atomic<uint64_t> underrunCount_;
    std::atomic<uint64_t> overrunCount_;

    mutable std::mutex paramMutex_;

    // PulseAudio callbacks
    static void contextStateCallback(pa_context* c, void* userdata);
    static void streamStateCallback(pa_stream* s, void* userdata);
    static void streamWriteCallback(pa_stream* s, size_t nbytes, void* userdata);
    static void streamUnderflowCallback(pa_stream* s, void* userdata);

    void onContextState(pa_context* c);
    void onStreamState(pa_stream* s);
    void onStreamWrite(pa_stream* s, size_t nbytes);
    void onStreamUnderflow(pa_stream* s);

    // Helpers
    bool createContext();
    bool createStream();
    void destroyStream();
    void destroyContext();
    bool waitForContextReady();
    bool waitForStreamReady();
    void prefillStream();
};

} // namespace nda
```

**Parameters:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `device` | string | "auto" | PulseAudio sink name. "auto" = find AIOC device |
| `sampleRate` | int | 48000 | Sample rate in Hz |
| `channels` | int | 1 | Channel count (mono for radio) |
| `bufferSize` | int | 512 | Frames per buffer |
| `ptt_mode` | string | "auto" | "auto", "hid", or "cdc" |
| `cdc_port` | string | "auto" | Serial port path or "auto" |

---

## Build System Integration

### CMakeLists.txt Changes

```cmake
# Option to build Linux AIOC plugins
option(BUILD_LINUX_AIOC_PLUGINS "Build Linux AIOC plugins" ON)

if(UNIX AND NOT APPLE AND BUILD_LINUX_PLUGINS)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(PULSEAUDIO REQUIRED libpulse)

    if(BUILD_LINUX_AIOC_PLUGINS)
        # hidapi for HID PTT control
        pkg_check_modules(HIDAPI REQUIRED hidapi-libusb)

        message(STATUS "hidapi found: ${HIDAPI_VERSION}")
        message(STATUS "  Include dirs: ${HIDAPI_INCLUDE_DIRS}")
        message(STATUS "  Libraries: ${HIDAPI_LIBRARIES}")

        # Linux AIOC Source plugin
        add_library(LinuxAIOCSource SHARED
            plugins_src/linux/LinuxAIOCSourcePlugin.cpp
            plugins_src/linux/LinuxAIOCSession.cpp
        )
        target_include_directories(LinuxAIOCSource PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${PULSEAUDIO_INCLUDE_DIRS}
            ${HIDAPI_INCLUDE_DIRS}
        )
        target_link_libraries(LinuxAIOCSource
            ${PULSEAUDIO_LIBRARIES}
            ${HIDAPI_LIBRARIES}
        )
        set_target_properties(LinuxAIOCSource PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugins
        )

        # Linux AIOC Sink plugin
        add_library(LinuxAIOCSink SHARED
            plugins_src/linux/LinuxAIOCSinkPlugin.cpp
            plugins_src/linux/LinuxAIOCSession.cpp
        )
        target_include_directories(LinuxAIOCSink PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${PULSEAUDIO_INCLUDE_DIRS}
            ${HIDAPI_INCLUDE_DIRS}
        )
        target_link_libraries(LinuxAIOCSink
            ${PULSEAUDIO_LIBRARIES}
            ${HIDAPI_LIBRARIES}
        )
        set_target_properties(LinuxAIOCSink PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugins
        )
    endif()
endif()
```

### Package Dependencies

**Fedora:**
```bash
sudo dnf install pulseaudio-libs-devel hidapi-devel
```

**Ubuntu/Debian:**
```bash
sudo apt install libpulse-dev libhidapi-dev
```

**Arch:**
```bash
sudo pacman -S libpulse hidapi
```

### udev Rules (for non-root HID access)

Create `/etc/udev/rules.d/99-aioc.rules`:

```udev
# AIOC device - allow non-root access to HID and serial
SUBSYSTEM=="usb", ATTR{idVendor}=="1209", ATTR{idProduct}=="7388", MODE="0666"
SUBSYSTEM=="hidraw", ATTRS{idVendor}=="1209", ATTRS{idProduct}=="7388", MODE="0666"
KERNEL=="ttyACM[0-9]*", ATTRS{idVendor}=="1209", ATTRS{idProduct}=="7388", MODE="0666", SYMLINK+="aioc"
```

Then reload rules:
```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

---

## UI Integration

### Device Selection Dropdown

The UI should show detected AIOC devices prominently:

```cpp
void populateDeviceDropdown(QComboBox* combo, bool isInput) {
    combo->clear();

    // Add "Auto (AIOC)" option first
    combo->addItem("Auto-detect AIOC", QString("auto"));

    auto devices = isInput ? enumeratePulseSources() : enumeratePulseSinks();

    for (const auto& dev : devices) {
        QString label = QString::fromStdString(dev.description);

        // Highlight AIOC devices
        std::string descLower = dev.description;
        std::transform(descLower.begin(), descLower.end(), descLower.begin(), ::tolower);
        if (descLower.find("aioc") != std::string::npos) {
            label = "[AIOC] " + label;
        }

        if (dev.isDefault) {
            label += " (Default)";
        }

        combo->addItem(label, QString::fromStdString(dev.name));
    }
}
```

### PTT Status Indicator

Add a visual indicator for PTT state in the UI:

```cpp
// In PluginSidebar or MainWindow
void updatePttIndicator(bool asserted) {
    if (asserted) {
        pttIndicator_->setStyleSheet("background-color: red; border-radius: 8px;");
        pttLabel_->setText("PTT: ON");
    } else {
        pttIndicator_->setStyleSheet("background-color: gray; border-radius: 8px;");
        pttLabel_->setText("PTT: OFF");
    }
}
```

### PTT Mode Selection

```cpp
QComboBox* pttModeCombo = new QComboBox();
pttModeCombo->addItem("Auto (HID → CDC)", "auto");
pttModeCombo->addItem("HID Only", "hid");
pttModeCombo->addItem("CDC/Serial Only", "cdc");

connect(pttModeCombo, &QComboBox::currentTextChanged, [this](const QString& text) {
    if (aiocSink_) {
        aiocSink_->setParameter("ptt_mode", text.toStdString());
    }
});
```

---

## Implementation Phases

### Phase 1: LinuxAIOCSession - HID PTT

**Files:** `LinuxAIOCSession.h`, `LinuxAIOCSession.cpp`

**Tasks:**
1. Implement `openHidDevice()` using hidapi
2. Implement `sendHidPttReport()`
3. Implement `closeHidDevice()`
4. Implement `isHidDevicePresent()`
5. Unit test: Open AIOC HID, toggle PTT, verify with multimeter/radio

**Verification:**
- Connect AIOC device
- Call `setPttState(true)` → PTT LED on / radio transmits
- Call `setPttState(false)` → PTT LED off

### Phase 2: LinuxAIOCSession - CDC PTT

**Tasks:**
1. Implement `findAIOCSerialPort()` (scan /sys/class/tty)
2. Implement `openSerialPort()` with termios configuration
3. Implement `setSerialPttState()` with DTR/RTS control
4. Implement `closeSerialPort()`
5. Test PTT via serial on /dev/ttyACM0

**Verification:**
- Set `ptt_mode` to "cdc"
- Toggle PTT via serial DTR
- Verify radio transmits

### Phase 3: AIOC Device Auto-Detection

**Tasks:**
1. Add `findAIOCPulseSources()` to PulseDeviceEnum
2. Add `findAIOCPulseSinks()` to PulseDeviceEnum
3. Test detection with AIOC connected vs disconnected

**Verification:**
- Run enumeration, verify AIOC devices are found
- Check description matching ("AIOC" substring)

### Phase 4: LinuxAIOCSourcePlugin

**Tasks:**
1. Create plugin header and implementation
2. Integrate PulseAudio capture (reuse from PulseAudioSource)
3. Add AIOC auto-detection on initialize()
4. Export plugin factory functions
5. Build as shared library

**Verification:**
- Load plugin dynamically
- Capture audio from AIOC microphone
- Verify audio quality and latency

### Phase 5: LinuxAIOCSinkPlugin

**Tasks:**
1. Create plugin header and implementation
2. Integrate PulseAudio playback (reuse from PulseAudioSink)
3. Add AIOC auto-detection
4. Integrate LinuxAIOCSession for PTT
5. Add `setPttState()` and `isPttAsserted()` methods
6. Build as shared library

**Verification:**
- Load plugin, play audio to AIOC speaker
- Toggle PTT during playback
- Verify radio transmission

### Phase 6: UI Integration

**Tasks:**
1. Add AIOC device dropdown to PluginSidebar
2. Add PTT mode selection
3. Add PTT indicator (red/gray circle)
4. Wire up parameter changes

**Verification:**
- Select AIOC device from dropdown
- Change PTT mode, verify behavior changes
- PTT indicator reflects actual state

### Phase 7: Full Integration Testing

**Tasks:**
1. Full TX pipeline: LinuxAIOCSource → Encryptor → Network Sink
2. Full RX pipeline: Network Source → Decryptor → LinuxAIOCSink
3. PTT control during transmission
4. Hot-plug testing (connect/disconnect AIOC during operation)
5. Long-duration stability test (1+ hours)

---

## Error Handling

### HID Errors

```cpp
bool LinuxAIOCSession::openHidDevice() {
    if (hid_init() < 0) {
        lastError_ = "Failed to initialize hidapi";
        return false;
    }

    hidDevice_ = hid_open(AIOC_VID, AIOC_PID, nullptr);
    if (!hidDevice_) {
        lastError_ = "AIOC HID device not found (VID=1209, PID=7388)";
        return false;
    }

    // Set non-blocking mode
    hid_set_nonblocking(static_cast<hid_device*>(hidDevice_), 1);

    return true;
}
```

### Serial Port Errors

```cpp
bool LinuxAIOCSession::openSerialPort() {
    if (cdcPort_.empty() || cdcPort_ == "auto") {
        cdcPort_ = findAIOCSerialPort();
    }

    if (cdcPort_.empty()) {
        lastError_ = "No AIOC serial port found. Check udev rules and device connection.";
        return false;
    }

    serialFd_ = open(cdcPort_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (serialFd_ < 0) {
        lastError_ = "Failed to open " + cdcPort_ + ": " + std::string(strerror(errno));
        if (errno == EACCES) {
            lastError_ += ". Check udev rules for non-root access.";
        }
        return false;
    }

    return true;
}
```

### Graceful Degradation

If HID fails but CDC works (or vice versa), the plugin should continue with the available PTT method:

```cpp
bool LinuxAIOCSession::connect() {
    bool hidOk = false;
    bool cdcOk = false;

    if (pttMode_ == LinuxPttMode::Auto || pttMode_ == LinuxPttMode::HidManual) {
        hidOk = openHidDevice();
    }

    if (pttMode_ == LinuxPttMode::Auto || pttMode_ == LinuxPttMode::CdcManual) {
        cdcOk = openSerialPort();
    }

    if (pttMode_ == LinuxPttMode::Auto) {
        // Prefer HID if both available
        activePttMode_ = hidOk ? "hid" : (cdcOk ? "cdc" : "none");
    } else if (pttMode_ == LinuxPttMode::HidManual) {
        activePttMode_ = hidOk ? "hid" : "none";
    } else {
        activePttMode_ = cdcOk ? "cdc" : "none";
    }

    connected_ = (activePttMode_ != "none");
    return connected_;
}
```

---

## Testing Checklist

### Unit Tests

- [ ] `LinuxAIOCSession::openHidDevice()` - Opens AIOC HID
- [ ] `LinuxAIOCSession::sendHidPttReport()` - Sends correct HID report
- [ ] `LinuxAIOCSession::findAIOCSerialPort()` - Finds /dev/ttyACM*
- [ ] `LinuxAIOCSession::setSerialPttState()` - Toggles DTR correctly
- [ ] `findAIOCPulseSources()` - Returns AIOC audio input
- [ ] `findAIOCPulseSinks()` - Returns AIOC audio output

### Integration Tests

- [ ] LinuxAIOCSourcePlugin: Initialize → Start → Read 1000 buffers → Stop
- [ ] LinuxAIOCSinkPlugin: Initialize → Start → Write 1000 buffers → Stop
- [ ] PTT via HID: Assert → Verify radio TX → Release
- [ ] PTT via CDC: Assert → Verify radio TX → Release
- [ ] Auto-detection: Plugin finds AIOC without explicit device parameter

### Manual Tests

- [ ] Audio quality: Play audio through AIOC, verify clarity
- [ ] PTT timing: Measure assert-to-TX delay (<50ms target)
- [ ] Hot-plug: Connect/disconnect AIOC during operation
- [ ] Multiple devices: System with multiple audio devices
- [ ] Permissions: Non-root user can access HID and serial
- [ ] Long-duration: Run for 1+ hour without crashes/leaks

---

## Appendix: AIOC Hardware Reference

### USB Identifiers

| Field | Value |
|-------|-------|
| Vendor ID | 0x1209 |
| Product ID | 0x7388 |
| Manufacturer | AIOC Project |
| Product | All-In-One-Cable |

### Audio Interfaces

The AIOC appears as a USB Audio Class device with:
- **Capture**: Mono or stereo microphone input (from radio)
- **Playback**: Mono or stereo speaker output (to radio)
- **Sample Rates**: Typically 8000, 16000, 44100, 48000 Hz

### PTT Interface Options

1. **HID**: 4-byte reports, Report ID 0x00, Byte 1 = PTT state
2. **CDC ACM**: Virtual COM port, DTR line controls PTT
3. **GPIO**: Some variants have physical GPIO (not covered here)

### Serial Port Naming on Linux

| Kernel Name | Device Path | Common For |
|-------------|-------------|------------|
| ttyACM0 | /dev/ttyACM0 | CDC ACM (AIOC default) |
| ttyUSB0 | /dev/ttyUSB0 | FTDI/CH340 adapters |

---

## Conclusion

This plan provides a clear path to full Linux AIOC support while maintaining the clean separation of concerns established in the Windows implementation. The use of hidapi ensures consistent HID handling across platforms, while Linux-specific serial handling provides reliable CDC PTT control.

**Key benefits:**
- Feature parity with Windows AIOC plugins
- Clean plugin architecture (separate AIOC plugins)
- Auto-detection for ease of use
- Fallback between HID and CDC PTT methods
- Comprehensive error handling

**Next steps:** Begin Phase 1 implementation after plan approval.
