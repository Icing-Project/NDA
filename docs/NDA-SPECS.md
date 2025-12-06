# NDA Desktop Application for Windows
## Professional Audio Encryption Bridge System

---

## Executive Summary

The NDA Desktop Application is a Windows and Linux audio processing system designed to provide real-time encryption for various audio communication channels. Built with C++ and Qt6 for maximum performance, it leverages platform audio APIs for optimal throughput while maintaining a flexible plugin architecture for different audio sources and encryption methods. The processing core is a full duplex pipeline where the source → encryptor → bearer → sink chain simultaneously handles outbound capture and inbound playback with symmetric plugin responsibilities.

### Key Features
- **Windows Native Performance**: C++ implementation with direct Windows API access
- **Universal Audio Support**: WASAPI, WDM, ASIO support for professional audio
- **Plugin Architecture**: Hot-swappable DLL modules for different audio sources
- **Ultra-Low Latency**: <5ms with ASIO, <10ms with WASAPI
- **Modern UI**: Dark-themed Qt6 interface with native performance
- **Military-Grade Encryption**: AES-256-GCM with hardware acceleration

---

## Windows Platform Architecture

### Technology Stack

**Selected: C++ with Qt6 (Maximum Performance)**

**Technology Components:**
- **Language**: C++17/20 for modern features and performance
- **GUI Framework**: Qt6 Widgets for cross-platform UI development
- **Build System**: CMake 3.16+ for cross-platform builds
- **Encryption**: OpenSSL 3.x for AES-256-GCM
- **Audio APIs**: WASAPI, ASIO, WDM-KS (Windows native)

**Architecture Advantages:**
- Direct Windows API access
- Minimal latency (<5ms achievable)
- Low memory footprint (50-80MB)
- Hardware-accelerated encryption (AES-NI)
- Native performance without runtime overhead
- Professional-grade audio processing

---

## Windows Audio Architecture

### Audio APIs Available on Windows

```
┌─────────────────────────────────────────────────────────┐
│                   NDA Windows Application              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │            Windows Audio APIs                     │  │
│  ├──────────────────────────────────────────────────┤  │
│  │                                                  │  │
│  │  WASAPI (Windows Audio Session API)              │  │
│  │  - Lowest latency native Windows API             │  │
│  │  - Exclusive/Shared mode                         │  │
│  │  - Default for Windows 10/11                     │  │
│  │                                                  │  │
│  │  WDM-KS (Kernel Streaming)                       │  │
│  │  - Direct kernel access                          │  │
│  │  - Very low latency                              │  │
│  │                                                  │  │
│  │  ASIO (Audio Stream Input/Output)                │  │
│  │  - Professional audio interfaces                 │  │
│  │  - Ultra-low latency (<5ms)                      │  │
│  │  - Requires ASIO4ALL or device drivers           │  │
│  │                                                  │  │
│  │  DirectSound (Legacy)                            │  │
│  │  - Backward compatibility                        │  │
│  │  - Higher latency                                │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Implementation Architecture

### Full Duplex Pipeline Architecture

The processing pipeline is composed of four cooperating plugin slots—**audio source**, **encryptor**, **bearer**, and **audio sink**—and every pipeline run wires them in both the send and receive directions. During capture, the source fills the working buffer, the encryptor signs+encrypts that frame (emitting nonce and tag metadata), the bearer transmits the packet, and the sink plays or monitors the same buffer locally for sidetone workflows. At the same time, the bearer exposes a packet-receive callback so inbound frames can be fed back through the decrypt → render branch: `bearer.setPacketReceivedCallback()` hands the packet to the pipeline, which invokes the encryptor’s `decrypt()` to validate nonce/tag pairs and then writes the restored samples into the sink. This means the bearer acts as a true duplex transport (send APIs plus receive callbacks) and the encryptor is logically positioned between both transport directions.

#### Encryptor responsibilities
- Plugins supply both `encrypt()` and `decrypt()` along with `setKey()` / `generateKey()` helpers so negotiated keys, nonces, and authentication tags remain synchronized across the two legs of the pipeline.
- Nonces are generated per transmit frame (the default implementation uses a 96-bit counter-based nonce) and tags are appended to every bearer packet; decrypt uses the same tag length reported by `getTagSize()`.
- The receive loop extracts nonce+tag from the bearer packet header, calls `decrypt()` inside the callback thread, and only forwards clean PCM frames to the sink when authentication succeeds; failures update bearer statistics and can raise UI alarms.

#### Lifecycle and threading
- During `ProcessingPipeline::start()` both source and sink threads are launched and the bearer registers its receive callback before audio capture begins to guarantee that inbound audio can be rendered immediately, even if capture is not yet delivering frames.
- The processing thread handles outbound capture, but received packets are handled on the bearer’s networking thread; each callback writes into an AudioBuffer pool, decrypts, and schedules a sink write via a lock-free queue so the sink’s render thread never blocks on network I/O.
- Stopping the pipeline tears down callbacks in reverse order (sink → encryptor → bearer → source) ensuring no pending decrypt jobs attempt to touch a destroyed sink.

#### UI and plugin examples
- `PipelineView` presents the source/encryptor/bearer/sink combo boxes as a duplex chain: the hint text explains that selecting a bearer enables both send and receive, and the status label switches to “Pipeline Running (Full Duplex)” after `start()` so operators know inbound audio is armed.
- The sample `UDPBearerPlugin` implements both `sendPacket()` and `setPacketReceivedCallback()` by spinning a receive thread that hands packets to the pipeline; its configuration dialog now shows remote and local endpoints to reinforce the duplex role.
- Dashboard latency meters aggregate outbound buffer depth plus inbound decrypt/render latency, and tooltips highlight when only one direction is active (e.g., “Receive muted—no sink selected”).

### C++/Qt6 Application Structure

```cpp
// main.cpp - Application Entry Point
#include <QApplication>
#include "ui/MainWindow.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    app.setApplicationName("NDA");
    app.setApplicationVersion("1.0.0");

    // Set dark theme
    app.setStyle(QStyleFactory::create("Fusion"));
    QPalette darkPalette;
    darkPalette.setColor(QPalette::Window, QColor(53, 53, 53));
    darkPalette.setColor(QPalette::Base, QColor(25, 25, 25));
    darkPalette.setColor(QPalette::Text, Qt::white);
    app.setPalette(darkPalette);

    MainWindow window;
    window.show();

    return app.exec();
}
```

### Windows-Specific Audio Implementation

```cpp
// AudioEngine.cpp - WASAPI/ASIO Audio Processing
#include "audio/AudioEngine.h"
#include <windows.h>
#include <audioclient.h>
#include <mmdeviceapi.h>

class AudioEngine {
private:
    IAudioClient* pAudioClient;
    IAudioRenderClient* pRenderClient;
    IAudioCaptureClient* pCaptureClient;
    WAVEFORMATEX* pwfx;
    Encryptor* encryptor_;

public:
    bool initialize(int sampleRate, int bufferSize) {
        // Initialize COM
        CoInitialize(nullptr);

        // Get default audio endpoint
        IMMDeviceEnumerator* pEnumerator;
        IMMDevice* pDevice;

        CoCreateInstance(
            __uuidof(MMDeviceEnumerator),
            nullptr,
            CLSCTX_ALL,
            __uuidof(IMMDeviceEnumerator),
            (void**)&pEnumerator
        );

        // Get default audio device
        pEnumerator->GetDefaultAudioEndpoint(
            eRender,
            eConsole,
            &pDevice
        );

        // Initialize audio client
        pDevice->Activate(
            __uuidof(IAudioClient),
            CLSCTX_ALL,
            nullptr,
            (void**)&pAudioClient
        );

        // Configure for low latency
        REFERENCE_TIME hnsRequestedDuration = 10000; // 1ms
        pAudioClient->Initialize(
            AUDCLNT_SHAREMODE_EXCLUSIVE,  // Low latency
            AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
            hnsRequestedDuration,
            0,
            pwfx,
            nullptr
        );

        return true;
    }

    void processAudio(const AudioBuffer& input, AudioBuffer& output) {
        // Copy input to output
        output.copyFrom(input);

        // Apply encryption
        if (encryptor_) {
            float* data = output.getChannelData(0);
            encryptor_->encryptAudio(data, output.getFrameCount(),
                                    output.getChannelCount());
        }
    }
};
```

### Plugin System Architecture

```cpp
// PluginInterface.h - Base Plugin Interface
#ifndef PLUGININTERFACE_H
#define PLUGININTERFACE_H

#include <string>
#include "audio/AudioBuffer.h"

enum class PluginType {
    AudioSource,    // Spotify, YouTube Music, etc.
    AudioEffect,    // Equalizer, compressor, etc.
    Transport       // Network streaming, file output, etc.
};

struct PluginInfo {
    std::string name;
    std::string version;
    std::string author;
    std::string description;
    PluginType type;
    int apiVersion;
};

class PluginInterface {
public:
    virtual ~PluginInterface() = default;

    // Plugin lifecycle
    virtual bool initialize() = 0;
    virtual void shutdown() = 0;

    // Plugin information
    virtual PluginInfo getInfo() const = 0;

    // Audio processing
    virtual void processAudio(AudioBuffer& buffer) = 0;

    // Plugin-specific settings
    virtual void configure(const std::string& key,
                          const std::string& value) = 0;

    // State
    virtual bool isActive() const = 0;
    virtual void setActive(bool active) = 0;
};

// Plugin factory macros
#define NDA_PLUGIN_EXPORT extern "C" __declspec(dllexport)

#define NDA_DECLARE_PLUGIN(PluginClass) \
    NDA_PLUGIN_EXPORT PluginInterface* createPlugin() { \
        return new PluginClass(); \
    } \
    NDA_PLUGIN_EXPORT void destroyPlugin(PluginInterface* plugin) { \
        delete plugin; \
    }

#endif // PLUGININTERFACE_H
```

### Example Plugin Implementation

```cpp
// SpotifySourcePlugin.cpp
#include "PluginInterface.h"

class SpotifySourcePlugin : public PluginInterface {
private:
    bool active_;
    // Spotify SDK integration here

public:
    bool initialize() override {
        // Initialize Spotify SDK
        active_ = false;
        return true;
    }

    void shutdown() override {
        // Cleanup Spotify connection
    }

    PluginInfo getInfo() const override {
        return {
            "Spotify Source",
            "1.0.0",
            "Icing Project",
            "Captures audio from Spotify playback",
            PluginType::AudioSource,
            1
        };
    }

    void processAudio(AudioBuffer& buffer) override {
        if (!active_) return;
        // Capture Spotify audio and fill buffer
    }

    void configure(const std::string& key,
                  const std::string& value) override {
        // Handle configuration
    }

    bool isActive() const override { return active_; }
    void setActive(bool active) override { active_ = active; }
};

// Export plugin
NDA_DECLARE_PLUGIN(SpotifySourcePlugin)
```

---

## Windows-Specific Features

### 1. Encryption Implementation

```cpp
// Encryptor.cpp - AES-256-GCM with Hardware Acceleration
#include "crypto/Encryptor.h"
#include <openssl/evp.h>
#include <openssl/aes.h>

class Encryptor {
private:
    EVP_CIPHER_CTX* ctx_;
    std::vector<uint8_t> key_;
    bool hardwareAccelerated_;

public:
    bool encryptAudio(float* audioData, size_t sampleCount, int channels) {
        // Convert float samples to bytes
        size_t totalSamples = sampleCount * channels;
        uint8_t* byteData = reinterpret_cast<uint8_t*>(audioData);
        size_t byteSize = totalSamples * sizeof(float);

        // Generate unique nonce for this buffer
        uint8_t nonce[12];
        generateNonce(nonce, 12);

        // Encrypt using AES-256-GCM (hardware accelerated if available)
        EVP_EncryptInit_ex(ctx_, EVP_aes_256_gcm(), nullptr,
                          key_.data(), nonce);

        int outLen;
        EVP_EncryptUpdate(ctx_, byteData, &outLen, byteData, byteSize);

        // Get authentication tag
        uint8_t tag[16];
        EVP_CIPHER_CTX_ctrl(ctx_, EVP_CTRL_GCM_GET_TAG, 16, tag);

        return true;
    }

    bool isHardwareAccelerated() const {
        // Check for AES-NI support
        return hardwareAccelerated_;
    }
};
```

### 2. Qt UI Components

```cpp
// Dashboard.cpp - Main Dashboard View
#include "ui/Dashboard.h"

class Dashboard : public QWidget {
private:
    QPushButton* startStopButton;
    QLabel* statusLabel;
    QProgressBar* inputMeterL;
    QProgressBar* inputMeterR;
    QProgressBar* outputMeterL;
    QProgressBar* outputMeterR;
    bool isStreaming;

public:
    void setupUI() {
        QVBoxLayout* mainLayout = new QVBoxLayout(this);

        // Control buttons
        QGroupBox* controlGroup = new QGroupBox("Stream Control");
        startStopButton = new QPushButton("Start Stream");
        startStopButton->setMinimumHeight(50);
        connect(startStopButton, &QPushButton::clicked,
                this, &Dashboard::onStartStopClicked);

        // Audio meters for real-time monitoring
        inputMeterL = new QProgressBar();
        inputMeterR = new QProgressBar();
        outputMeterL = new QProgressBar();
        outputMeterR = new QProgressBar();

        // Timer for updating meters
        QTimer* meterTimer = new QTimer(this);
        connect(meterTimer, &QTimer::timeout,
                this, &Dashboard::updateAudioMeters);
        meterTimer->start(50); // 50ms updates (20 FPS)
    }

    void onStartStopClicked() {
        isStreaming = !isStreaming;
        if (isStreaming) {
            startStopButton->setText("Stop Stream");
            statusLabel->setText("Status: Running");
            emit streamStarted();
        } else {
            startStopButton->setText("Start Stream");
            statusLabel->setText("Status: Stopped");
            emit streamStopped();
        }
    }
};
```

---

## Directory Structure (C++/Qt Project)

```
NDA/
├── src/                         # Source files
│   ├── main.cpp                 # Application entry point
│   │
│   ├── ui/                      # Qt UI components
│   │   ├── MainWindow.cpp
│   │   ├── Dashboard.cpp
│   │   ├── AudioDevicesView.cpp
│   │   ├── EncryptionView.cpp
│   │   ├── PluginsView.cpp
│   │   └── SettingsView.cpp
│   │
│   ├── audio/                   # Audio engine
│   │   ├── AudioEngine.cpp      # Main audio processing
│   │   ├── AudioDevice.cpp      # WASAPI/ASIO device handling
│   │   └── AudioBuffer.cpp      # Audio buffer management
│   │
│   ├── crypto/                  # Encryption engine
│   │   ├── Encryptor.cpp        # AES-256-GCM encryption
│   │   └── KeyExchange.cpp      # ECDH key exchange
│   │
│   └── plugins/                 # Plugin system
│       ├── PluginInterface.cpp
│       └── PluginManager.cpp    # Plugin loading/management
│
├── include/                     # Public headers
│   ├── ui/                      # UI headers
│   │   ├── MainWindow.h
│   │   ├── Dashboard.h
│   │   ├── AudioDevicesView.h
│   │   ├── EncryptionView.h
│   │   ├── PluginsView.h
│   │   └── SettingsView.h
│   │
│   ├── audio/                   # Audio headers
│   │   ├── AudioEngine.h
│   │   ├── AudioDevice.h
│   │   └── AudioBuffer.h
│   │
│   ├── crypto/                  # Crypto headers
│   │   ├── Encryptor.h
│   │   └── KeyExchange.h
│   │
│   └── plugins/                 # Plugin headers
│       ├── PluginInterface.h
│       └── PluginManager.h
│
├── resources/                   # Application resources
│   ├── icons/                   # Application icons
│   ├── images/                  # UI images
│   └── styles/                  # Qt stylesheets
│
├── cmake/                       # CMake modules
│   └── FindQt6.cmake
│
├── tests/                       # Unit tests
│   ├── audio/
│   ├── crypto/
│   └── plugins/
│
├── docs/                        # Documentation
│   ├── API.md
│   ├── BUILDING.md
│   └── PLUGINS.md
│
├── CMakeLists.txt               # Main CMake configuration
├── README.md
├── NDA-SPECS.md
└── .gitignore
```

---

## Building and Distribution

### Development Setup

**Prerequisites:**
- Qt6 (6.2 or later)
- CMake 3.16 or later
- OpenSSL 3.x
- C++ compiler (MSVC 2019/2022 on Windows, GCC/Clang on Linux)
- Visual Studio 2022 (Windows) or build-essential (Linux)

**Linux Development (Cross-platform):**
```bash
# Install dependencies (Fedora)
sudo dnf install qt6-qtbase-devel qt6-qtbase-gui cmake openssl-devel gcc-c++

# Or Ubuntu/Debian
sudo apt install qt6-base-dev cmake libssl-dev build-essential

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build
make -j$(nproc)

# Run
./NDA
```

**Windows Build (Target Platform):**
```cmd
REM Install Qt6 from qt.io
REM Install Visual Studio 2022 with C++ workload
REM Install OpenSSL for Windows

REM Create build directory
mkdir build
cd build

REM Configure with CMake (use Qt6 path)
cmake .. -G "Visual Studio 17 2022" -DCMAKE_PREFIX_PATH=C:\Qt\6.5.0\msvc2019_64

REM Build
cmake --build . --config Release

REM Run
Release\NDA.exe
```

### CMake Build Configuration

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.16)
project(NDA VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Find Qt6
find_package(Qt6 REQUIRED COMPONENTS Core Widgets Gui Network)

# Find OpenSSL
find_package(OpenSSL REQUIRED)

# Add executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(${PROJECT_NAME}
    Qt6::Core
    Qt6::Widgets
    Qt6::Gui
    Qt6::Network
    OpenSSL::SSL
    OpenSSL::Crypto
)

# Windows-specific linking
if(WIN32)
    target_link_libraries(${PROJECT_NAME} winmm dsound)
endif()
```

### Creating Windows Installer

Use NSIS or WiX Toolset for creating professional Windows installers:

```nsis
# installer.nsi (NSIS Script)
!define APP_NAME "NDA Desktop"
!define APP_VERSION "1.0.0"
!define APP_PUBLISHER "Icing Project"

Name "${APP_NAME}"
OutFile "NDA-Setup-${APP_VERSION}.exe"
InstallDir "$PROGRAMFILES64\NDA"

Section "Install"
    SetOutPath "$INSTDIR"
    File "Release\NDA.exe"
    File "*.dll"  # Qt and OpenSSL DLLs

    CreateDirectory "$SMPROGRAMS\${APP_NAME}"
    CreateShortcut "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk" "$INSTDIR\NDA.exe"
    CreateShortcut "$DESKTOP\${APP_NAME}.lnk" "$INSTDIR\NDA.exe"

    WriteUninstaller "$INSTDIR\Uninstall.exe"
SectionEnd
```

---

## Windows-Specific Optimizations

### 1. Process Priority (C++)

```cpp
// Set high priority for audio processing thread
#include <windows.h>

void AudioEngine::audioThread() {
    // Set thread priority to time-critical
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);

    // Set process priority to high
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);

    // Audio processing loop
    while (isRunning_) {
        processAudio(inputBuffer, outputBuffer);
    }
}
```

### 2. Memory Locking

```cpp
// Lock memory pages to prevent paging (reduce latency)
#include <windows.h>

bool AudioEngine::lockMemory() {
    // Increase working set size
    SIZE_T minSize, maxSize;
    GetProcessWorkingSetSize(GetCurrentProcess(), &minSize, &maxSize);
    SetProcessWorkingSetSize(GetCurrentProcess(),
                            minSize + (50 * 1024 * 1024),  // +50MB
                            maxSize + (100 * 1024 * 1024)); // +100MB

    // Lock audio buffer memory
    VirtualLock(audioBuffer_.data(), audioBuffer_.size());

    return true;
}
```

### 3. SIMD Optimization

```cpp
// Use SSE/AVX for audio processing
#include <immintrin.h>

void AudioEngine::processAudioSIMD(float* input, float* output, size_t samples) {
    // Process 8 samples at once with AVX
    size_t i = 0;
    for (; i + 8 <= samples; i += 8) {
        __m256 data = _mm256_load_ps(&input[i]);
        // Apply processing with SIMD instructions
        _mm256_store_ps(&output[i], data);
    }

    // Process remaining samples
    for (; i < samples; ++i) {
        output[i] = input[i];
    }
}
```

---

## Performance Specifications (Windows)

### System Requirements

| Component | Minimum | Recommended | Professional |
|-----------|---------|-------------|--------------|
| **OS** | Windows 10 20H2 | Windows 11 | Windows 11 Pro |
| **CPU** | Intel i3/AMD Ryzen 3 | Intel i5/AMD Ryzen 5 | Intel i7/AMD Ryzen 7 |
| **RAM** | 4GB | 8GB | 16GB |
| **Audio** | Onboard audio | USB Audio Interface | ASIO-compatible interface |
| **Dependencies** | Qt6 Runtime, OpenSSL | Qt6 Runtime, OpenSSL | Qt6 Runtime, OpenSSL |
| **Visual C++** | 2019 Redistributable | 2022 Redistributable | 2022 Redistributable |

### Performance Metrics (C++/Qt Implementation)

| Configuration | Latency | CPU Usage | RAM Usage |
|--------------|---------|-----------|-----------|
| **WASAPI Shared** | 10-15ms | 2-3% | 50MB |
| **WASAPI Exclusive** | 5-10ms | 3-5% | 60MB |
| **ASIO** | 2-5ms | 5-8% | 70MB |
| **WDM-KS** | 3-8ms | 4-7% | 65MB |

**Note:** C++/Qt implementation provides significantly better performance than Electron:
- 50% lower latency
- 50% less memory usage
- 40% lower CPU usage
- Direct hardware access for encryption (AES-NI)

---

## Troubleshooting (Windows)

### Common Issues

1. **"Cannot find Qt6 libraries"**
   - Solution: Install Qt6 runtime or deploy with application
   - Set Qt6_DIR environment variable

2. **"OpenSSL not found"**
   - Install OpenSSL for Windows
   - Add OpenSSL bin directory to PATH

3. **Audio device not detected**
   - Check Windows audio device drivers
   - Run application as Administrator for exclusive mode

4. **High latency or audio dropouts**
   - Reduce buffer size in settings
   - Use ASIO driver for professional audio interfaces
   - Disable Windows audio enhancements

5. **Plugin loading fails**
   - Ensure plugin API version matches
   - Check plugin dependencies (DLLs)
   - Verify plugin is 64-bit (or 32-bit if using x86 build)

---

## Security Considerations (Windows)

### Windows-Specific Security

1. **Code Signing**
   - EV Certificate recommended for immediate SmartScreen trust
   - Standard certificate requires reputation building

2. **Encryption Key Storage**
   - Use Windows Data Protection API (DPAPI)
   - Integrate with Windows Hello for biometric authentication
   - Store keys in encrypted format

3. **Memory Protection**
   - Use SecureZeroMemory for sensitive data cleanup
   - Lock audio buffer memory to prevent paging
   - Enable DEP (Data Execution Prevention)

4. **Plugin Security**
   - Verify plugin signatures before loading
   - Sandbox plugin execution
   - Validate plugin API version compatibility

---

## Conclusion

The NDA Desktop Application for Windows leverages C++ and Qt6 to provide maximum performance and native Windows integration. By using direct Windows audio APIs (WASAPI, ASIO, WDM-KS) and hardware-accelerated encryption (AES-NI), the application achieves professional-grade performance with ultra-low latency and minimal resource usage.

The C++/Qt architecture provides:
- **Superior Performance**: <5ms latency, <10% CPU usage, <100MB RAM
- **Native Integration**: Direct Windows API access without runtime overhead
- **Cross-Platform Development**: Develop on Linux, deploy on Windows
- **Plugin Extensibility**: Hot-swappable DLL modules for audio sources
- **Military-Grade Security**: AES-256-GCM with hardware acceleration

This implementation is ideal for professional audio encryption applications requiring real-time processing with minimal latency.

---

*Version: 1.0 - C++/Qt Edition*
*Platform: Windows 10/11 (x64)*
*Development: Linux/Windows*
*Stack: C++17, Qt6, CMake, OpenSSL*
