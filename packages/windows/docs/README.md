# NADE Desktop Application

Professional Audio Encryption Bridge System for Windows - **C++/Qt6 Implementation**

## Overview

NADE Desktop is a high-performance, Windows-native audio processing system designed to provide real-time encryption for various audio communication channels. Built with modern C++17/20 and Qt6 for maximum performance, it leverages Windows audio APIs (WASAPI/ASIO) for ultra-low latency while maintaining a flexible plugin architecture for extensibility.

## Key Features

- **Native Windows Performance**: Direct C++ implementation with <5ms latency
- **Professional Audio APIs**: WASAPI Exclusive Mode, ASIO, WDM-KS support
- **Modern Qt UI**: Native Windows 11 styled interface with Qt6
- **Plugin Architecture**: Hot-swappable modules for different audio sources
- **Real-time Encryption**: AES-256-GCM with hardware acceleration
- **Cross-Platform Development**: Develop on Linux, deploy on Windows
- **Low Memory Footprint**: ~50-100MB RAM usage

## Technology Stack

- **Language**: C++17 (minimum), C++20 (preferred)
- **GUI Framework**: Qt6 Widgets (6.2+)
- **Build System**: CMake 3.16+
- **Audio APIs**:
  - WASAPI (Windows Core Audio)
  - ASIO SDK (optional, for pro audio)
  - WDM-KS (kernel streaming)
- **Encryption**: OpenSSL 3.x (AES-256-GCM)

## System Requirements

### Development (Linux)
- Fedora 41+ / Ubuntu 22.04+ / Debian 12+
- GCC 11+ or Clang 14+
- Qt6 (6.2+)
- CMake 3.16+
- OpenSSL 3.x
- Git

### Target Platform (Windows)
**Minimum**:
- Windows 10 20H2 (64-bit)
- Intel i3 / AMD Ryzen 3 or equivalent
- 4GB RAM
- Onboard audio device

**Recommended**:
- Windows 11 (64-bit)
- Intel i5 / AMD Ryzen 5 or equivalent
- 8GB RAM
- USB Audio Interface

**Professional**:
- Windows 11 Pro (64-bit)
- Intel i7 / AMD Ryzen 7 or equivalent
- 16GB RAM
- ASIO-compatible audio interface

## Installation (Development on Linux)

### 1. Install Dependencies (Fedora)

```bash
# Install Qt6 and development tools
sudo dnf install \
    qt6-qtbase-devel \
    qt6-qtbase-gui \
    cmake \
    gcc-c++ \
    git \
    openssl-devel

# Optional: for cross-compilation to Windows
sudo dnf install mingw64-qt6-qtbase mingw64-gcc-c++
```

### 2. Build the Project

```bash
# From project root directory
cd /home/bartosz/delivery/NDA

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build
make -j$(nproc)

# Run (on Linux for testing UI)
./NADE
```

## Building for Windows

### Option 1: Cross-Compile from Linux

```bash
# Using MinGW cross-compiler
mkdir build-windows && cd build-windows
cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/mingw-w64.cmake
cmake --build . -j$(nproc)
```

### Option 2: Build on Windows

```powershell
# Install Qt 6 from https://www.qt.io/download-open-source
# Install Visual Studio 2022 or MinGW

# In project directory
mkdir build
cd build
cmake -G "Visual Studio 17 2022" ..
cmake --build . --config Release
```

## Project Structure

```
NDA/
├── CMakeLists.txt               # Main CMake build configuration
├── README.md                    # This file
├── NDA-SPECS.md                 # Detailed specifications
├── .gitignore
│
├── src/                         # Source files
│   ├── main.cpp                 # Application entry point
│   │
│   ├── ui/                      # Qt UI components
│   │   ├── MainWindow.cpp
│   │   ├── Dashboard.cpp        # Stream control & audio meters
│   │   ├── AudioDevicesView.cpp # Audio device management
│   │   ├── EncryptionView.cpp   # Encryption settings
│   │   ├── PluginsView.cpp      # Plugin management
│   │   └── SettingsView.cpp     # Application settings
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
│       ├── PluginInterface.cpp  # Plugin base interface
│       └── PluginManager.cpp    # Plugin loader/manager
│
├── include/                     # Public headers
│   ├── ui/                      # UI headers
│   ├── audio/                   # Audio headers
│   ├── crypto/                  # Crypto headers
│   └── plugins/                 # Plugin headers
│
├── resources/                   # Application resources
│   ├── icons/
│   ├── images/
│   └── styles/
│
├── cmake/                       # CMake modules
├── tests/                       # Unit tests
└── docs/                        # Documentation
```

## Architecture

### Core Components

1. **Qt Main Window** (`MainWindow`)
   - Modern dark theme UI
   - Tab-based navigation
   - Real-time status updates
   - System tray integration

2. **Audio Engine** (`AudioEngine`)
   - Multi-threaded audio processing
   - Lock-free ring buffers
   - Device enumeration and management
   - Real-time latency monitoring

3. **Plugin System** (`PluginManager`)
   - Dynamic plugin loading (.dll/.so)
   - Version checking
   - Dependency resolution
   - Hot-swapping support

4. **Encryption Core** (`Encryptor`)
   - AES-256-GCM hardware accelerated
   - ChaCha20-Poly1305 support
   - ECDH/RSA/X25519 key exchange
   - Minimal processing overhead

### Threading Model

```
Main Thread (Qt GUI)
├─> Audio Thread (High Priority)
│   ├─> Input Callback
│   ├─> Processing Pipeline
│   └─> Output Callback
├─> Plugin Thread (Normal Priority)
└─> Network Thread (for key exchange)
```

## Usage

### Basic Operation

1. Launch NADE Desktop
2. **Dashboard Tab**: View stream status, audio meters, and performance metrics
3. **Audio Devices Tab**: Select input/output devices and configure sample rate/buffer size
4. **Encryption Tab**: Generate or import encryption keys, select algorithm
5. **Plugins Tab**: Load audio source plugins (Spotify, YouTube Music, etc.)
6. **Settings Tab**: Configure application preferences
7. Click **"Start Stream"** on Dashboard to begin encrypted audio processing

### UI Components

**Dashboard View:**
- Start/Stop stream control
- Real-time audio level meters (L/R input and output)
- Performance metrics (latency, CPU, memory)
- Stream status indicator

**Audio Devices View:**
- Device selection (WASAPI, ASIO)
- Sample rate configuration (44.1kHz, 48kHz, 96kHz, 192kHz)
- Buffer size selection (64, 128, 256, 512, 1024 samples)
- Device information display

**Encryption View:**
- Algorithm selection (AES-256-GCM, ChaCha20-Poly1305)
- Key generation and management
- Import/Export encryption keys
- Hardware acceleration status

**Plugins View:**
- Load/unload plugins
- View installed plugins
- Plugin information and status

**Settings View:**
- General settings (auto-start, minimize to tray)
- Performance settings (latency, CPU priority)
- System information

## Performance Targets

| Metric | Target | Typical |
|--------|--------|---------|
| **Latency** | <5ms | 3-4ms (ASIO) |
| **CPU Usage** | <10% | 5-7% |
| **RAM Usage** | <100MB | 50-70MB |
| **Buffer Size** | 64-256 samples | 128 samples |

## Development

### Code Style

- **C++ Standard**: C++17 minimum, C++20 preferred
- **Naming**: CamelCase for classes, camelCase for methods
- **Headers**: Use `#pragma once`
- **Smart Pointers**: Prefer `std::unique_ptr` and `std::shared_ptr`

### Building with Debug Symbols

```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake --build .
```

### Running Tests

```bash
ctest --output-on-failure
```

## Troubleshooting

### Linux Development

**Qt not found**:
```bash
export CMAKE_PREFIX_PATH=/usr/lib64/cmake/Qt6
```

**Build errors**:
```bash
# Clean build
rm -rf build && mkdir build && cd build
cmake .. && cmake --build .
```

### Windows Deployment

**Missing DLL errors**:
```bash
# Use windeployqt to copy Qt DLLs
windeployqt.exe nade-desktop.exe
```

## License

MIT License - See LICENSE file for details

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## Roadmap

- [x] Project structure and CMake setup
- [x] Qt6 UI framework (MainWindow, Dashboard, Views)
- [x] Audio engine framework (AudioEngine, AudioDevice, AudioBuffer)
- [x] Encryption framework (Encryptor, KeyExchange)
- [x] Plugin system (PluginInterface, PluginManager)
- [ ] WASAPI audio implementation (Windows-specific)
- [ ] ASIO support (Windows-specific)
- [ ] OpenSSL integration for encryption
- [ ] Plugin DLL loading implementation
- [ ] Windows installer (NSIS)
- [ ] Cross-platform build testing
- [ ] API documentation

## Support

For issues and feature requests, please use the GitHub issue tracker.

---

**Built with ❤️ using C++ and Qt**
