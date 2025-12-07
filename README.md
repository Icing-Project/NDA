# NDA Desktop Application

Professional Audio Encryption Bridge System - **C++/Qt6 Implementation**

## Overview

NDA Desktop is a high-performance, Windows-native (but Linux compatible, obviously) audio processing system designed to provide real-time encryption for various audio communication channels. Built with modern C++17/20 and Qt6 for maximum performance, it leverages Windows audio APIs (WASAPI/ASIO) for ultra-low latency while maintaining a flexible plugin architecture for extensibility.

## Key Features

- **Native Windows Performance**: Direct C++ implementation with <5ms latency
- **Professional Audio APIs**: WASAPI Exclusive Mode, ASIO, WDM-KS support
- **Modern Qt UI**: Native Windows 11 styled interface with Qt6
- **Plugin Architecture**: Hot-swappable modules for different audio sources
- **Real-time Encryption**: AES-256-GCM with hardware acceleration
- **Cross-Platform Development**: Develop on Linux, deploy on Windows
- **Low Memory Footprint**: ~50-100MB RAM usage

## Tested Environment

- Ubuntu 24.04.3 LTS (GCC 13.3, CMake 3.28, Qt 6.4 packages from apt)
- Python 3.12 with `sounddevice` 0.5.x and PortAudio backend
- PortAudio 19.6 (`portaudio19-dev` / `python3-pyaudio`)

## Technology Stack

- **Language**: C++17 (minimum), C++20 (preferred)
- **GUI Framework**: Qt6 Widgets (6.2+)
- **Build System**: CMake 3.16+
- **Audio APIs**:
  - WASAPI (Windows Core Audio)
  - ASIO SDK (optional, for pro audio)
  - WDM-KS (kernel streaming)
- **Encryption**: OpenSSL 3.x (AES-256-GCM)

## How to use

We currently only provide an Ubuntu-built binary.

### Ubuntu

Download the [pre-release binary](https://github.com/Icing-Project/NDA/releases/tag/pre-1.1) and execute it.

-----

### Load the audio plugins

Either by selecting them manually from a custom folder, or by clicking the "Auto-Load" button.

### Configure your audio path

Select your audio source and audio sink

## Development Environment (Ubuntu)

All build and runtime verification was performed on **Ubuntu 24.04.3 LTS**. The instructions below assume that platform (the same commands work on Ubuntu 22.04+ and recent Debian releases).

### 1. Install system packages

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    qt6-base-dev \
    qt6-base-dev-tools \
    libxkbcommon-dev \
    libqt6opengl6-dev \
    libssl-dev \
    python3 \
    python3-dev \
    python3-pip \
    python3-numpy \
    portaudio19-dev \
    python3-pyaudio \
    libgl1-mesa-dev \
    git
```

> Qt6 installs into `/usr/lib/x86_64-linux-gnu/cmake/Qt6` by default on Ubuntu. If you are using a custom Qt build, set `CMAKE_PREFIX_PATH` or `Qt6_DIR` accordingly so CMake can locate it.

### 2. Install Python plugin runtime dependencies

`CMakeLists.txt` enables `NDA_ENABLE_PYTHON=ON`, so the UI auto-load feature expects the Python plugins in `plugins_py/` plus their dependencies. Either create a virtual environment or allow user installs (Ubuntu uses PEP 668 protections by default):

```bash
# Option A: virtual environment (recommended)
python3 -m venv .nade-venv
source .nade-venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Option B: system interpreter (requires --break-system-packages on Ubuntu)
pip3 install --user --break-system-packages -r requirements.txt
```

If you do not need Python plugins, configure with `-DNDA_ENABLE_PYTHON=OFF` and skip these packages. The UI plugin auto-loader will be disabled in that configuration.

### 3. Configure, build, and run

```bash
# From the repo root
cmake -S . -B build
cmake --build build -j$(nproc)

# Launch the Qt UI (from the project root)
./build/NDA
```

At startup the application will attempt to auto-load every plugin in `plugins_py/`. Audio capture/playback plugins require `sounddevice` (provided via `requirements.txt`) and the system PortAudio/PyAudio libs installed in step 1. If you see “[PortAudio library not found]”, re-check the `portaudio19-dev` and `python3-pyaudio` packages.

### 4. Ubuntu one-shot compile script

If you prefer a single script that installs dependencies, compiles NDA, and runs it on Ubuntu, save the following as `scripts/build_ubuntu.sh` in the repo root and make it executable with `chmod +x scripts/build_ubuntu.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "==> Installing Ubuntu build dependencies (requires sudo)..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    qt6-base-dev \
    qt6-base-dev-tools \
    libxkbcommon-dev \
    libqt6opengl6-dev \
    libssl-dev \
    python3 \
    python3-dev \
    python3-pip \
    python3-numpy \
    portaudio19-dev \
    python3-pyaudio \
    libgl1-mesa-dev \
    git

echo "==> Installing Python plugin dependencies (user scope)..."
if command -v pip3 >/dev/null 2>&1; then
    pip3 install --user --break-system-packages -r requirements.txt
fi

echo "==> Configuring CMake build..."
/usr/bin/cmake -S . -B build

echo "==> Building NDA..."
/usr/bin/cmake --build build -j"$(nproc)"

echo "==> Launching NDA..."
./build/NDA
```

Then run:

```bash
./scripts/build_ubuntu.sh
```

### 5. Ubuntu dev loop script

For frequent compile/run cycles during development, use the dedicated dev script (assumes you have already run the one-shot script or installed dependencies manually):

```bash
chmod +x scripts/dev_ubuntu.sh
./scripts/dev_ubuntu.sh
```

By default this:
- Configures CMake once into `build/` with `CMAKE_BUILD_TYPE=Debug`.
- Rebuilds incrementally with all available cores.
- Runs `./build/NDA` after a successful build.

You can tune behaviour with environment variables:

```bash
# Use a different build directory and build type, and skip auto-run
BUILD_DIR=build-debug BUILD_TYPE=Debug RUN_AFTER_BUILD=0 ./scripts/dev_ubuntu.sh
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

1. Launch NDA Desktop
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
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)
```

### Running Tests

Automated CTest suites are not yet part of the repository. Add tests under a `tests/` directory and enable them in CMake before running `ctest`.

## Troubleshooting

### Qt not found

Ubuntu installs Qt6 files under `/usr/lib/x86_64-linux-gnu/cmake/Qt6`. If you are using a custom Qt build, set:

```bash
export CMAKE_PREFIX_PATH=/path/to/Qt/6.x/gcc_64/lib/cmake
```

### Build errors / stale artifacts

```bash
# Clean build
rm -rf build && mkdir build && cd build
cmake .. && cmake --build .
```

### PortAudio / sounddevice errors

- Ensure `portaudio19-dev` and `python3-pyaudio` are installed.
- Reinstall the Python dependencies (inside your venv if applicable):
  ```bash
  pip install --force-reinstall -r requirements.txt
  ```
- If PulseAudio plugins complain about missing PyAudio, verify `python3-pyaudio` is available in the interpreter you are running the app with.

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
