# NDA v2.0 - Real-Time Audio Encryption Bridge

Professional Audio Processing Framework - **C++17 + Qt6**

## Overview

NDA (Nade Desktop Application) is a **real-time audio encryption bridge** that sits transparently between audio devices, providing encryption/decryption for secure communication. Built with C++17 and Qt6, NDA features a clean 3-slot plugin architecture, dual independent pipelines for full-duplex operation, and automatic sample rate adaptation for universal device compatibility.

**v2.0 Major Improvements:**
- ✅ Simplified 3-slot architecture (Source → Processor → Sink)
- ✅ Dual independent pipelines (TX + RX simultaneously)
- ✅ Automatic sample rate conversion (works with any device)
- ✅ Python processor plugins (equal to C++)
- ✅ 35% code reduction (bearer/crypto removed from core)
- ✅ Production-ready stability (<50ms latency, <30% CPU)

## Key Features

- **Dual Pipeline Architecture**: Independent TX and RX processing chains
- **Plugin-Based Encryption**: Encryption is plugin-provided, not hardcoded
- **Universal Audio Compatibility**: Auto-resampling handles any sample rate
- **Python & C++ Plugins**: Equal support for rapid prototyping and production
- **Automatic Sample Rate Adaptation**: 48kHz internal, adapts to any endpoint
- **Modular 3-Stage Design**: Source → Processor (optional) → Sink
- **Cross-Platform**: Linux and Windows support
- **Stable Long-Running**: Designed for hours of glitch-free operation
- **Low Resource Usage**: <100MB RAM, <30% CPU (dual pipelines)

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

## Quick Start (v2.0)

### Core Concept

NDA provides **two independent audio pipelines** for full-duplex encrypted communication:

```
TX Pipeline (Transmit):
  Device Mic → [Encryptor] → AIOC/VB-Cable Output
                     ↓
              Encrypted audio sent via external transport

RX Pipeline (Receive):
  AIOC/VB-Cable Input → [Decryptor] → Device Speaker
           ↑
    Encrypted audio received via external transport
```

### Example: Encrypted AIOC Radio

```bash
# 1. Build and run
mkdir build && cd build
cmake .. && make -j
./NDA

# 2. Load plugins
Click "Load Plugins" or "Auto-Load Python Plugins"

# 3. Configure TX Pipeline
Source:    Device Microphone
Processor: AES-256 Encryptor
Sink:      AIOC USB Output

# 4. Configure RX Pipeline  
Source:    AIOC USB Input
Processor: AES-256 Decryptor
Sink:      Device Speaker

# 5. Start both pipelines
Click "Start Both Pipelines"

# Result: Secure two-way radio communication
```

### Example: Encrypted Discord/VoIP

Use virtual audio cables (VB-Cable, PulseAudio loopback):

```
TX: Device Mic → Encryptor → VB-Cable Input
                                  ↓
                         (Discord reads from VB-Cable)

RX: VB-Cable Output → Decryptor → Device Speaker
          ↑
    (Discord writes to VB-Cable)
```

### How to use

-----

### Load the audio plugins

Either by selecting them manually from a custom folder, or by clicking the "Auto-Load" button.

Python plugins in `plugins_py/`:
- `sine_wave_source.py` - Test signal generator
- `sounddevice_microphone.py` - System microphone
- `sounddevice_speaker.py` - System speaker
- `examples/passthrough.py` - No-op processor (testing)
- `examples/simple_gain.py` - Volume control processor
- `examples/fernet_encryptor.py` - Python encryption (demo)
- `examples/fernet_decryptor.py` - Python decryption (demo)

C++ plugins in `build/plugins/`:
- `libSineWaveSourcePlugin.so` - Test signal
- `libNullSinkPlugin.so` - Silent output (testing)
- `libWavFileSinkPlugin.so` - Record to WAV file
- `libAES256EncryptorPlugin.so` - Production encryption
- `libAES256DecryptorPlugin.so` - Production decryption

### Configure your audio pipelines

**v2.0 uses dual pipelines (TX + RX):**

1. **TX Pipeline** - Outbound audio (encrypt before send)
   - Source: Your microphone
   - Processor: Encryptor (optional)
   - Sink: Output device/cable

2. **RX Pipeline** - Inbound audio (decrypt after receive)
   - Source: Input device/cable  
   - Processor: Decryptor (optional)
   - Sink: Your speaker

3. Click "Start Both" to run simultaneously

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

## Development Environment (Windows, MSVC/Ninja)

1. Install prerequisites:
   - Visual Studio Build Tools 2022 with the C++ workload and Windows 10 SDK (10.0.18362+)
   - Qt 6.6.3 MSVC toolchain at `C:\Qt\6.6.3\msvc2019_64`
   - OpenSSL Win64 at `C:\Program Files\OpenSSL-Win64`
   - Ninja available on `PATH`
2. Build (from the VS 2022 Developer Command Prompt):
   - Run `scripts\build_windows_ninja.bat` **or**
   - Manual commands:
     ```bat
     cmake -S . -B build -G "Ninja" ^
       -DCMAKE_PREFIX_PATH="C:/Qt/6.6.3/msvc2019_64" ^
       -DOPENSSL_ROOT_DIR="C:/Program Files/OpenSSL-Win64" ^
       -DCMAKE_BUILD_TYPE=Release
     cmake --build build --config Release
     ```
3. Outputs:
   - App: `build\NDA.exe`
   - C++ plugins: `build\plugins\AIOCSourcePlugin.dll`, `AIOCSinkPlugin.dll`, `SineWaveSourcePlugin.dll`, `NullSinkPlugin.dll`, `WavFileSinkPlugin.dll`
4. Run:
   - Add Qt to `PATH`: `set PATH=C:\Qt\6.6.3\msvc2019_64\bin;%PATH%`
   - Launch `build\NDA.exe`
   - In the UI, choose **Load Plugins from Directory** and select `build\plugins` to load the AIOC source/sink (and sample) plugins.
5. Deploy:
   - `scripts\deploy_windows.bat` or `python scripts\deploy.py` packages the binary and plugins; afterwards run `windeployqt` from `readytoship\bin` to copy Qt DLLs and drop in Python/OpenSSL runtimes as noted in the deploy script output.

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
