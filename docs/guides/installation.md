# Installation & Build Guide

Build NDA from source for your platform, or use the pre-built standalone package.

---

## 🚀 Quick Start: Standalone Portable Package (Windows Only)

**For end users who just want to run NDA without building from source.**

### Download and Run

1. **Download** the standalone package: `NDA-Windows-Portable-v2.0.0.zip`
2. **Extract** the ZIP to any location (e.g., Desktop, USB drive, etc.)
3. **Launch** by double-clicking `NDA.bat`

That's it! No installation, no dependencies to install, everything is included.

**Package includes:**
- ✅ NDA.exe with all plugins (C++ and Python)
- ✅ Qt6 framework
- ✅ OpenSSL libraries
- ✅ Python runtime (if enabled)
- ✅ All required DLLs and dependencies

**For developers:** See [Standalone Windows Package Guide](standalone-windows.md) for how to build this package yourself.

---

## Prerequisites

### Windows
- Visual Studio 2022 or later (C++17 support)
- CMake 3.16+
- Qt 6.2+ (MSVC build)
  - Download: https://www.qt.io/download
  - Install to: `C:\Qt\6.x\msvc2019_64` (or configure path in build script)
- OpenSSL 3.x
  - Download: https://slproweb.com/products/Win32OpenSSL.html
  - Install to: `C:\Program Files\OpenSSL-Win64`
- Python 3.8+ (optional, for Python plugin support)

### Linux (Ubuntu/Debian)
```bash
sudo apt-get install -y \
  build-essential cmake \
  qt6-base-dev qt6-tools-dev \
  libssl-dev \
  python3-dev python3-pip \
  portaudio19-dev
```

### macOS
```bash
# Using Homebrew
brew install cmake qt@6 openssl python@3.11 portaudio
```

---

## Quick Build (Windows)

### Fresh Installation (First Time Setup)

If this is your first time building NDA, run the setup script to install all dependencies:

```batch
# From repository root
scripts\setup_windows.bat
```

This will:
- Check for all required dependencies
- Guide you through installing missing components
- Provide direct download links
- Install Python packages automatically

**Then proceed to build using one of the options below.**

### Option 1: Use Build Script (Recommended)

```batch
# From repository root
scripts\build_windows.bat

# Built executable: build\Release\NDA.exe
```

The build script will automatically:
- Detect Qt installation (checks 6.5.3, 6.6.3, 6.7.0)
- Detect OpenSSL location
- Verify all prerequisites
- Build application and plugins

### Option 2: Fast Build with Ninja (Recommended for Developers)

```batch
# Requires Ninja build system
scripts\build_windows_ninja.bat

# Built executable: build\NDA.exe
```

Ninja provides faster incremental builds for development.

### Option 3: Manual CMake

```bash
# Configure
cmake -S . -B build ^
  -G "Visual Studio 17 2022" ^
  -A x64 ^
  -DCMAKE_PREFIX_PATH="C:/Qt/6.x/msvc2019_64" ^
  -DOPENSSL_ROOT_DIR="C:/Program Files/OpenSSL-Win64"

# Build
cmake --build build --config Release

# Run
build\Release\NDA.exe
```

### Option 3: With Python Support

```bash
cmake -S . -B build ^
  -G "Visual Studio 17 2022" ^
  -A x64 ^
  -DCMAKE_PREFIX_PATH="C:/Qt/6.x/msvc2019_64" ^
  -DOPENSSL_ROOT_DIR="C:/Program Files/OpenSSL-Win64" ^
  -DNDA_ENABLE_PYTHON=ON

cmake --build build --config Release
```

---

## Quick Build (Linux)

```bash
# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run
./NDA
```

### With Python Support

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DNDA_ENABLE_PYTHON=ON
make -j$(nproc)

./NDA
```

---

## Quick Build (macOS)

```bash
# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/usr/local/opt/qt
make -j$(nproc)

# Run
./NDA.app/Contents/MacOS/NDA
```

---

## Deployment (Packaging)

### Windows Standalone Package (Recommended)

**Single-click automated build** that creates a complete portable ZIP package:

```batch
# One-time setup (detects Qt, Python, OpenSSL paths)
scripts\windows-packaging\setup_build_config.bat

# Build complete standalone package
scripts\windows-packaging\build_release_package.bat

# Output: NDA-Windows-Portable-v2.0.0.zip
```

This creates a production-ready package with:
- NDA.exe and all plugins
- All dependencies bundled (Qt, OpenSSL, VCRUNTIME, Python)
- Automatic verification
- Ready-to-distribute ZIP archive

**See the complete guide:** [Standalone Windows Package Guide](standalone-windows.md)

### Windows (Legacy Deployment)

```batch
# Creates basic package (manual DLL copying required)
python scripts\deploy.py

# Or use batch script
scripts\deploy_windows.bat

# Output: readytoship/
```

### Linux
```bash
python scripts\deploy.py

# Output: packages/NDA-v2.0-linux/
```

---

## Verification

After build, verify it works:

```bash
# 1. Run the application
./NDA  # Linux/macOS
build\Release\NDA.exe  # Windows

# 2. In the UI:
#    - Click "Load Plugins" or "Auto-Load Python Plugins"
#    - Select Source, Processor, Sink for both TX and RX pipelines
#    - Click "Start Both"
#    - Verify both show "Running" status

# 3. Check console for any errors
```

---

## Troubleshooting

### "Qt not found"
**Solution:** Update CMake prefix path
```bash
cmake -DCMAKE_PREFIX_PATH="C:/Qt/6.x/msvc2019_64" ...
```

### "OpenSSL not found"
**Solution:** Install OpenSSL or set path
```bash
cmake -DOPENSSL_ROOT_DIR="C:/Program Files/OpenSSL-Win64" ...
```

### "Python support enabled" but Python plugins don't load
**Solution:** Verify Python path and dependencies
```bash
python -m pip install -r plugins_py/requirements.txt
```

### Build fails with "BearerPlugin not found"
**Solution:** You're on an old branch. This is expected - v2.0 removed Bearer.
Switch to correct branch or see `development/migration-v1-to-v2.md`

### Application crashes on startup
**Solution:** Check console output for missing dependencies
- Verify Qt libraries are in PATH
- Verify OpenSSL DLLs are in PATH
- Run `scripts\deploy_windows.bat` to package with all dependencies

---

## Enabling Python Plugin Support

To use Python plugins (not required for C++ plugins):

1. Install Python 3.8+
2. Install dependencies:
   ```bash
   pip install -r plugins_py/requirements.txt
   ```

3. Build with Python support:
   ```bash
   cmake ... -DNDA_ENABLE_PYTHON=ON
   ```

4. Verify in CMake output:
   ```
   -- Python support enabled
   -- Python version: 3.11
   ```

5. Run NDA and click "Auto-Load Python Plugins"

---

## Next Steps

1. **Build succeeded?** → Run `./NDA` and launch the UI
2. **Want to use NDA?** → See `getting-started/README.md` and `use-cases.md`
3. **Want to write plugins?** → See `development/plugin-development.md`
4. **Something broken?** → See `development/troubleshooting.md`

---

For detailed architecture and specifications, see:
- `technical/ARCHITECTURE.md`
- `technical/specifications.md`

For deployment details, see:
- `scripts/deploy.py` (source code)
- `scripts/build_windows.bat` (batch script reference)
