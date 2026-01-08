# Installation & Build Guide

Build NDA from source for your platform.

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

### Option 1: Use Build Script (Recommended)

```bash
# From repository root
scripts\build_windows.bat

# Built executable: build\Release\NDA.exe
```

### Option 2: Manual CMake

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

### Windows
```bash
# Creates standalone package with all dependencies
python scripts\deploy.py

# Or use batch script
scripts\deploy_windows.bat

# Output: packages/NDA-v2.0-win64/
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
