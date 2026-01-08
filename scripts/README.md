# NDA Build Scripts

This directory contains build and deployment scripts for Windows and Linux platforms.

## Windows Scripts

### Fresh Installation (Recommended for New Users)

If you've just cloned the repository and need to set up your build environment:

```batch
scripts\setup_windows.bat
```

This script will:
- Check for all required dependencies
- Guide you through installing missing components
- Provide direct download links
- Install Python packages automatically

**Required Dependencies:**
- Visual Studio 2022 Build Tools (with C++ workload and Windows 10 SDK)
- CMake 3.16+
- Qt 6.2+ (MSVC toolchain)
- OpenSSL 3.x Win64
- Python 3.8+ (with development headers)
- Python packages (numpy, sounddevice, soundcard)

### Building the Project

After running the setup script, use one of these build methods:

#### Option 1: Visual Studio Generator (Standard)

```batch
scripts\build_windows.bat
```

- Uses Visual Studio 2022 project generator
- Reliable and well-tested
- Outputs to: `build\Release\NDA.exe`

#### Option 2: Ninja Generator (Faster Builds)

```batch
scripts\build_windows_ninja.bat
```

- Uses Ninja build system (faster incremental builds)
- Requires Ninja to be installed
- Outputs to: `build\NDA.exe`

Both scripts will:
- Auto-detect Qt installation (checks 6.5.3, 6.6.3, 6.7.0)
- Auto-detect OpenSSL location
- Verify all prerequisites before building
- Build both the application and plugins

### Deployment

After building, create a distributable package:

```batch
scripts\deploy_windows.bat
```

This will:
- Copy the executable and plugins to `readytoship\`
- Run `windeployqt` to include Qt dependencies
- Copy Python and OpenSSL DLLs
- Install Python packages
- Create a ready-to-distribute folder

Output: `readytoship\` directory with everything needed to run NDA

### Troubleshooting

#### "CMake configuration failed"
- Run `scripts\setup_windows.bat` to verify dependencies
- Ensure Python development headers are installed (not just Python)
- Check that Visual Studio has C++ workload installed

#### "Qt not found"
- Install Qt to `C:\Qt\` directory (recommended)
- Or update the Qt path in the build script

#### "OpenSSL not found"
- Install to `C:\Program Files\OpenSSL-Win64` (recommended)
- Or update the OpenSSL path in the build script

#### "Python support enabled" but plugins don't load
- Install Python packages: `pip install -r requirements.txt`
- Ensure NumPy is installed: `pip install numpy`

#### Build fails with missing Python headers
- Reinstall Python with "Development headers/libraries" option checked
- Or install from python.org (not Microsoft Store version)

## Linux Scripts

### Ubuntu/Debian Quick Build

```bash
# One-shot: Install deps + build + run
scripts/build_ubuntu.sh

# Development: Fast incremental builds
scripts/dev_ubuntu.sh
```

See `docs/guides/installation.md` for detailed Linux instructions.

## Manual CMake Configuration

If you prefer to configure manually:

### Windows (Visual Studio)
```batch
cmake -S . -B build ^
  -G "Visual Studio 17 2022" ^
  -A x64 ^
  -DCMAKE_PREFIX_PATH="C:/Qt/6.6.3/msvc2019_64" ^
  -DOPENSSL_ROOT_DIR="C:/Program Files/OpenSSL-Win64"

cmake --build build --config Release
```

### Windows (Ninja)
```batch
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

cmake -S . -B build ^
  -G "Ninja" ^
  -DCMAKE_PREFIX_PATH="C:/Qt/6.6.3/msvc2019_64" ^
  -DOPENSSL_ROOT_DIR="C:/Program Files/OpenSSL-Win64" ^
  -DCMAKE_BUILD_TYPE=Release

cmake --build build
```

### Linux
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## CMake Options

### Enable/Disable Python Support
```batch
-DNDA_ENABLE_PYTHON=ON   # Enable Python plugins (default)
-DNDA_ENABLE_PYTHON=OFF  # Disable Python plugins
```

### Specify Python Installation
```batch
-DNDA_PYTHON_EXECUTABLE="C:/Python311/python.exe"
-DNDA_PYTHON_ROOT_DIR="C:/Python311"
```

### Build Types
```batch
-DCMAKE_BUILD_TYPE=Release  # Optimized build
-DCMAKE_BUILD_TYPE=Debug    # Debug symbols
```

## Script Execution Order

For a fresh Windows setup:

1. **`setup_windows.bat`** - Check and install dependencies
2. **`build_windows.bat`** or **`build_windows_ninja.bat`** - Build the project
3. **`deploy_windows.bat`** - Create distributable package

## File Structure

```
scripts/
├── README.md                   # This file
├── setup_windows.bat           # Dependency installer (Windows)
├── build_windows.bat           # Build script - VS generator
├── build_windows_ninja.bat     # Build script - Ninja generator
├── deploy_windows.bat          # Deployment script (Windows)
├── build_ubuntu.sh             # One-shot build (Linux)
├── dev_ubuntu.sh               # Dev loop script (Linux)
└── deploy.py                   # Deployment helper (cross-platform)
```

## Additional Resources

- **Installation Guide:** `docs/guides/installation.md`
- **Troubleshooting:** `docs/guides/troubleshooting.md`
- **Development Guide:** `docs/development/plugins.md`
- **Architecture:** `docs/technical/ARCHITECTURE.md`

## Support

If you encounter issues:
1. Run `scripts\setup_windows.bat` to verify dependencies
2. Check the troubleshooting section above
3. See `docs/guides/troubleshooting.md` for detailed solutions
4. Open an issue on GitHub with build output

---

**Note:** All scripts automatically detect dependency locations where possible. If auto-detection fails, you can manually edit the path variables in the scripts or use CMake command-line options.
