# NDA Windows Quick Start Guide

Get up and running with NDA on Windows in minutes.

## Prerequisites

- Windows 10 or 11 (64-bit)
- Administrator access (for installing dependencies)
- Internet connection (for downloading dependencies)

## Step-by-Step Setup

### 1. Clone the Repository

```batch
git clone <repository-url>
cd NDA
```

### 2. Run the Setup Script

The setup script will check for and guide you through installing all required dependencies:

```batch
scripts\setup_windows.bat
```

This script will verify and help you install:
- **Visual Studio Build Tools 2022** (C++ compiler and Windows SDK)
- **CMake 3.16+** (build system)
- **Qt 6.6.3+** (GUI framework)
- **OpenSSL 3.x** (encryption library)
- **Python 3.8+** (with development headers for plugin support)
- **Python packages** (numpy, sounddevice, soundcard)

The script provides direct download links and installation instructions for any missing components.

**Installation Tips:**
- When installing Python, check "Add Python to PATH" and "Install development headers/libraries"
- When installing Qt, select "MSVC 2019 64-bit" component
- Install Qt to `C:\Qt\` (recommended for auto-detection)
- When installing Visual Studio, select "Desktop development with C++" workload

### 3. Build the Project

After all dependencies are installed, build NDA:

**Option A: Standard Build (Visual Studio Generator)**
```batch
scripts\build_windows.bat
```

**Option B: Fast Build (Ninja Generator - Requires Ninja)**
```batch
scripts\build_windows_ninja.bat
```

Both will:
- Auto-detect Qt and OpenSSL installations
- Configure CMake with correct paths
- Build the application and all plugins
- Output to `build\Release\NDA.exe` (VS) or `build\NDA.exe` (Ninja)

Build time: ~2-5 minutes on modern hardware

### 4. Run NDA

```batch
build\Release\NDA.exe
```

Or for Ninja builds:
```batch
build\NDA.exe
```

### 5. Load Plugins

In the NDA UI:
1. Click **"Auto-Load Python Plugins"** to load Python plugins from `plugins_py/`
2. Or click **"Load Plugins from Directory"** and select `build\plugins` for C++ plugins

Available plugins:
- **Sources:** Microphone, AIOC USB Input, Sine Wave Generator
- **Processors:** AES-256 Encryptor/Decryptor, Simple Gain, Passthrough
- **Sinks:** Speaker, AIOC USB Output, WAV File Recorder, Null Sink

### 6. Configure and Start

1. **TX Pipeline** (Transmit):
   - Source: Select your microphone
   - Processor: (Optional) Select encryptor
   - Sink: Select output device/cable

2. **RX Pipeline** (Receive):
   - Source: Select input device/cable
   - Processor: (Optional) Select decryptor
   - Sink: Select your speaker

3. Click **"Start Both"** to begin audio processing

## Optional: Virtual Environment (Recommended for Developers)

For cleaner Python dependency management:

```batch
scripts\setup_venv_windows.bat
```

This creates a Python virtual environment in `venv/`. CMake will auto-detect and use it.

To activate manually:
```batch
venv\Scripts\activate.bat
```

## Deployment (Creating Distributable Package)

To create a standalone package for distribution:

```batch
scripts\deploy_windows.bat
```

Output: `readytoship\` folder containing:
- NDA.exe
- All required DLLs (Qt, OpenSSL, Python)
- Plugins
- Python runtime packages

This package can be zipped and distributed to users without requiring them to install dependencies.

## Troubleshooting

### "CMake not found" or "Python not found"
- Rerun `scripts\setup_windows.bat`
- Ensure you selected "Add to PATH" during installation
- Restart your command prompt or PC

### "Qt not found" during build
- Verify Qt is installed to `C:\Qt\` directory
- Or manually edit the Qt path in `scripts\build_windows.bat`

### "OpenSSL not found" during build
- Install to `C:\Program Files\OpenSSL-Win64` (default location)
- Or manually edit the OpenSSL path in `scripts\build_windows.bat`

### Python plugins don't load
- Run: `pip install -r requirements.txt`
- Ensure NumPy is installed: `pip list | findstr numpy`
- Verify Python development headers are installed (not just Python runtime)

### Build fails with "Python.h not found"
- Reinstall Python with "Development headers/libraries" option checked
- Avoid using Python from Microsoft Store (use python.org installer)

### Application crashes on startup
- Ensure Visual Studio is installed with Windows 10 SDK
- Run `scripts\deploy_windows.bat` to test with all dependencies bundled

## Next Steps

- **Usage Examples:** See `docs/guides/use-cases.md`
- **Plugin Development:** See `docs/development/plugins.md`
- **Architecture Details:** See `docs/technical/ARCHITECTURE.md`
- **Troubleshooting:** See `docs/guides/troubleshooting.md`

## Quick Reference

| Task | Command |
|------|---------|
| Setup dependencies | `scripts\setup_windows.bat` |
| Build (standard) | `scripts\build_windows.bat` |
| Build (fast) | `scripts\build_windows_ninja.bat` |
| Run application | `build\Release\NDA.exe` |
| Deploy package | `scripts\deploy_windows.bat` |
| Setup venv | `scripts\setup_venv_windows.bat` |

## Support

For issues or questions:
1. Check the troubleshooting section above
2. See detailed docs in `docs/guides/`
3. Run `scripts\setup_windows.bat` to verify all dependencies
4. Open an issue on GitHub with build output

---

**Welcome to NDA! Happy audio processing! 🎵🔒**
