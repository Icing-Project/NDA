# Windows Standalone Package Guide

## Overview

This guide explains how to build and distribute NDA as a **standalone portable Windows package** that includes all dependencies bundled together. Users can simply extract and run the application without installing anything.

**Package Features:**
- ✅ Fully portable (no installation required)
- ✅ All dependencies bundled (Qt, OpenSSL, VCRUNTIME, Python)
- ✅ Single-click build automation
- ✅ Includes all plugins (C++ and Python)
- ✅ Automatic verification
- ✅ ZIP archive ready for distribution

---

## For Developers: Building the Standalone Package

### Prerequisites

Before building, you need the following installed:

1. **CMake** 3.16 or later
2. **Qt6** (6.5.3 or later) with MSVC 2019 64-bit
3. **OpenSSL 3.x** for Windows (64-bit)
4. **Python 3.8+** with development headers
5. **Visual Studio 2022** or VS Build Tools with C++ workload
6. **Git** (for cloning the repository)

### Step 1: Initial Setup

Run the configuration setup script to detect your environment:

```batch
cd /path/to/NDA
scripts\windows-packaging\setup_build_config.bat
```

This script will:
- Auto-detect Qt, OpenSSL, Python, and Visual Studio installations
- Create `scripts/build_config.json` with detected paths
- Report any missing components

**Review the configuration:**

Open `scripts/build_config.json` and verify all paths are correct:

```json
{
  "qt_path": "C:/Qt/6.6.3/msvc2019_64",
  "openssl_path": "C:/Program Files/OpenSSL-Win64",
  "python_path": "C:/Python312",
  "visual_studio_path": "C:/Program Files/Microsoft Visual Studio/2022/Community",
  "build_config": "Release",
  "enable_python": true,
  "package_version": "2.0.0",
  "package_name": "NDA-Windows-Portable"
}
```

**Important settings:**
- `enable_python`: Set to `true` to include Python plugin support
- `package_version`: Update for each release
- `build_config`: Use `Release` for production builds

### Step 2: Build the Standalone Package

Run the master build script:

```batch
scripts\windows-packaging\build_release_package.bat
```

This automated script will:

1. **Clean** previous build artifacts
2. **Configure** CMake with correct paths
3. **Build** NDA.exe and all plugins
4. **Deploy** the standalone package:
   - Copy NDA.exe
   - Copy all plugins (C++ DLLs and Python .py files)
   - Collect Qt dependencies using `windeployqt`
   - Collect OpenSSL DLLs
   - Collect Visual C++ runtime DLLs
   - Bundle Python runtime and site-packages
5. **Verify** package contents
6. **Create** ZIP archive

**Expected output:**

```
Building NDA Standalone Package...
[1/5] Cleaning build directory...
[2/5] Configuring with CMake...
[3/5] Building NDA.exe and plugins...
[4/5] Deploying standalone package...
[5/5] Verifying package...

✓ Package created: package/
✓ ZIP archive: NDA-Windows-Portable-v2.0.0.zip

Package size: 125.3 MB
Total files: 1,247
Verification: 42/42 checks passed (100%)

Ready for distribution!
```

### Step 3: Verify the Package

The build script automatically runs verification, but you can run it manually:

```batch
python scripts\windows-packaging\verify_package.py
```

**Verification checks:**
- ✓ NDA.exe exists
- ✓ All required DLLs present (Qt, OpenSSL, VCRUNTIME)
- ✓ Python runtime bundled (if enabled)
- ✓ All plugins copied
- ✓ Documentation included
- ✓ Package size reasonable

### Step 4: Test the Package

Before distribution, test the package locally:

```batch
cd package
NDA.bat
```

**Test checklist:**
1. Application launches successfully
2. Click "Auto-Load Python Plugins" (if Python enabled)
3. Verify all plugins appear in dropdowns
4. Create test pipeline: Sine Wave → WAV File Sink
5. Start pipeline, wait 5 seconds, stop
6. Verify WAV file created successfully

### Step 5: Distribute

The final ZIP archive is ready for distribution:

```
NDA-Windows-Portable-v2.0.0.zip
```

**Distribution checklist:**
- ✓ Package tested on clean Windows machine
- ✓ README.txt included with quick start instructions
- ✓ Version number updated in package name
- ✓ ZIP archive scanned for malware (optional)
- ✓ Release notes prepared

---

## Package Structure

```
NDA-Windows-Portable-v2.0.0/
├── NDA.exe                      # Main application (353 KB)
├── NDA.bat                      # Launcher script (sets PATH)
├── README.txt                   # Quick start guide
│
├── lib/                         # All dependencies (~100 MB)
│   ├── qt/                      # Qt6 framework
│   │   ├── Qt6Core.dll
│   │   ├── Qt6Gui.dll
│   │   ├── Qt6Widgets.dll
│   │   ├── Qt6Network.dll
│   │   └── platforms/           # Platform plugins
│   │       └── qwindows.dll
│   ├── openssl/                 # OpenSSL libraries
│   │   ├── libcrypto-3-x64.dll
│   │   └── libssl-3-x64.dll
│   ├── vcruntime/               # Visual C++ runtime
│   │   ├── vcruntime140.dll
│   │   ├── msvcp140.dll
│   │   └── vcruntime140_1.dll
│   └── python/                  # Python runtime (if enabled)
│       ├── python312.dll
│       ├── python3.dll
│       └── site-packages/       # Python packages
│           ├── numpy/
│           ├── sounddevice/
│           └── soundcard/
│
├── plugins/                     # Audio plugins (~5 MB)
│   ├── # C++ Plugins
│   ├── AIOCSourcePlugin.dll
│   ├── AIOCSinkPlugin.dll
│   ├── WindowsMicrophoneSourcePlugin.dll
│   ├── WindowsSpeakerSinkPlugin.dll
│   ├── SineWaveSourcePlugin.dll
│   ├── NullSinkPlugin.dll
│   └── WavFileSinkPlugin.dll
│   │
│   └── # Python Plugins (if enabled)
│       ├── sounddevice_microphone.py
│       ├── sounddevice_speaker.py
│       └── ... (all plugins from plugins_py/)
│
└── docs/                        # Documentation
    ├── README.md
    ├── LICENSE.txt
    ├── installation.md
    └── troubleshooting.md
```

**Size breakdown:**
- **Core application:** ~5 MB (NDA.exe + plugins)
- **Qt dependencies:** ~80 MB
- **Python runtime:** ~30 MB (if enabled)
- **OpenSSL + VCRUNTIME:** ~5 MB
- **Documentation:** <1 MB
- **Total:** ~120-130 MB (compressed), ~200-250 MB (extracted)

---

## For End Users: Using the Portable Package

### Installation

1. **Download** the ZIP archive: `NDA-Windows-Portable-v2.0.0.zip`
2. **Extract** to any location on your computer (e.g., `C:\NDA\` or `Desktop\NDA\`)
3. **Done!** No installation required.

### Launching NDA

**Recommended method:**

Double-click `NDA.bat` in the package folder.

This launcher script:
- Sets up the correct PATH for all DLLs
- Configures Python environment (if enabled)
- Launches NDA.exe

**Alternative method:**

You can also run `NDA.exe` directly, but make sure to run it from within the package folder (not from a shortcut elsewhere).

### Loading Plugins

#### C++ Plugins (Automatic)

C++ plugins load automatically at startup. You'll see them in the dropdown menus:

- **Source Plugins:** AIOC Source, Windows Microphone, Sine Wave
- **Sink Plugins:** AIOC Sink, Windows Speaker, WAV File, Null Sink

#### Python Plugins (Manual)

If Python support is enabled, load Python plugins by clicking:

**UI → "Auto-Load Python Plugins" button**

After loading, Python plugins appear alongside C++ plugins:
- **Source Plugins:** Sounddevice Microphone, Soundcard Microphone
- **Sink Plugins:** Sounddevice Speaker, Soundcard Speaker

### Troubleshooting

#### Application Won't Start

**Problem:** Double-clicking NDA.exe does nothing or shows DLL errors.

**Solutions:**
1. Run `NDA.bat` instead of `NDA.exe` directly
2. Make sure you extracted the **entire ZIP**, not just NDA.exe
3. Check that `lib/` folder exists alongside NDA.exe
4. Disable antivirus temporarily (some AV software blocks extracted executables)

#### Missing VCRUNTIME140.dll Error

**Problem:** "The code execution cannot proceed because VCRUNTIME140.dll was not found."

**Solutions:**
1. Run `NDA.bat` which sets up the correct PATH
2. Verify `lib/vcruntime/` folder exists
3. If still failing, install [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)

#### Python Plugins Don't Load

**Problem:** Python plugins don't appear in dropdown menus.

**Solutions:**
1. Click "Auto-Load Python Plugins" button in the UI
2. Verify `lib/python/` folder exists
3. Check that `lib/python/python312.dll` (or similar) exists
4. Check logs for Python initialization errors

#### No Audio Devices Shown

**Problem:** Dropdowns show "No source/sink plugins available".

**Solutions:**
1. Click "Reload C++ Plugins" or "Auto-Load Python Plugins"
2. Check that `plugins/` folder exists with DLL/PY files
3. Verify audio devices work in Windows Sound settings
4. Restart NDA

#### Package Is Too Large

**Problem:** Package size exceeds 200 MB.

**For developers:**
- Disable Python support: Set `enable_python: false` in `build_config.json`
- Remove unnecessary Qt modules (check `windeployqt` output)
- Exclude Qt translations (already done by default)
- Use maximum ZIP compression (7-Zip with LZMA2)

---

## Advanced Configuration

### Customizing the Build

Edit `scripts/build_config.json` to customize the build:

```json
{
  "enable_python": false,         // Disable Python (reduces size ~30 MB)
  "package_version": "2.1.0",    // Update version number
  "package_name": "NDA-Minimal",  // Rename package
  "build_config": "RelWithDebInfo" // Include debug symbols
}
```

### Adding Custom Plugins

To include custom plugins in the package:

1. **C++ plugins:** Place compiled `.dll` in `build/plugins/` before running deployment
2. **Python plugins:** Place `.py` files in `plugins_py/` before running deployment

Both will be automatically copied to the package.

### Excluding Plugins

To exclude specific plugins, remove them from:
- **C++:** Edit `CMakeLists.txt` to disable plugin compilation
- **Python:** Delete unwanted `.py` files from `plugins_py/` before deployment

### Building Without Python

To create a minimal package without Python support:

```json
{
  "enable_python": false
}
```

This reduces package size by ~30 MB and removes dependency on Python DLLs.

---

## Build Scripts Reference

### Main Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `setup_build_config.bat` | Initial environment detection and configuration | Run once to create `build_config.json` |
| `build_release_package.bat` | Master automation script (builds everything) | Run to create complete package |
| `collect_dependencies.py` | Collects Qt, OpenSSL, VCRUNTIME DLLs | Called automatically by deployment |
| `bundle_python.py` | Bundles Python runtime and site-packages | Called automatically if Python enabled |
| `deploy_standalone.py` | Main deployment orchestration | Called automatically by master script |
| `verify_package.py` | Package validation and verification | Run manually or automatically |

### Manual Build Steps (Advanced)

If you need to run steps manually instead of using `build_release_package.bat`:

```batch
# 1. Configure and build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_PREFIX_PATH="C:/Qt/6.6.3/msvc2019_64" ^
    -DOPENSSL_ROOT_DIR="C:/Program Files/OpenSSL-Win64" ^
    -DNDA_ENABLE_PYTHON=ON
cmake --build . --config Release

# 2. Deploy
cd ..
python scripts\windows-packaging\deploy_standalone.py

# 3. Verify
python scripts\windows-packaging\verify_package.py

# 4. Create ZIP
cd package
tar -a -c -f ../NDA-Windows-Portable-v2.0.0.zip *
```

---

## Continuous Integration

### GitHub Actions Example

```yaml
name: Build Windows Standalone Package

on:
  push:
    tags:
      - 'v*'

jobs:
  build-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Qt
        uses: jurplel/install-qt-action@v3
        with:
          version: '6.6.3'
          arch: 'win64_msvc2019_64'

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install Python dependencies
        run: pip install -r requirements.txt

      - name: Setup build configuration
        run: scripts\windows-packaging\setup_build_config.bat

      - name: Build standalone package
        run: scripts\windows-packaging\build_release_package.bat

      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: NDA-Windows-Portable
          path: NDA-Windows-Portable-v*.zip
```

---

## FAQ

### Q: Can I run the package from a USB drive?

Yes! The package is fully portable. Copy the entire folder to a USB drive and run `NDA.bat` from there.

### Q: Do I need administrator rights?

No. The package runs with standard user permissions. No registry modifications or system file changes are made.

### Q: Can I create a desktop shortcut?

Yes. Right-click `NDA.bat` → Send to → Desktop (create shortcut).

### Q: How do I update to a newer version?

Extract the new version to a separate folder. Your old version remains intact until you delete it.

### Q: Can I customize the included plugins?

Yes. Edit `plugins_py/` for Python plugins, or modify `CMakeLists.txt` for C++ plugins, then rebuild the package.

### Q: Why is the package so large?

The package includes Qt GUI framework (~80 MB), which provides the entire user interface. This is necessary for a standalone package. To reduce size, disable Python support.

### Q: Can I redistribute this package?

Yes, if you have the appropriate licenses for all bundled components (Qt, OpenSSL, Python). Review the LICENSE file and individual component licenses before commercial redistribution.

---

## See Also

- [Installation Guide](installation.md) - Build from source without packaging
- [Troubleshooting Guide](troubleshooting.md) - Common issues and solutions
- [Plugin Development](../development/plugins.md) - Create custom plugins
- [Architecture Documentation](../technical/ARCHITECTURE.md) - Technical details

---

**Last Updated:** January 2026
**Version:** 2.0
**Status:** Production Ready
