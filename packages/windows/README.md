# NADE - Windows Package

**Version 1.0.0**

Cross-platform audio encryption system - Windows build-ready package

## Build Instructions

### Prerequisites

1. Visual Studio 2019/2022 with C++ Desktop Development
2. CMake 3.16+
3. Qt 6.x for MSVC (msvc2019_64 or msvc2022_64)
4. Python 3.7+ with development headers
5. OpenSSL for Windows (Win64 full version)

### Build Steps

```cmd
cd build_scripts
build_windows.bat
```

### Deploy

```cmd
cd build_scripts
deploy_windows.bat
```

This will copy NADE.exe, Qt DLLs, Python DLL, and OpenSSL DLLs to bin/ folder.

### Run

```cmd
pip install -r requirements.txt
run_nade.bat
```

## Documentation

- **docs/BUILD_WINDOWS.md** - Detailed build instructions
- **docs/WINDOWS_README.md** - Windows setup guide
- **docs/USER_GUIDE.md** - Complete usage guide

## Pre-built Binary

If you have a pre-built NADE.exe, place it in bin/ folder along with required DLLs.

## Support

See docs/BUILD_WINDOWS.md for troubleshooting and detailed build instructions.
