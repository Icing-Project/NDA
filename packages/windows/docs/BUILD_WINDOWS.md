# Building NADE on Windows

## Prerequisites

### Required Software

1. **Visual Studio 2019 or 2022** (Community Edition or higher)
   - Install "Desktop development with C++"
   - Install "C++ CMake tools for Windows"
   - Download: https://visualstudio.microsoft.com/

2. **CMake 3.16+**
   - Download: https://cmake.org/download/
   - Add to PATH during installation

3. **Qt 6.x for MSVC**
   - Download Qt Online Installer: https://www.qt.io/download-open-source
   - Install Qt 6.5+ with MSVC 2019 64-bit component
   - Default location: `C:\Qt\6.5.3\msvc2019_64`

4. **Python 3.7+**
   - Download: https://www.python.org/downloads/
   - **IMPORTANT**: Check "Add Python to PATH" during installation
   - Install development headers (included by default)

5. **OpenSSL for Windows**
   - Download: https://slproweb.com/products/Win32OpenSSL.html
   - Install "Win64 OpenSSL v3.x" (full, not Light)
   - Default location: `C:\Program Files\OpenSSL-Win64`

### Install Python Dependencies

```cmd
pip install numpy
```

## Build Steps

### Option 1: Using Visual Studio GUI

1. **Open CMake GUI**
   - Source: `C:\path\to\NDA`
   - Build: `C:\path\to\NDA\build`

2. **Configure**
   - Click "Configure"
   - Generator: "Visual Studio 17 2022"
   - Platform: x64
   - Click "Finish"

3. **Set Paths** (if not auto-detected)
   - `CMAKE_PREFIX_PATH`: `C:/Qt/6.5.3/msvc2019_64`
   - `OPENSSL_ROOT_DIR`: `C:/Program Files/OpenSSL-Win64`

4. **Generate**
   - Click "Generate"

5. **Build**
   - Click "Open Project" (opens Visual Studio)
   - Select "Release" configuration
   - Build > Build Solution (or press F7)

### Option 2: Using Command Line

```cmd
# Run from NADE directory
build_windows.bat
```

Or manually:

```cmd
mkdir build
cd build

cmake .. -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_PREFIX_PATH="C:/Qt/6.5.3/msvc2019_64" ^
    -DOPENSSL_ROOT_DIR="C:/Program Files/OpenSSL-Win64"

cmake --build . --config Release
```

## Deploy

### Create Deployment Package

```cmd
# From NADE directory
python deploy_windows.py
```

This creates `readytoship/` folder with:
- NADE.exe
- All required DLLs (Qt, Python, OpenSSL)
- Python plugins
- Documentation

### Manual Deployment

1. **Copy Executable**
   ```cmd
   copy build\Release\NADE.exe readytoship\bin\
   ```

2. **Deploy Qt DLLs**
   ```cmd
   cd readytoship\bin
   C:\Qt\6.5.3\msvc2019_64\bin\windeployqt.exe NADE.exe
   ```

3. **Copy Python DLL**
   ```cmd
   copy C:\Python3x\python3x.dll readytoship\bin\
   ```

4. **Copy OpenSSL DLLs**
   ```cmd
   copy "C:\Program Files\OpenSSL-Win64\bin\*.dll" readytoship\bin\
   ```

5. **Copy Plugins**
   ```cmd
   xcopy /E /I plugins_py readytoship\plugins
   ```

## Troubleshooting

### CMake Can't Find Qt
- Set environment variable:
  ```cmd
  set CMAKE_PREFIX_PATH=C:\Qt\6.5.3\msvc2019_64
  ```

### CMake Can't Find Python
- Ensure Python is in PATH:
  ```cmd
  python --version
  ```
- Set manually:
  ```cmd
  set Python3_ROOT_DIR=C:\Python3x
  ```

### CMake Can't Find OpenSSL
- Set environment variable:
  ```cmd
  set OPENSSL_ROOT_DIR=C:\Program Files\OpenSSL-Win64
  ```

### Build Errors
- **LNK2001 errors**: Install Visual C++ Redistributable
- **Qt MOC errors**: Ensure Qt bin is in PATH
- **Python.h not found**: Install Python development package

### Runtime Errors
- **DLL not found**: Run `windeployqt.exe` on NADE.exe
- **Python import errors**: Copy all .py files to plugins/ folder
- **Audio errors**: Install sounddevice: `pip install sounddevice`

## Testing

```cmd
cd readytoship\bin
NADE.exe
```

Expected behavior:
1. Application window opens
2. Click "Auto-Load Python Plugins"
3. Select SoundDevice Microphone → WAV File Recorder
4. Click "Start Pipeline"
5. Should record audio successfully

## Packaging for Distribution

### Create Installer (Optional)

Use **Inno Setup** or **WiX Toolset**:

1. Install Inno Setup: https://jrsoftware.org/isinfo.php
2. Create installer script (NADE.iss)
3. Include all files from readytoship/
4. Build installer

### Create ZIP Package

```cmd
cd readytoship
7z a -r NADE-Windows-x64.zip *
```

## Notes

- **Debug builds**: Change `--config Release` to `--config Debug`
- **Console output**: Debug builds show console, Release can hide it
- **Antivirus**: May need to add exception for NADE.exe
- **Code signing**: Sign .exe for production with certificate
