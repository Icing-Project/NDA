@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM NDA Windows Setup Script
REM ============================================================================
REM This script helps fresh users install all required dependencies for NDA
REM Requirements it checks/installs:
REM   - Visual Studio Build Tools 2022 with C++ workload
REM   - CMake 3.16+
REM   - Qt 6.6.3 MSVC
REM   - OpenSSL Win64
REM   - Python 3.8+ with development headers
REM   - Python packages (numpy, sounddevice, etc.)
REM ============================================================================

echo.
echo ============================================================================
echo NDA Windows Setup - Dependency Installer
echo ============================================================================
echo.
echo This script will check for and help you install all required dependencies.
echo Administrative privileges may be required for some installations.
echo.
pause

REM ============================================================================
REM 1. Check for Python
REM ============================================================================
echo.
echo [1/6] Checking for Python...
python --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYTHON_VERSION=%%v
    echo [OK] Python !PYTHON_VERSION! found
    set PYTHON_FOUND=1
) else (
    echo [MISSING] Python not found
    echo.
    echo Python 3.8+ is required for NDA Python plugin support.
    echo.
    echo Please download and install Python from:
    echo   https://www.python.org/downloads/
    echo.
    echo IMPORTANT: During installation, check "Add Python to PATH"
    echo           and check "Install development headers/libraries"
    echo.
    set /p INSTALL_PYTHON="Open download page now? (Y/N): "
    if /i "!INSTALL_PYTHON!"=="Y" (
        start https://www.python.org/downloads/
    )
    echo.
    echo Please install Python and run this script again.
    pause
    exit /b 1
)

REM ============================================================================
REM 2. Check for pip and install Python dependencies
REM ============================================================================
echo.
echo [2/6] Checking Python pip...
pip --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] pip found
    echo.
    echo Installing Python dependencies...
    if exist "%~dp0..\requirements.txt" (
        pip install -r "%~dp0..\requirements.txt"
        if %ERRORLEVEL% EQU 0 (
            echo [OK] Python dependencies installed
        ) else (
            echo [WARNING] Failed to install some Python dependencies
            echo You may need to install them manually: pip install -r requirements.txt
        )
    ) else (
        echo [WARNING] requirements.txt not found
    )
) else (
    echo [ERROR] pip not found - Python installation may be incomplete
    pause
    exit /b 1
)

REM ============================================================================
REM 3. Check for CMake
REM ============================================================================
echo.
echo [3/6] Checking for CMake...
cmake --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    for /f "tokens=3" %%v in ('cmake --version 2^>^&1 ^| findstr /C:"cmake version"') do set CMAKE_VERSION=%%v
    echo [OK] CMake !CMAKE_VERSION! found
) else (
    echo [MISSING] CMake not found
    echo.
    echo CMake 3.16+ is required to build NDA.
    echo.
    echo Please download and install CMake from:
    echo   https://cmake.org/download/
    echo.
    echo IMPORTANT: During installation, select "Add CMake to system PATH"
    echo.
    set /p INSTALL_CMAKE="Open download page now? (Y/N): "
    if /i "!INSTALL_CMAKE!"=="Y" (
        start https://cmake.org/download/
    )
    echo.
    echo Please install CMake and run this script again.
    pause
    exit /b 1
)

REM ============================================================================
REM 4. Check for Visual Studio Build Tools
REM ============================================================================
echo.
echo [4/6] Checking for Visual Studio Build Tools...
set VS_FOUND=0

REM Check VS 2022
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" (
    echo [OK] Visual Studio Build Tools 2022 found
    set VS_FOUND=1
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    echo [OK] Visual Studio 2022 Community found
    set VS_FOUND=1
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" (
    echo [OK] Visual Studio 2022 Professional found
    set VS_FOUND=1
)

if !VS_FOUND! EQU 0 (
    echo [MISSING] Visual Studio 2022 Build Tools not found
    echo.
    echo Visual Studio Build Tools 2022 with C++ workload is required.
    echo.
    echo Please download and install from:
    echo   https://visualstudio.microsoft.com/downloads/
    echo.
    echo Choose either:
    echo   - Visual Studio 2022 Community ^(free^)
    echo   - Build Tools for Visual Studio 2022
    echo.
    echo During installation, select:
    echo   - Desktop development with C++
    echo   - Windows 10 SDK ^(10.0.18362 or later^)
    echo.
    set /p INSTALL_VS="Open download page now? (Y/N): "
    if /i "!INSTALL_VS!"=="Y" (
        start https://visualstudio.microsoft.com/downloads/
    )
    echo.
    echo Please install Visual Studio and run this script again.
    pause
    exit /b 1
)

REM ============================================================================
REM 5. Check for Qt6
REM ============================================================================
echo.
echo [5/6] Checking for Qt6...
set QT_FOUND=0

REM Check common Qt installation paths
if exist "C:\Qt\6.6.3\msvc2019_64\bin\qmake.exe" (
    echo [OK] Qt 6.6.3 found at C:\Qt\6.6.3\msvc2019_64
    set QT_FOUND=1
    set QT_PATH=C:\Qt\6.6.3\msvc2019_64
) else if exist "C:\Qt\6.7.0\msvc2019_64\bin\qmake.exe" (
    echo [OK] Qt 6.7.0 found at C:\Qt\6.7.0\msvc2019_64
    set QT_FOUND=1
    set QT_PATH=C:\Qt\6.7.0\msvc2019_64
) else if exist "C:\Qt\6.5.3\msvc2019_64\bin\qmake.exe" (
    echo [OK] Qt 6.5.3 found at C:\Qt\6.5.3\msvc2019_64
    set QT_FOUND=1
    set QT_PATH=C:\Qt\6.5.3\msvc2019_64
)

if !QT_FOUND! EQU 0 (
    echo [MISSING] Qt6 not found
    echo.
    echo Qt 6.2+ with MSVC toolchain is required.
    echo.
    echo Please download and install Qt from:
    echo   https://www.qt.io/download-open-source
    echo.
    echo During installation:
    echo   1. Create a Qt account ^(free^)
    echo   2. Install Qt 6.6.3 or later
    echo   3. Select "MSVC 2019 64-bit" component
    echo   4. Install to C:\Qt\ ^(recommended^)
    echo.
    set /p INSTALL_QT="Open download page now? (Y/N): "
    if /i "!INSTALL_QT!"=="Y" (
        start https://www.qt.io/download-open-source
    )
    echo.
    echo Please install Qt and run this script again.
    pause
    exit /b 1
)

REM ============================================================================
REM 6. Check for OpenSSL
REM ============================================================================
echo.
echo [6/6] Checking for OpenSSL...
set OPENSSL_FOUND=0

if exist "C:\Program Files\OpenSSL-Win64\bin\openssl.exe" (
    echo [OK] OpenSSL found at C:\Program Files\OpenSSL-Win64
    set OPENSSL_FOUND=1
) else if exist "C:\OpenSSL-Win64\bin\openssl.exe" (
    echo [OK] OpenSSL found at C:\OpenSSL-Win64
    set OPENSSL_FOUND=1
)

if !OPENSSL_FOUND! EQU 0 (
    echo [MISSING] OpenSSL not found
    echo.
    echo OpenSSL 3.x Win64 is required for encryption support.
    echo.
    echo Please download and install OpenSSL from:
    echo   https://slproweb.com/products/Win32OpenSSL.html
    echo.
    echo Download: Win64 OpenSSL v3.x.x
    echo Install to: C:\Program Files\OpenSSL-Win64
    echo.
    set /p INSTALL_OPENSSL="Open download page now? (Y/N): "
    if /i "!INSTALL_OPENSSL!"=="Y" (
        start https://slproweb.com/products/Win32OpenSSL.html
    )
    echo.
    echo Please install OpenSSL and run this script again.
    pause
    exit /b 1
)

REM ============================================================================
REM Optional: Check for Ninja (recommended for faster builds)
REM ============================================================================
echo.
echo [OPTIONAL] Checking for Ninja build system...
ninja --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    for /f "tokens=1" %%v in ('ninja --version 2^>^&1') do set NINJA_VERSION=%%v
    echo [OK] Ninja !NINJA_VERSION! found
    echo      You can use scripts\build_windows_ninja.bat for faster builds
) else (
    echo [INFO] Ninja not found ^(optional^)
    echo       Ninja provides faster builds than Visual Studio generator
    echo       Download from: https://github.com/ninja-build/ninja/releases
    echo       Extract ninja.exe to a folder in your PATH
)

REM ============================================================================
REM Summary
REM ============================================================================
echo.
echo ============================================================================
echo Setup Complete - All Required Dependencies Found!
echo ============================================================================
echo.
echo You can now build NDA using one of these methods:
echo.
echo   1. Quick build ^(Visual Studio generator^):
echo      scripts\build_windows.bat
echo.
echo   2. Fast build ^(Ninja generator, if installed^):
echo      scripts\build_windows_ninja.bat
echo.
echo   3. Manual CMake configuration:
echo      See docs\guides\installation.md
echo.
echo After building, run the deployment script to create a distributable package:
echo   scripts\deploy_windows.bat
echo.
echo ============================================================================
pause
