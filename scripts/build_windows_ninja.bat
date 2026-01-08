@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM NDA Windows Build Script (Ninja Generator - Faster Builds)
REM ============================================================================
REM Configure and build NDA (app + plugins) with Ninja using MSVC toolchain.
REM
REM Prerequisites:
REM   - VS Build Tools 2022 (C++ workload + Windows 10 SDK)
REM   - Qt 6.6.3+ msvc2019_64
REM   - OpenSSL Win64
REM   - Ninja build system
REM   - Python 3.8+ with development headers
REM
REM For fresh setup, run: scripts\setup_windows.bat
REM ============================================================================

echo ================================================
echo NDA - Windows Build Script (Ninja)
echo ================================================
echo.

set ROOT=%~dp0..
cd /d "%ROOT%"

REM ============================================================================
REM Check Prerequisites
REM ============================================================================
echo Checking prerequisites...
echo.

REM Check Ninja
ninja --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Ninja not found!
    echo Ninja is required for this build script.
    echo.
    echo Download from: https://github.com/ninja-build/ninja/releases
    echo Or use scripts\build_windows.bat instead (uses Visual Studio generator)
    pause
    exit /b 1
)

REM Check Python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found!
    echo Please run scripts\setup_windows.bat to install dependencies.
    pause
    exit /b 1
)

REM Auto-detect Qt installation
set QT_PREFIX=
if exist "C:\Qt\6.6.3\msvc2019_64\bin\qmake.exe" (
    set QT_PREFIX=C:/Qt/6.6.3/msvc2019_64
    echo [OK] Found Qt 6.6.3
) else if exist "C:\Qt\6.7.0\msvc2019_64\bin\qmake.exe" (
    set QT_PREFIX=C:/Qt/6.7.0/msvc2019_64
    echo [OK] Found Qt 6.7.0
) else if exist "C:\Qt\6.5.3\msvc2019_64\bin\qmake.exe" (
    set QT_PREFIX=C:/Qt/6.5.3/msvc2019_64
    echo [OK] Found Qt 6.5.3
) else (
    echo [ERROR] Qt6 not found!
    echo Please run scripts\setup_windows.bat to install dependencies.
    pause
    exit /b 1
)

REM Check OpenSSL
set OPENSSL_ROOT=
if exist "C:\Program Files\OpenSSL-Win64\bin\openssl.exe" (
    set OPENSSL_ROOT=C:/Program Files/OpenSSL-Win64
    echo [OK] Found OpenSSL
) else if exist "C:\OpenSSL-Win64\bin\openssl.exe" (
    set OPENSSL_ROOT=C:/OpenSSL-Win64
    echo [OK] Found OpenSSL
) else (
    echo [ERROR] OpenSSL not found!
    echo Please run scripts\setup_windows.bat to install dependencies.
    pause
    exit /b 1
)

echo.
echo All prerequisites found!
echo.

REM ============================================================================
REM Initialize Visual Studio Build Environment
REM ============================================================================
echo Initializing Visual Studio build environment...

REM Try different VS 2022 installation paths
set VCVARSALL=
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" (
    set VCVARSALL=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    set VCVARSALL=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" (
    set VCVARSALL=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat
)

if "!VCVARSALL!"=="" (
    echo [ERROR] Visual Studio 2022 not found!
    echo Please run scripts\setup_windows.bat to install dependencies.
    pause
    exit /b 1
)

REM Update the SDK version below if a newer Windows 10/11 SDK is installed.
call "!VCVARSALL!" x64 10.0.18362.0
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to initialize Visual Studio build environment.
    echo Check that Windows 10 SDK is installed.
    pause
    exit /b 1
)

echo [OK] Build environment initialized
echo.

REM ============================================================================
REM Configure and Build
REM ============================================================================

if not exist build mkdir build

echo Configuring with CMake (Ninja)...
cmake -S . -B build -G "Ninja" ^
    -DCMAKE_PREFIX_PATH="%QT_PREFIX%" ^
    -DOPENSSL_ROOT_DIR="%OPENSSL_ROOT%" ^
    -DCMAKE_BUILD_TYPE=Release
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ================================================
    echo CMake Configuration Failed!
    echo ================================================
    echo.
    echo Troubleshooting:
    echo   1. Run scripts\setup_windows.bat to verify all dependencies
    echo   2. Check that Python development headers are installed
    echo   3. Ensure Windows 10 SDK is installed with Visual Studio
    echo.
    echo For detailed setup instructions, see: docs\guides\installation.md
    echo.
    pause
    exit /b 1
)

cmake --build build --config Release
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Build failed.
    exit /b 1
)

echo.
echo [OK] Build artifacts:
echo   - App: %ROOT%\build\NDA.exe
echo   - Plugins: %ROOT%\build\plugins\*.dll
echo.
