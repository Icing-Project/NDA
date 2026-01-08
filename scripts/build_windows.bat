@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM NDA Windows Build Script
REM ============================================================================
REM Requirements:
REM   - CMake 3.16+
REM   - Qt6 (with MSVC)
REM   - Python 3.8+ with development headers
REM   - Visual Studio 2022 or Build Tools
REM   - OpenSSL 3.x Win64
REM
REM For fresh setup, run: scripts\setup_windows.bat
REM ============================================================================

echo ================================================
echo NDA - Windows Build Script
echo ================================================
echo.

REM ============================================================================
REM Check Prerequisites
REM ============================================================================
echo Checking prerequisites...
echo.

REM Check CMake
cmake --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] CMake not found!
    echo Please run scripts\setup_windows.bat to install dependencies.
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
set QT_PATH=
if exist "C:\Qt\6.6.3\msvc2019_64\bin\qmake.exe" (
    set QT_PATH=C:/Qt/6.6.3/msvc2019_64
    echo [OK] Found Qt 6.6.3
) else if exist "C:\Qt\6.7.0\msvc2019_64\bin\qmake.exe" (
    set QT_PATH=C:/Qt/6.7.0/msvc2019_64
    echo [OK] Found Qt 6.7.0
) else if exist "C:\Qt\6.5.3\msvc2019_64\bin\qmake.exe" (
    set QT_PATH=C:/Qt/6.5.3/msvc2019_64
    echo [OK] Found Qt 6.5.3
) else (
    echo [ERROR] Qt6 not found!
    echo Please run scripts\setup_windows.bat to install dependencies.
    pause
    exit /b 1
)

REM Check OpenSSL
set OPENSSL_PATH=
if exist "C:\Program Files\OpenSSL-Win64\bin\openssl.exe" (
    set OPENSSL_PATH=C:/Program Files/OpenSSL-Win64
    echo [OK] Found OpenSSL
) else if exist "C:\OpenSSL-Win64\bin\openssl.exe" (
    set OPENSSL_PATH=C:/OpenSSL-Win64
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
REM Build
REM ============================================================================

REM Check if build directory exists
if not exist build (
    mkdir build
    echo Created build directory
)

cd build

REM Configure with CMake
echo Configuring with CMake...
cmake .. -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_PREFIX_PATH="%QT_PATH%" ^
    -DOPENSSL_ROOT_DIR="%OPENSSL_PATH%"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ================================================
    echo CMake Configuration Failed!
    echo ================================================
    echo.
    echo Troubleshooting:
    echo   1. Run scripts\setup_windows.bat to verify all dependencies
    echo   2. Check that Python development headers are installed
    echo   3. Ensure Visual Studio 2022 with C++ workload is installed
    echo.
    echo For detailed setup instructions, see: docs\guides\installation.md
    echo.
    pause
    exit /b 1
)

echo.
echo Building NDA...
cmake --build . --config Release

if %ERRORLEVEL% NEQ 0 (
    echo Build failed!
    pause
    exit /b 1
)

echo.
echo ================================================
echo Build completed successfully!
echo ================================================
echo.
echo Executable: build\Release\NDA.exe
echo.
echo To install Python dependencies:
echo   pip install -r requirements.txt
echo.
pause
