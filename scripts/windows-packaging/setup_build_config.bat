@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo NDA Build Configuration Setup
echo ============================================================
echo.
echo This script will detect your development environment and
echo create build_config.json for automated packaging.
echo.

set "SCRIPT_DIR=%~dp0"
set "CONFIG_FILE=%SCRIPT_DIR%build_config.json"

REM Check if config already exists
if exist "%CONFIG_FILE%" (
    echo WARNING: build_config.json already exists!
    echo.
    set /p "OVERWRITE=Overwrite existing configuration? (y/N): "
    if /i not "!OVERWRITE!"=="y" (
        echo.
        echo Configuration cancelled.
        pause
        exit /b 0
    )
    echo.
)

echo Detecting installations...
echo.

REM ============================================================
REM Detect Qt Installation
REM ============================================================
set "QT_PATH="
set "QT_VERSIONS=6.8.0 6.7.0 6.6.3 6.5.3"

for %%v in (%QT_VERSIONS%) do (
    if exist "C:\Qt\%%v\msvc2019_64\bin\qmake.exe" (
        set "QT_PATH=C:/Qt/%%v/msvc2019_64"
        echo [FOUND] Qt %%v at !QT_PATH!
        goto :qt_found
    )
)

echo [NOT FOUND] Qt6 - Checking common locations...
if exist "C:\Qt" (
    echo   Qt folder exists but no compatible version found
    echo   Please install Qt 6.5+ with MSVC 2019 64-bit
)
:qt_found

REM ============================================================
REM Detect OpenSSL
REM ============================================================
set "OPENSSL_PATH="

if exist "C:\Program Files\OpenSSL-Win64\bin\openssl.exe" (
    set "OPENSSL_PATH=C:/Program Files/OpenSSL-Win64"
    echo [FOUND] OpenSSL at !OPENSSL_PATH!
) else if exist "C:\OpenSSL-Win64\bin\openssl.exe" (
    set "OPENSSL_PATH=C:/OpenSSL-Win64"
    echo [FOUND] OpenSSL at !OPENSSL_PATH!
) else (
    echo [NOT FOUND] OpenSSL - Please install OpenSSL 3.x Win64
)

REM ============================================================
REM Detect Python
REM ============================================================
set "PYTHON_PATH="

REM Try to get Python path from where command
for /f "delims=" %%i in ('where python 2^>nul') do (
    set "PYTHON_EXE=%%i"
    for %%j in ("!PYTHON_EXE!") do set "PYTHON_PATH=%%~dpj"
    REM Remove trailing backslash and convert to forward slashes
    set "PYTHON_PATH=!PYTHON_PATH:~0,-1!"
    set "PYTHON_PATH=!PYTHON_PATH:\=/!"
    echo [FOUND] Python at !PYTHON_PATH!
    goto :python_found
)

echo [NOT FOUND] Python - Please install Python 3.8+ with development headers
:python_found

REM ============================================================
REM Detect Visual Studio
REM ============================================================
set "VS_PATH="
set "VS_YEARS=2022 2019"

for %%y in (%VS_YEARS%) do (
    if exist "C:\Program Files\Microsoft Visual Studio\%%y\Community\VC\Auxiliary\Build\vcvars64.bat" (
        set "VS_PATH=C:/Program Files/Microsoft Visual Studio/%%y/Community"
        echo [FOUND] Visual Studio %%y Community at !VS_PATH!
        goto :vs_found
    )
    if exist "C:\Program Files\Microsoft Visual Studio\%%y\Professional\VC\Auxiliary\Build\vcvars64.bat" (
        set "VS_PATH=C:/Program Files/Microsoft Visual Studio/%%y/Professional"
        echo [FOUND] Visual Studio %%y Professional at !VS_PATH!
        goto :vs_found
    )
    if exist "C:\Program Files\Microsoft Visual Studio\%%y\Enterprise\VC\Auxiliary\Build\vcvars64.bat" (
        set "VS_PATH=C:/Program Files/Microsoft Visual Studio/%%y/Enterprise"
        echo [FOUND] Visual Studio %%y Enterprise at !VS_PATH!
        goto :vs_found
    )
)

echo [NOT FOUND] Visual Studio - Please install VS 2019/2022 with C++ workload
:vs_found

echo.
echo ============================================================
echo Configuration Summary
echo ============================================================
echo.

REM Check if all required components are found
set "MISSING_COMPONENTS="
if "!QT_PATH!"=="" (
    echo [MISSING] Qt6
    set "MISSING_COMPONENTS=1"
)
if "!OPENSSL_PATH!"=="" (
    echo [MISSING] OpenSSL
    set "MISSING_COMPONENTS=1"
)
if "!PYTHON_PATH!"=="" (
    echo [MISSING] Python
    set "MISSING_COMPONENTS=1"
)
if "!VS_PATH!"=="" (
    echo [MISSING] Visual Studio
    set "MISSING_COMPONENTS=1"
)

if defined MISSING_COMPONENTS (
    echo.
    echo WARNING: Some components are missing!
    echo You can continue and manually edit build_config.json,
    echo or install the missing components and run this script again.
    echo.
    set /p "CONTINUE=Continue anyway? (y/N): "
    if /i not "!CONTINUE!"=="y" (
        echo.
        echo Setup cancelled.
        pause
        exit /b 1
    )
)

REM ============================================================
REM Create build_config.json
REM ============================================================
echo.
echo Creating build_config.json...

REM Set defaults for missing components
if "!QT_PATH!"=="" set "QT_PATH=C:/Qt/6.6.3/msvc2019_64"
if "!OPENSSL_PATH!"=="" set "OPENSSL_PATH=C:/Program Files/OpenSSL-Win64"
if "!PYTHON_PATH!"=="" set "PYTHON_PATH=C:/Python312"
if "!VS_PATH!"=="" set "VS_PATH=C:/Program Files/Microsoft Visual Studio/2022/Community"

(
echo {
echo   "_comment": "NDA Build Configuration - Auto-generated by setup_build_config.bat",
echo   "qt_path": "!QT_PATH!",
echo   "openssl_path": "!OPENSSL_PATH!",
echo   "python_path": "!PYTHON_PATH!",
echo   "visual_studio_path": "!VS_PATH!",
echo   "build_config": "Release",
echo   "enable_python": true,
echo   "package_version": "2.0.0",
echo   "package_name": "NDA-Windows-Portable"
echo }
) > "%CONFIG_FILE%"

echo.
echo ============================================================
echo Configuration Complete!
echo ============================================================
echo.
echo Configuration file created: %CONFIG_FILE%
echo.
echo Next steps:
echo   1. Review and edit build_config.json if needed
echo   2. Run: scripts\build_release_package.bat
echo.
echo To rebuild configuration, delete build_config.json and run this script again.
echo.
pause
