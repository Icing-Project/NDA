@echo off
setlocal

REM Configure and build NDA (app + plugins) with Ninja using MSVC toolchain.
REM Prereqs: VS Build Tools 2022 (C++ workload + Windows 10 SDK), Qt 6.6.3 msvc2019_64, OpenSSL Win64.

set ROOT=%~dp0..
cd /d "%ROOT%"

REM Update the SDK version below if a newer Windows 10/11 SDK is installed.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 10.0.18362.0
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to initialize Visual Studio build environment.
    exit /b 1
)

set QT_PREFIX=C:/Qt/6.6.3/msvc2019_64
set OPENSSL_ROOT=C:/Program Files/OpenSSL-Win64

if not exist build mkdir build

cmake -S . -B build -G "Ninja" ^
    -DCMAKE_PREFIX_PATH="%QT_PREFIX%" ^
    -DOPENSSL_ROOT_DIR="%OPENSSL_ROOT%" ^
    -DCMAKE_BUILD_TYPE=Release
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] CMake configure failed.
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
