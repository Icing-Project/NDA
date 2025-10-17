@echo off
REM NADE Windows Build Script
REM Requirements:
REM   - CMake 3.16+
REM   - Qt6 (with MSVC)
REM   - Python 3.7+ with development headers
REM   - Visual Studio 2019+ or Build Tools
REM   - OpenSSL

echo ================================================
echo NADE - Windows Build Script
echo ================================================
echo.

REM Check if build directory exists
if not exist build (
    mkdir build
    echo Created build directory
)

cd build

REM Configure with CMake
echo Configuring with CMake...
cmake .. -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_PREFIX_PATH="C:/Qt/6.x/msvc2019_64" ^
    -DOPENSSL_ROOT_DIR="C:/OpenSSL-Win64"

if %ERRORLEVEL% NEQ 0 (
    echo CMake configuration failed!
    echo.
    echo Please check:
    echo   1. Qt6 is installed at C:/Qt/6.x/msvc2019_64
    echo   2. OpenSSL is installed at C:/OpenSSL-Win64
    echo   3. Python 3.x is in PATH
    echo   4. Visual Studio 2022 is installed
    pause
    exit /b 1
)

echo.
echo Building NADE...
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
echo Executable: build\Release\NADE.exe
echo.
echo To install Python dependencies:
echo   pip install -r requirements.txt
echo.
pause
