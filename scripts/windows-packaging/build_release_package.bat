@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM NDA - Master Release Package Builder
REM ============================================================================
REM This script automates the complete build and packaging process:
REM   1. Verifies prerequisites and configuration
REM   2. Cleans and rebuilds NDA.exe and all plugins
REM   3. Creates standalone package with all dependencies
REM   4. Bundles Python runtime (if enabled)
REM   5. Verifies package integrity
REM   6. Creates distributable ZIP archive
REM
REM Prerequisites:
REM   - Run scripts\windows-packaging\setup_build_config.bat once to create build_config.json
REM   - All development tools installed (CMake, Qt, OpenSSL, Python, VS)
REM
REM Output:
REM   - package/ - Standalone package directory
REM   - NDA-Windows-Portable-v{version}.zip - Distributable archive
REM ============================================================================

echo.
echo ========================================================================
echo NDA - MASTER RELEASE PACKAGE BUILDER
echo ========================================================================
echo.
echo This script will build a complete standalone Windows package
echo with all dependencies bundled.
echo.

set "SCRIPT_DIR=%~dp0"
set "BASE_DIR=%SCRIPT_DIR%..\..
set "CONFIG_FILE=%SCRIPT_DIR%build_config.json"

REM ============================================================================
REM Check Configuration
REM ============================================================================

echo [1/6] Checking configuration...
echo.

if not exist "%CONFIG_FILE%" (
    echo ERROR: build_config.json not found!
    echo.
    echo Please run: scripts\windows-packaging\setup_build_config.bat
    echo.
    echo This will detect your development environment and create
    echo the configuration file needed for automated packaging.
    echo.
    pause
    exit /b 1
)

echo   Found build_config.json
echo.

REM Parse configuration (simple JSON parsing for batch)
REM We'll just verify the file exists and let Python scripts handle parsing

REM ============================================================================
REM Verify Prerequisites
REM ============================================================================

echo [2/6] Verifying prerequisites...
echo.

REM Check CMake
cmake --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo   ERROR: CMake not found in PATH
    echo   Please install CMake 3.16 or later
    pause
    exit /b 1
)
echo   CMake: OK

REM Check Python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo   ERROR: Python not found in PATH
    echo   Please install Python 3.8 or later
    pause
    exit /b 1
)
echo   Python: OK

REM Check Visual Studio
where cl.exe >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo   WARNING: Visual Studio compiler not in PATH
    echo   Attempting to continue anyway...
)

echo.
echo   All prerequisites OK
echo.

REM ============================================================================
REM Clean Build Directory
REM ============================================================================

echo [3/6] Cleaning build directory...
echo.

cd /d "%BASE_DIR%"

if exist "build" (
    echo   Removing old build directory...
    rmdir /s /q build 2>nul
    if exist "build" (
        echo   WARNING: Could not fully clean build directory
        echo   Some files may be locked. Continuing anyway...
    )
)

if exist "package" (
    echo   Removing old package directory...
    rmdir /s /q package 2>nul
    if exist "package" (
        echo   WARNING: Could not fully clean package directory
        echo   Some files may be locked. Continuing anyway...
    )
)

mkdir build 2>nul
echo   Build directory ready
echo.

REM ============================================================================
REM Configure and Build
REM ============================================================================

echo [4/6] Building NDA and plugins...
echo.

cd /d "%BASE_DIR%\build"

REM Read configuration values from JSON
REM For batch simplicity, we use Python to extract values
for /f "delims=" %%i in ('python -c "import json; f=open(r'%CONFIG_FILE%'); c=json.load(f); print(c['qt_path'])"') do set "QT_PATH=%%i"
for /f "delims=" %%i in ('python -c "import json; f=open(r'%CONFIG_FILE%'); c=json.load(f); print(c['openssl_path'])"') do set "OPENSSL_PATH=%%i"
for /f "delims=" %%i in ('python -c "import json; f=open(r'%CONFIG_FILE%'); c=json.load(f); print(c.get('enable_python', False))"') do set "ENABLE_PYTHON=%%i"
for /f "delims=" %%i in ('python -c "import json; f=open(r'%CONFIG_FILE%'); c=json.load(f); print(c.get('build_config', 'Release'))"') do set "BUILD_CONFIG=%%i"

echo   Qt Path: %QT_PATH%
echo   OpenSSL Path: %OPENSSL_PATH%
echo   Python Support: %ENABLE_PYTHON%
echo   Build Config: %BUILD_CONFIG%
echo.

REM Configure with CMake
echo   Configuring with CMake...
cmake .. -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_PREFIX_PATH="%QT_PATH%" ^
    -DOPENSSL_ROOT_DIR="%OPENSSL_PATH%" ^
    -DNDA_ENABLE_PYTHON=%ENABLE_PYTHON%

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo   ERROR: CMake configuration failed!
    echo.
    echo   Troubleshooting:
    echo     - Verify paths in build_config.json are correct
    echo     - Check that Qt, OpenSSL are properly installed
    echo     - Review CMake output above for specific errors
    echo.
    pause
    exit /b 1
)

echo.
echo   Building NDA.exe and plugins...
cmake --build . --config %BUILD_CONFIG% --parallel

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo   ERROR: Build failed!
    echo.
    echo   Review compiler errors above.
    echo.
    pause
    exit /b 1
)

echo.
echo   Build completed successfully!
echo.

cd /d "%BASE_DIR%"

REM ============================================================================
REM Deploy Standalone Package
REM ============================================================================

echo [5/6] Deploying standalone package...
echo.

python "%SCRIPT_DIR%deploy_standalone.py"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo   ERROR: Deployment failed!
    echo.
    echo   Check Python script output above for details.
    echo.
    pause
    exit /b 1
)

echo.

REM ============================================================================
REM Verify Package
REM ============================================================================

echo [6/6] Verifying package...
echo.

python "%SCRIPT_DIR%verify_package.py"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo   WARNING: Package verification failed or has warnings
    echo.
    echo   Review verification output above.
    echo   The package may still work but should be tested manually.
    echo.
)

echo.

REM ============================================================================
REM Create ZIP Archive
REM ============================================================================

echo Creating ZIP archive...
echo.

REM Get package name and version from config
for /f "delims=" %%i in ('python -c "import json; f=open(r'%CONFIG_FILE%'); c=json.load(f); print(c.get('package_name', 'NDA-Windows-Portable'))"') do set "PACKAGE_NAME=%%i"
for /f "delims=" %%i in ('python -c "import json; f=open(r'%CONFIG_FILE%'); c=json.load(f); print(c.get('package_version', '2.0.0'))"') do set "VERSION=%%i"

set "ZIP_NAME=%PACKAGE_NAME%-v%VERSION%.zip"

REM Check if package directory exists
if not exist "package" (
    echo   ERROR: Package directory not found!
    echo   Deployment may have failed.
    pause
    exit /b 1
)

REM Create ZIP using PowerShell (available on Windows 10+)
echo   Creating %ZIP_NAME%...

powershell -Command "Compress-Archive -Path 'package\*' -DestinationPath '%ZIP_NAME%' -Force"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo   ERROR: Failed to create ZIP archive!
    echo.
    echo   You can manually create the ZIP:
    echo     1. Right-click 'package' folder
    echo     2. Send to ^> Compressed (zipped) folder
    echo     3. Rename to %ZIP_NAME%
    echo.
    pause
    exit /b 1
)

echo   ZIP archive created successfully
echo.

REM ============================================================================
REM Calculate Package Statistics
REM ============================================================================

echo ========================================================================
echo BUILD COMPLETE!
echo ========================================================================
echo.

REM Get package size
for /f "delims=" %%i in ('python -c "import os; print(f'{sum(f.stat().st_size for f in pathlib.Path(\"package\").rglob(\"*\") if f.is_file()) / (1024*1024):.1f}'); import pathlib"') do set "PKG_SIZE=%%i"

REM Get ZIP size
for %%A in ("%ZIP_NAME%") do set "ZIP_SIZE=%%~zA"
set /a "ZIP_SIZE_MB=%ZIP_SIZE% / 1048576"

echo Package Details:
echo   Location: %CD%\package\
echo   Size (extracted): %PKG_SIZE% MB
echo.
echo ZIP Archive:
echo   File: %ZIP_NAME%
echo   Size: %ZIP_SIZE_MB% MB
echo.

REM ============================================================================
REM Next Steps
REM ============================================================================

echo ========================================================================
echo NEXT STEPS
echo ========================================================================
echo.
echo 1. TEST THE PACKAGE:
echo    cd package
echo    NDA.bat
echo.
echo 2. VERIFY FUNCTIONALITY:
echo    - Application launches
echo    - Plugins load successfully
echo    - Create test audio pipeline
echo    - Verify audio processing works
echo.
echo 3. DISTRIBUTE:
echo    Upload %ZIP_NAME% to your distribution channel
echo.
echo ========================================================================

REM Ask if user wants to open package folder
echo.
set /p "OPEN_FOLDER=Open package folder in Explorer? (Y/n): "
if /i "%OPEN_FOLDER%"=="n" goto :skip_open
if /i "%OPEN_FOLDER%"=="no" goto :skip_open

explorer "package"

:skip_open

echo.
echo Done!
echo.
pause
