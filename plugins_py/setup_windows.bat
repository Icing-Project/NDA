@echo off
REM Windows setup script for NADE Python plugins

echo ======================================
echo NADE Python Plugins Setup (Windows)
echo ======================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found!
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

REM Install requirements
echo Installing Python dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to install dependencies
    echo.
    echo Trying alternative method for PyAudio on Windows...
    python -m pip install pipwin
    python -m pipwin install pyaudio
)

echo.
echo ======================================
echo Testing plugins...
echo ======================================
python test_plugins.py

if %errorlevel% equ 0 (
    echo.
    echo ======================================
    echo Setup completed successfully!
    echo ======================================
    echo.
    echo You can now:
    echo   1. Run test_plugins.py to test the plugins
    echo   2. Use the plugins from the NADE C++ application
    echo.
) else (
    echo.
    echo ======================================
    echo Setup completed with warnings
    echo ======================================
    echo.
    echo The basic plugins work, but PyAudio might not be available.
    echo This means microphone and speaker plugins won't work.
    echo You can still use: sine_wave_source, null_sink, wav_file_sink
    echo.
)

pause
