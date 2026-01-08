@echo off
setlocal

REM ============================================================================
REM Python Virtual Environment Setup (Windows)
REM ============================================================================
REM This script creates a Python virtual environment for NDA development.
REM Using a venv is optional but recommended for clean dependency management.
REM ============================================================================

echo ================================================
echo NDA - Python Virtual Environment Setup
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found!
    echo Please install Python 3.8+ first.
    echo Run scripts\setup_windows.bat for complete setup.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYTHON_VERSION=%%v
echo [OK] Python %PYTHON_VERSION% found
echo.

REM Check if venv already exists
if exist "venv" (
    echo [INFO] Virtual environment already exists at: venv\
    echo.
    set /p RECREATE="Do you want to recreate it? This will delete existing venv. (Y/N): "
    if /i "!RECREATE!"=="Y" (
        echo Removing existing venv...
        rmdir /s /q venv
    ) else (
        echo Keeping existing venv.
        goto :activate_instructions
    )
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to create virtual environment
    echo Make sure Python venv module is installed
    pause
    exit /b 1
)
echo [OK] Virtual environment created
echo.

REM Activate venv and install dependencies
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Installing NDA Python dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Some packages may have failed to install
    echo You may need to install them manually later
) else (
    echo [OK] All dependencies installed successfully
)

echo.
echo ================================================
echo Virtual Environment Setup Complete!
echo ================================================
echo.

:activate_instructions
echo To use the virtual environment:
echo.
echo   1. Activate it:
echo      venv\Scripts\activate.bat
echo.
echo   2. Build NDA (venv will be auto-detected by CMake):
echo      scripts\build_windows.bat
echo.
echo   3. When done, deactivate:
echo      deactivate
echo.
echo Benefits of using venv:
echo   - Isolated Python dependencies
echo   - No system Python pollution
echo   - Easy to reset (just delete venv folder)
echo   - CMake auto-detects and uses venv Python
echo.
pause
