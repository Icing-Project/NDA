@echo off
REM NADE Windows Launcher

cd /d "%~dp0"

REM Check if built
if not exist "bin\NADE.exe" (
    echo ERROR: NADE.exe not found!
    echo.
    echo Please build first:
    echo   1. cd build_scripts
    echo   2. build_windows.bat
    echo   3. deploy_windows.bat
    pause
    exit /b 1
)

REM Check dependencies
python -c "import sounddevice" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Installing Python dependencies...
    pip install -r requirements.txt
)

REM Run NADE
bin\NADE.exe
