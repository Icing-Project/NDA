@echo off
REM NDA Windows Deployment Script
REM Packages everything into readytoship folder

echo ================================================
echo NDA - Windows Deployment Script
echo ================================================
echo.

REM Check if executable exists (Visual Studio or Ninja layout)
set EXE_PATH=build\Release\NDA.exe
if not exist "%EXE_PATH%" (
    set EXE_PATH=build\NDA.exe
)
if not exist "%EXE_PATH%" (
    echo ERROR: NDA.exe not found in build\Release or build\
    echo Please build the project first (scripts\build_windows.bat or scripts\build_windows_ninja.bat)
    pause
    exit /b 1
)

REM Run Python deployment script
echo Running deployment script...
python deploy.py
if %ERRORLEVEL% NEQ 0 (
    echo Deployment script failed!
    pause
    exit /b 1
)

echo.
echo ================================================
echo Deploying Qt Dependencies
echo ================================================
echo.

REM Find Qt installation
set QT_DIR=
if exist "C:\Qt\6.5.3\msvc2019_64\bin\windeployqt.exe" (
    set QT_DIR=C:\Qt\6.5.3\msvc2019_64
) else if exist "C:\Qt\6.6.0\msvc2019_64\bin\windeployqt.exe" (
    set QT_DIR=C:\Qt\6.6.0\msvc2019_64
) else if exist "C:\Qt\6.7.0\msvc2019_64\bin\windeployqt.exe" (
    set QT_DIR=C:\Qt\6.7.0\msvc2019_64
)

if "%QT_DIR%"=="" (
    echo WARNING: Qt installation not found automatically
    echo Please run windeployqt manually:
    echo   cd readytoship\bin
    echo   "C:\Qt\6.x\msvc2019_64\bin\windeployqt.exe" NDA.exe
    echo.
) else (
    echo Found Qt at: %QT_DIR%
    echo Running windeployqt...
    cd readytoship\bin
    "%QT_DIR%\bin\windeployqt.exe" NDA.exe --release --no-translations
    cd ..\..
    echo ✓ Qt dependencies deployed
)

echo.
echo ================================================
echo Deploying Python Runtime
echo ================================================
echo.

REM Find Python DLL
for /f "delims=" %%i in ('python -c "import sys; print(sys.base_prefix)"') do set PYTHON_DIR=%%i
set PYTHON_DLL=%PYTHON_DIR%\python3*.dll

if exist "%PYTHON_DLL%" (
    copy "%PYTHON_DLL%" readytoship\bin\
    echo ✓ Python DLL deployed
) else (
    echo WARNING: Python DLL not found
    echo Please copy manually: %PYTHON_DIR%\python3*.dll
)

echo.
echo ================================================
echo Deploying OpenSSL
echo ================================================
echo.

if exist "C:\Program Files\OpenSSL-Win64\bin" (
    copy "C:\Program Files\OpenSSL-Win64\bin\libcrypto-3-x64.dll" readytoship\bin\ 2>nul
    copy "C:\Program Files\OpenSSL-Win64\bin\libssl-3-x64.dll" readytoship\bin\ 2>nul
    echo ✓ OpenSSL DLLs deployed
) else (
    echo WARNING: OpenSSL not found at standard location
    echo Please copy libcrypto and libssl DLLs manually
)

echo.
echo ================================================
echo Installing Python Dependencies
echo ================================================
echo.

pip install -r requirements.txt
echo ✓ Python packages installed

echo.
echo ================================================
echo DEPLOYMENT COMPLETE
echo ================================================
echo.
echo Package location: readytoship\
echo.
echo To test:
echo   cd readytoship
echo   NDA.bat
echo.
echo To distribute:
echo   1. Test the application
echo   2. Create ZIP: 7z a NDA-Windows-x64.zip readytoship\*
echo   3. Or create installer with Inno Setup
echo.
pause
