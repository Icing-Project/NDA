#!/usr/bin/env python3
"""
Create separate Linux and Windows deployment packages
"""

import os
import shutil
import sys
from pathlib import Path

def main():
    print("=" * 60)
    print("Creating Platform-Specific Packages")
    print("=" * 60)
    print()

    base_dir = Path(__file__).parent

    # Source directories
    build_dir = base_dir / "build"
    plugins_dir = base_dir / "plugins_py"

    # Destination directories
    linux_dir = base_dir / "readytoship_linux"
    windows_dir = base_dir / "readytoship_windows"

    # Clean and create destination directories
    for dest_dir in [linux_dir, windows_dir]:
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        dest_dir.mkdir()

    print("Creating Linux package...")
    create_linux_package(base_dir, linux_dir, build_dir, plugins_dir)

    print("\nCreating Windows package...")
    create_windows_package(base_dir, windows_dir, plugins_dir)

    print("\n" + "=" * 60)
    print("PLATFORM PACKAGES CREATED")
    print("=" * 60)
    print(f"\nLinux package:   {linux_dir}")
    print(f"Windows package: {windows_dir}")
    print()

    # Show structure
    print("Linux package structure:")
    os.system(f"tree -L 2 {linux_dir} 2>/dev/null || find {linux_dir} -maxdepth 2 | head -20")

    print("\nWindows package structure:")
    os.system(f"tree -L 2 {windows_dir} 2>/dev/null || find {windows_dir} -maxdepth 2 | head -20")

    print("\n✓ Done!")
    return 0

def create_linux_package(base_dir, linux_dir, build_dir, plugins_dir):
    """Create Linux deployment package"""

    # Create structure
    (linux_dir / "bin").mkdir()
    (linux_dir / "plugins").mkdir()
    (linux_dir / "docs").mkdir()

    # Copy executable
    exe_src = build_dir / "NDA"
    if exe_src.exists():
        shutil.copy2(exe_src, linux_dir / "bin" / "NDA")
        os.chmod(linux_dir / "bin" / "NDA", 0o755)
        print("  ✓ Executable copied")
    else:
        print("  ✗ Executable not found!")
        return

    # Copy plugins
    if plugins_dir.exists():
        for py_file in plugins_dir.glob("*.py"):
            shutil.copy2(py_file, linux_dir / "plugins" / py_file.name)
        print(f"  ✓ {len(list((linux_dir / 'plugins').glob('*.py')))} plugins copied")

    # Copy documentation
    docs = {
        "README.md": "README.md",
        "NDA-SPECS.md": "SPECIFICATIONS.md",
    }
    for src, dst in docs.items():
        src_file = base_dir / src
        if src_file.exists():
            shutil.copy2(src_file, linux_dir / "docs" / dst)

    # Copy readytoship docs
    readytoship_docs = base_dir / "readytoship" / "docs"
    if readytoship_docs.exists():
        for doc in ["USER_GUIDE.md"]:
            src_file = readytoship_docs / doc
            if src_file.exists():
                shutil.copy2(src_file, linux_dir / "docs" / doc)

    print(f"  ✓ Documentation copied")

    # Create Linux-specific README
    create_linux_readme(linux_dir)

    # Copy requirements.txt
    req_file = base_dir / "requirements.txt"
    if req_file.exists():
        shutil.copy2(req_file, linux_dir / "requirements.txt")

    # Create launcher script
    create_linux_launcher(linux_dir)

    # Create VERSION file
    create_version_file(linux_dir, "Linux")

    print("  ✓ Linux package complete")

def create_windows_package(base_dir, windows_dir, plugins_dir):
    """Create Windows deployment package (build-ready)"""

    # Create structure
    (windows_dir / "bin").mkdir()
    (windows_dir / "plugins").mkdir()
    (windows_dir / "docs").mkdir()
    (windows_dir / "build_scripts").mkdir()

    # Copy plugins
    if plugins_dir.exists():
        for py_file in plugins_dir.glob("*.py"):
            shutil.copy2(py_file, windows_dir / "plugins" / py_file.name)
        print(f"  ✓ {len(list((windows_dir / 'plugins').glob('*.py')))} plugins copied")

    # Copy documentation
    docs = {
        "README.md": "README.md",
        "NDA-SPECS.md": "SPECIFICATIONS.md",
    }
    for src, dst in docs.items():
        src_file = base_dir / src
        if src_file.exists():
            shutil.copy2(src_file, windows_dir / "docs" / dst)

    # Copy Windows-specific docs
    readytoship_docs = base_dir / "readytoship" / "docs"
    if readytoship_docs.exists():
        for doc in ["USER_GUIDE.md", "WINDOWS_README.md", "BUILD_WINDOWS.md"]:
            src_file = readytoship_docs / doc
            if src_file.exists():
                shutil.copy2(src_file, windows_dir / "docs" / doc)

    print(f"  ✓ Documentation copied")

    # Copy build scripts
    build_scripts = ["build_windows.bat", "deploy_windows.bat", "CMakeLists.txt"]
    for script in build_scripts:
        src_file = base_dir / script
        if src_file.exists():
            shutil.copy2(src_file, windows_dir / "build_scripts" / script)

    # Copy entire source tree for Windows build
    copy_source_tree(base_dir, windows_dir)

    print(f"  ✓ Build scripts copied")

    # Copy requirements.txt
    req_file = base_dir / "requirements.txt"
    if req_file.exists():
        shutil.copy2(req_file, windows_dir / "requirements.txt")

    # Create Windows-specific README
    create_windows_readme(windows_dir)

    # Create launcher script
    create_windows_launcher(windows_dir)

    # Create VERSION file
    create_version_file(windows_dir, "Windows")

    print("  ✓ Windows package complete")

def copy_source_tree(base_dir, windows_dir):
    """Copy source tree needed for Windows compilation"""

    # Copy source directories
    for src_dir in ["src", "include"]:
        src_path = base_dir / src_dir
        if src_path.exists():
            shutil.copytree(src_path, windows_dir / src_dir, dirs_exist_ok=True)

    print("  ✓ Source tree copied")

def create_linux_launcher(linux_dir):
    """Create Linux launcher script"""
    launcher = linux_dir / "run_nda.sh"
    with open(launcher, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('# NDA Linux Launcher\n\n')
        f.write('cd "$(dirname "$0")"\n\n')
        f.write('# Add plugins to Python path\n')
        f.write('export PYTHONPATH="$PWD/plugins:$PYTHONPATH"\n\n')
        f.write('# Check dependencies\n')
        f.write('if ! python3 -c "import sounddevice" 2>/dev/null; then\n')
        f.write('    echo "Installing Python dependencies..."\n')
        f.write('    pip3 install --user -r requirements.txt\n')
        f.write('fi\n\n')
        f.write('# Run NDA\n')
        f.write('./bin/NDA\n')
    os.chmod(launcher, 0o755)
    print("  ✓ Launcher created")

def create_windows_launcher(windows_dir):
    """Create Windows launcher script"""
    launcher = windows_dir / "run_nda.bat"
    with open(launcher, 'w') as f:
        f.write('@echo off\n')
        f.write('REM NDA Windows Launcher\n\n')
        f.write('cd /d "%~dp0"\n\n')
        f.write('REM Check if built\n')
        f.write('if not exist "bin\\NDA.exe" (\n')
        f.write('    echo ERROR: NDA.exe not found!\n')
        f.write('    echo.\n')
        f.write('    echo Please build first:\n')
        f.write('    echo   1. cd build_scripts\n')
        f.write('    echo   2. build_windows.bat\n')
        f.write('    echo   3. deploy_windows.bat\n')
        f.write('    pause\n')
        f.write('    exit /b 1\n')
        f.write(')\n\n')
        f.write('REM Check dependencies\n')
        f.write('python -c "import sounddevice" 2>nul\n')
        f.write('if %ERRORLEVEL% NEQ 0 (\n')
        f.write('    echo Installing Python dependencies...\n')
        f.write('    pip install -r requirements.txt\n')
        f.write(')\n\n')
        f.write('REM Run NDA\n')
        f.write('bin\\NDA.exe\n')
    print("  ✓ Launcher created")

def create_linux_readme(linux_dir):
    """Create Linux-specific README"""
    readme = linux_dir / "README.md"
    with open(readme, 'w') as f:
        f.write('# NDA - Linux Package\n\n')
        f.write('**Version 1.0.0**\n\n')
        f.write('Cross-platform audio encryption system - Linux build\n\n')
        f.write('## Quick Start\n\n')
        f.write('```bash\n')
        f.write('# Install dependencies\n')
        f.write('pip3 install --user -r requirements.txt\n\n')
        f.write('# Run NDA\n')
        f.write('./run_nda.sh\n')
        f.write('```\n\n')
        f.write('## Manual Installation\n\n')
        f.write('```bash\n')
        f.write('# Install Python packages\n')
        f.write('pip3 install --user sounddevice numpy\n\n')
        f.write('# Run executable\n')
        f.write('./bin/NDA\n')
        f.write('```\n\n')
        f.write('## Documentation\n\n')
        f.write('- **docs/USER_GUIDE.md** - Complete usage guide\n')
        f.write('- **docs/SPECIFICATIONS.md** - Technical specs\n\n')
        f.write('## Support\n\n')
        f.write('See docs/USER_GUIDE.md for troubleshooting and usage instructions.\n')
    print("  ✓ README created")

def create_windows_readme(windows_dir):
    """Create Windows-specific README"""
    readme = windows_dir / "README.md"
    with open(readme, 'w') as f:
        f.write('# NDA - Windows Package\n\n')
        f.write('**Version 1.0.0**\n\n')
        f.write('Cross-platform audio encryption system - Windows build-ready package\n\n')
        f.write('## Build Instructions\n\n')
        f.write('### Prerequisites\n\n')
        f.write('1. Visual Studio 2019/2022 with C++ Desktop Development\n')
        f.write('2. CMake 3.16+\n')
        f.write('3. Qt 6.x for MSVC (msvc2019_64 or msvc2022_64)\n')
        f.write('4. Python 3.7+ with development headers\n')
        f.write('5. OpenSSL for Windows (Win64 full version)\n\n')
        f.write('### Build Steps\n\n')
        f.write('```cmd\n')
        f.write('cd build_scripts\n')
        f.write('build_windows.bat\n')
        f.write('```\n\n')
        f.write('### Deploy\n\n')
        f.write('```cmd\n')
        f.write('cd build_scripts\n')
        f.write('deploy_windows.bat\n')
        f.write('```\n\n')
        f.write('This will copy NDA.exe, Qt DLLs, Python DLL, and OpenSSL DLLs to bin/ folder.\n\n')
        f.write('### Run\n\n')
        f.write('```cmd\n')
        f.write('pip install -r requirements.txt\n')
        f.write('run_nda.bat\n')
        f.write('```\n\n')
        f.write('## Documentation\n\n')
        f.write('- **docs/BUILD_WINDOWS.md** - Detailed build instructions\n')
        f.write('- **docs/WINDOWS_README.md** - Windows setup guide\n')
        f.write('- **docs/USER_GUIDE.md** - Complete usage guide\n\n')
        f.write('## Pre-built Binary\n\n')
        f.write('If you have a pre-built NDA.exe, place it in bin/ folder along with required DLLs.\n\n')
        f.write('## Support\n\n')
        f.write('See docs/BUILD_WINDOWS.md for troubleshooting and detailed build instructions.\n')
    print("  ✓ README created")

def create_version_file(dest_dir, platform):
    """Create version file"""
    version = dest_dir / "VERSION.txt"
    with open(version, 'w') as f:
        f.write('NDA - Plugin-Based Audio Encryption System\n')
        f.write('Version: 1.0.0\n')
        f.write(f'Platform: {platform}\n')
        f.write('Build Date: October 2025\n')
        f.write('\nFeatures:\n')
        f.write('- Real-time audio recording\n')
        f.write('- WAV file export (32-bit float)\n')
        f.write('- AES-256-GCM encryption\n')
        f.write('- Network streaming\n')
        f.write('- Python plugin system\n')
    print("  ✓ VERSION file created")

if __name__ == "__main__":
    sys.exit(main())
