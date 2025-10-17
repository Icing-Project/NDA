#!/usr/bin/env python3
"""
NADE Deployment Script
Creates a ready-to-ship package with all necessary files
"""

import os
import shutil
import sys
import platform
from pathlib import Path

def main():
    print("=" * 60)
    print("NADE Deployment Script")
    print("=" * 60)
    print()

    # Determine platform
    is_windows = platform.system() == "Windows"
    exe_name = "NADE.exe" if is_windows else "NADE"

    # Paths
    base_dir = Path(__file__).parent
    build_dir = base_dir / "build"
    ship_dir = base_dir / "readytoship"

    # Create deployment structure
    print("Creating deployment structure...")
    (ship_dir / "bin").mkdir(parents=True, exist_ok=True)
    (ship_dir / "plugins").mkdir(parents=True, exist_ok=True)
    (ship_dir / "lib").mkdir(parents=True, exist_ok=True)
    (ship_dir / "docs").mkdir(parents=True, exist_ok=True)

    # Copy executable
    print(f"Copying {exe_name}...")
    if is_windows:
        exe_src = build_dir / "Release" / exe_name
    else:
        exe_src = build_dir / exe_name

    if exe_src.exists():
        shutil.copy2(exe_src, ship_dir / "bin" / exe_name)
        print(f"  ✓ {exe_name} copied")
    else:
        print(f"  ✗ {exe_name} not found at {exe_src}")
        print(f"    Build the application first!")
        return 1

    # Copy Python plugins
    print("Copying Python plugins...")
    plugins_src = base_dir / "plugins_py"
    plugins_dest = ship_dir / "plugins"

    if plugins_src.exists():
        # Copy all .py files
        for py_file in plugins_src.glob("*.py"):
            shutil.copy2(py_file, plugins_dest / py_file.name)
            print(f"  ✓ {py_file.name}")
    else:
        print(f"  ✗ plugins_py directory not found")
        return 1

    # Copy requirements.txt
    print("Copying requirements.txt...")
    req_file = base_dir / "requirements.txt"
    if req_file.exists():
        shutil.copy2(req_file, ship_dir / "requirements.txt")
        print("  ✓ requirements.txt")

    # Copy documentation
    print("Copying documentation...")
    docs = [
        ("README.md", "README.md"),
        ("NDA-SPECS.md", "SPECIFICATIONS.md"),
    ]

    for src_name, dest_name in docs:
        src_file = base_dir / src_name
        if src_file.exists():
            shutil.copy2(src_file, ship_dir / "docs" / dest_name)
            print(f"  ✓ {dest_name}")

    # Platform-specific steps
    if is_windows:
        print("\nWindows-specific deployment:")
        print("  Run the following commands to complete deployment:")
        print()
        print(f"  1. Deploy Qt DLLs:")
        print(f"     cd {ship_dir / 'bin'}")
        print(f"     windeployqt.exe {exe_name}")
        print()
        print(f"  2. Copy Python DLL:")
        print(f"     copy C:\\Python3x\\python3*.dll {ship_dir / 'bin'}")
        print()
        print(f"  3. Copy OpenSSL DLLs:")
        print(f'     copy "C:\\Program Files\\OpenSSL-Win64\\bin\\*.dll" {ship_dir / "bin"}')
        print()
    else:
        print("\nLinux deployment:")
        print("  Note: Linux binaries typically include all needed libraries")
        print("  or use system libraries via package manager")

    # Create launcher scripts
    print("\nCreating launcher scripts...")

    # Windows launcher
    launcher_bat = ship_dir / "NADE.bat"
    with open(launcher_bat, 'w') as f:
        f.write('@echo off\n')
        f.write('cd /d "%~dp0"\n')
        f.write('bin\\NADE.exe\n')
        f.write('pause\n')
    print(f"  ✓ NADE.bat")

    # Linux/Mac launcher
    launcher_sh = ship_dir / "NADE.sh"
    with open(launcher_sh, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('cd "$(dirname "$0")"\n')
        f.write('export PYTHONPATH="$PWD/plugins:$PYTHONPATH"\n')
        f.write('./bin/NADE\n')
    os.chmod(launcher_sh, 0o755)
    print(f"  ✓ NADE.sh")

    # Create version info
    print("\nCreating version info...")
    version_file = ship_dir / "VERSION.txt"
    with open(version_file, 'w') as f:
        f.write("NADE - Plugin-Based Audio Encryption System\n")
        f.write("Version: 1.0.0\n")
        f.write(f"Platform: {platform.system()} {platform.machine()}\n")
        f.write(f"Python: {sys.version.split()[0]}\n")
    print("  ✓ VERSION.txt")

    # Summary
    print("\n" + "=" * 60)
    print("DEPLOYMENT COMPLETE")
    print("=" * 60)
    print(f"\nPackage location: {ship_dir}")
    print("\nContents:")
    print(f"  - bin/{exe_name}")
    print(f"  - plugins/ ({len(list(plugins_dest.glob('*.py')))} files)")
    print(f"  - docs/ (documentation)")
    print(f"  - requirements.txt")
    print(f"  - Launcher scripts")

    print("\nNext steps:")
    if is_windows:
        print("  1. Run windeployqt.exe to copy Qt DLLs")
        print("  2. Copy Python and OpenSSL DLLs")
        print("  3. Test: readytoship\\NADE.bat")
        print("  4. Package: zip -r NADE-Windows.zip readytoship/")
    else:
        print("  1. Test: cd readytoship && ./NADE.sh")
        print("  2. Package: tar -czf NADE-Linux.tar.gz readytoship/")

    print("\n✓ Done!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
