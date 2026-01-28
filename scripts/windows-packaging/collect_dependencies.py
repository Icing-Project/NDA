#!/usr/bin/env python3
"""
NDA Dependency Collection Script
Automatically collects all required DLLs and dependencies for Windows packaging
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'ignore')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'ignore')


def load_config() -> Dict:
    """Load build configuration from build_config.json"""
    script_dir = Path(__file__).parent
    config_file = script_dir / "build_config.json"

    if not config_file.exists():
        print("ERROR: build_config.json not found!")
        print("Run scripts/windows-packaging/setup_build_config.bat first to create configuration.")
        sys.exit(1)

    with open(config_file, 'r') as f:
        config = json.load(f)

    return config


def ensure_dir(path: Path) -> None:
    """Ensure directory exists"""
    path.mkdir(parents=True, exist_ok=True)


def collect_qt_dependencies(config: Dict, package_dir: Path, exe_path: Path) -> Tuple[bool, List[str]]:
    """
    Collect Qt dependencies using windeployqt
    Returns: (success, list of messages)
    """
    messages = []
    qt_path = Path(config['qt_path'].replace('/', '\\'))
    windeployqt = qt_path / "bin" / "windeployqt.exe"

    if not windeployqt.exists():
        messages.append(f"ERROR: windeployqt.exe not found at {windeployqt}")
        return False, messages

    messages.append(f"Running windeployqt on {exe_path.name}...")

    # Create lib/qt directory
    qt_lib_dir = package_dir / "lib" / "qt"
    ensure_dir(qt_lib_dir)

    try:
        # Run windeployqt to deploy Qt dependencies
        # --no-translations: Skip translation files to reduce size
        # --no-system-d3d-compiler: Skip D3D compiler DLLs
        # --no-opengl-sw: Skip software OpenGL
        cmd = [
            str(windeployqt),
            "--release",
            "--no-translations",
            "--no-system-d3d-compiler",
            "--no-opengl-sw",
            "--dir", str(qt_lib_dir),
            str(exe_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            messages.append("✓ Qt dependencies collected successfully")

            # Count collected files
            dll_count = len(list(qt_lib_dir.glob("*.dll")))
            messages.append(f"  - Collected {dll_count} Qt DLLs")

            # Check for platform plugins
            platform_dir = qt_lib_dir / "platforms"
            if platform_dir.exists():
                platform_count = len(list(platform_dir.glob("*.dll")))
                messages.append(f"  - Collected {platform_count} platform plugins")

            return True, messages
        else:
            messages.append(f"ERROR: windeployqt failed: {result.stderr}")
            return False, messages

    except Exception as e:
        messages.append(f"ERROR: Failed to run windeployqt: {e}")
        return False, messages


def collect_openssl_dependencies(config: Dict, package_dir: Path) -> Tuple[bool, List[str]]:
    """
    Collect OpenSSL DLLs
    Returns: (success, list of messages)
    """
    messages = []
    openssl_path = Path(config['openssl_path'].replace('/', '\\'))
    openssl_bin = openssl_path / "bin"

    if not openssl_bin.exists():
        messages.append(f"ERROR: OpenSSL bin directory not found at {openssl_bin}")
        return False, messages

    messages.append("Collecting OpenSSL dependencies...")

    # Create lib/openssl directory
    openssl_lib_dir = package_dir / "lib" / "openssl"
    ensure_dir(openssl_lib_dir)

    # Required OpenSSL DLLs
    required_dlls = [
        "libcrypto-3-x64.dll",
        "libssl-3-x64.dll"
    ]

    copied_count = 0
    for dll_name in required_dlls:
        src = openssl_bin / dll_name
        dst = openssl_lib_dir / dll_name

        if src.exists():
            shutil.copy2(src, dst)
            messages.append(f"  ✓ {dll_name}")
            copied_count += 1
        else:
            messages.append(f"  ✗ {dll_name} not found")

    if copied_count == len(required_dlls):
        messages.append(f"✓ OpenSSL dependencies collected ({copied_count} DLLs)")
        return True, messages
    else:
        messages.append(f"WARNING: Only {copied_count}/{len(required_dlls)} OpenSSL DLLs found")
        return False, messages


def collect_vcruntime_dependencies(config: Dict, package_dir: Path) -> Tuple[bool, List[str]]:
    """
    Collect Visual C++ runtime DLLs
    Returns: (success, list of messages)
    """
    messages = []
    vs_path = Path(config['visual_studio_path'].replace('/', '\\'))

    messages.append("Collecting Visual C++ runtime...")

    # Try to find vcruntime DLLs in Visual Studio redist folders
    # VS2022 structure: VS\2022\Community\VC\Redist\MSVC\<version>\x64\Microsoft.VC143.CRT\
    vc_redist_base = vs_path / "VC" / "Redist" / "MSVC"

    vcruntime_lib_dir = package_dir / "lib" / "vcruntime"
    ensure_dir(vcruntime_lib_dir)

    # Required runtime DLLs
    required_dlls = [
        "vcruntime140.dll",
        "msvcp140.dll",
        "vcruntime140_1.dll"
    ]

    # Try to find the DLLs
    found_dlls = []

    if vc_redist_base.exists():
        # Find the version folder (should be something like 14.xx.xxxxx)
        version_folders = [d for d in vc_redist_base.iterdir() if d.is_dir()]

        for version_folder in sorted(version_folders, reverse=True):
            crt_folder = version_folder / "x64" / "Microsoft.VC143.CRT"

            if not crt_folder.exists():
                # Try VC142 for VS2019
                crt_folder = version_folder / "x64" / "Microsoft.VC142.CRT"

            if crt_folder.exists():
                for dll_name in required_dlls:
                    src = crt_folder / dll_name
                    if src.exists() and dll_name not in found_dlls:
                        dst = vcruntime_lib_dir / dll_name
                        shutil.copy2(src, dst)
                        messages.append(f"  ✓ {dll_name}")
                        found_dlls.append(dll_name)

                if len(found_dlls) == len(required_dlls):
                    break

    # Fallback: Try system32 (not recommended but works for testing)
    if len(found_dlls) < len(required_dlls):
        system32 = Path("C:/Windows/System32")
        for dll_name in required_dlls:
            if dll_name not in found_dlls:
                src = system32 / dll_name
                if src.exists():
                    dst = vcruntime_lib_dir / dll_name
                    shutil.copy2(src, dst)
                    messages.append(f"  ✓ {dll_name} (from System32)")
                    found_dlls.append(dll_name)

    if len(found_dlls) == len(required_dlls):
        messages.append(f"✓ Visual C++ runtime collected ({len(found_dlls)} DLLs)")
        return True, messages
    else:
        messages.append(f"WARNING: Only {len(found_dlls)}/{len(required_dlls)} runtime DLLs found")
        messages.append("  Missing DLLs may already be on target systems")
        return True, messages  # Not critical, return success anyway


def collect_python_dependencies(config: Dict, package_dir: Path) -> Tuple[bool, List[str]]:
    """
    Collect Python runtime DLLs (core Python only, not site-packages)
    Returns: (success, list of messages)
    """
    messages = []

    if not config.get('enable_python', False):
        messages.append("Python support disabled, skipping Python DLL collection")
        return True, messages

    python_path = Path(config['python_path'].replace('/', '\\'))

    if not python_path.exists():
        messages.append(f"ERROR: Python path not found at {python_path}")
        return False, messages

    messages.append("Collecting Python runtime DLLs...")

    # Create lib/python directory
    python_lib_dir = package_dir / "lib" / "python"
    ensure_dir(python_lib_dir)

    # Detect Python version (e.g., python312.dll)
    python_dlls = list(python_path.glob("python3*.dll"))

    copied_count = 0
    for dll_path in python_dlls:
        dst = python_lib_dir / dll_path.name
        shutil.copy2(dll_path, dst)
        messages.append(f"  ✓ {dll_path.name}")
        copied_count += 1

    if copied_count > 0:
        messages.append(f"✓ Python runtime collected ({copied_count} DLLs)")
        return True, messages
    else:
        messages.append("ERROR: No Python DLLs found")
        return False, messages


def main():
    print("=" * 70)
    print("NDA Dependency Collection")
    print("=" * 70)
    print()

    # Load configuration
    try:
        config = load_config()
        print("✓ Configuration loaded from build_config.json")
        print()
    except Exception as e:
        print(f"ERROR: Failed to load configuration: {e}")
        return 1

    # Determine paths
    script_dir = Path(__file__).parent  # scripts/windows-packaging/
    base_dir = script_dir.parent.parent  # repo root (up from windows-packaging/ -> scripts/ -> root)
    package_dir = base_dir / "package"
    exe_path = package_dir / "NDA.exe"

    if not exe_path.exists():
        print(f"ERROR: NDA.exe not found at {exe_path}")
        print("Run the build script first to create NDA.exe")
        return 1

    print(f"Package directory: {package_dir}")
    print(f"Executable: {exe_path}")
    print()

    # Collect all dependencies
    all_success = True
    all_messages = []

    # 1. Qt Dependencies
    print("=" * 70)
    success, messages = collect_qt_dependencies(config, package_dir, exe_path)
    all_success &= success
    all_messages.extend(messages)
    for msg in messages:
        print(msg)
    print()

    # 2. OpenSSL Dependencies
    print("=" * 70)
    success, messages = collect_openssl_dependencies(config, package_dir)
    all_success &= success
    all_messages.extend(messages)
    for msg in messages:
        print(msg)
    print()

    # 3. Visual C++ Runtime
    print("=" * 70)
    success, messages = collect_vcruntime_dependencies(config, package_dir)
    all_success &= success
    all_messages.extend(messages)
    for msg in messages:
        print(msg)
    print()

    # 4. Python Runtime (if enabled)
    if config.get('enable_python', False):
        print("=" * 70)
        success, messages = collect_python_dependencies(config, package_dir)
        all_success &= success
        all_messages.extend(messages)
        for msg in messages:
            print(msg)
        print()

    # Summary
    print("=" * 70)
    if all_success:
        print("✓ Dependency collection completed successfully!")
        print()
        print("All required dependencies have been collected to:")
        print(f"  {package_dir / 'lib'}")
        return 0
    else:
        print("⚠ Dependency collection completed with warnings/errors")
        print()
        print("Some dependencies may be missing. Check messages above.")
        print("The package may still work if dependencies are on target system.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
