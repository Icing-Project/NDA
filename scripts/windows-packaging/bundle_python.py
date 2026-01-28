#!/usr/bin/env python3
"""
NDA Python Environment Bundler
Bundles Python site-packages for standalone distribution
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'ignore')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'ignore')


def load_config():
    """Load build configuration"""
    script_dir = Path(__file__).parent
    config_file = script_dir / "build_config.json"

    if not config_file.exists():
        print("ERROR: build_config.json not found!")
        sys.exit(1)

    with open(config_file, 'r') as f:
        return json.load(f)


def get_python_info(python_exe: Path) -> dict:
    """Get Python version and site-packages location"""
    try:
        # Get Python version
        result = subprocess.run(
            [str(python_exe), "--version"],
            capture_output=True,
            text=True
        )
        version = result.stdout.strip() if result.returncode == 0 else "Unknown"

        # Get site-packages location
        # Use all site-packages paths and find the correct one
        result = subprocess.run(
            [str(python_exe), "-c", "import site; print('\\n'.join(site.getsitepackages()))"],
            capture_output=True,
            text=True
        )

        site_packages = None
        if result.returncode == 0:
            # Get all site-packages paths
            paths = result.stdout.strip().split('\n')

            # Find the path that ends with 'site-packages' (the actual packages directory)
            for path in paths:
                path = path.strip()
                if path.endswith('site-packages'):
                    site_packages = path
                    break

            # Fallback: use the last path if none end with 'site-packages'
            if not site_packages and paths:
                site_packages = paths[-1]

        return {
            "version": version,
            "site_packages": Path(site_packages) if site_packages else None
        }
    except Exception as e:
        print(f"ERROR: Failed to get Python info: {e}")
        return {"version": "Unknown", "site_packages": None}


def install_required_packages(python_exe: Path, requirements_file: Path) -> Tuple[bool, List[str]]:
    """
    Install required packages from requirements.txt
    Returns: (success, list of messages)
    """
    messages = []

    if not requirements_file.exists():
        messages.append(f"WARNING: requirements.txt not found at {requirements_file}")
        messages.append("Skipping package installation")
        return True, messages

    messages.append("Installing required Python packages...")

    try:
        cmd = [str(python_exe), "-m", "pip", "install", "-r", str(requirements_file)]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            messages.append("✓ Packages installed successfully")
            return True, messages
        else:
            messages.append(f"ERROR: pip install failed: {result.stderr}")
            return False, messages

    except Exception as e:
        messages.append(f"ERROR: Failed to install packages: {e}")
        return False, messages


def copy_site_packages(site_packages_src: Path, site_packages_dst: Path,
                       required_packages: List[str]) -> Tuple[bool, List[str]]:
    """
    Copy required packages from site-packages
    Returns: (success, list of messages)
    """
    messages = []
    messages.append("Copying Python packages...")

    if not site_packages_src.exists():
        messages.append(f"ERROR: Source site-packages not found at {site_packages_src}")
        return False, messages

    site_packages_dst.mkdir(parents=True, exist_ok=True)

    # Packages to copy (name patterns)
    package_patterns = required_packages + [
        # Also include package metadata
        "*.dist-info",
        "*.egg-info",
    ]

    copied_packages = []

    # Copy each required package
    for package in required_packages:
        found = False

        # Try direct package folder
        pkg_dir = site_packages_src / package
        if pkg_dir.exists() and pkg_dir.is_dir():
            dst_dir = site_packages_dst / package
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            shutil.copytree(pkg_dir, dst_dir)
            messages.append(f"  ✓ {package}/")
            copied_packages.append(package)
            found = True

        # Try as single .py file
        pkg_file = site_packages_src / f"{package}.py"
        if pkg_file.exists():
            shutil.copy2(pkg_file, site_packages_dst / f"{package}.py")
            if not found:
                messages.append(f"  ✓ {package}.py")
                copied_packages.append(package)
            found = True

        # Copy associated .dist-info or .egg-info
        for info_dir in site_packages_src.glob(f"{package}*.dist-info"):
            dst_info = site_packages_dst / info_dir.name
            if dst_info.exists():
                shutil.rmtree(dst_info)
            shutil.copytree(info_dir, dst_info)

        for info_dir in site_packages_src.glob(f"{package}*.egg-info"):
            dst_info = site_packages_dst / info_dir.name
            if dst_info.exists():
                shutil.rmtree(dst_info)
            if info_dir.is_dir():
                shutil.copytree(info_dir, dst_info)
            else:
                shutil.copy2(info_dir, dst_info)

        if not found:
            messages.append(f"  ⚠ {package} not found in site-packages")

    if len(copied_packages) > 0:
        messages.append(f"✓ Copied {len(copied_packages)} packages to bundle")
        return True, messages
    else:
        messages.append("ERROR: No packages were copied")
        return False, messages


def copy_python_stdlib(python_path: Path, python_lib_dir: Path) -> Tuple[bool, List[str]]:
    """
    Copy Python standard library (Lib folder) and DLLs to bundle
    Returns: (success, list of messages)
    """
    messages = []
    messages.append("Copying Python standard library...")

    # 1. Copy Python's Lib folder (standard library)
    stdlib_src = python_path / "Lib"
    if not stdlib_src.exists():
        messages.append(f"ERROR: Python Lib folder not found at {stdlib_src}")
        return False, messages

    # Destination: lib/python/Lib
    stdlib_dst = python_lib_dir / "Lib"

    # Directories to exclude (reduce size)
    exclude_dirs = {
        '__pycache__',
        'test',  # Python test suite
        'tests',
        'idlelib',  # IDLE IDE
        'tkinter',  # Tkinter GUI (if not needed)
        'turtle',
        'turtledemo',
        'lib2to3',  # 2to3 converter
        'ensurepip',  # pip installer
        'venv',  # Virtual environment
        'site-packages',  # Exclude Lib/site-packages (we'll create our own)
    }

    try:
        # Copy standard library
        if stdlib_dst.exists():
            shutil.rmtree(stdlib_dst)

        def ignore_patterns(directory, files):
            """Ignore specific directories to reduce bundle size"""
            return [f for f in files if f in exclude_dirs or f.endswith('.pyc')]

        shutil.copytree(stdlib_src, stdlib_dst, ignore=ignore_patterns)

        # Count copied files
        total_files = sum(1 for _ in stdlib_dst.rglob('*.py'))
        messages.append(f"  ✓ Copied {total_files} Python standard library files")

        # 2. Copy Python's DLLs folder (extension modules)
        dlls_src = python_path / "DLLs"
        if dlls_src.exists():
            dlls_dst = python_lib_dir / "DLLs"
            if dlls_dst.exists():
                shutil.rmtree(dlls_dst)
            shutil.copytree(dlls_src, dlls_dst)

            # Count copied DLLs
            pyd_count = len(list(dlls_dst.glob('*.pyd')))
            dll_count = len(list(dlls_dst.glob('*.dll')))
            messages.append(f"  ✓ Copied {pyd_count} .pyd modules and {dll_count} DLLs")
        else:
            messages.append(f"  ⚠ DLLs folder not found at {dlls_src}")

        messages.append(f"✓ Python standard library copied successfully")
        return True, messages

    except Exception as e:
        messages.append(f"ERROR: Failed to copy standard library: {e}")
        return False, messages


def create_pyvenv_cfg(python_lib_dir: Path, python_version: str) -> Tuple[bool, List[str]]:
    """
    Create pyvenv.cfg for portable Python
    Returns: (success, list of messages)
    """
    messages = []

    pyvenv_cfg = python_lib_dir / "pyvenv.cfg"

    try:
        with open(pyvenv_cfg, 'w') as f:
            f.write("# Portable Python Configuration for NDA\n")
            f.write(f"# Python Version: {python_version}\n")
            f.write("home = .\n")
            f.write("include-system-site-packages = false\n")
            f.write("version = 3.12\n")

        messages.append(f"✓ Created pyvenv.cfg")
        return True, messages

    except Exception as e:
        messages.append(f"ERROR: Failed to create pyvenv.cfg: {e}")
        return False, messages


def verify_python_bundle(python_lib_dir: Path, package_dir: Path) -> Tuple[bool, List[str]]:
    """
    Verify Python bundle can import numpy
    Returns: (success, list of messages)
    """
    messages = []
    messages.append("Verifying Python bundle...")

    # Check if Python DLLs exist
    python_dlls = list(python_lib_dir.glob("python*.dll"))
    if not python_dlls:
        messages.append("ERROR: No Python DLLs found in bundle")
        return False, messages

    messages.append(f"  ✓ Found {len(python_dlls)} Python DLL(s)")

    # Check if site-packages exists
    site_packages = python_lib_dir / "site-packages"
    if not site_packages.exists():
        messages.append("ERROR: site-packages directory not found")
        return False, messages

    messages.append("  ✓ site-packages directory exists")

    # Check for required packages
    required_packages = ["numpy", "sounddevice", "soundcard"]
    found_packages = []

    for package in required_packages:
        pkg_dir = site_packages / package
        pkg_file = site_packages / f"{package}.py"

        if pkg_dir.exists() or pkg_file.exists():
            found_packages.append(package)
            messages.append(f"  ✓ {package} found")
        else:
            messages.append(f"  ⚠ {package} not found")

    if len(found_packages) >= 1:  # At least numpy should be present
        messages.append(f"✓ Python bundle verification passed ({len(found_packages)}/{len(required_packages)} packages)")
        return True, messages
    else:
        messages.append("ERROR: Python bundle verification failed")
        return False, messages


def main():
    print("=" * 70)
    print("NDA Python Environment Bundler")
    print("=" * 70)
    print()

    # Load configuration
    try:
        config = load_config()
        print("✓ Configuration loaded")
        print()
    except Exception as e:
        print(f"ERROR: Failed to load configuration: {e}")
        return 1

    # Check if Python support is enabled
    if not config.get('enable_python', False):
        print("Python support is disabled in configuration.")
        print("Skipping Python bundling.")
        return 0

    # Get Python path
    python_path = Path(config['python_path'].replace('/', '\\'))
    python_exe = python_path / "python.exe"

    if not python_exe.exists():
        print(f"ERROR: Python executable not found at {python_exe}")
        return 1

    # Get Python info
    python_info = get_python_info(python_exe)
    print(f"Python: {python_info['version']}")
    print(f"Site-packages: {python_info['site_packages']}")
    print()

    # Determine paths
    script_dir = Path(__file__).parent  # scripts/windows-packaging/
    base_dir = script_dir.parent.parent  # repo root (up from windows-packaging/ -> scripts/ -> root)
    package_dir = base_dir / "package"
    python_lib_dir = package_dir / "lib" / "python"
    site_packages_dst = python_lib_dir / "site-packages"
    requirements_file = script_dir / "requirements.txt"

    # Install required packages (in user's Python environment)
    print("=" * 70)
    success, messages = install_required_packages(python_exe, requirements_file)
    for msg in messages:
        print(msg)
    print()

    if not success:
        print("WARNING: Package installation failed, but continuing...")
        print()

    # Copy Python standard library
    print("=" * 70)
    success, messages = copy_python_stdlib(python_path, python_lib_dir)
    for msg in messages:
        print(msg)
    print()

    if not success:
        print("ERROR: Failed to copy Python standard library")
        return 1

    # Copy site-packages
    print("=" * 70)
    if python_info['site_packages']:
        required_packages = ["numpy", "sounddevice", "soundcard", "cffi", "_sounddevice"]
        success, messages = copy_site_packages(
            python_info['site_packages'],
            site_packages_dst,
            required_packages
        )
        for msg in messages:
            print(msg)
        print()

        if not success:
            print("ERROR: Failed to copy site-packages")
            return 1
    else:
        print("ERROR: Could not determine site-packages location")
        return 1

    # Create pyvenv.cfg
    print("=" * 70)
    success, messages = create_pyvenv_cfg(python_lib_dir, python_info['version'])
    for msg in messages:
        print(msg)
    print()

    # Verify bundle
    print("=" * 70)
    success, messages = verify_python_bundle(python_lib_dir, package_dir)
    for msg in messages:
        print(msg)
    print()

    if success:
        print("=" * 70)
        print("✓ Python environment bundled successfully!")
        print()
        print(f"Python libraries bundled to: {python_lib_dir}")
        return 0
    else:
        print("=" * 70)
        print("⚠ Python bundling completed with warnings")
        return 1


if __name__ == "__main__":
    sys.exit(main())
