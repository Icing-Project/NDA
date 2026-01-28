#!/usr/bin/env python3
"""
NDA Standalone Deployment Script
Creates a complete standalone Windows package with all dependencies
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

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
        print("Run: scripts\\windows-packaging\\setup_build_config.bat")
        sys.exit(1)

    with open(config_file, 'r') as f:
        return json.load(f)


def create_package_structure(package_dir: Path):
    """Create the package directory structure"""
    print("Creating package structure...")

    # Clean existing package directory
    if package_dir.exists():
        print(f"  Removing existing package: {package_dir}")
        shutil.rmtree(package_dir)

    # Create directory structure
    dirs = [
        package_dir,
        package_dir / "lib" / "qt",
        package_dir / "lib" / "openssl",
        package_dir / "lib" / "vcruntime",
        package_dir / "lib" / "python" / "site-packages",
        package_dir / "plugins",
        package_dir / "docs",
    ]

    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory.relative_to(package_dir.parent)}")

    print()


def copy_executable(build_dir: Path, package_dir: Path, build_config: str):
    """Copy NDA.exe to package"""
    print("Copying NDA.exe...")

    # Try Release/Debug subdirectories first (Visual Studio)
    exe_locations = [
        build_dir / build_config / "NDA.exe",
        build_dir / "NDA.exe",
    ]

    for exe_path in exe_locations:
        if exe_path.exists():
            dst = package_dir / "NDA.exe"
            shutil.copy2(exe_path, dst)
            print(f"  ✓ NDA.exe ({exe_path.stat().st_size // 1024} KB)")
            print()
            return True

    print("  ✗ NDA.exe not found!")
    print(f"    Checked: {exe_locations}")
    print()
    return False


def copy_plugins(base_dir: Path, build_dir: Path, package_dir: Path, build_config: str, enable_python: bool):
    """Copy all plugins (C++ DLLs and Python .py files)"""
    print("Copying plugins...")

    plugins_dst = package_dir / "plugins"
    copied_cpp = 0
    copied_py = 0

    # Copy C++ plugin DLLs
    cpp_plugin_locations = [
        build_dir / "plugins",
        build_dir / build_config / "plugins",
    ]

    for plugin_dir in cpp_plugin_locations:
        if plugin_dir.exists():
            for dll in plugin_dir.glob("*.dll"):
                shutil.copy2(dll, plugins_dst / dll.name)
                print(f"  ✓ {dll.name} (C++)")
                copied_cpp += 1
            break

    # Copy Python plugins (if enabled)
    if enable_python:
        plugins_py_dir = base_dir / "plugins_py"

        if plugins_py_dir.exists():
            # Copy all .py files except examples
            for py_file in plugins_py_dir.glob("*.py"):
                # Skip __pycache__ and test files
                if py_file.name.startswith("test_"):
                    continue

                shutil.copy2(py_file, plugins_dst / py_file.name)
                print(f"  ✓ {py_file.name} (Python)")
                copied_py += 1

    print()
    print(f"  Total: {copied_cpp} C++ plugins, {copied_py} Python plugins")
    print()

    return copied_cpp > 0 or copied_py > 0


def copy_documentation(base_dir: Path, package_dir: Path):
    """Copy documentation files"""
    print("Copying documentation...")

    docs_dst = package_dir / "docs"

    # Core documentation files
    doc_files = [
        ("README.md", "README.md"),
        ("LICENSE", "LICENSE.txt"),
    ]

    copied = 0
    for src_name, dst_name in doc_files:
        src_file = base_dir / src_name
        if src_file.exists():
            shutil.copy2(src_file, docs_dst / dst_name)
            print(f"  ✓ {dst_name}")
            copied += 1

    # Copy key documentation from docs/
    docs_dir = base_dir / "docs" / "guides"
    if docs_dir.exists():
        guide_files = ["installation.md", "troubleshooting.md"]
        for guide in guide_files:
            src_file = docs_dir / guide
            if src_file.exists():
                shutil.copy2(src_file, docs_dst / guide)
                print(f"  ✓ {guide}")
                copied += 1

    print()
    return copied > 0


def create_launcher_script(package_dir: Path, enable_python: bool):
    """Create NDA.bat launcher script"""
    print("Creating launcher script...")

    launcher_path = package_dir / "NDA.bat"

    with open(launcher_path, 'w') as f:
        f.write('@echo off\n')
        f.write('REM NDA Standalone Launcher\n')
        f.write('REM Auto-generated by deploy_standalone.py\n')
        f.write('\n')
        f.write('REM Change to package directory\n')
        f.write('cd /d "%~dp0"\n')
        f.write('\n')
        f.write('REM Set up environment\n')
        f.write('set PATH=%~dp0lib\\qt;%~dp0lib\\openssl;%~dp0lib\\vcruntime;%PATH%\n')
        f.write('\n')

        if enable_python:
            f.write('REM Python environment\n')
            f.write('set PYTHONHOME=%~dp0lib\\python\n')
            f.write('set PYTHONPATH=%~dp0lib\\python\\site-packages;%~dp0plugins\n')
            f.write('set PATH=%~dp0lib\\python;%PATH%\n')
            f.write('\n')

        f.write('REM Launch NDA\n')
        f.write('"%~dp0NDA.exe"\n')
        f.write('\n')
        f.write('REM Pause on error to see any messages\n')
        f.write('if errorlevel 1 pause\n')

    print(f"  ✓ NDA.bat")
    print()


def create_qt_conf(package_dir: Path):
    """Create qt.conf to tell Qt where to find plugins and libraries"""
    print("Creating qt.conf...")

    qt_conf_path = package_dir / "qt.conf"

    with open(qt_conf_path, 'w') as f:
        f.write('[Paths]\n')
        f.write('Prefix = .\n')
        f.write('Binaries = lib/qt\n')
        f.write('Libraries = lib/qt\n')
        f.write('Plugins = lib/qt\n')
        f.write('Translations = lib/qt/translations\n')

    print(f"  ✓ qt.conf")
    print()


def create_readme_txt(package_dir: Path, config: dict):
    """Create README.txt for the package"""
    print("Creating README.txt...")

    readme_path = package_dir / "README.txt"
    version = config.get('package_version', '2.0.0')

    with open(readme_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"NDA - Nade Desktop Application v{version}\n")
        f.write("Windows Standalone Portable Package\n")
        f.write("=" * 70 + "\n")
        f.write("\n")
        f.write("WHAT IS NDA?\n")
        f.write("-" * 70 + "\n")
        f.write("NDA is a real-time audio encryption bridge for secure communication.\n")
        f.write("It processes audio through a plugin-based pipeline architecture.\n")
        f.write("\n")
        f.write("KEY FEATURES:\n")
        f.write("  - Dual independent TX/RX pipelines\n")
        f.write("  - Plugin-based encryption\n")
        f.write("  - Python & C++ plugin support\n")
        f.write("  - Low latency (<50ms)\n")
        f.write("  - Automatic sample rate adaptation\n")
        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write("QUICK START\n")
        f.write("=" * 70 + "\n")
        f.write("\n")
        f.write("1. EXTRACT THIS PACKAGE\n")
        f.write("   Extract the entire ZIP to any location on your computer.\n")
        f.write("   No installation required!\n")
        f.write("\n")
        f.write("2. LAUNCH NDA\n")
        f.write("   Double-click: NDA.bat\n")
        f.write("\n")
        f.write("   Alternative: Run NDA.exe directly from this folder\n")
        f.write("\n")
        f.write("3. LOAD PLUGINS\n")
        f.write("   - C++ plugins load automatically\n")
        f.write("   - Python plugins: Click \"Auto-Load Python Plugins\" in the UI\n")
        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write("SYSTEM REQUIREMENTS\n")
        f.write("=" * 70 + "\n")
        f.write("\n")
        f.write("  - Windows 10 or Windows 11 (64-bit)\n")
        f.write("  - 4 GB RAM minimum\n")
        f.write("  - Audio input/output device\n")
        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write("PACKAGE CONTENTS\n")
        f.write("=" * 70 + "\n")
        f.write("\n")
        f.write("  NDA.exe              - Main application\n")
        f.write("  NDA.bat              - Launcher script (recommended)\n")
        f.write("  README.txt           - This file\n")
        f.write("  lib/                 - All dependencies (Qt, OpenSSL, Python, etc.)\n")
        f.write("  plugins/             - Audio processing plugins\n")
        f.write("  docs/                - Documentation\n")
        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write("TROUBLESHOOTING\n")
        f.write("=" * 70 + "\n")
        f.write("\n")
        f.write("Q: Application won't start\n")
        f.write("A: - Make sure you extracted the ENTIRE ZIP file\n")
        f.write("   - Run NDA.bat instead of NDA.exe directly\n")
        f.write("   - Check Windows Defender / antivirus hasn't blocked files\n")
        f.write("\n")
        f.write("Q: Python plugins don't load\n")
        f.write("A: - Click \"Auto-Load Python Plugins\" button in the UI\n")
        f.write("   - Check that lib/python/ folder exists and contains DLLs\n")
        f.write("\n")
        f.write("Q: Missing VCRUNTIME140.dll error\n")
        f.write("A: - Make sure lib/vcruntime/ folder is present\n")
        f.write("   - Run NDA.bat which sets up the correct PATH\n")
        f.write("   - Download Visual C++ Redistributable from Microsoft if needed\n")
        f.write("\n")
        f.write("Q: No audio devices shown\n")
        f.write("A: - Check that audio devices are working in Windows\n")
        f.write("   - Try restarting NDA\n")
        f.write("   - Load plugins using the buttons in the UI\n")
        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write("MORE INFORMATION\n")
        f.write("=" * 70 + "\n")
        f.write("\n")
        f.write("Full documentation: docs/\n")
        f.write("\n")
        f.write("Key documents:\n")
        f.write("  - docs/installation.md     - Detailed setup guide\n")
        f.write("  - docs/troubleshooting.md  - Common issues and solutions\n")
        f.write("  - docs/README.md           - User guide\n")
        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write(f"Package created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n")

    print(f"  ✓ README.txt")
    print()


def run_script(script_path: Path, script_name: str):
    """Run a Python script and return success status"""
    print("=" * 70)
    print(f"Running {script_name}...")
    print("=" * 70)
    print()

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=script_path.parent.parent
        )
        print()
        return result.returncode == 0
    except Exception as e:
        print(f"ERROR: Failed to run {script_name}: {e}")
        print()
        return False


def main():
    print("=" * 70)
    print("NDA Standalone Deployment")
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
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent.parent
    build_dir = base_dir / "build"
    package_name = config.get('package_name', 'NDA-Windows-Portable')
    version = config.get('package_version', '2.0.0')
    package_dir = base_dir / "package"
    enable_python = config.get('enable_python', False)
    build_config = config.get('build_config', 'Release')

    print(f"Package name: {package_name}")
    print(f"Version: {version}")
    print(f"Python support: {'Enabled' if enable_python else 'Disabled'}")
    print(f"Build configuration: {build_config}")
    print()

    # Phase 1: Create package structure
    print("=" * 70)
    print("PHASE 1: Package Structure")
    print("=" * 70)
    print()
    create_package_structure(package_dir)

    # Phase 2: Copy executable
    print("=" * 70)
    print("PHASE 2: Copy Executable")
    print("=" * 70)
    print()
    if not copy_executable(build_dir, package_dir, build_config):
        print("ERROR: Failed to copy NDA.exe")
        return 1

    # Phase 3: Copy plugins
    print("=" * 70)
    print("PHASE 3: Copy Plugins")
    print("=" * 70)
    print()
    if not copy_plugins(base_dir, build_dir, package_dir, build_config, enable_python):
        print("WARNING: No plugins copied")

    # Phase 4: Collect dependencies
    if not run_script(script_dir / "collect_dependencies.py", "collect_dependencies.py"):
        print("ERROR: Dependency collection failed")
        return 1

    # Phase 5: Bundle Python (if enabled)
    if enable_python:
        if not run_script(script_dir / "bundle_python.py", "bundle_python.py"):
            print("ERROR: Python bundling failed")
            return 1

    # Phase 6: Copy documentation
    print("=" * 70)
    print("PHASE: Documentation")
    print("=" * 70)
    print()
    copy_documentation(base_dir, package_dir)

    # Phase 7: Create launcher script and Qt configuration
    print("=" * 70)
    print("PHASE: Launcher Script & Qt Configuration")
    print("=" * 70)
    print()
    create_launcher_script(package_dir, enable_python)
    create_qt_conf(package_dir)

    # Phase 8: Create README.txt
    print("=" * 70)
    print("PHASE: Package README")
    print("=" * 70)
    print()
    create_readme_txt(package_dir, config)

    # Summary
    print("=" * 70)
    print("DEPLOYMENT COMPLETE!")
    print("=" * 70)
    print()
    print(f"Package created at: {package_dir}")
    print()

    # Calculate package size
    total_size = sum(f.stat().st_size for f in package_dir.rglob('*') if f.is_file())
    total_size_mb = total_size / (1024 * 1024)
    print(f"Package size: {total_size_mb:.1f} MB")
    print()

    # Count files
    file_count = len(list(package_dir.rglob('*')))
    print(f"Total files: {file_count}")
    print()

    print("Next steps:")
    print("  1. Run verification: python scripts/windows-packaging/verify_package.py")
    print(f"  2. Test package: cd package && NDA.bat")
    print(f"  3. Create ZIP: Compress 'package' folder as '{package_name}-v{version}.zip'")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
