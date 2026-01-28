#!/usr/bin/env python3
"""
NDA Package Verification Script
Verifies that the standalone package contains all required files
"""

import sys
import json
from pathlib import Path
from typing import List, Tuple


def load_config():
    """Load build configuration"""
    script_dir = Path(__file__).parent
    config_file = script_dir / "build_config.json"

    if config_file.exists():
        with open(config_file, 'r') as f:
            return json.load(f)
    else:
        # Return default config if file doesn't exist
        return {"enable_python": False}


def check_file_exists(file_path: Path, description: str) -> Tuple[bool, str]:
    """Check if a file exists and return result"""
    if file_path.exists():
        size_kb = file_path.stat().st_size // 1024
        return True, f"✓ {description} ({size_kb} KB)"
    else:
        return False, f"✗ {description} - NOT FOUND"


def check_directory_exists(dir_path: Path, description: str) -> Tuple[bool, str]:
    """Check if a directory exists and count contents"""
    if dir_path.exists() and dir_path.is_dir():
        file_count = len(list(dir_path.rglob('*')))
        return True, f"✓ {description} ({file_count} items)"
    else:
        return False, f"✗ {description} - NOT FOUND"


def verify_core_files(package_dir: Path) -> Tuple[int, int, List[str]]:
    """Verify core application files"""
    checks = []
    passed = 0
    total = 0

    # Core executable
    total += 1
    success, msg = check_file_exists(package_dir / "NDA.exe", "NDA.exe")
    checks.append(msg)
    if success:
        passed += 1

    # Launcher script
    total += 1
    success, msg = check_file_exists(package_dir / "NDA.bat", "NDA.bat")
    checks.append(msg)
    if success:
        passed += 1

    # README
    total += 1
    success, msg = check_file_exists(package_dir / "README.txt", "README.txt")
    checks.append(msg)
    if success:
        passed += 1

    return passed, total, checks


def verify_qt_dependencies(package_dir: Path) -> Tuple[int, int, List[str]]:
    """Verify Qt dependencies"""
    checks = []
    passed = 0
    total = 0

    qt_dir = package_dir / "lib" / "qt"

    # Qt directory
    total += 1
    success, msg = check_directory_exists(qt_dir, "Qt library directory")
    checks.append(msg)
    if success:
        passed += 1

        # Check for essential Qt DLLs
        qt_dlls = [
            "Qt6Core.dll",
            "Qt6Gui.dll",
            "Qt6Widgets.dll",
            "Qt6Network.dll",
        ]

        for dll in qt_dlls:
            total += 1
            success, msg = check_file_exists(qt_dir / dll, f"Qt: {dll}")
            checks.append(f"  {msg}")
            if success:
                passed += 1

        # Check for platform plugins
        total += 1
        platform_dir = qt_dir / "platforms"
        success, msg = check_directory_exists(platform_dir, "Qt platform plugins")
        checks.append(f"  {msg}")
        if success:
            passed += 1

    return passed, total, checks


def verify_openssl_dependencies(package_dir: Path) -> Tuple[int, int, List[str]]:
    """Verify OpenSSL dependencies"""
    checks = []
    passed = 0
    total = 0

    openssl_dir = package_dir / "lib" / "openssl"

    # OpenSSL directory
    total += 1
    success, msg = check_directory_exists(openssl_dir, "OpenSSL directory")
    checks.append(msg)
    if success:
        passed += 1

        # Check for OpenSSL DLLs
        openssl_dlls = [
            "libcrypto-3-x64.dll",
            "libssl-3-x64.dll",
        ]

        for dll in openssl_dlls:
            total += 1
            success, msg = check_file_exists(openssl_dir / dll, f"OpenSSL: {dll}")
            checks.append(f"  {msg}")
            if success:
                passed += 1

    return passed, total, checks


def verify_vcruntime_dependencies(package_dir: Path) -> Tuple[int, int, List[str]]:
    """Verify Visual C++ runtime dependencies"""
    checks = []
    passed = 0
    total = 0

    vcruntime_dir = package_dir / "lib" / "vcruntime"

    # VCRUNTIME directory
    total += 1
    success, msg = check_directory_exists(vcruntime_dir, "VCRUNTIME directory")
    checks.append(msg)
    if success:
        passed += 1

        # Check for runtime DLLs
        runtime_dlls = [
            "vcruntime140.dll",
            "msvcp140.dll",
            "vcruntime140_1.dll",
        ]

        for dll in runtime_dlls:
            total += 1
            success, msg = check_file_exists(vcruntime_dir / dll, f"VCRUNTIME: {dll}")
            checks.append(f"  {msg}")
            if success:
                passed += 1

    return passed, total, checks


def verify_python_dependencies(package_dir: Path, enable_python: bool) -> Tuple[int, int, List[str]]:
    """Verify Python dependencies"""
    checks = []
    passed = 0
    total = 0

    if not enable_python:
        checks.append("⊘ Python support disabled - skipping checks")
        return 0, 0, checks

    python_dir = package_dir / "lib" / "python"
    site_packages = python_dir / "site-packages"

    # Python directory
    total += 1
    success, msg = check_directory_exists(python_dir, "Python directory")
    checks.append(msg)
    if success:
        passed += 1

        # Check for Python DLLs
        python_dlls = list(python_dir.glob("python*.dll"))
        total += 1
        if python_dlls:
            checks.append(f"  ✓ Found {len(python_dlls)} Python DLL(s)")
            passed += 1
        else:
            checks.append("  ✗ No Python DLLs found")

        # Check site-packages
        total += 1
        success, msg = check_directory_exists(site_packages, "site-packages")
        checks.append(f"  {msg}")
        if success:
            passed += 1

            # Check for required packages
            required_packages = ["numpy", "sounddevice", "soundcard"]
            for package in required_packages:
                total += 1
                pkg_dir = site_packages / package
                pkg_file = site_packages / f"{package}.py"
                if pkg_dir.exists() or pkg_file.exists():
                    checks.append(f"    ✓ {package}")
                    passed += 1
                else:
                    checks.append(f"    ⚠ {package} (optional)")

    return passed, total, checks


def verify_plugins(package_dir: Path) -> Tuple[int, int, List[str]]:
    """Verify plugins"""
    checks = []
    passed = 0
    total = 0

    plugins_dir = package_dir / "plugins"

    # Plugins directory
    total += 1
    success, msg = check_directory_exists(plugins_dir, "Plugins directory")
    checks.append(msg)
    if success:
        passed += 1

        # Count C++ plugins
        cpp_plugins = list(plugins_dir.glob("*.dll"))
        total += 1
        if cpp_plugins:
            checks.append(f"  ✓ C++ plugins: {len(cpp_plugins)}")
            passed += 1
            for plugin in sorted(cpp_plugins):
                checks.append(f"    - {plugin.name}")
        else:
            checks.append("  ⚠ No C++ plugins found")

        # Count Python plugins
        py_plugins = list(plugins_dir.glob("*.py"))
        # Filter out non-plugin files
        py_plugins = [p for p in py_plugins if not p.name.startswith('__') and not p.name.startswith('test_')]

        total += 1
        if py_plugins:
            checks.append(f"  ✓ Python plugins: {len(py_plugins)}")
            passed += 1
            for plugin in sorted(py_plugins):
                checks.append(f"    - {plugin.name}")
        else:
            checks.append("  ⊘ No Python plugins (may be disabled)")

    return passed, total, checks


def verify_documentation(package_dir: Path) -> Tuple[int, int, List[str]]:
    """Verify documentation"""
    checks = []
    passed = 0
    total = 0

    docs_dir = package_dir / "docs"

    # Docs directory
    total += 1
    success, msg = check_directory_exists(docs_dir, "Documentation directory")
    checks.append(msg)
    if success:
        passed += 1

        # Check for key documentation files
        doc_files = [
            "README.md",
            "LICENSE.txt",
        ]

        for doc in doc_files:
            total += 1
            success, msg = check_file_exists(docs_dir / doc, f"Doc: {doc}")
            checks.append(f"  {msg}")
            if success:
                passed += 1

    return passed, total, checks


def calculate_package_size(package_dir: Path) -> Tuple[float, int]:
    """Calculate total package size and file count"""
    total_size = 0
    file_count = 0

    for file_path in package_dir.rglob('*'):
        if file_path.is_file():
            total_size += file_path.stat().st_size
            file_count += 1

    size_mb = total_size / (1024 * 1024)
    return size_mb, file_count


def main():
    print("=" * 70)
    print("NDA Package Verification")
    print("=" * 70)
    print()

    # Load configuration
    config = load_config()
    enable_python = config.get('enable_python', False)

    # Determine package directory
    script_dir = Path(__file__).parent
    package_dir = script_dir.parent.parent / "package"

    if not package_dir.exists():
        print(f"ERROR: Package directory not found: {package_dir}")
        print()
        print("Run deploy_standalone.py first to create the package.")
        return 1

    print(f"Package directory: {package_dir}")
    print(f"Python support: {'Enabled' if enable_python else 'Disabled'}")
    print()

    # Run all verification checks
    total_passed = 0
    total_checks = 0
    all_checks = []

    # Core files
    print("=" * 70)
    print("CORE FILES")
    print("=" * 70)
    passed, total, checks = verify_core_files(package_dir)
    total_passed += passed
    total_checks += total
    all_checks.extend(checks)
    for check in checks:
        print(check)
    print()

    # Qt dependencies
    print("=" * 70)
    print("QT DEPENDENCIES")
    print("=" * 70)
    passed, total, checks = verify_qt_dependencies(package_dir)
    total_passed += passed
    total_checks += total
    all_checks.extend(checks)
    for check in checks:
        print(check)
    print()

    # OpenSSL dependencies
    print("=" * 70)
    print("OPENSSL DEPENDENCIES")
    print("=" * 70)
    passed, total, checks = verify_openssl_dependencies(package_dir)
    total_passed += passed
    total_checks += total
    all_checks.extend(checks)
    for check in checks:
        print(check)
    print()

    # VCRUNTIME dependencies
    print("=" * 70)
    print("VISUAL C++ RUNTIME")
    print("=" * 70)
    passed, total, checks = verify_vcruntime_dependencies(package_dir)
    total_passed += passed
    total_checks += total
    all_checks.extend(checks)
    for check in checks:
        print(check)
    print()

    # Python dependencies
    print("=" * 70)
    print("PYTHON DEPENDENCIES")
    print("=" * 70)
    passed, total, checks = verify_python_dependencies(package_dir, enable_python)
    if total > 0:
        total_passed += passed
        total_checks += total
        all_checks.extend(checks)
    for check in checks:
        print(check)
    print()

    # Plugins
    print("=" * 70)
    print("PLUGINS")
    print("=" * 70)
    passed, total, checks = verify_plugins(package_dir)
    total_passed += passed
    total_checks += total
    all_checks.extend(checks)
    for check in checks:
        print(check)
    print()

    # Documentation
    print("=" * 70)
    print("DOCUMENTATION")
    print("=" * 70)
    passed, total, checks = verify_documentation(package_dir)
    total_passed += passed
    total_checks += total
    all_checks.extend(checks)
    for check in checks:
        print(check)
    print()

    # Package statistics
    print("=" * 70)
    print("PACKAGE STATISTICS")
    print("=" * 70)
    size_mb, file_count = calculate_package_size(package_dir)
    print(f"Total size: {size_mb:.1f} MB")
    print(f"Total files: {file_count}")
    print()

    # Size warnings
    if size_mb > 200:
        print("⚠ WARNING: Package size exceeds 200 MB")
        print("  Consider removing unnecessary files")
    elif size_mb > 150:
        print("⚠ Package size is large (>150 MB)")
        print("  This is acceptable but consider optimization")
    else:
        print("✓ Package size is reasonable")
    print()

    # Summary
    print("=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print()
    print(f"Checks passed: {total_passed}/{total_checks}")
    pass_rate = (total_passed / total_checks * 100) if total_checks > 0 else 0
    print(f"Success rate: {pass_rate:.1f}%")
    print()

    # Determine overall status
    if pass_rate >= 95:
        print("✓ VERIFICATION PASSED")
        print()
        print("The package is ready for distribution!")
        return 0
    elif pass_rate >= 80:
        print("⚠ VERIFICATION PASSED WITH WARNINGS")
        print()
        print("The package is mostly complete but has some issues.")
        print("Review the warnings above before distribution.")
        return 0
    else:
        print("✗ VERIFICATION FAILED")
        print()
        print("The package has critical issues and is not ready for distribution.")
        print("Fix the errors above and re-run verification.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
