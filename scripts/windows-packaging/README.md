# Windows Standalone Packaging Scripts

This directory contains all scripts for building standalone Windows portable packages of NDA.

## Files

### Configuration
- **build_config.json.template** - Template for build configuration
- **build_config.json** - Generated configuration (created by setup script, not in git)

### Scripts

#### Setup
- **setup_build_config.bat** - Auto-detect development environment and create configuration

#### Build & Package
- **build_release_package.bat** - Master script - runs complete build and packaging pipeline

#### Deployment Components
- **collect_dependencies.py** - Collect Qt, OpenSSL, and VCRUNTIME DLLs
- **bundle_python.py** - Bundle Python runtime and site-packages
- **deploy_standalone.py** - Main deployment orchestrator
- **verify_package.py** - Comprehensive package verification (42 checks)

## Quick Start

### First Time Setup

```batch
# Run from repository root
scripts\windows-packaging\setup_build_config.bat
```

This will:
- Auto-detect Qt, OpenSSL, Python, Visual Studio installations
- Create `build_config.json` with detected paths
- Report any missing components

### Build Release Package

```batch
# Run from repository root
scripts\windows-packaging\build_release_package.bat
```

This will:
1. Clean build directory
2. Configure with CMake
3. Build NDA.exe and all plugins
4. Deploy standalone package
5. Bundle Python (if enabled)
6. Verify package integrity
7. Create ZIP archive

**Output:**
- `package/` - Extracted standalone package
- `NDA-Windows-Portable-v{version}.zip` - Distributable archive

## Configuration

Edit `build_config.json` to customize:

```json
{
  "qt_path": "C:/Qt/6.6.3/msvc2019_64",
  "openssl_path": "C:/Program Files/OpenSSL-Win64",
  "python_path": "C:/Python312",
  "visual_studio_path": "C:/Program Files/Microsoft Visual Studio/2022/Community",
  "build_config": "Release",
  "enable_python": true,
  "package_version": "2.0.0",
  "package_name": "NDA-Windows-Portable"
}
```

**Key settings:**
- `enable_python`: Set to `true` to include Python support (~30 MB larger)
- `package_version`: Update for each release
- `build_config`: Use `Release` for production, `Debug` for development

## Manual Execution

You can run individual scripts if needed:

```batch
# From repository root

# 1. Deploy package (assumes NDA.exe already built)
python scripts\windows-packaging\deploy_standalone.py

# 2. Verify package
python scripts\windows-packaging\verify_package.py

# 3. Collect dependencies only
python scripts\windows-packaging\collect_dependencies.py

# 4. Bundle Python only
python scripts\windows-packaging\bundle_python.py
```

## Documentation

- **Full Guide:** `docs/guides/standalone-windows.md`
- **Implementation Report:** `docs/reports/WINDOWS_PACKAGING_COMPLETE.md`
- **Installation Guide:** `docs/guides/installation.md`

## Troubleshooting

### Error: build_config.json not found
**Solution:** Run `setup_build_config.bat` first

### Error: Qt/OpenSSL not found
**Solution:** Edit `build_config.json` with correct paths

### Package verification fails
**Solution:** Review verification output for missing dependencies

### Package too large
**Solution:** Set `"enable_python": false` in `build_config.json` (saves ~30 MB)

## Architecture

```
User runs: build_release_package.bat
    ├─> Verifies prerequisites
    ├─> Cleans build directory
    ├─> Runs CMake configure
    ├─> Builds NDA.exe + plugins
    ├─> Calls deploy_standalone.py
    │    ├─> Creates package structure
    │    ├─> Copies NDA.exe + plugins
    │    ├─> Calls collect_dependencies.py
    │    │    ├─> Runs windeployqt (Qt DLLs)
    │    │    ├─> Copies OpenSSL DLLs
    │    │    └─> Copies VCRUNTIME DLLs
    │    ├─> Calls bundle_python.py (if enabled)
    │    │    ├─> Installs required packages
    │    │    ├─> Copies Python DLLs
    │    │    └─> Copies site-packages
    │    ├─> Generates NDA.bat launcher
    │    └─> Generates README.txt
    ├─> Calls verify_package.py
    │    └─> 42-point verification
    └─> Creates ZIP archive

Output: NDA-Windows-Portable-v{version}.zip
```

## See Also

- [Standalone Windows Guide](../../docs/guides/standalone-windows.md) - Complete user guide
- [Installation Guide](../../docs/guides/installation.md) - Build from source instructions
- [Troubleshooting](../../docs/guides/troubleshooting.md) - Common issues
