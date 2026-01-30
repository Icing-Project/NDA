# Update Log

## 2026-01-30: GitHub Actions CI for Windows Releases

### What was done
Added GitHub Actions CI workflow for automated Windows releases.

### Files created
- `.github/workflows/release.yml` - Main CI workflow

### Workflow features
1. **Trigger**: Runs on push to `main` branch and on version tags (`v*`)
2. **Build environment**: Uses Fedora 41 container (matches existing Fedora-based build scripts)
3. **Cross-compilation**: Uses MinGW to compile Windows executable from Linux
4. **Versioning**:
   - Tagged releases: Uses the tag name (e.g., `v2.0.1`)
   - Push to main: Auto-generates version from commit count and SHA (e.g., `v2.0.42-abc1234`)
5. **Package contents**:
   - NADE.exe with all required DLLs (Qt6, OpenSSL, MinGW runtime)
   - Embedded Python 3.11 (users don't need to install Python)
   - Python plugins from `plugins_py/`
   - C++ plugin DLLs
   - Nade Python package from https://github.com/Icing-Project/Nade-Python
   - Documentation
   - NADE.bat launcher script
6. **Release**: Creates GitHub release with the Windows ZIP package

### How to use
- Push to `main` → Automatic prerelease with auto-generated version
- Create a tag (e.g., `git tag v2.0.1 && git push --tags`) → Full release with that version

### Dependencies included in release
- Qt6 (Core, Gui, Widgets)
- OpenSSL
- MinGW runtime DLLs
- Embedded Python 3.11
- Nade Python package (cloned from Nade-Python repo)
