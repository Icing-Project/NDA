# Repository Guidelines

## Project Structure & Module Organization
- C++ app lives in `src/` with headers in `include/` (Qt6 UI under `src/ui`, audio/crypto/core modules under matching folders). Entry point: `src/main.cpp`.
- Native plugins (C++) in `plugins_src/`; Python plugins in `plugins_py/`.
- Packaging and helper tools in `scripts/` and prebuilt artifacts in `packages/`.
- Docs in `docs/`; example snippets in `examples/`.

## Build, Test, and Development Commands
- Configure + build (Windows, MSVC):
  - `cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_PREFIX_PATH="C:/Qt/6.x/msvc2019_64" -DOPENSSL_ROOT_DIR="C:/Program Files/OpenSSL-Win64"`
  - `cmake --build build --config Release`
- One‑shot Windows build script: `scripts\build_windows.bat`
- Run app (Windows): `build\Release\NDA.exe`
- Enable Python bridge: add `-DNDA_ENABLE_PYTHON=ON` to the CMake configure step.
- Deploy package: `python scripts\deploy.py` or `scripts\deploy_windows.bat` (adds Qt/Python/OpenSSL runtime files).
- Linux (example): `mkdir build && cd build && cmake .. && make -j`.

## Coding Style & Naming Conventions
- C++17, 4‑space indent, brace on new line for functions; include paths relative to `include/`.
- Classes: UpperCamelCase (e.g., `ProcessingPipeline`); methods: lowerCamelCase; private members end with underscore (e.g., `pluginManager_`).
- C++ files use PascalCase (e.g., `MainWindow.cpp/.h`). Python plugins: snake_case modules and functions, PEP8‑style.
- Keep changes minimal and consistent with surrounding code; prefer `std::` over custom utilities.

## Testing Guidelines
- No formal unit test harness is present yet. Provide manual repro steps in PRs and validate core flows: build, launch UI, start/stop pipeline, and load a Python plugin.
- For Python plugins, see `plugins_py/QUICKSTART.md` and run quick tests after `pip install -r requirements.txt`.

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise subject (≤72 chars), optional scoped prefix like `[ui]`, `[core]`, `[plugins]`.
- PRs must include: summary, rationale, test steps, screenshots for UI, and any doc updates. Ensure Release build succeeds and deployment script runs.

## Security & Configuration Tips
- Do not commit secrets or packaged binaries outside `packages/`.
- Ensure Qt6 and OpenSSL paths are configured (see build script). When enabling Python, verify CMake reports “Python support enabled”.
