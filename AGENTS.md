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
- Ensure Qt6 and OpenSSL paths are configured (see build script). When enabling Python, verify CMake reports "Python support enabled".

---

## 📚 Documentation Structure

Documentation is organized into 5 main subdirectories for easy navigation:

### docs/guides/
Getting started, installation, use cases, and examples
- **README.md** - Introduction to Icing
- **installation.md** - Platform-specific setup instructions
- **use-cases.md** - Common application scenarios
- **troubleshooting.md** - Solutions to common problems
- **examples/** - Real-world examples (encrypted AIOC radio, Discord VoIP encryption)
- **nda-beta-tester.md** - Beta tester information

### docs/technical/
Core architecture, specifications, and technical details
- **ARCHITECTURE.md** - System design, components, and plugin architecture
- **specifications.md** - Complete API reference and design specification
- **python-bridge.md** - Python optimization details and performance characteristics
- **aioc-io-ptt-spec.md** - AIOC I/O and PTT specification

### docs/development/
Plugin development, building, and migration guides
- **plugins.md** - C++ plugin interface specification and authoring guide
- **python-plugins.md** - Python plugin development guide
- **migration.md** - v1.x → v2.0 migration guide
- **aioc-plugin-plan.md** - AIOC plugin planning and design
- **aioc-plugin-implementation.md** - AIOC plugin implementation details

### docs/strategy/
Design decisions, roadmap, and implementation planning
- **decisions.md** - Why design decisions were made (understand the rationale)
- **summary.md** - Strategic overview and executive summary
- **roadmap.md** - Detailed implementation roadmap and planning

### docs/reports/
Analysis, audits, and technical findings
- **ARCHITECTURE_REPORT.md** - Detailed architecture analysis
- **PERFORMANCE_ANALYSIS_V2.1.md** - Performance metrics and optimization attempts
- **V2.1_EXECUTIVE_SUMMARY.md** - Performance limitations and recommendations
- **PYTHON_PLUGINS_V2_AUDIT.md** - Python plugin audit findings
- **DOCUMENTATION_AUDIT_REPORT.md** - Documentation quality assessment
- Plus 20+ additional analysis and findings reports

### docs/START_HERE.md
Navigation guide for all documentation - read this first!

---

## 📚 Documentation Reference for AI Assistants

### Quick Start for Different Tasks

| Task | Read First |
|------|-----------|
| **Getting started** | `docs/START_HERE.md` → `docs/guides/README.md` |
| **Install & run** | `docs/guides/installation.md` |
| **Understand design** | `docs/strategy/decisions.md` + `docs/technical/ARCHITECTURE.md` |
| **Fix bug in core pipeline** | `docs/technical/ARCHITECTURE.md` → `docs/technical/specifications.md` |
| **Write C++ plugin** | `docs/development/plugins.md` → `docs/technical/specifications.md` |
| **Write Python plugin** | `docs/development/python-plugins.md` → `docs/technical/python-bridge.md` |
| **Add UI feature** | `docs/technical/ARCHITECTURE.md` → `docs/guides/use-cases.md` |
| **Optimize performance** | `docs/technical/python-bridge.md` + `docs/reports/PERFORMANCE_ANALYSIS_V2.1.md` |
| **See examples** | `docs/guides/examples/` |
| **Migrate from v1** | `docs/development/migration.md` |

### Core Technical References
- **[docs/technical/ARCHITECTURE.md](docs/technical/ARCHITECTURE.md)** - System design, components, and plugin architecture
- **[docs/technical/specifications.md](docs/technical/specifications.md)** - Complete API reference and design specification
- **[docs/technical/python-bridge.md](docs/technical/python-bridge.md)** - Python optimization details and performance characteristics

### Strategic & Implementation Context
- **[docs/strategy/decisions.md](docs/strategy/decisions.md)** - Why design decisions were made
- **[docs/strategy/roadmap.md](docs/strategy/roadmap.md)** - Detailed implementation roadmap
- **[docs/reports/v2-implementation-report.md](docs/reports/v2-implementation-report.md)** - Current build status and completed tasks

### Performance Analysis
- **[docs/reports/PERFORMANCE_ANALYSIS_V2.1.md](docs/reports/PERFORMANCE_ANALYSIS_V2.1.md)** - Detailed performance analysis and optimization attempts
- **[docs/reports/V2.1_EXECUTIVE_SUMMARY.md](docs/reports/V2.1_EXECUTIVE_SUMMARY.md)** - Performance limitations and recommendations

---

## 🎯 Key Concepts Summary

**NDA v2.0 is a 3-slot audio processing pipeline with dual independent pipelines:**

```
Source → [Processor (optional)] → Sink
```

- **Source:** Audio input (mic, file, network)
- **Processor:** Audio transformation (encryption, effects) - optional
- **Sink:** Audio output (speaker, file, network)

**Two instances run simultaneously:**
- **TX Pipeline:** Device Mic → Encryptor → AIOC/Network Output
- **RX Pipeline:** AIOC/Network Input → Decryptor → Device Speaker

**Key implementation notes:**
- All processing at 48kHz internally (auto-resamples source/sink)
- Plugins are C++/Python implementations of the three interfaces
- Python bridge is optimized (zero-copy, cached objects, batch GIL)
- Bearer and core crypto removed (v2.0 decision) - encryption is plugin-only
- Sample rate adaptation is transparent (no plugin changes needed)

---

## ⚠️ Important Constraints

1. **Do not make unnecessary changes** - Keep modifications minimal and focused
2. **Do not refactor surrounding code** - Only change what's needed for the task
3. **Do not add features beyond the request** - Stick to the scope
4. **Do not remove documented decisions** - Changes must align with `docs/strategy/decisions.md`
5. **Do not add complex abstractions** - Keep it simple
6. **Do not break existing tests** - If tests exist, keep them passing
7. **Do not add new dependencies** - Use what's already in CMakeLists.txt

---

## 🚀 Common AI Implementation Patterns

### Reading Documentation
```
1. Start with docs/START_HERE.md (2 min)
2. Check docs/ folder structure for your task type (2 min)
3. Read primary reference doc (15-30 min)
4. Read related strategy doc if needed (10 min)
5. Review code examples in repository (10 min)
6. Implement (varies)
```

### Verifying Correctness
```
1. Check against docs/technical/specifications.md
2. Verify against docs/strategy/decisions.md
3. Review similar existing code
4. Test with build: cmake --build build --config Release
5. For Python: verify plugins_py/examples/ still work
```

---

Last updated: January 2026 (Documentation reorganization complete)
