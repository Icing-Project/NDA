# NDA Codebase, Plugin & Audio Report

This document expands the previous summary to provide an explicit, step-by-step reference for the UI entry points, audio engine, plugin infrastructure, encryption/key-exchange components, and every supported Python plugin path. Each section highlights the primary files and the behaviors they implement so you can orient yourself rapidly before making changes.

---

## 1. Application Foundations

### 1.1 Build Configuration
- **`CMakeLists.txt`** configures the Qt6 project (Core, Widgets, Gui, Network) and links OpenSSL (`SSL` + `Crypto`).
  - Defines `NDA_ENABLE_PYTHON` as an option. When enabled, CMake finds the Python interpreter + NumPy, defines `NDA_ENABLE_PYTHON`, and injects `PythonPluginBridge` sources and headers.
  - When the option is disabled, `PythonPluginBridge` is excluded, no Python linker flags are emitted, and only native C++ plugins and features remain.

### 1.2 Application Entry Point
- **`src/main.cpp`** sets up `QApplication`, applies the Fusion dark palette and the global stylesheet, instantiates `MainWindow`, and enters `app.exec()`. It is the only file that interacts with the Qt event loop directly.

- **`include/ui/MainWindow.h`** / **`src/ui/MainWindow.cpp`**
  - Owns the `QTabWidget` with the three primary tabs: `PipelineView`, `Dashboard`, and `SettingsView`.
  - Shares `nda::PluginManager` and `nda::ProcessingPipeline` as `std::shared_ptr` so that every tab manipulates the same runtime state.
  - Handles actions such as `pipelineStarted` / `pipelineStopped` to update the status bar and switch to the dashboard automatically whenever streaming begins.

### 1.3 UI Modules and Controls
- **Pipeline configuration**
  - `src/ui/PipelineView.cpp` manages plugin loading (via directory selection or the `plugins_py` auto-load), populates the four combo boxes (source, encryptor, bearer, sink), and enforces the “source + sink required” rule before enabling `Start Pipeline`. The view emits `pipelineStarted`/`pipelineStopped` signals for the main window.
  - `src/ui/PluginsView.cpp` presents a list view alternative for inspecting loaded plugins and shows plugin metadata; it mirrors the auto-load logic and helps visualize what the manager currently holds.

- **Live monitoring**
  - `src/ui/Dashboard.cpp` pulls latency/CPU/memory stats from `nda::ProcessingPipeline`, animates audio level meters (currently with random data), and exposes a stop-only control so users can halt streaming from the monitor tab.

- **Application settings**
  - `src/ui/SettingsView.cpp` renders several cards for general, performance, and system information. It stores values in the form (auto-start, minimize to tray, hardware acceleration toggle, latency target, CPU priority). All persistence behavior is stubbed but hooks exist for future config file integration.

---

## 2. Audio Engine & Core Processing

### 2.1 Buffers, Devices, and Threading
- **`include/audio/AudioBuffer.h`**
  - Defines a multi-channel float container usable on both the C++ and Python sides.
  - Offers helpers for resizing, copying, mixing, and retrieving per-channel pointers, so plugins and the pipeline do not have to reimplement buffer management.

- **`include/audio/AudioDevice.h` / `src/audio/AudioDevice.cpp`**
  - Provides `AudioDeviceInfo` and `AudioDevice` for enumerating devices (placeholder WASAPI entries for now).
  - `open()`/`close()` are currently stubs, but `getLatency()` derives latency from buffer size/sample rate to give the UI something to display.

### 2.2 Audio Engine
- **`include/audio/AudioEngine.h` / `src/audio/AudioEngine.cpp`**
  - Manages dedicated input/output `AudioDevice` instances, the optional `Encryptor`, and a callback that plugins or the UI can install.
  - `initialize()` opens default devices (fallback stubs), `start()` sets `isRunning_` and is designed to spawn an audio thread, and `processAudio()` is the hook where encryption and user callbacks would eventually run.
  - Provides straightforward getters for latency (input + output), CPU load (placeholder 8.5%), and underrun counters.

### 2.3 Processing Pipeline
- **`include/core/ProcessingPipeline.h` / `src/core/ProcessingPipeline.cpp`**
  - The pipeline keeps pointers to the source, bearer (network), encryptor, and sink, plus a working `AudioBuffer` for frame transfers.
  - `initialize()` ensures every plugin is in the `Initialized` state, resizes the work buffer based on the source configuration, and writes informative logs about the pipeline layout.
  - `start()` begins plugin execution, spins up `processingThread()`, and resets the processed-samples counter.
  - `processAudioFrame()`:
      1. Reads audio from the source (handles failure counters and sleeps on repeated misses).
      2. Encrypts the buffer in-place if an encryptor is attached (stubs for nonce/tag handling exist).
      3. Builds a `Packet`, serializes the frame, and sends it through the bearer if connected.
      4. Writes the buffer to the sink.
      5. Increments `processedSamples_` for monitoring.
  - `stop()` halts the source first (avoids blocking reads), joins the background thread, and then stops the remainder of the components so shutdown is deterministic.
  - Metrics such as `getLatency()`, `getCPULoad()`, and `getProcessedSamples()` provide the Dashboard with live values (currently based on buffer sizes and a placeholder CPU load).

### 2.4 Encryption & Key Exchange
- **`include/crypto/Encryptor.h` / `src/crypto/Encryptor.cpp`**
  - Defines AES/ChaCha algorithms (AES-128/192/256-GCM, ChaCha20-Poly1305), wrappers for `encrypt`, `decrypt`, `encryptAudio`, `decryptAudio`, and flags for hardware acceleration (placeholder assumes true).
  - Supports key generation, export/import to disk, and random nonce creation. TODO: replace the memcpy placeholders with OpenSSL `EVP_Cipher` calls.

- **`include/crypto/KeyExchange.h` / `src/crypto/KeyExchange.cpp`**
  - Houses ECDH/DH abstractions (P-256, P-384, DH-2048/4096), key-pair generation, shared-secret derivation, and serialization helpers for public keys. For now it uses dummy data but highlights where OpenSSL integration will go.

---

## 3. Plugin Infrastructure

### 3.1 Base Contracts
- **`include/plugins/BasePlugin.h`**
  - Defines `BasePlugin`, `PluginInfo`, and the `PluginState` enum.
  - Declares lifecycle methods (`initialize`, `shutdown`, `start`, `stop`), parameter helpers (`setParameter`, `getParameter`), and API version enforcement via `NDA_DECLARE_PLUGIN`.

- **`include/plugins/PluginTypes.h`**
  - Enumerates plugin categories (`AudioSource`, `AudioSink`, `Bearer`, `Encryptor`, `Processor`) plus the state machine (`Unloaded`, `Loaded`, `Initialized`, `Running`, `Error`).

- **Specialized Interfaces**
  - `AudioSourcePlugin` includes callback wiring plus sample rate/channel configuration.
  - `AudioSinkPlugin` defines `writeAudio`, buffer-size introspection, and available space tracking.
  - `BearerPlugin` handles network connection, packet send/receive, and simulated latency/packet loss.
  - `EncryptorPlugin` exposes encryption/decryption methods, key management, algorithms, and hardware acceleration info.

### 3.2 Plugin Manager
- **`include/plugins/PluginManager.h` / `src/plugins/PluginManager.cpp`**
  - Loads C++ plugins via platform-specific APIs (`dlopen` / `LoadLibrary`). It locates `createPlugin`/`destroyPlugin`, validates the plugin’s API version, and wraps the instance into a `std::shared_ptr` with a custom deleter that calls `destroyPlugin`.
  - When `NDA_ENABLE_PYTHON` is defined, it also loads `.py` plugins through `PythonPluginBridge`. Python plugins are stored with `libraryHandle = nullptr` but still expose metadata and lifecycle hooks.
  - Offers directory scanning (filters helper scripts), typed getters (e.g., `getAudioSourcePlugin`), and `unloadPlugin`/`unloadAll`.

### 3.3 Sample Plugin Code
- **`plugins_src/`**
  - Contains C++ plugin samples such as `SineWaveSourcePlugin.cpp` (generates a sine tone), `NullSinkPlugin.cpp` (logs RMS), `WavFileSinkPlugin.cpp`, and Linux-specific microphone/speaker stubs to illustrate real implementations of the interfaces.
  - Each sample respects the life-cycle states and demonstrates parameter handling and buffer manipulation.

- **`examples/`**
  - `MicrophoneSourcePlugin.h`, `AES256EncryptorPlugin.h`, `UDPBearerPlugin.h`, etc., serve as header-only templates and show how to implement state tracking, send/receive logic, and hardware acceleration flags.
  - `examples/python_plugin_example.cpp` illustrates how to instantiate Python plugins via `PythonPluginBridge`, wire a sine wave generator to a sink, record to WAV, and list plugin metadata. It requires `NDA_ENABLE_PYTHON` during build.

---

## 4. Python Plugin Ecosystem

### 4.1 C++ ↔ Python Bridge
- **`include/plugins/PythonPluginBridge.h` / `src/plugins/PythonPluginBridge.cpp`**
  - Initializes the Python interpreter on the first bridge instance (with NumPy via `import_array1`), manages the GIL, and keeps references to the imported module and plugin instance.
  - Implements both the `AudioSourcePlugin` and `AudioSinkPlugin` interfaces so Python code can be slotted at either end of the pipeline.
  - Bridges `AudioBuffer` objects by creating `base_plugin.AudioBuffer` instances (NumPy-backed) and copying data back/forth when `read_audio`/`write_audio` are invoked.
  - Forwards lifecycle and parameter calls (`initialize`, `start`, `stop`, `get_parameter`, etc.) and prints Python tracebacks when calls fail.

### 4.2 Python Plugin Contracts
- **`plugins_py/base_plugin.py`**
  - Mirrors the C++ plugin API: `PluginType`, `PluginState`, `PluginInfo`, abstract `BasePlugin`, and specialized `AudioSourcePlugin`/`AudioSinkPlugin` classes.
  - `AudioBuffer` wraps a NumPy array and provides metadata readers plus `copy_from`.
  - Each plugin must implement `initialize`, `shutdown`, `start`, `stop`, `get_info`, parameter setters/getters, and audio I/O methods.

- **`plugins_py/plugin_loader.py`**
  - Dynamic loader for running Python plugins independently. It handles discovery, `importlib` loading, factory invocation, and state tracking for debugging.

### 4.3 Python Plugin Catalog
- **Audio Sources**
  1. `plugins_py/sine_wave_source.py` – stereo sine-wave generator with adjustable frequency and state safeguards.
  2. `plugins_py/pulseaudio_microphone.py` – PyAudio callback-based microphone capture with ring-buffer deque and underrun tracking.
  3. `plugins_py/sounddevice_microphone.py` – sounddevice-based capture with queue-backed reads and timeout handling.

- **Audio Sinks**
  1. `plugins_py/null_sink.py` – console monitor sink that drops audio but logs RMS every tenth of a second.
  2. `plugins_py/wav_file_sink.py` – records float PCM to timestamped WAV files, writes proper headers, and flushes on stop.
  3. `plugins_py/pulseaudio_speaker.py` – PyAudio output stream (blocking) for real-time playback.
  4. `plugins_py/sounddevice_speaker.py` – sounddevice callback/queue playback for latency control.

- **Helper instructions**
  - `plugins_py/QUICKSTART.md` steps through dependency installation (`sounddevice`, `numpy`, optional `pyaudio`), running `test_plugins.py`, and using the Python loader or the C++ bridge.
  - `plugins_py/README.md` explains plugin categories, how to author a plugin (`create_plugin()` factory), and when to prefer Python for prototyping vs. C++ for tight latency.
  - `requirements.txt` lists the runtime libraries expected by the Python plugin folder (`sounddevice`, `numpy`, optional `pyaudio`).

---

## 5. Deployment, Packaging & References

- **`scripts/`**
  - `build_windows.bat`, `deploy.py`, `deploy_windows.bat`, and `create_platform_packages.py` automate packaging and deployment of NDA binaries. Refer to `docs/PLATFORM_PACKAGES_READY.md` for the release steps.

- **`docs/`**
  - In addition to this report, the folder already contains `NDA-SPECS.md`, `DEPLOYMENT_COMPLETE.md`, and `PLATFORM_PACKAGES_READY.md`. Keep those checklists synchronized with this overview when you adjust critical subsystems; they document release milestones, QA steps, and deployment sign-off criteria.

- **`packages/`**
  - Stores the built artifacts: `NADE-v1.0.0-Linux-x64.tar.gz` and `NADE-v1.0.0-Windows-Source.tar.gz`, plus the unpacked directories under `packages/linux` and `packages/windows`.

- **`examples/`**
  - Reference headers and the `python_plugin_example.cpp` file are ready-to-copy templates for new plugins. Rerun `cmake`/`ninja` with `-DNDA_ENABLE_PYTHON=ON` if you want to compile the example against the bridge.

---

Keep this document nearby so anyone scanning the repo knows which files to open when investigating UI wiring, plugin lifecycles, audio threading, encryption, or Python integrations. Update the referenced sections every time you modify the pipeline, plugin manager, or Python bridge so this report stays accurate.
