# Python Bridge & Plugin Loading System - Complete Technical Guide

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Loading Sequence](#loading-sequence)
3. [Python Plugin Structure](#python-plugin-structure)
4. [Plugin Types](#plugin-types)
5. [Bridge Caching Optimization](#bridge-caching-optimization)
6. [Buffer Exchange Mechanism](#buffer-exchange-mechanism)
7. [Plugin Loader (Dual-Mode)](#plugin-loader-dual-mode)
8. [GIL Management](#gil-management)
9. [Complete Loading Flow](#complete-loading-flow)
10. [Runtime Audio Flow](#runtime-audio-flow)
11. [Key Optimizations](#key-optimizations)
12. [Concrete Examples](#concrete-examples)
13. [Buffer Size Requirements](#buffer-size-requirements)
14. [Error Handling & State Machine](#error-handling--state-machine)

---

## Architecture Overview

The NDA system uses a sophisticated **polymorphic bridge pattern** where:

- **C++ Side**: `PythonPluginBridge` class implements all three plugin interfaces (`AudioSourcePlugin`, `AudioSinkPlugin`, `AudioProcessorPlugin`)
- **Python Side**: Plugins inherit from corresponding base classes
- **Bridge Role**: Acts as a wrapper that lets Python plugins work anywhere C++ plugins are expected
- **Integration**: Seamlessly plugs into the 3-slot `ProcessingPipeline` (source → processor → sink)

```
User Application (C++)
    ↓
ProcessingPipeline (holds 3 slots for plugins)
    ├── Source Plugin (C++ or Python via Bridge)
    ├── Processor Plugin (C++ or Python via Bridge)  [v2.0+ feature]
    └── Sink Plugin (C++ or Python via Bridge)
```

### Key Design Goals

1. **Transparency**: Python plugins work exactly like C++ plugins from the pipeline's perspective
2. **Performance**: Aggressive caching delivers 6-30x performance improvement (v2.0+)
3. **Safety**: GIL management ensures thread-safe Python API calls
4. **Flexibility**: Support for source, processor, and sink plugins
5. **Fallback**: Pure Python runs fine when Cython compilation unavailable

---

## Loading Sequence

### Phase 1: Plugin Discovery & Detection

When the application requests to load a plugin, `PluginManager` inspects the file extension:

```cpp
// src/plugins/PluginManager.cpp:27
bool PluginManager::loadPlugin(const std::string& path) {
    std::filesystem::path pluginPath(path);
    std::string extension = pluginPath.extension().string();

    // Detect plugin type by extension
    if (extension == ".py") {
        return loadPythonPlugin(path);  // Route to Python loader
    } else {
        return loadCppPlugin(path);     // Route to C++ dynamic library loader
    }
}
```

**Python plugins are identified by `.py` extension and routed to `loadPythonPlugin()`**

### Phase 2: Python Plugin Loading

Once identified as a Python plugin, `PluginManager` creates a `PythonPluginBridge` wrapper:

```cpp
// src/plugins/PluginManager.cpp:116
bool PluginManager::loadPythonPlugin(const std::string& path) {
    std::filesystem::path pluginPath(path);
    std::string filename = pluginPath.filename().string();      // e.g., "sine_wave_source.py"
    std::string directory = pluginPath.parent_path().string();  // e.g., "plugins_py"

    // Step 1: Create bridge wrapper
    PythonPluginBridge* bridge =
        PythonPluginFactory::createPlugin(filename, directory);

    if (!bridge) {
        std::cerr << "[PluginManager] Failed to load Python plugin: " << path << std::endl;
        return false;
    }

    // Step 2: Initialize the bridge (loads Python module inside)
    if (!bridge->initialize()) {
        std::cerr << "[PluginManager] Python plugin initialization failed" << std::endl;
        delete bridge;
        return false;
    }

    // Step 3: Store as shared_ptr for memory management
    std::shared_ptr<BasePlugin> instance(bridge);

    LoadedPlugin loadedPlugin;
    loadedPlugin.path = path;
    loadedPlugin.name = instance->getInfo().name;  // From Python plugin's get_info()
    loadedPlugin.libraryHandle = nullptr;          // N/A for Python
    loadedPlugin.instance = instance;
    loadedPlugin.info = instance->getInfo();
    loadedPlugin.type = instance->getType();

    plugins_[loadedPlugin.name] = loadedPlugin;  // Register globally
    return true;
}
```

### Phase 3: PythonPluginBridge::loadPlugin() - The Core Magic

This is where Python modules are dynamically imported. The bridge:

1. **Adds plugin directory to Python path**
   ```python
   sys.path.insert(0, "/path/to/plugins_py")
   ```

2. **Dynamically imports the module**
   ```python
   import importlib
   module = importlib.import_module("sine_wave_source")  # basename without .py
   ```

3. **Calls the factory function**
   ```python
   plugin_instance = module.create_plugin()  # Must exist in every plugin
   ```

4. **Validates the instance**
   Ensures it has the right methods:
   - `initialize()`, `start()`, `stop()`, `shutdown()`
   - `read_audio()` or `write_audio()` or `process_audio()`
   - `get_info()`, `get_type()`, etc.

5. **Initializes the caching layer** (v2.0+ optimization)
   ```cpp
   initializeCache();  // Cache Python objects & methods for performance
   ```

---

## Python Plugin Structure

Every Python plugin must follow this pattern:

```python
# Example: plugins_py/sine_wave_source.py
from base_plugin import AudioSourcePlugin, PluginInfo, PluginType, AudioBuffer, PluginState

class SineWaveSourcePlugin(AudioSourcePlugin):
    def __init__(self):
        super().__init__()
        self.sample_rate = 48000
        self.channel_count = 2
        self.buffer_size = 512  # CRITICAL: Must be 512 (v2.1 requirement)
        self.frequency = 440.0
        self.phase = 0.0

    def initialize(self) -> bool:
        """Called once after plugin is loaded"""
        self.state = PluginState.INITIALIZED
        return True

    def start(self) -> bool:
        """Called when pipeline starts playing"""
        self.state = PluginState.RUNNING
        return True

    def stop(self):
        """Called when pipeline stops"""
        self.state = PluginState.INITIALIZED

    def shutdown(self):
        """Called when plugin is unloaded"""
        self.state = PluginState.UNLOADED

    def read_audio(self, buffer: AudioBuffer) -> bool:
        """Fill buffer with audio data (called every frame)"""
        frame_count = buffer.get_frame_count()  # Usually 512
        channels = buffer.get_channel_count()    # Usually 2

        # Generate sine wave and fill buffer
        # buffer.data is NumPy array: shape (channels, frames)
        phase_increment = (2π * frequency) / sample_rate
        # ... generate samples ...
        buffer.data[:channels, :frame_count] = samples

        return True  # Success

    def get_info(self) -> PluginInfo:
        """Return plugin metadata"""
        return PluginInfo(
            name="Sine Wave Generator",
            version="1.0.0",
            author="Icing Project",
            description="Generates 440Hz sine wave for testing",
            plugin_type=PluginType.AUDIO_SOURCE,
            api_version=1
        )

    def get_sample_rate(self) -> int:
        return self.sample_rate

    def get_channel_count(self) -> int:
        return self.channel_count

    def set_sample_rate(self, sample_rate: int):
        self.sample_rate = sample_rate

    def set_channel_count(self, channels: int):
        self.channel_count = channels

    def get_buffer_size(self) -> int:
        return self.buffer_size

    def set_buffer_size(self, samples: int):
        self.buffer_size = max(64, int(samples))

    def set_audio_callback(self, callback):
        self.callback = callback

    def set_parameter(self, key: str, value: str):
        if key == "frequency":
            self.frequency = float(value)

    def get_parameter(self, key: str) -> str:
        if key == "frequency":
            return str(self.frequency)
        return ""

    def get_type(self) -> PluginType:
        return PluginType.AUDIO_SOURCE

    def get_state(self) -> PluginState:
        return self.state

# REQUIRED: Factory function
def create_plugin():
    """Called by PythonPluginBridge to instantiate the plugin"""
    return SineWaveSourcePlugin()
```

### Required Methods Summary

Every Python plugin must implement:

| Method | Required? | Purpose |
|--------|-----------|---------|
| `__init__()` | Yes | Constructor - initialize state |
| `initialize()` | Yes | One-time setup (called once) |
| `start()` | Yes | Start operation (called when pipeline starts) |
| `stop()` | Yes | Stop operation (called when pipeline stops) |
| `shutdown()` | Yes | Cleanup (called when plugin unloaded) |
| `get_info()` | Yes | Return PluginInfo metadata |
| `get_type()` | Yes | Return PluginType (SOURCE/SINK/PROCESSOR) |
| `get_state()` | Yes | Return current PluginState |
| `set_parameter(key, value)` | Yes | Handle parameter changes |
| `get_parameter(key)` | Yes | Query parameter values |
| `read_audio()` | If SOURCE | Fill buffer with audio |
| `write_audio()` | If SINK | Consume audio from buffer |
| `process_audio()` | If PROCESSOR | Modify buffer in-place |
| `get_sample_rate()` | Yes | Return current sample rate |
| `get_channel_count()` | Yes | Return current channel count |
| `set_sample_rate()` | Yes | Handle sample rate changes |
| `set_channel_count()` | Yes | Handle channel count changes |
| `get_buffer_size()` | If SOURCE | Return buffer size (must be 512) |
| `set_buffer_size()` | If SOURCE | Handle buffer size changes |
| `set_audio_callback()` | If SOURCE | Optional: set callback for push model |
| `get_available_space()` | If SINK | Return available buffer space |
| `create_plugin()` | Yes | Module-level factory function |

---

## Plugin Types

### AudioSourcePlugin (Generates Audio)

Generates or reads audio data from an external source:

```python
class AudioSourcePlugin(BasePlugin):
    def read_audio(self, buffer: AudioBuffer) -> bool:
        """
        Fill buffer with audio data.

        This is the "pull" model - the pipeline requests data.
        Called repeatedly by the audio thread.

        Args:
            buffer: AudioBuffer to fill (shape: [channels, frames])

        Returns:
            True if successful, False on error
        """
        # Example: microphone, sine wave, file, network stream
        return True

    def get_buffer_size(self) -> int:
        """MUST return 512 in v2.1"""
        return 512
```

**Real Examples in codebase:**
- `sine_wave_source.py`: Generates test tones
- `sounddevice_microphone.py`: Captures from system microphone
- `pulseaudio_microphone.py`: Captures from PulseAudio

### AudioProcessorPlugin (Transforms Audio)

Modifies audio in-place (v2.0+ feature):

```python
class AudioProcessorPlugin(BasePlugin):
    def process_audio(self, buffer: AudioBuffer) -> bool:
        """
        Process audio buffer in-place.

        Args:
            buffer: AudioBuffer to process
                    Modify buffer.data (NumPy array) directly

        Returns:
            True on success, False on error (pipeline passthrough on failure)
        """
        # Example: encryption, volume adjustment, effects, resampling
        buffer.data *= 0.5  # Reduce volume by half
        return True

    def get_processing_latency(self) -> float:
        """Return algorithmic latency in seconds"""
        return 0.0  # Zero-latency for most processors
```

**Real Examples in codebase:**
- `simple_gain.py`: Volume adjustment with parameter control
- `fernet_encryptor.py`: Encrypts audio stream
- `fernet_decryptor.py`: Decrypts audio stream

### AudioSinkPlugin (Consumes Audio)

Outputs audio to an external destination:

```python
class AudioSinkPlugin(BasePlugin):
    def write_audio(self, buffer: AudioBuffer) -> bool:
        """
        Consume audio data.

        This is the "pull" model - the pipeline provides data.

        Args:
            buffer: AudioBuffer with data to write

        Returns:
            True if successful, False on error
        """
        # Example: speakers, file, network stream
        return True

    def get_available_space(self) -> int:
        """Return bytes available before blocking"""
        return 65536
```

**Real Examples in codebase:**
- `pulseaudio_speaker.py`: Outputs to system speakers
- `wav_file_sink.py`: Writes to WAV file
- `null_sink.py`: Discards audio (for testing)

---

## Bridge Caching Optimization

The v2.0/v2.1 optimization delivers **6-30x performance improvement** via aggressive caching. This is critical because the bridge is called 48,000 times per second at 48kHz sample rate.

### What Gets Cached

```cpp
// include/plugins/PythonPluginBridge.h:117-132
// v2.0 Optimization: Object and method caching (6-30x performance improvement)
PyObject* cachedBasePluginModule_;     // base_plugin module (reused across calls)
PyObject* cachedAudioBufferClass_;     // AudioBuffer class object
PyObject* cachedBufferInstance_;       // Reused buffer object (recreated only on size change)
PyArrayObject* cachedNumpyArray_;      // Reused NumPy array reference
float* cachedDataPtr_ = nullptr;       // Direct pointer to NumPy data (avoids per-frame lookup)

// Cached method objects (avoid repeated attribute lookup)
PyObject* cachedReadAudioMethod_;      // plugin.read_audio method
PyObject* cachedWriteAudioMethod_;     // plugin.write_audio method
PyObject* cachedProcessAudioMethod_;   // plugin.process_audio method

// Buffer dimension tracking for cache invalidation
int cachedChannels_;
int cachedFrames_;
```

### Initialization

```cpp
// Called once after plugin load
void initializeCache() {
    // 1. Cache the base_plugin module (reused across all operations)
    cachedBasePluginModule_ = PyImport_ImportModule("base_plugin");

    // 2. Cache AudioBuffer class object
    cachedAudioBufferClass_ = PyObject_GetAttrString(
        cachedBasePluginModule_, "AudioBuffer");

    // 3. Pre-create reusable buffer instance
    cachedBufferInstance_ = PyObject_CallObject(
        cachedAudioBufferClass_, argsWithChannelsAndFrames);

    // 4. Cache the NumPy array inside the buffer
    cachedNumpyArray_ = (PyArrayObject*)PyObject_GetAttrString(
        cachedBufferInstance_, "data");

    // 5. Cache DIRECT POINTER to NumPy's underlying data
    // This is critical - avoids per-frame Python API call
    cachedDataPtr_ = (float*)PyArray_DATA(cachedNumpyArray_);

    // 6. Cache method references (avoid attribute lookup per frame)
    cachedReadAudioMethod_ = PyObject_GetAttrString(
        pPluginInstance_, "read_audio");
    cachedWriteAudioMethod_ = PyObject_GetAttrString(
        pPluginInstance_, "write_audio");
    cachedProcessAudioMethod_ = PyObject_GetAttrString(
        pPluginInstance_, "process_audio");
}
```

### Per-Frame Operation

```cpp
// Called 48,000 times per second at 48kHz sample rate
bool PythonPluginBridge::readAudio(AudioBuffer& buffer) {
    // 1. Acquire GIL once (not per attribute lookup!)
    PyGILState_STATE gstate = PyGILState_Ensure();

    // 2. Get or recreate cached buffer (only if dimensions changed)
    PyObject* pBuffer = getOrCreateCachedBuffer(buffer);

    // 3. Update cached NumPy pointer with new data
    updateCachedBufferData(buffer, pBuffer);

    // 4. Call CACHED method (no attribute lookup!)
    // This is dramatically faster than PyObject_GetAttrString + call
    PyObject* result = PyObject_CallFunctionObjArgs(
        cachedReadAudioMethod_, pBuffer, nullptr);

    // 5. Process result
    bool success = PyObject_IsTrue(result);

    // 6. Copy data back via cached pointer (fast memcpy)
    std::memcpy(buffer.data(),
                cachedDataPtr_,
                buffer.frames() * buffer.channels() * sizeof(float));

    // Cleanup
    Py_DECREF(result);
    PyGILState_Release(gstate);

    return success;
}
```

### Performance Impact

| Approach | Time per Frame |
|----------|----------------|
| Uncached (attribute lookup + import + resolution) | ~100-200μs |
| Cached (direct pointer dereference) | ~5-10μs |
| **Speedup** | **10-20x faster** |

For context:
- 512 samples @ 48kHz = 10.67ms per frame
- Uncached overhead = 100-200μs (1-2% of frame time)
- Cached overhead = 5-10μs (0.05% of frame time)

---

## Buffer Exchange Mechanism

### C++ → Python (Writing/Processing)

The flow for sending data from C++ to Python:

```
C++ AudioBuffer (float array)
    ↓
Create Python AudioBuffer object
    ↓
Copy data: std::memcpy → NumPy array
    ↓
Call Python method: write_audio(buffer) or process_audio(buffer)
    ↓
Python modifies buffer.data (NumPy array)
    ↓
Copy back: std::memcpy from NumPy → C++ buffer
    ↓
C++ uses modified data
```

**Implementation:**

```cpp
// Create Python buffer and copy data
PyObject* createPythonAudioBuffer(const AudioBuffer& cppBuffer) const {
    // Create: AudioBuffer(channels, frames)
    PyObject* pArgs = PyTuple_Pack(2,
        PyLong_FromLong(cppBuffer.channels()),
        PyLong_FromLong(cppBuffer.frames()));

    PyObject* pBuffer = PyObject_CallObject(
        cachedAudioBufferClass_, pArgs);

    // Copy C++ data to NumPy array
    float* numpyPtr = (float*)PyArray_DATA(cachedNumpyArray_);
    std::memcpy(numpyPtr, cppBuffer.data(),
                cppBuffer.channels() * cppBuffer.frames() * sizeof(float));

    Py_DECREF(pArgs);
    return pBuffer;
}
```

### Python → C++ (Reading)

The flow for getting data from Python back to C++:

```
Python AudioBuffer with NumPy data
    ↓
C++ receives Python buffer reference
    ↓
Access cached NumPy data pointer
    ↓
Copy data: std::memcpy from NumPy → C++ buffer
    ↓
C++ processes data
```

**Optimization in v2.1:**

From `base_plugin.py:64`:
```python
class AudioBuffer:
    def __init__(self, channels: int, frame_count: int):
        # OPTIMIZATION: Use np.empty() instead of np.zeros()
        # The C++ bridge always fills the buffer via memcpy before Python sees it
        self.data = np.empty((channels, frame_count), dtype=np.float32)
```

This saves initialization time because:
1. C++ will immediately overwrite with `memcpy` anyway
2. No need to zero-initialize (saves time)
3. Memory is already allocated

---

## Plugin Loader (Dual-Mode)

The Python plugin loader supports two modes for maximum flexibility:

### Plugin Loading (Direct Import)

Plugins are loaded directly via Python's import system:

```cpp
// src/plugins/PythonPluginBridge.cpp
PyObject* pName = PyUnicode_FromString(moduleName.c_str());
pModule_ = PyImport_Import(pName);
Py_DECREF(pName);

// Get factory function and create instance
PyObject* pFunc = PyObject_GetAttrString(pModule_, "create_plugin");
pPluginInstance_ = PyObject_CallObject(pFunc, nullptr);
```

**Plugin structure:**

```
plugins_py/
├── sine_wave_source.py      # Source plugin
├── simple_gain.py           # Processor plugin
├── null_sink.py             # Sink plugin
└── examples/
    ├── simple_gain.py
    └── fernet_encryptor.py
```

### Optional: Cython Compilation (Manual)

For CPU-intensive plugins, manual Cython compilation is available via `cython_compiler.py`:

```python
from cython_compiler import compile_plugin, get_cache_dir
from pathlib import Path

plugin_dir = Path("plugins_py")
cache_dir = get_cache_dir(plugin_dir)

# Compile a plugin (uses SHA256-based cache invalidation)
compiled_path = compile_plugin(plugin_dir / "simple_gain.py", cache_dir)
```

**Note:** Automatic Cython compilation during loading is deprecated. Use manual compilation for performance-critical plugins.

---

## GIL Management

Python's Global Interpreter Lock (GIL) requires careful handling for thread-safe Python API calls. The bridge manages this automatically:

```cpp
// Acquire GIL for thread-safe Python API calls
PyGILState_STATE gstate = PyGILState_Ensure();

try {
    // Now it's safe to call Python API
    PyObject* result = PyObject_CallFunctionObjArgs(
        cachedReadAudioMethod_, pBuffer, nullptr);

    // ... process result ...

} finally {
    // ALWAYS release GIL
    PyGILState_Release(gstate);
}
```

### Optimization Strategy

Rather than acquiring/releasing GIL per sample, the bridge **batches operations**:

```cpp
// Per audio frame (512 samples @ 48kHz):
// - 1 GIL acquire/release
// - 1 Python method call
// - 1 memcpy operation
```

This is tracked via optional profiling:

```cpp
// Enable detailed profiling
NDA_PROFILE_PYBRIDGE=1 ./nda_app

// Output includes:
// [PythonBridgeProfile] read(avgUs total/gil/buf/py/copy)=20.5/2.1/0.3/15.2/2.8
```

Where:
- `total`: Total time for read_audio
- `gil`: GIL acquire/release overhead
- `buf`: Buffer creation/management
- `py`: Python method execution
- `copy`: Data copying (memcpy)

---

## Complete Loading Flow

Here's the full sequence from application start to audio playing:

```
1. User calls: PluginManager::loadPlugin("sine_wave_source.py")
    ↓
2. PluginManager detects .py extension
    ↓
3. PluginManager::loadPythonPlugin()
    ├─ Extract filename: "sine_wave_source.py"
    └─ Extract directory: "plugins_py"
    ↓
4. PythonPluginFactory::createPlugin()
    └─→ new PythonPluginBridge()
    ↓
5. PythonPluginBridge::loadPlugin()
    ├─ Add plugin directory to sys.path
    │   sys.path.insert(0, "/path/to/plugins_py")
    │
    ├─ importlib.import_module("sine_wave_source")
    │   → Finds sine_wave_source.py
    │   → Executes module code
    │   → Creates module object
    │
    ├─ Call module.create_plugin()
    │   → Returns SineWaveSourcePlugin() instance
    │
    └─ Validate plugin instance
        ├─ Has read_audio() method? ✓
        ├─ Has initialize() method? ✓
        └─ Is AudioSourcePlugin? ✓
    ↓
6. PythonPluginBridge::initialize()
    ├─ Store Python module reference
    ├─ Store Python plugin instance reference
    ├─ Call plugin.initialize()  ← Python code runs
    │   └─ Returns True (success)
    │
    └─ initializeCache()
        ├─ Cache base_plugin module
        ├─ Cache AudioBuffer class
        ├─ Cache buffer instance
        ├─ Cache NumPy array
        ├─ Cache direct data pointer
        └─ Cache method references
    ↓
7. PluginManager::registerPlugin()
    └─ Store in plugins_["Sine Wave Generator"] = LoadedPlugin
    ↓
8. User calls: pipeline.setSource(pluginInstance)
    ↓
9. Pipeline::initialize()
    ├─ Call plugin.initialize()  ← Already done, but verified
    ├─ Query sample rate: plugin.get_sample_rate() → 48000
    ├─ Query channels: plugin.get_channel_count() → 2
    └─ Query buffer size: plugin.get_buffer_size() → 512
    ↓
10. User calls: pipeline.start()
    ├─ Call plugin.start()  ← Python code runs
    │  └─ SineWaveSourcePlugin sets state = RUNNING
    │
    └─ Start audio thread
    ↓
11. Audio thread runs in loop:
    ├─ Call plugin.readAudio(buffer)
    │  └─ PythonPluginBridge::readAudio() with caching
    │
    ├─ Create output buffer
    │  └─ Write to speaker/file
    │
    └─ Repeat 48,000 times per second
```

---

## Runtime Audio Flow

Here's what happens during real-time audio processing:

```
ProcessingThread (running at 48 fps, 512 samples per frame = 10.67ms per frame):

    loop {
        1. Source readAudio(buffer)
            ├─ If Python: PythonPluginBridge::readAudio()
            │   ├─ Acquire GIL once
            │   ├─ Get/create cached buffer
            │   ├─ Call cached plugin.read_audio(buffer) method
            │   │   └─ Python generates sine wave
            │   ├─ Fast memcpy from NumPy to C++ buffer
            │   ├─ Release GIL
            │   └─ Return 512 frames of audio
            │
            └─ If C++: Direct C++ plugin call (no bridge overhead)

        2. Processor processAudio(buffer)  [optional, v2.0+]
            ├─ If Python: PythonPluginBridge::processAudio()
            │   ├─ Acquire GIL
            │   ├─ Create Python buffer with NumPy array
            │   ├─ Fast memcpy from C++ to NumPy
            │   ├─ Call cached plugin.process_audio(buffer) method
            │   │   └─ Python applies gain or encryption
            │   ├─ Fast memcpy from NumPy back to C++
            │   ├─ Release GIL
            │   └─ Return modified audio
            │
            └─ If C++: Direct C++ plugin call

        3. Sink writeAudio(buffer)
            ├─ If Python: PythonPluginBridge::writeAudio()
            │   ├─ Acquire GIL
            │   ├─ Create Python buffer with NumPy array
            │   ├─ Fast memcpy from C++ to NumPy
            │   ├─ Call cached plugin.write_audio(buffer) method
            │   │   └─ Python writes to file or speaker
            │   ├─ Release GIL
            │   └─ Return success
            │
            └─ If C++: Direct C++ plugin call

        // Repeat every 10.67ms
        wait for next frame
    }
```

### Timing Breakdown (per frame)

At 48kHz with 512-sample buffer (one frame every 10.67ms):

| Operation | Time | % of Frame |
|-----------|------|-----------|
| Source readAudio (Python cached) | 20μs | 0.2% |
| Processor processAudio (if enabled) | 15μs | 0.1% |
| Sink writeAudio (Python cached) | 25μs | 0.2% |
| Total bridge overhead | ~60μs | 0.6% |
| **Available for plugin logic** | ~10,600μs | 99.4% |

This shows the caching optimization is extremely effective.

---

## Key Optimizations

| Optimization | v1.0 Impact | v2.0/v2.1 Impact | Implementation |
|---|---|---|---|
| **Object Caching** | Not available | 6-30x | Cache module, class, methods, NumPy array |
| **Direct Pointer** | Per-frame lookup | Eliminate attribute lookup | `cachedDataPtr_` = direct NumPy data pointer |
| **Batch GIL** | Per-sample lock/unlock | Single per-frame | Acquire/release once per audio frame |
| **Fast memcpy** | Per-element access | Bulk transfer | `std::memcpy` instead of element-by-element |
| **Buffer Reuse** | Allocate every frame | Allocate only on size change | Recreate buffer only when dimensions change |
| **Cython Support** | Pure Python only | 10-50x speedup available | SHA256-based cache invalidation |
| **512-sample buffers** | Variable sizes | Zero-copy pipeline | Enforce buffer size matches pipeline |

---

## Concrete Examples

### Example 1: Simple Gain Processor

Here's a complete working processor plugin:

```python
# plugins_py/examples/simple_gain.py
"""
Simple Gain Processor Plugin
Adjusts audio volume/gain with parameter control.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from base_plugin import AudioProcessorPlugin, PluginInfo, PluginType, PluginState, AudioBuffer


class SimpleGainPlugin(AudioProcessorPlugin):
    """
    Basic volume/gain adjustment processor.

    Demonstrates parameter handling and in-place audio processing.
    Supports gain values from 0.0 (mute) to 2.0 (200% volume).
    """

    def __init__(self):
        super().__init__()
        self.state = PluginState.UNLOADED
        self.gain = 1.0  # Unity gain (no change) by default
        self.sample_rate = 48000
        self.channels = 2
        self.frames_processed = 0

    def initialize(self) -> bool:
        """Initialize processor"""
        self.state = PluginState.INITIALIZED
        self.frames_processed = 0
        print(f"[SimpleGainPlugin] Initialized at {self.sample_rate}Hz, "
              f"{self.channels} channels, gain={self.gain}")
        return True

    def start(self) -> bool:
        """Start processing"""
        if self.state != PluginState.INITIALIZED:
            return False
        self.state = PluginState.RUNNING
        print(f"[SimpleGainPlugin] Started with gain={self.gain}")
        return True

    def stop(self):
        """Stop processing"""
        if self.state == PluginState.RUNNING:
            self.state = PluginState.INITIALIZED
            seconds = self.frames_processed / self.sample_rate
            print(f"[SimpleGainPlugin] Stopped after processing {self.frames_processed} frames ({seconds:.2f}s)")

    def shutdown(self):
        """Shutdown processor"""
        self.state = PluginState.UNLOADED
        print("[SimpleGainPlugin] Shutdown")

    def process_audio(self, buffer: AudioBuffer) -> bool:
        """
        Apply gain to audio buffer in-place.

        Args:
            buffer: AudioBuffer to process (modified in-place)

        Returns:
            True on success, False on error
        """
        if self.state != PluginState.RUNNING:
            return False

        try:
            # Apply gain to all samples
            buffer.data *= self.gain

            # Clip to prevent overflow (hard limiting)
            np.clip(buffer.data, -1.0, 1.0, out=buffer.data)

            self.frames_processed += buffer.get_frame_count()
            return True

        except Exception as e:
            print(f"[SimpleGainPlugin] Error processing audio: {e}")
            return False

    def get_info(self) -> PluginInfo:
        """Get plugin information"""
        return PluginInfo(
            name="Simple Gain",
            version="1.0.0",
            author="NDA Team",
            description="Basic volume/gain adjustment processor",
            plugin_type=PluginType.PROCESSOR,
            api_version=1
        )

    def get_type(self) -> PluginType:
        """Get plugin type"""
        return PluginType.PROCESSOR

    def get_state(self) -> PluginState:
        """Get current state"""
        return self.state

    def get_sample_rate(self) -> int:
        """Get sample rate"""
        return self.sample_rate

    def get_channel_count(self) -> int:
        """Get channel count"""
        return self.channels

    def set_sample_rate(self, rate: int) -> None:
        """Set sample rate"""
        self.sample_rate = rate

    def set_channel_count(self, channels: int) -> None:
        """Set channel count"""
        self.channels = channels

    def set_parameter(self, key: str, value: str) -> bool:
        """
        Set plugin parameter.

        Supported parameters:
            - "gain": float value (0.0 to 2.0)
        """
        if key == "gain":
            try:
                new_gain = float(value)
                if 0.0 <= new_gain <= 2.0:
                    self.gain = new_gain
                    print(f"[SimpleGainPlugin] Gain set to {self.gain}")
                    return True
                else:
                    print(f"[SimpleGainPlugin] Invalid gain value: {new_gain} (must be 0.0-2.0)")
                    return False
            except ValueError:
                print(f"[SimpleGainPlugin] Invalid gain format: {value}")
                return False

        return False

    def get_parameter(self, key: str) -> str:
        """Get plugin parameter"""
        if key == "gain":
            return str(self.gain)
        return ""


def create_plugin():
    """Factory function called by plugin loader"""
    return SimpleGainPlugin()
```

**Usage from C++:**

```cpp
// Load plugin
auto plugin = loadPlugin("simple_gain.py");

// Initialize
plugin->initialize();
plugin->start();

// Adjust gain during playback
plugin->setParameter("gain", "0.5");  // Reduce to 50% volume

// Audio processing happens automatically in background thread
// Each frame: buffer.data *= 0.5 (in-place modification)
// With NumPy: extremely fast operation

// Stop
plugin->stop();
plugin->shutdown();
```

### Example 2: Sine Wave Generator

From `plugins_py/sine_wave_source.py`:

```python
class SineWaveSourcePlugin(AudioSourcePlugin):
    """Generates a sine wave (default 440Hz A4 note)"""

    def __init__(self):
        super().__init__()
        self.sample_rate = 48000
        self.channel_count = 2
        self.buffer_size = 512  # CRITICAL: Must be 512
        self.frequency = 440.0  # A4 note
        self.phase = 0.0
        self.callback = None

        # Pre-allocate work buffers for fast NumPy operations
        self._offsets = None
        self._phase_work = None
        self._samples = None

    def read_audio(self, buffer: AudioBuffer) -> bool:
        """Read audio data - generates sine wave"""
        if self.state != PluginState.RUNNING:
            buffer.clear()
            return False

        frame_count = int(buffer.get_frame_count())
        if frame_count <= 0:
            return True

        # Ensure work buffers are correct size
        self._ensure_work_buffers(frame_count)

        # Calculate phase increment per sample
        phase_increment = (math.tau * float(self.frequency)) / float(self.sample_rate)
        phase0 = float(self.phase)

        # Use NumPy for fast vectorized sine generation
        phase_increment_f32 = np.float32(phase_increment)
        phase0_f32 = np.float32(phase0)

        # Phase offsets for all frames: phase = phase0 + [0,1,2,...] * phase_increment
        np.multiply(self._offsets, phase_increment_f32, out=self._phase_work)
        self._phase_work += phase0_f32

        # Calculate sine for all frames at once (vectorized)
        np.sin(self._phase_work, out=self._samples)
        self._samples *= np.float32(0.5)  # Amplitude 0.5

        # Copy to output buffer (all channels)
        channels = int(buffer.get_channel_count())
        buffer.data[:channels, :frame_count] = self._samples

        # Advance phase robustly
        self.phase = (phase0 + (frame_count * phase_increment)) % math.tau

        return True
```

This demonstrates:
- Pre-allocated NumPy buffers for performance
- Vectorized sine generation (very fast)
- Proper buffer size handling (512 samples)
- Phase management for continuous tone generation

---

## Buffer Size Requirements

### Critical v2.1 Requirement

**All audio source plugins MUST use 512-sample buffers:**

```python
def __init__(self):
    self.buffer_size = 512  # MANDATORY in v2.1

def get_buffer_size(self) -> int:
    return self.buffer_size  # Always return 512
```

### Why 512 Samples?

- **Pipeline internal buffer**: Uses 512-sample blocks internally
- **Mismatched sizes cause**: Queue starvation, timing issues, latency spikes
- **Enforced at**: Pipeline initialization
- **Sink flexibility**: Sinks can accept any size (they rebuffer internally)

### Implementation

If your audio library uses different buffer sizes:

```python
class MicrophonePlugin(AudioSourcePlugin):
    def __init__(self):
        self.buffer_size = 512       # Expose as 512
        self.internal_buffer = []    # Internal queue for rebuffering

    def read_audio(self, buffer: AudioBuffer) -> bool:
        # Your library may produce 256 or 1024 samples at a time
        # Rebuffer to 512 samples for the pipeline

        while len(self.internal_buffer) < 512:
            raw_samples = self.audio_device.read()  # May be 256 or 1024
            self.internal_buffer.extend(raw_samples)

        # Extract 512 samples
        samples_512 = self.internal_buffer[:512]
        self.internal_buffer = self.internal_buffer[512:]

        # Fill output buffer
        buffer.data[:, :512] = samples_512
        return True
```

---

## Error Handling & State Machine

### Plugin State Machine

Every plugin has a lifecycle:

```python
class PluginState(Enum):
    UNLOADED = "Unloaded"         # Just created
    LOADED = "Loaded"              # Module loaded
    INITIALIZED = "Initialized"    # initialize() succeeded
    RUNNING = "Running"            # start() succeeded, active
    ERROR = "Error"                # Something failed
```

### State Transitions

```
UNLOADED ─initialize()→ INITIALIZED ─start()→ RUNNING
           (success)                    (success)
             ↓                            ↓
           ERROR                        ERROR

From RUNNING:
    stop() → INITIALIZED
                  ↓
            shutdown() → UNLOADED
```

### Proper State Checking

Methods must verify state before operating:

```python
def read_audio(self, buffer: AudioBuffer) -> bool:
    # Verify we're in the right state
    if self.state != PluginState.RUNNING:
        buffer.clear()  # Return silence on error
        return False

    # Only read if running
    return self._read_impl(buffer)

def process_audio(self, buffer: AudioBuffer) -> bool:
    # Verify we're in the right state
    if self.state != PluginState.RUNNING:
        return False  # Pipeline will passthrough

    # Only process if running
    return self._process_impl(buffer)
```

### Error Recovery

The pipeline handles plugin errors gracefully:

```cpp
// From ProcessingPipeline
if (!processor->processAudio(buffer)) {
    // Plugin returned False, likely an error
    // Pipeline passes audio through unmodified
    // No exception, continues running
}
```

Plugins should:
1. Log errors internally
2. Return False to indicate failure
3. Maintain valid state for recovery
4. Avoid throwing exceptions

---

## Key Files Reference

| File | Lines | Purpose |
|------|-------|---------|
| `include/plugins/PythonPluginBridge.h` | 170 | C++ bridge header with cache definitions |
| `src/plugins/PythonPluginBridge.cpp` | 1300+ | Bridge implementation (highly optimized) |
| `include/plugins/PluginManager.h` | ~50 | Plugin loader interface |
| `src/plugins/PluginManager.cpp` | ~200 | Plugin loading logic (C++ and Python) |
| `plugins_py/base_plugin.py` | 345 | Python base classes and enums |
| `plugins_py/cython_compiler.py` | ~300 | Optional Cython compilation with SHA256 caching |
| `plugins_py/examples/simple_gain.py` | 193 | Example processor plugin |
| `plugins_py/sine_wave_source.py` | 158 | Example source plugin |
| `plugins_py/pulseaudio_speaker.py` | ~100 | Example sink plugin |
| `include/core/ProcessingPipeline.h` | ~150 | Pipeline that uses plugins |
| `src/audio/AudioEngine.cpp` | ~400 | Audio thread that calls pipeline |

---

## Performance Characteristics

### Measured Performance (v2.0+)

From profiling data in the codebase:

```
[PythonBridgeProfile] Enabled (interval 1000ms)
[PythonBridgeProfile:sine_wave_source] dt=1000.00ms calls=48000/0/0
read(avgUs total/gil/buf/py/copy)=20.5/2.1/0.3/15.2/2.8
maxUs(totalR/pyR)=156/149
errors(r/w/a)=0/0/0
```

**Breakdown per frame (512 samples):**

| Component | Time | % of 10.67ms frame |
|-----------|------|-------------------|
| GIL acquire/release | 2.1μs | 0.02% |
| Buffer creation | 0.3μs | <0.01% |
| Python method call | 15.2μs | 0.14% |
| Data copying | 2.8μs | 0.03% |
| **Total overhead** | **20.5μs** | **0.19%** |
| **Available for plugin** | ~10,646μs | **99.81%** |

This confirms the bridge has minimal overhead relative to available CPU time.

### Bottleneck Analysis

In order of importance:

1. **Plugin logic** (depends on implementation)
   - Sine generation: Very fast (vectorized NumPy)
   - Encryption: Moderate (depends on algorithm)
   - File I/O: Slow (depends on storage)

2. **GIL overhead** (2-3μs)
   - Profiled separately
   - Minimal with caching

3. **NumPy operations** (fast memcpy)
   - Typically <1μs per frame @ 512 samples
   - Optimized with direct pointer access

4. **Python method invocation** (cached to avoid lookup)
   - Minimal with cached method references

---

## Conclusion

The NDA Python bridge architecture demonstrates sophisticated C++/Python interoperability with careful attention to:

1. **Performance**: 6-30x faster than naive implementation via aggressive caching
2. **Thread safety**: Proper GIL management for multi-threaded audio processing
3. **Memory safety**: Smart pointers, proper cleanup, no leaks
4. **Flexibility**: Supports source, processor, and sink plugins
5. **Reliability**: Graceful error handling, state machine, fallback modes
6. **Developer experience**: Simple plugin API, comprehensive examples, clear documentation

The bridge is production-ready and can handle real-time audio processing at 48kHz with minimal latency impact.
