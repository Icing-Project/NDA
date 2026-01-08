# NDA v2.0 - Real-Time Audio Encryption Bridge
## Modular Audio Processing System

---

## Executive Summary

NDA (Nade Desktop Application) is a **real-time audio encryption bridge** designed to provide transparent encryption/decryption between any audio devices. Built with C++17 and Qt6, it enables secure audio communication by sitting between audio endpoints—encrypting outbound audio and decrypting inbound audio through a clean, modular plugin architecture.

### Core Use Case

```
┌─────────────────────────────────────────────────────────────┐
│  TRANSMIT (TX) Pipeline                                     │
│  Device Microphone → Encryptor → AIOC Sink                  │
│                                                             │
│  RECEIVE (RX) Pipeline                                      │
│  AIOC Source → Decryptor → Device Speaker                   │
└─────────────────────────────────────────────────────────────┘
```

**NDA handles audio transformation only—no network, no transport layer.**

The external transport (AIOC hardware, Discord, VoIP software) handles actual data transmission. NDA provides the encryption/decryption layer that sits transparently in front of these services.

### Key Features

- **Dual Independent Pipelines**: Simultaneous TX and RX processing
- **Universal Audio Compatibility**: Works with any audio device or virtual cable
- **Plugin-Based Encryption**: Encryption logic lives in plugins, not core
- **Automatic Sample Rate Handling**: Pipeline manages 48kHz normalization and adaptation
- **Python & C++ Plugins**: Equal support for rapid prototyping and production performance
- **Modular 3-Stage Architecture**: Source → Processor → Sink
- **Stable Long-Running**: Designed for hours of glitch-free operation

### Non-Goals

- ❌ Network transport (use external tools: AIOC, VB-Cable, Discord, etc.)
- ❌ Sub-5ms latency guarantee (target: <50ms end-to-end)
- ❌ DAW-grade audio processing (focus: communication, not music production)
- ❌ Built-in encryption algorithms (all encryption is plugin-provided)

---

## Architecture Overview

### System Design Philosophy

**NDA is an audio transform engine, not a communication stack.**

The core responsibility is moving audio buffers through a processing chain:
```
Read from Source → Transform (optional) → Write to Sink
```

Everything else—encryption algorithms, device drivers, network protocols—is delegated to plugins or external systems.

### Dual Pipeline Model

NDA runs **two independent pipelines simultaneously**:

#### TX Pipeline (Transmit)
```
Local Microphone → [Encryptor Plugin] → Output Device/Cable
                        ↓
               Encrypted audio sent to external transport
```

#### RX Pipeline (Receive)
```
Input Device/Cable → [Decryptor Plugin] → Local Speaker
        ↑
External transport delivers encrypted audio
```

**Each pipeline is a separate `ProcessingPipeline` instance with its own thread.**

### Three-Slot Architecture

Each pipeline has exactly **three slots**:

```
┌──────────┐    ┌───────────┐    ┌──────────┐
│  SOURCE  │ ─→ │ PROCESSOR │ ─→ │   SINK   │
│ (INPUT)  │    │ (OPTIONAL)│    │ (OUTPUT) │
└──────────┘    └───────────┘    └──────────┘
```

**Slot 1: Source (Required)**
- Audio input provider
- Examples: Microphone, AIOC Input, File, Sine Generator

**Slot 2: Processor (Optional)**
- Audio transformation
- Examples: Encryptor, Decryptor, Resampler, EQ, Compressor
- **Can be empty** for direct passthrough

**Slot 3: Sink (Required)**
- Audio output consumer
- Examples: Speaker, AIOC Output, File, Null (monitor)

### Sample Rate Adaptation

**All audio processing happens at 48kHz internally** (configurable).

The pipeline core handles sample rate conversion:
- If source provides 44.1kHz → pipeline upsamples to 48kHz
- If sink expects 96kHz → pipeline upsamples from 48kHz
- Processors always receive 48kHz buffers

**Resampling Strategy:**
- **Simple (default)**: Linear interpolation for small mismatches
- **Quality**: Windowed sinc resampler (libsamplerate) for large mismatches
- **Plugin**: User can insert explicit resampler processor for control

Buffer size mismatches are handled via:
- **Padding**: Silence added to fill required buffer size
- **Chunking**: Large buffers split across multiple iterations
- **Accumulation**: Small buffers accumulated until threshold reached

---

## Technology Stack

### Core Technologies

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **Language** | C++17 | Performance, direct OS API access |
| **GUI Framework** | Qt6 Widgets | Cross-platform, native performance |
| **Build System** | CMake 3.16+ | Standard, cross-platform |
| **Audio I/O** | WASAPI (Win), ALSA (Linux) | Low-latency platform APIs |
| **Threading** | std::thread | Deterministic lifecycle |

### Plugin Technologies

| Type | C++ Plugins | Python Plugins |
|------|-------------|----------------|
| **Purpose** | Production performance | Rapid prototyping |
| **Latency** | <5ms overhead | 10-20ms overhead |
| **Loading** | Dynamic library (DLL/SO) | Python interpreter bridge |
| **Examples** | AIOC, WAV File, Native devices | Sounddevice, Test generators |

### Removed from Core

- ❌ OpenSSL direct integration (moved to plugin examples)
- ❌ Bearer/Network abstractions (deleted entirely)
- ❌ Built-in crypto classes (plugins provide encryption)

---

## Plugin Architecture

### Plugin Types (Simplified)

```cpp
enum class PluginType {
    AudioSource,    // Input: Microphone, File, Network receiver
    AudioSink,      // Output: Speaker, File, Network sender  
    Processor       // Transform: Encryptor, Decryptor, Effects
};
```

**Removed:** `Bearer`, `Encryptor` (merged into `Processor`)

### Plugin Lifecycle

```
Unloaded → Loaded → Initialized → Running → Stopped → Shutdown
```

**State Transitions:**
- `Unloaded`: Not yet discovered
- `Loaded`: DLL/module loaded, factory called
- `Initialized`: `initialize()` succeeded, ready to start
- `Running`: `start()` succeeded, actively processing
- `Error`: Any failure occurred

### Base Plugin Interface

```cpp
class BasePlugin {
public:
    // Lifecycle
    virtual bool initialize() = 0;
    virtual void shutdown() = 0;
    virtual bool start() = 0;
    virtual void stop() = 0;
    
    // Metadata
    virtual PluginInfo getInfo() const = 0;
    virtual PluginState getState() const = 0;
    
    // Configuration
    virtual bool setParameter(const std::string& key, const std::string& value) = 0;
    virtual std::string getParameter(const std::string& key) const = 0;
};
```

### Audio Source Plugin

```cpp
class AudioSourcePlugin : public BasePlugin {
public:
    virtual bool readAudio(AudioBuffer& buffer) = 0;
    
    virtual int getSampleRate() const = 0;
    virtual int getChannelCount() const = 0;
    virtual int getBufferSize() const = 0;
    
    virtual void setSampleRate(int rate) = 0;
    virtual void setChannelCount(int channels) = 0;
    virtual void setBufferSize(int frames) = 0;
};
```

### Audio Sink Plugin

```cpp
class AudioSinkPlugin : public BasePlugin {
public:
    virtual bool writeAudio(const AudioBuffer& buffer) = 0;
    
    virtual int getSampleRate() const = 0;
    virtual int getChannelCount() const = 0;
    virtual int getBufferSize() const = 0;
    virtual int getAvailableSpace() const = 0;  // For backpressure
    
    virtual void setSampleRate(int rate) = 0;
    virtual void setChannelCount(int channels) = 0;
    virtual void setBufferSize(int frames) = 0;
};
```

### Audio Processor Plugin (New)

```cpp
class AudioProcessorPlugin : public BasePlugin {
public:
    // Process audio in-place
    virtual bool processAudio(AudioBuffer& buffer) = 0;
    
    // Metadata
    virtual int getSampleRate() const = 0;
    virtual int getChannelCount() const = 0;
    
    // Configuration
    virtual void setSampleRate(int rate) = 0;
    virtual void setChannelCount(int channels) = 0;
    
    // Processors can declare latency they add
    virtual double getProcessingLatency() const { return 0.0; }
};
```

**Encryptor and Decryptor are just processors:**
```cpp
class AES256EncryptorPlugin : public AudioProcessorPlugin {
    // Encrypts buffer in-place
    bool processAudio(AudioBuffer& buffer) override;
};

class AES256DecryptorPlugin : public AudioProcessorPlugin {
    // Decrypts buffer in-place
    bool processAudio(AudioBuffer& buffer) override;
};
```

---

## Processing Pipeline (Redesigned)

### Pipeline Structure

```cpp
class ProcessingPipeline {
public:
    // Simplified configuration (3 slots)
    bool setSource(std::shared_ptr<AudioSourcePlugin> source);
    bool setProcessor(std::shared_ptr<AudioProcessorPlugin> processor);
    bool setSink(std::shared_ptr<AudioSinkPlugin> sink);
    
    // Lifecycle
    bool initialize();
    bool start();
    void stop();
    void shutdown();
    
    // Monitoring
    bool isRunning() const;
    uint64_t getProcessedSamples() const;
    uint64_t getDroppedSamples() const;      // New: failed writes
    double getActualLatency() const;          // New: measured, not estimated
    float getActualCPULoad() const;           // New: measured, not hardcoded
    
private:
    void processingThread();
    void processAudioFrame();
    
    std::shared_ptr<AudioSourcePlugin> source_;
    std::shared_ptr<AudioProcessorPlugin> processor_;  // Optional
    std::shared_ptr<AudioSinkPlugin> sink_;
    
    AudioBuffer workBuffer_;
    
    // Sample rate adaptation
    int targetSampleRate_;      // Pipeline internal rate (48kHz default)
    Resampler sourceResampler_; // Source rate → target rate
    Resampler sinkResampler_;   // Target rate → sink rate
};
```

### Processing Loop (Fixed)

```cpp
void ProcessingPipeline::processAudioFrame() {
    // 1. Read from source
    if (!source_->readAudio(workBuffer_)) {
        consecutiveFailures_++;
        if (consecutiveFailures_ > 10) {
            droppedSamples_ += workBuffer_.getFrameCount();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return;
    }
    consecutiveFailures_ = 0;
    
    // 2. Resample source → 48kHz if needed
    if (source_->getSampleRate() != targetSampleRate_) {
        sourceResampler_.process(workBuffer_);
    }
    
    // 3. Apply processor (encryptor/decryptor/effects)
    if (processor_ && processor_->getState() == PluginState::Running) {
        if (!processor_->processAudio(workBuffer_)) {
            // Processor failed, but continue (passthrough)
            processorFailures_++;
        }
    }
    
    // 4. Resample 48kHz → sink if needed
    if (sink_->getSampleRate() != targetSampleRate_) {
        sinkResampler_.process(workBuffer_);
    }
    
    // 5. Check backpressure
    if (sink_->getAvailableSpace() < workBuffer_.getFrameCount()) {
        // Sink queue full, wait briefly
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    
    // 6. Write to sink
    if (sink_->writeAudio(workBuffer_)) {
        processedSamples_ += workBuffer_.getFrameCount();
    } else {
        droppedSamples_ += workBuffer_.getFrameCount();
    }
    
    // 7. Pacing: sleep to maintain real-time cadence
    auto targetTime = startTime_ + std::chrono::microseconds(
        (processedSamples_ * 1000000) / targetSampleRate_
    );
    auto now = std::chrono::steady_clock::now();
    
    if (now < targetTime) {
        std::this_thread::sleep_until(targetTime);
    } else {
        // We're behind schedule, log drift
        auto drift = std::chrono::duration_cast<std::chrono::milliseconds>(now - targetTime);
        if (drift.count() > 50) {
            // More than 50ms behind, warn
            driftWarnings_++;
        }
    }
}
```

**Key Improvements:**
- ✅ Real-time pacing (sleep to maintain cadence)
- ✅ Backpressure handling (check sink space)
- ✅ Automatic resampling (source/sink rate mismatch)
- ✅ Accurate sample counting (only increment on success)
- ✅ Failure tracking (dropped samples, processor failures)

---

## Python Plugin Bridge (Optimized)

### Current Performance Problem

**Per-buffer overhead: 3-15ms** due to:
- Fresh Python object allocation every call
- Element-by-element data copying
- Repeated module imports
- GIL acquisition per call

### Optimization Strategy

#### 1. Object Caching
```cpp
class PythonPluginBridge {
private:
    // Cache Python AudioBuffer objects
    PyObject* cachedPyBuffer_;
    
    // Cache NumPy array objects
    PyArrayObject* cachedNumpyArray_;
    
    // Reuse instead of recreating
    void updateCachedBuffer(const AudioBuffer& buffer);
};
```

#### 2. Zero-Copy Data Sharing
```cpp
// Instead of copying element-by-element:
// BAD (current):
for (int c = 0; c < channels; ++c) {
    for (int f = 0; f < frames; ++f) {
        *pyData++ = cppBuffer[c][f];  // Slow!
    }
}

// GOOD (optimized):
// Share C++ memory directly with NumPy
npy_intp dims[2] = {channels, frames};
PyObject* array = PyArray_SimpleNewFromData(
    2, dims, NPY_FLOAT32, cppBuffer.data()
);
```

#### 3. Batch GIL Operations
```cpp
// BAD: Acquire/release GIL multiple times per frame
auto state1 = PyGILState_Ensure();
// ... read metadata
PyGILState_Release(state1);

auto state2 = PyGILState_Ensure();
// ... process audio
PyGILState_Release(state2);

// GOOD: Hold GIL for entire frame
auto state = PyGILState_Ensure();
// ... all Python operations
PyGILState_Release(state);
```

#### 4. Module Import Caching
```cpp
// BAD: Import base_plugin every call
PyObject* module = PyImport_ImportModule("base_plugin");

// GOOD: Import once, cache reference
if (!cachedBasePluginModule_) {
    cachedBasePluginModule_ = PyImport_ImportModule("base_plugin");
    Py_INCREF(cachedBasePluginModule_);  // Keep alive
}
```

**Expected improvement: 3-15ms → 0.5-2ms per buffer**

---

## User Interface (Dual Pipeline)

### New Layout

```
┌─────────────────────────────────────────────────────────────┐
│  NDA - Real-Time Audio Encryption Bridge                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  TRANSMIT Pipeline (TX)                             │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │  Source:    [Device Microphone          ▼]         │   │
│  │  Processor: [AES-256 Encryptor          ▼]         │   │
│  │  Sink:      [AIOC Output                ▼]         │   │
│  │                                                     │   │
│  │  Status: 🟢 Running  │  Latency: 23ms  │ CPU: 8%  │   │
│  │  [■ Stop TX]                                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  RECEIVE Pipeline (RX)                              │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │  Source:    [AIOC Input                 ▼]         │   │
│  │  Processor: [AES-256 Decryptor          ▼]         │   │
│  │  Sink:      [Device Speaker             ▼]         │   │
│  │                                                     │   │
│  │  Status: 🟢 Running  │  Latency: 19ms  │ CPU: 7%  │   │
│  │  [■ Stop RX]                                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  [▶ Start Both]  [■ Stop Both]  [📁 Load Plugins]        │
└─────────────────────────────────────────────────────────────┘
```

### UI Components

**PipelineView** (redesigned):
- Two separate pipeline configuration sections
- Each has 3 combo boxes (Source, Processor, Sink)
- Independent status displays
- Individual and combined start/stop controls

**Dashboard** (enhanced):
- Dual pipeline metrics side-by-side
- Real-time level meters for all 4 audio paths
- Latency/CPU graphs over time
- Error/warning log viewer

**SettingsView** (expanded):
- Global sample rate (default 48kHz)
- Buffer size (128, 256, 512, 1024 frames)
- Resampling quality (Simple, Medium, High)
- Plugin directories configuration

---

## Example Configurations

### Configuration 1: Encrypted Voice to AIOC Radio

**TX Pipeline:**
```
Local Microphone → AES-256 Encryptor → AIOC USB Output
```

**RX Pipeline:**
```
AIOC USB Input → AES-256 Decryptor → Local Speaker
```

**Use Case:** Secure two-way radio communication. AIOC hardware handles RF transmission; NDA provides encryption layer.

### Configuration 2: Encrypted Discord/VoIP

**TX Pipeline:**
```
Local Microphone → AES-256 Encryptor → VB-Cable Input
```
*(Discord reads from VB-Cable)*

**RX Pipeline:**
```
VB-Cable Output → AES-256 Decryptor → Local Speaker
```
*(Discord writes to VB-Cable)*

**Use Case:** Add encryption to any voice chat app using virtual audio cables.

### Configuration 3: Encrypted File Recording/Playback

**TX Pipeline:**
```
Microphone → AES-256 Encryptor → WAV File Sink
```

**RX Pipeline:**
```
WAV File Source → AES-256 Decryptor → Speaker
```

**Use Case:** Record encrypted audio for later playback.

### Configuration 4: Passthrough Testing (No Encryption)

**TX Pipeline:**
```
Sine Generator → (None) → Null Sink
```

**RX Pipeline:**
```
File Source → (None) → Speaker
```

**Use Case:** Test audio quality and latency without encryption overhead.

---

## Performance Targets

### Latency Goals

| Configuration | Target Latency | Acceptable Range |
|--------------|----------------|------------------|
| C++ plugins only | <20ms | <30ms |
| Python plugins | <50ms | <80ms |
| With encryption (C++) | <25ms | <40ms |
| With encryption (Py) | <60ms | <100ms |

**Measurement:** End-to-end from source read to sink write, including processing and buffer overhead.

### CPU Usage

| Pipeline State | Target CPU | Maximum |
|---------------|------------|---------|
| Single pipeline (idle source) | <5% | 10% |
| Single pipeline (active) | <10% | 20% |
| Dual pipelines (both active) | <15% | 30% |

**On:** Quad-core 2.5GHz processor (typical laptop)

### Memory Usage

| Component | Target | Maximum |
|-----------|--------|---------|
| Core application | 30MB | 50MB |
| Per pipeline | 10MB | 20MB |
| Per C++ plugin | 2MB | 5MB |
| Per Python plugin | 10MB | 30MB |
| **Total (typical)** | **60MB** | **100MB** |

### Stability

**Success Criteria:**
- ✅ Run dual pipelines continuously for **24 hours** without crash
- ✅ Process **10 million samples** without audio glitches
- ✅ Handle source/sink failures gracefully (no crash)
- ✅ Support start/stop cycles **1000+ times** without memory leaks

---

## Directory Structure (Updated)

```
NDA/
├── src/                         # C++ source files
│   ├── main.cpp                 # Application entry
│   │
│   ├── ui/                      # Qt UI components
│   │   ├── MainWindow.cpp       # Main window with tabs
│   │   ├── PipelineView.cpp     # DUAL pipeline configuration (redesigned)
│   │   ├── Dashboard.cpp        # Live monitoring (dual metrics)
│   │   └── SettingsView.cpp     # Global settings
│   │
│   ├── core/                    # Core processing
│   │   └── ProcessingPipeline.cpp  # Simplified 3-slot pipeline
│   │
│   ├── audio/                   # Audio infrastructure
│   │   ├── AudioBuffer.cpp      # Multi-channel buffer
│   │   ├── AudioDevice.cpp      # Device enumeration (stubs)
│   │   └── Resampler.cpp        # NEW: Sample rate conversion
│   │
│   └── plugins/                 # Plugin system
│       ├── PluginManager.cpp    # Plugin loading/lifecycle
│       └── PythonPluginBridge.cpp  # Optimized Python bridge
│
├── include/                     # Public headers
│   ├── ui/                      # UI headers
│   ├── core/                    # Core headers
│   ├── audio/                   # Audio headers
│   └── plugins/                 # Plugin headers
│       ├── BasePlugin.h         # Base plugin interface
│       ├── AudioSourcePlugin.h
│       ├── AudioSinkPlugin.h
│       ├── AudioProcessorPlugin.h  # NEW: Processor interface
│       └── PluginTypes.h        # Updated enum (no Bearer)
│
├── plugins_src/                 # C++ plugin implementations
│   ├── SineWaveSourcePlugin.cpp
│   ├── WavFileSinkPlugin.cpp
│   ├── NullSinkPlugin.cpp
│   ├── AIOCSourcePlugin.cpp
│   ├── AIOCSinkPlugin.cpp
│   └── examples/                # NEW: Crypto examples moved here
│       ├── AES256EncryptorPlugin.cpp
│       └── AES256DecryptorPlugin.cpp
│
├── plugins_py/                  # Python plugin implementations
│   ├── base_plugin.py           # Python plugin contracts
│   ├── sine_wave_source.py
│   ├── sounddevice_microphone.py
│   ├── sounddevice_speaker.py
│   ├── wav_file_sink.py
│   ├── null_sink.py
│   └── examples/                # NEW: Crypto examples
│       ├── aes256_encryptor.py
│       └── aes256_decryptor.py
│
├── docs/                        # Documentation
│   ├── NDA-SPECS-v2.md          # This document
│   ├── MIGRATION_GUIDE.md       # NEW: v1 → v2 migration
│   ├── PLUGIN_DEVELOPMENT.md    # Plugin authoring guide
│   └── ARCHITECTURE_REPORT.md   # Updated architecture doc
│
├── scripts/                     # Build and deployment
│   ├── build_windows.bat
│   ├── deploy.py
│   └── migrate_v1_to_v2.py      # NEW: Automated migration script
│
├── tests/                       # Testing (future)
│   ├── test_pipeline.cpp
│   ├── test_resampling.cpp
│   └── test_python_bridge.cpp
│
├── CMakeLists.txt               # Build configuration
└── README.md                    # Getting started

REMOVED:
  ❌ include/crypto/              # Moved to plugin examples
  ❌ src/crypto/                  # Deleted
  ❌ include/plugins/BearerPlugin.h
  ❌ include/plugins/EncryptorPlugin.h  # Now just Processor
```

---

## Implementation Roadmap

### Phase 1: Core Refactoring (Week 1)

**Tasks:**
1. ✅ Delete Bearer plugin type and all references
2. ✅ Remove `include/crypto/` and `src/crypto/`
3. ✅ Create `AudioProcessorPlugin` interface
4. ✅ Simplify `ProcessingPipeline` to 3 slots
5. ✅ Remove hardcoded encryption logic from pipeline
6. ✅ Update `PluginTypes.h` (remove Bearer, Encryptor)

**Deliverable:** Clean core that compiles without bearer/crypto

### Phase 2: Sample Rate Adaptation (Week 1-2)

**Tasks:**
1. ✅ Implement `Resampler` class (simple linear interpolation)
2. ✅ Add resampling to `processAudioFrame()`
3. ✅ Integrate libsamplerate (optional, high-quality mode)
4. ✅ Add buffer padding/chunking for size mismatches
5. ✅ Test with mismatched source/sink rates

**Deliverable:** Pipeline handles any sample rate/buffer size

### Phase 3: Python Bridge Optimization (Week 2)

**Tasks:**
1. ✅ Cache Python `AudioBuffer` objects
2. ✅ Implement zero-copy NumPy data sharing
3. ✅ Batch GIL acquisition
4. ✅ Cache module imports
5. ✅ Profile and measure improvement

**Deliverable:** Python overhead reduced to <2ms per buffer

### Phase 4: Real-Time Pacing & Metrics (Week 2-3)

**Tasks:**
1. ✅ Implement sleep-to-target pacing in processing loop
2. ✅ Add backpressure handling (check sink space)
3. ✅ Fix sample counting (only on success)
4. ✅ Add `droppedSamples_` counter
5. ✅ Measure actual CPU load (not hardcoded)
6. ✅ Measure actual latency (not estimated)

**Deliverable:** Accurate, stable pipeline timing

### Phase 5: Dual Pipeline Architecture (Week 3)

**Tasks:**
1. ✅ Create two `ProcessingPipeline` instances in `MainWindow`
2. ✅ Redesign `PipelineView` for dual pipeline UI
3. ✅ Wire independent start/stop controls
4. ✅ Add combined "Start Both" / "Stop Both" buttons
5. ✅ Update `Dashboard` to show dual metrics

**Deliverable:** Full TX/RX dual pipeline support

### Phase 6: Crypto Plugin Examples (Week 3-4)

**Tasks:**
1. ✅ Move old `Encryptor.h/cpp` to `plugins_src/examples/`
2. ✅ Implement `AES256EncryptorPlugin` (processor plugin)
3. ✅ Implement `AES256DecryptorPlugin` (processor plugin)
4. ✅ Integrate OpenSSL EVP API (real encryption)
5. ✅ Add Python crypto examples (Fernet, ChaCha20)
6. ✅ Document key exchange (out of band)

**Deliverable:** Working encryption/decryption plugins

### Phase 7: Testing & Stability (Week 4)

**Tasks:**
1. ✅ 24-hour soak test (dual pipelines)
2. ✅ Memory leak detection (Valgrind, sanitizers)
3. ✅ Error injection testing (disconnect sources mid-run)
4. ✅ Performance profiling (CPU, latency graphs)
5. ✅ Cross-platform validation (Windows & Linux)

**Deliverable:** Production-ready stability

### Phase 8: Documentation & Release (Week 4)

**Tasks:**
1. ✅ Update `README.md` with v2 goals
2. ✅ Write `MIGRATION_GUIDE.md` (v1 → v2)
3. ✅ Update `PLUGIN_DEVELOPMENT.md` for processors
4. ✅ Create example configurations (encrypted AIOC, Discord)
5. ✅ Package v2.0.0 release

**Deliverable:** Public v2.0 release

---

## Migration from v1.x

### Breaking Changes

**1. Bearer Plugin Removed**
- **Old:** `pipeline->setBearer(bearerPlugin)`
- **New:** Network is external; use AIOC/VB-Cable as sink/source

**2. Encryptor is Now a Processor**
- **Old:** `pipeline->setEncryptor(encryptorPlugin)`  
- **New:** `pipeline->setProcessor(processorPlugin)`

**3. Single Pipeline → Dual Pipeline**
- **Old:** One pipeline instance in UI
- **New:** Two pipelines (TX and RX)

**4. Crypto Moved to Plugins**
- **Old:** `#include "crypto/Encryptor.h"`
- **New:** `#include "plugins/examples/AES256EncryptorPlugin.h"`

### Migration Script

A Python script `scripts/migrate_v1_to_v2.py` will:
- Update `#include` statements
- Rename `setEncryptor` → `setProcessor`
- Remove `setBearer` calls
- Add dual pipeline boilerplate to UI code

**Usage:**
```bash
python scripts/migrate_v1_to_v2.py --input src/ --dry-run
python scripts/migrate_v1_to_v2.py --input src/ --apply
```

---

## FAQ

### Q: Why remove Bearer?

**A:** Bearer was a premature abstraction. NDA's job is audio processing, not network transport. External tools (AIOC hardware, VB-Cable, VoIP apps) already handle transport. Mixing audio processing and network semantics created unnecessary complexity.

### Q: Why not <5ms latency?

**A:** Sub-5ms requires:
- Exclusive-mode ASIO drivers (not universally available)
- Real-time OS scheduling (Linux RT kernel, Windows MMCSS)
- Zero-copy buffer chains (no Python bridge)
- Lock-free data structures

For **encrypted communication**, 20-50ms is acceptable and much simpler to achieve reliably.

### Q: Why dual pipelines instead of one bidirectional pipeline?

**A:** Simplicity and independence:
- TX and RX have different sources, sinks, processors
- Failures in one direction don't crash the other
- UI is clearer (two simple chains vs. one complex bidirectional graph)
- Threading is easier (two independent threads vs. complex coordination)

### Q: What about key exchange?

**A:** **Out of band.** NDA doesn't handle key distribution—that's a separate problem.

Options:
- Manual: Users share keys via secure channel (Signal, in person)
- External tool: Use a key exchange daemon (Diffie-Hellman server)
- Plugin parameter: Set `encryptor.setParameter("key", "hex_key_here")`

### Q: Can I chain multiple processors?

**Not in v2.0.** The processor slot is single. Future enhancement could allow:
```
Source → [Processor 1] → [Processor 2] → ... → Sink
```

For now, create a composite processor plugin that chains internally.

### Q: How do I debug Python plugin issues?

1. Check console logs (Python exceptions are printed)
2. Use `null_sink` to isolate source issues
3. Use `sine_wave_source` to isolate sink issues
4. Enable Python logging in plugin (`import logging`)
5. Run Python plugin directly via `plugin_loader.py`

---

## Success Criteria (Final Checklist)

### Functional Requirements

- ✅ **Dual pipelines run simultaneously** without interference
- ✅ **Sample rate mismatches handled** automatically (44.1 ↔ 48 ↔ 96 kHz)
- ✅ **Encryption/decryption works** end-to-end (identical output)
- ✅ **Python and C++ plugins** have equal status and performance
- ✅ **UI shows both pipelines** clearly in one screen
- ✅ **Plugins hot-load** without application restart
- ✅ **Buffer size mismatches** handled gracefully

### Performance Requirements

- ✅ **Latency <50ms** (Python plugins, encrypted, dual pipeline)
- ✅ **CPU <30%** on typical quad-core laptop
- ✅ **Memory <100MB** total (dual pipelines, multiple plugins)
- ✅ **No dropouts** during 1 hour continuous operation

### Stability Requirements

- ✅ **24-hour soak test** passes without crash
- ✅ **1000+ start/stop cycles** without memory leak
- ✅ **Source disconnect** handled gracefully (no crash)
- ✅ **Sink failure** logged but pipeline continues
- ✅ **Plugin crash** isolated (doesn't crash core)

### Code Quality Requirements

- ✅ **Bearer deleted** entirely from codebase
- ✅ **Crypto removed** from core (plugins only)
- ✅ **Python bridge optimized** (<2ms overhead)
- ✅ **Metrics accurate** (measured, not hardcoded)
- ✅ **Documentation updated** (README, guides, examples)

---

## Conclusion

**NDA v2.0 is a focused, achievable system.**

By removing network complexity (bearer), delegating encryption to plugins, and embracing a dual-pipeline model, we've simplified the architecture while making it more powerful.

**Core Principles:**
1. **Separation of Concerns**: Audio processing ≠ Network transport
2. **Plugin Everything**: Encryption, devices, effects—all plugins
3. **Dual Pipelines**: Independent TX/RX for clarity
4. **Sample Rate Flexibility**: 48kHz internal, adapt to anything
5. **Python = C++**: Equal support, optimized bridge
6. **Stability > Speed**: 50ms latency is fine; crashes are not

**Implementation Timeline:** 4 weeks to production-ready v2.0

**Next Steps:**
1. Review and approve this spec
2. Begin Phase 1 (core refactoring)
3. Iterate based on testing feedback

---

*NDA v2.0 Specification*  
*Revised: December 2025*  
*Target Release: January 2026*

