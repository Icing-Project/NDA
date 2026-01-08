# NDA v2.0 Architecture

This is the core architecture reference. For complete API and design details, see [`specifications.md`](./specifications.md).

---

## System Overview

NDA (Nade Desktop Application) is a **real-time audio encryption bridge** with a clean 3-slot plugin architecture.

```
INPUT          PROCESSING              OUTPUT
├─ Microphone   Source ──┐   Processor  ├─ AIOC USB
├─ AIOC In      Processor ├─ (optional) ├─ Speaker
├─ File         Sink ─────┘   (encryption) └─ Network
└─ Network                     (effects)
                               48kHz internal
                               Auto-resampling
```

---

## Core Architecture: 3-Slot Pipeline

Every pipeline has exactly **three slots**:

```
┌──────────┐    ┌───────────┐    ┌──────────┐
│  SOURCE  │ ─→ │ PROCESSOR │ ─→ │   SINK   │
│ (INPUT)  │    │ (optional)│    │ (OUTPUT) │
└──────────┘    └───────────┘    └──────────┘
    Audio           Transform        Audio
   Provider      (encryption,       Consumer
                 effects, etc.)
```

**Slot 1: Source (Required)**
- Provides audio input
- Examples: Microphone, File, AIOC Input, Sine Generator
- Interface: `AudioSourcePlugin`

**Slot 2: Processor (Optional)**
- Transforms audio in-place
- Examples: AES-256 Encryptor, AES-256 Decryptor, Gain, EQ
- Can be omitted for passthrough (no transformation)
- Interface: `AudioProcessorPlugin`

**Slot 3: Sink (Required)**
- Consumes audio output
- Examples: Speaker, File, AIOC Output, Null Sink
- Interface: `AudioSinkPlugin`

---

## Dual Independent Pipelines

NDA runs **two independent pipelines simultaneously** for full-duplex operation:

```
┌──────────────────────────────────────────────────────────┐
│ TX Pipeline (Transmit)                                   │
│ Device Mic → [Encryptor] → AIOC/Network Output           │
│                    ↓                                      │
│         Encrypted audio sent to transport                │
├──────────────────────────────────────────────────────────┤
│ RX Pipeline (Receive)                                    │
│ AIOC/Network Input → [Decryptor] → Device Speaker        │
│                    ↓                                      │
│       Encrypted audio from transport                     │
└──────────────────────────────────────────────────────────┘
```

**Key points:**
- Each pipeline runs in its own thread
- They operate independently (TX failure doesn't affect RX)
- They run simultaneously with no coordination
- Each has its own source, processor, and sink

---

## Sample Rate Adaptation

**All processing happens at 48kHz internally** (configurable default).

The pipeline automatically resamples:

```
Source rate (any) ──→ [Resampler] ──→ 48kHz internal ──→ [Processor]
                                                          ↓
                                                      [Resampler]
                                                          ↓
Sink rate (any) ←── [Resampler] ←── 48kHz processed ←────────
```

**Automatic behavior:**
- Source provides 44.1kHz → Auto-resample to 48kHz
- Sink expects 96kHz → Auto-resample from 48kHz
- Processors always see 48kHz
- No special handling needed in plugins

**No code changes required** - It's transparent to plugins.

---

## Plugin System

Three plugin types, all treated equally:

```cpp
enum class PluginType {
    AudioSource,    // Input (microphone, file, network)
    AudioSink,      // Output (speaker, file, network)
    Processor       // Transform (encryption, effects, resampling)
};
```

**Removed in v2.0:**
- ❌ Bearer (network transport removed - use external tools)
- ❌ Encryptor (merged into Processor)

**Plugin lifecycle:**
```
Unloaded → Loaded → Initialized → Running → Stopped → Shutdown
```

**Both C++ and Python plugins supported:**
- C++ plugins: Compiled DLL/SO files in `plugins_src/`
- Python plugins: `.py` files in `plugins_py/`
- Equal status and support

See [`specifications.md`](./specifications.md) § "Plugin Architecture" for complete interface definitions.

---

## Processing Loop (Simplified)

Each pipeline runs this loop continuously:

```cpp
void ProcessingPipeline::processAudioFrame() {
    // 1. Read from source
    source_->readAudio(buffer);
    
    // 2. Resample source → 48kHz if needed
    if (source_->getSampleRate() != 48000) {
        sourceResampler_.process(buffer);
    }
    
    // 3. Apply processor (encryption, effects, etc.)
    if (processor_ && processor_->isRunning()) {
        processor_->processAudio(buffer);  // In-place modification
    }
    
    // 4. Resample 48kHz → sink rate if needed
    if (sink_->getSampleRate() != 48000) {
        sinkResampler_.process(buffer);
    }
    
    // 5. Write to sink
    sink_->writeAudio(buffer);
    
    // 6. Sleep to maintain real-time cadence
    // (pace processing to 48kHz)
}
```

**Key insight:** Everything flows through source → processor → sink.

---

## Key Components

### ProcessingPipeline
- **File:** `src/core/ProcessingPipeline.cpp`
- **Header:** `include/core/ProcessingPipeline.h`
- **Responsibility:** Main audio processing loop, threading, pacing
- **Size:** ~430 lines (v2.0, simplified from ~800 in v1.x)

### PluginManager
- **File:** `src/plugins/PluginManager.cpp`
- **Header:** `include/plugins/PluginManager.h`
- **Responsibility:** Discover, load, and manage plugins (C++ and Python)

### AudioBuffer
- **File:** `src/audio/AudioBuffer.cpp`
- **Header:** `include/audio/AudioBuffer.h`
- **Responsibility:** Multi-channel audio data structure

### Resampler
- **File:** `src/audio/Resampler.cpp`
- **Header:** `include/audio/Resampler.h`
- **Responsibility:** Sample rate conversion (linear interpolation)

### PythonPluginBridge
- **File:** `src/plugins/PythonPluginBridge.cpp`
- **Header:** `include/plugins/PythonPluginBridge.h`
- **Responsibility:** Optimized C++/Python interop (zero-copy, caching)

### MainWindow
- **File:** `src/ui/MainWindow.cpp`
- **Header:** `include/ui/MainWindow.h`
- **Responsibility:** Application top-level UI, manages both pipelines

### UnifiedPipelineView
- **File:** `src/ui/UnifiedPipelineView.cpp`
- **Header:** `include/ui/UnifiedPipelineView.h`
- **Responsibility:** Dual pipeline configuration UI (source/processor/sink dropdowns)

---

## Design Decisions (Why?)

**Q: Why 3 slots instead of more?**
A: Simplicity. Complex chains can be implemented as composite plugins internally.
See [`strategy/v2-decisions-locked.md`](../strategy/v2-decisions-locked.md) for rationale.

**Q: Why dual pipelines instead of one bidirectional?**
A: Clarity and independence. TX and RX have different sources/sinks/processors.
See [`strategy/v2-decisions-locked.md`](../strategy/v2-decisions-locked.md).

**Q: Why remove Bearer?**
A: NDA's job is audio processing. Network transport is out of scope.
Use AIOC hardware, VB-Cable, Discord as transport. See [`strategy/v2-decisions-locked.md`](../strategy/v2-decisions-locked.md).

**Q: Why 48kHz internal, not 44.1kHz or 96kHz?**
A: Standard CD quality, good balance. Auto-adapts to any device anyway.
See [`strategy/v2-decisions-locked.md`](../strategy/v2-decisions-locked.md).

---

## Data Flow Example

**Configuration:** Encrypted AIOC Radio

```
TX Pipeline:
┌─────────────────────────────────────────┐
│ Device Microphone (44.1kHz)             │
│         ↓                               │
│ [Resampler] 44.1 → 48kHz                │
│         ↓                               │
│ [AES-256 Encryptor] encrypt in-place    │
│         ↓                               │
│ [Resampler] 48kHz → 48kHz (no-op)       │
│         ↓                               │
│ AIOC USB Output (48kHz)                 │
│  (encrypted audio sent to radio)        │
└─────────────────────────────────────────┘

RX Pipeline:
┌─────────────────────────────────────────┐
│ AIOC USB Input (48kHz)                  │
│  (encrypted audio from radio)           │
│         ↓                               │
│ [Resampler] 48kHz → 48kHz (no-op)       │
│         ↓                               │
│ [AES-256 Decryptor] decrypt in-place    │
│         ↓                               │
│ [Resampler] 48kHz → 48kHz (no-op)       │
│         ↓                               │
│ Device Speaker (48kHz)                  │
└─────────────────────────────────────────┘
```

---

## User Interface

**Main Window:** `MainWindow` (Qt6)
- Top-level application window
- Manages both TX and RX pipelines
- Tabs: Pipeline Config, Dashboard, Settings

**Pipeline Configuration:** `UnifiedPipelineView`
- Two sections: TX Pipeline, RX Pipeline
- Each has 3 combo boxes: Source, Processor, Sink
- Independent start/stop controls
- Combined "Start Both" / "Stop Both" buttons

**Dashboard:** Metrics display
- Live status for both pipelines
- Latency, CPU load, sample count
- Audio level meters

**Settings:** Global configuration
- Default sample rate (48kHz)
- Buffer size, resampling quality
- Plugin directories

---

## Performance Targets

| Metric | Target | Maximum |
|--------|--------|---------|
| **Latency (both pipelines)** | <50ms | <100ms |
| **CPU (both active)** | <15% | <30% |
| **Memory (total)** | <60MB | <100MB |
| **Stability** | 24-hour continuous | No crashes |

See [`reports/v2.1-performance-analysis.md`](../reports/v2.1-performance-analysis.md) for current status.

---

## Version History

**v2.0 (Current):**
- 3-slot pipeline architecture
- Dual independent pipelines (TX + RX)
- Plugin-only encryption
- Sample rate auto-adaptation
- Python processor support
- ~35% code reduction vs v1.x

**v1.x (Legacy):**
- 4-slot pipeline (Bearer removed in v2.0)
- Single pipeline
- Core crypto + plugins
- Manual sample rate handling
- Limited Python support

See [`development/migration-v1-to-v2.md`](../development/migration-v1-to-v2.md) for migration details.

---

## Next Steps

- **Want complete API reference?** See [`specifications.md`](./specifications.md)
- **Want to write a plugin?** See [`development/plugin-development.md`](../development/plugin-development.md)
- **Want implementation details?** See [`development/plugin-development.md`](../development/plugin-development.md)
- **Want performance details?** See [`python-bridge.md`](./python-bridge.md)
- **Want design rationale?** See [`strategy/v2-decisions-locked.md`](../strategy/v2-decisions-locked.md)

---

**Last Updated:** January 2026
**Version:** 2.0
