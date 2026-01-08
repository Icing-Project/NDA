# 🤖 AI Documentation Index

**For:** Coding AI assistants, Claude, GPT, and other bots building/modifying NDA

This document provides optimized reading paths for different implementation tasks.

---

## 📋 Quick Reference (Copy-Paste Friendly)

**Most important files for AI development:**
1. `../AGENTS.md` - Rules & constraints (READ FIRST)
2. `../technical/ARCHITECTURE.md` - System design
3. `../technical/specifications.md` - Complete API reference
4. `../development/plugin-development.md` - Plugin interface spec
5. `../strategy/v2-decisions-locked.md` - Design decisions (why things are done)

---

## 🎯 Implementation Task Paths

### **Task: Fix a Bug in Core Audio Pipeline**

**Read in order:**
1. `../AGENTS.md` - Constraints & coding style
2. `../technical/ARCHITECTURE.md` - Understand ProcessingPipeline
3. `../technical/specifications.md` - Complete API reference for pipeline
4. `../development/troubleshooting.md` - Common issues (might be documented)
5. Code locations: `src/core/ProcessingPipeline.cpp`, `include/core/ProcessingPipeline.h`

**Key classes:**
- `ProcessingPipeline` - Main processing loop (src/core/ProcessingPipeline.cpp)
- `AudioBuffer` - Data structure (src/audio/AudioBuffer.cpp)
- `PluginManager` - Plugin loading (src/plugins/PluginManager.cpp)

---

### **Task: Write a C++ Plugin**

**Read in order:**
1. `../AGENTS.md` - Coding style & constraints
2. `../development/plugin-development.md` - Plugin interface spec
3. `../technical/specifications.md` § "Plugin Architecture" - Complete plugin API
4. `plugins_src/` folder - Study existing C++ plugin examples
5. Example: `plugins_src/SineWaveSourcePlugin.cpp` (clean, simple)

**Key interfaces:**
- `BasePlugin` (include/plugins/BasePlugin.h) - All plugins inherit
- `AudioSourcePlugin` - Input providers
- `AudioSinkPlugin` - Output consumers
- `AudioProcessorPlugin` - Transformations (encryption, effects)

---

### **Task: Write a Python Plugin**

**Read in order:**
1. `../AGENTS.md` - Rules & style
2. `../development/plugin-development.md` - Python plugin spec
3. `../development/python-processor-guide.md` - Detailed Python guide
4. `plugins_py/` folder - Study example plugins
5. Example: `plugins_py/examples/simple_gain.py` (clean, simple)

**Key interface:**
- `plugins_py/base_plugin.py` - Base classes for Python plugins

**Performance note:** See `../technical/python-bridge.md` for optimization details

---

### **Task: Add a New Feature to UI**

**Read in order:**
1. `../AGENTS.md` - Coding style
2. `../technical/ARCHITECTURE.md` § "User Interface" - UI design
3. `../technical/specifications.md` § "User Interface" - Detailed spec
4. `src/ui/` folder - Study existing components (MainWindow, PipelineView, etc.)

**Key components:**
- `MainWindow.cpp` - Top-level window (manages both pipelines)
- `UnifiedPipelineView.cpp` - Dual pipeline UI (TX and RX)
- `Dashboard.cpp` - Metrics display

---

### **Task: Optimize Performance (Python Bridge or Resampling)**

**Read in order:**
1. `../AGENTS.md` - Constraints
2. `../technical/python-bridge.md` - Python optimization details
3. `../technical/resampling.md` - Resampling algorithm details
4. `../reports/v2.1-performance-analysis.md` - What's been tried & findings
5. Code: `src/plugins/PythonPluginBridge.cpp` or `src/audio/Resampler.cpp`

**Current status:** Python bridge optimization complete but pending validation

---

### **Task: Add Support for New Sample Rate**

**Read in order:**
1. `../AGENTS.md` - Constraints
2. `../technical/resampling.md` - Resampling architecture
3. `../technical/specifications.md` § "Sample Rate Adaptation" - Full spec
4. Code: `src/audio/Resampler.cpp` and `src/core/ProcessingPipeline.cpp`

**Current support:** 44.1, 48, 96 kHz (auto-adapts via pipeline)

---

### **Task: Implement a New Plugin Type**

**Read in order:**
1. `../AGENTS.md` - Constraints
2. `../technical/ARCHITECTURE.md` § "Plugin Architecture" - Plugin design
3. `../technical/specifications.md` § "Plugin Architecture" - Complete spec
4. `../strategy/v2-decisions-locked.md` - Why current design was chosen
5. Code: `include/plugins/PluginTypes.h` (enum), `include/plugins/` (interfaces)

**Current types:** AudioSource, AudioSink, Processor (no more bearer/encryptor)

---

### **Task: Debug Performance Issues**

**Read in order:**
1. `../AGENTS.md` - Constraints
2. `../technical/python-bridge.md` - Known bottlenecks
3. `../reports/v2.1-performance-analysis.md` - Detailed analysis of limitations
4. `../reports/v2.1-executive-summary.md` - Root causes & recommendations
5. Code: `src/core/ProcessingPipeline.cpp` (pacing logic)

**Key finding:** Python is 10-400× slower than C++ for real-time audio. Design limitations, not implementation bugs.

---

## 🔍 File Locations Quick Map

```
For:                                    Look in:
─────────────────────────────────────────────────────────────
Pipeline logic                          src/core/ProcessingPipeline.cpp
Audio buffering                         src/audio/AudioBuffer.cpp
Plugin loading                          src/plugins/PluginManager.cpp
Python bridge                           src/plugins/PythonPluginBridge.cpp
Resampling                              src/audio/Resampler.cpp
UI - main window                        src/ui/MainWindow.cpp
UI - dual pipelines                     src/ui/UnifiedPipelineView.cpp
UI - metrics                            src/ui/Dashboard.cpp
Plugin interfaces (C++)                 include/plugins/*.h
Plugin implementations (C++)            plugins_src/*.cpp
Plugin implementations (Python)         plugins_py/*.py
Build config                            CMakeLists.txt
```

---

## 🏗️ Architecture Overview (AI-Optimized)

```
INPUT                 PROCESSING               OUTPUT
├─ Microphone  ──→  ┌──────────────────┐  ──→  ├─ AIOC USB
├─ AIOC Input  ──→  │ AudioBuffer:     │  ──→  ├─ VB-Cable
├─ File        ──→  │ ┌────────────────┤  ──→  ├─ Speaker
└─ Network     ──→  │ │ Resampler      │  ──→  └─ Network
                    │ │ (48kHz norm)   │
                    │ │                │
                    │ ├─ Processor    │
                    │ │ (optional)     │
                    │ │ - Encryptor   │
                    │ │ - Effects     │
                    │ │                │
                    │ └────────────────┤
                    │ Pipeline runs    │
                    │ in dedicated     │
                    │ thread at        │
                    │ 48000 Hz         │
                    └──────────────────┘
                           ↓
                    PluginManager
                    (discovers .dll/
                     .so, .py files)
```

**Two independent instances:**
- TX Pipeline: Device Mic → Processor → Network/AIOC
- RX Pipeline: Network/AIOC → Processor → Device Speaker

---

## 🔑 Key Concepts for AI Implementation

### **1. Plugin Architecture (Simplified)**

```cpp
// All plugins inherit from BasePlugin
class AudioSourcePlugin : public BasePlugin {
    bool readAudio(AudioBuffer& buffer) = 0;
    int getSampleRate() const = 0;
    // ...
};

class AudioSinkPlugin : public BasePlugin {
    bool writeAudio(const AudioBuffer& buffer) = 0;
    int getSampleRate() const = 0;
    // ...
};

class AudioProcessorPlugin : public BasePlugin {
    bool processAudio(AudioBuffer& buffer) = 0;  // In-place modification
    // ...
};
```

**Key insight:** Processor plugins transform audio in-place (same buffer, modified)

---

### **2. Processing Pipeline (Simplified)**

```cpp
class ProcessingPipeline {
    void processAudioFrame() {
        // 1. Read from source
        source_->readAudio(buffer);
        
        // 2. Resample to internal 48kHz if needed
        sourceResampler_.process(buffer);
        
        // 3. Process (encryption, effects, etc.)
        if (processor_) processor_->processAudio(buffer);
        
        // 4. Resample to sink's rate if needed
        sinkResampler_.process(buffer);
        
        // 5. Write to sink
        sink_->writeAudio(buffer);
        
        // 6. Sleep to maintain real-time cadence
        // ... (pacing logic)
    }
};
```

**Key insight:** Everything goes through source → processor → sink, with auto-resampling

---

### **3. Sample Rate Adaptation**

```
Source provides 44.1kHz → Pipeline auto-resamples to 48kHz
                          Processor sees 48kHz
                          Pipeline resamples to sink's rate
                          Sink receives expected rate
```

No manual intervention needed. The pipeline handles it transparently.

---

### **4. Python Bridge (Optimized)**

```cpp
// OLD (slow): ~5-40ms per call
PyObject* pyBuffer = PyList_New(channels);  // Allocate
for (...) *data++ = buffer[c][f];           // Copy element-by-element
PyObject_CallMethod(plugin, "process", "O", pyBuffer);  // Call
// Release and free

// NEW (fast): <1ms per call
PyArray_SimpleNewFromData(dims, NPY_FLOAT32, buffer.data());  // Zero-copy
PyObject_CallMethod(plugin, "process", "O", cachedArray);     // Reuse
// Cache held, no allocation/copy
```

Key: Zero-copy NumPy arrays + object caching + batch GIL operations

---

## 🚀 Quick Implementation Checklist

When implementing a feature, verify:

- [ ] Reviewed `../AGENTS.md` for constraints
- [ ] Checked `../technical/specifications.md` for API correctness
- [ ] Reviewed `../strategy/v2-decisions-locked.md` for design rationale
- [ ] Examined existing similar code in repository
- [ ] Follows coding style (4-space indent, PascalCase classes, lowerCamelCase methods)
- [ ] Tested with both C++ and Python plugins (if applicable)
- [ ] Memory safe (no raw pointers if std::shared_ptr available)
- [ ] Handles errors gracefully (no exceptions in audio loop)
- [ ] Minimal changes (don't refactor beyond scope)

---

## 📚 Documentation Cross-References

| Topic | Primary Docs | See Also |
|-------|--------------|----------|
| **Plugin system** | `../development/plugin-development.md` | `../technical/specifications.md` § Plugin Architecture |
| **Pipeline design** | `../technical/ARCHITECTURE.md` | `../technical/specifications.md` § Processing Pipeline |
| **Sample rates** | `../technical/resampling.md` | `../technical/specifications.md` § Sample Rate Adaptation |
| **Python performance** | `../technical/python-bridge.md` | `../reports/v2.1-performance-analysis.md` |
| **Design decisions** | `../strategy/v2-decisions-locked.md` | `../strategy/v2-strategic-summary.md` |
| **Implemented changes** | `../reports/v2-implementation-report.md` | `../development/migration-v1-to-v2.md` |
| **Known limitations** | `../reports/v2.1-executive-summary.md` | `../reports/v2.1-performance-analysis.md` |

---

## 🆘 When You Get Stuck

**Error: "Compiler error about BearerPlugin"**
→ Bearer was removed in v2.0. See `../development/migration-v1-to-v2.md`

**Error: "Plugin won't load"**
→ Check `../development/troubleshooting.md` for plugin loading issues

**Error: "Audio is glitchy"**
→ Usually performance-related. See `../reports/v2.1-performance-analysis.md`

**Question: "Why was [decision] made?"**
→ See `../strategy/v2-decisions-locked.md` with full rationale

**Question: "What's the complete API?"**
→ See `../technical/specifications.md`

---

## 📝 Notes for AI Assistants

1. **Always read `../AGENTS.md` first** - It contains critical constraints
2. **Reference the spec, not the code** - Avoid inferring API from implementation
3. **Check for decisions, not just implementation** - See `v2-decisions-locked.md` for rationale
4. **Sample rate is transparent** - Don't special-case it in plugins
5. **Python is supported** - Don't assume C++-only features
6. **Keep it simple** - Minimal changes preferred, no unnecessary refactoring
7. **Check reports** - Performance limitations are documented, not bugs

---

**Last Updated:** January 2026
**For questions:** See `../START_HERE.md`
