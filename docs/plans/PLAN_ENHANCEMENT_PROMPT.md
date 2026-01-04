# Plan Enhancement Prompt for Agent Planner

## Context
You are reviewing and enhancing the NDA v2.0 migration plan (`nda_v2.0_migration_008441a9.plan.md`). The plan currently covers core technical migration but needs additional detail and structure to be production-ready. Your task is to enhance it with missing implementation details, risk considerations, and comprehensive validation criteria.

## Enhancement Areas

### 3. Risk Mitigation Section
**Add a dedicated "Risk Mitigation" section** that identifies potential failure modes and mitigation strategies:

**Required Elements:**
- For each major component (Resampler, Python Bridge, Dual Pipeline UI, Crypto Plugins), identify:
  - **Risk**: What could go wrong? (e.g., "Resampling quality may be insufficient for production")
  - **Impact**: How severe is the failure? (High/Medium/Low)
  - **Mitigation**: What proactive steps prevent this? (e.g., "Start with linear interpolation, add libsamplerate as optional enhancement")
  - **Fallback**: What's Plan B if mitigation fails? (e.g., "Reject mismatched sample rates, require manual configuration")

**Example Structure:**
```markdown
## Risk Mitigation

### Risk 1: Resampling Quality Insufficient
- **Risk**: Linear interpolation may introduce audible artifacts
- **Impact**: Medium (affects audio quality)
- **Mitigation**: Implement libsamplerate integration as optional high-quality path
- **Fallback**: Force matching sample rates, show warning to user

### Risk 2: Python Bridge Performance Not Meeting Target
- **Risk**: Optimization may not achieve <500µs target
- **Impact**: High (blocks production use)
- **Mitigation**: Benchmark early, iterate on each optimization step
- **Fallback**: Recommend C++ plugins for production, document performance characteristics
```

**Questions to Consider:**
- What happens if resampling introduces latency?
- What if Python bridge optimization fails to meet targets?
- How do we handle plugin crashes gracefully?
- What if dual pipeline UI becomes too complex?

---

### 4. Success Metrics Enhancement
**Expand the "Success Criteria" section** with measurable, testable metrics:

**Required Elements:**
- **Code Quality Metrics**: Add specific targets (e.g., "ProcessingPipeline.cpp < 500 lines")
- **Performance Metrics**: Quantify targets with units (e.g., "Python bridge < 500µs per buffer")
- **Stability Metrics**: Define test scenarios (e.g., "24-hour soak test", "1000 start/stop cycles")
- **Usability Metrics**: Add user-facing criteria (e.g., "One-click 'Start Both' works", "Clear error messages")

**Example Enhancement:**
```markdown
## Success Criteria

### Code Quality
- ✅ Zero bearer references in codebase (verify with `grep -r "bearer"`)
- ✅ Zero crypto includes in core
- ✅ ProcessingPipeline.cpp < 500 lines (down from ~800)
- ✅ No compiler warnings
- ✅ All todos completed with status tracking

### Performance
- ✅ Python bridge < 500µs per buffer (512 frames, 2 channels)
- ✅ CPU usage < 30% (dual pipelines running simultaneously)
- ✅ Latency < 50ms end-to-end (measured loopback test)
- ✅ Memory < 100MB total (measured with valgrind)

### Stability
- ✅ 24-hour soak test passes (both TX and RX pipelines)
- ✅ 1000 start/stop cycles without memory leaks
- ✅ Source disconnect handled gracefully (no crashes)
- ✅ Plugin crashes isolated (don't crash main application)

### Usability
- ✅ Dual pipeline UI intuitive (user testing feedback)
- ✅ One-click "Start Both Pipelines" works reliably
- ✅ Metrics update in real-time (Dashboard refresh < 100ms)
- ✅ Clear error messages for common failure modes
```

**Questions to Consider:**
- How do we measure "intuitive"? (Add user testing criteria)
- What constitutes "graceful" error handling?
- How do we validate "real-time" metrics updates?

---

### 6. Resampler Implementation Details
**Enhance Phase 3 (Sample Rate Adaptation)** with complete implementation specifications:

**Required Elements:**
- **ResampleQuality enum**: Add `Medium` option (not just Simple/High)
- **Destructor**: Include `~Resampler()` in header specification
- **State management**: Document `lastSamples_` usage for continuity
- **libsamplerate integration**: Add complete CMake configuration example
- **Testing requirements**: Specify test cases (44.1→48, 96→48, pass-through)

**Example Enhancement:**
```markdown
### Create Resampler Class

**Create [include/audio/Resampler.h](include/audio/Resampler.h):**

```cpp
enum class ResampleQuality {
    Simple,   // Linear interpolation (fast, lower quality)
    Medium,   // Windowed sinc (balanced) - FUTURE
    High      // libsamplerate SINC_BEST_QUALITY (slow, best quality)
};

class Resampler {
public:
    Resampler();
    ~Resampler();  // ADD THIS
    
    void initialize(int inputRate, int outputRate, int channels,
                   ResampleQuality quality = ResampleQuality::Simple);
    void process(AudioBuffer& buffer);
    bool isActive() const { return inputRate_ != outputRate_; }
    
private:
    int inputRate_;
    int outputRate_;
    int channels_;
    ResampleQuality quality_;
    std::vector<float> lastSamples_;  // For continuity between buffers
    
    void processSimple(AudioBuffer& buffer);
    void processHighQuality(AudioBuffer& buffer);  // libsamplerate path
};
```

**Testing Requirements:**
- ✅ Upsample 44.1kHz → 48kHz (verify duration preserved ±1%)
- ✅ Downsample 96kHz → 48kHz (verify no aliasing artifacts via FFT)
- ✅ Pass-through when rates match (zero overhead, no processing)
- ✅ Multi-channel support (2, 4, 8 channels)
- ✅ Buffer boundary continuity (no clicks/pops between buffers)
```

**Questions to Consider:**
- How do we handle buffer boundaries to avoid discontinuities?
- What's the memory overhead of resampling?
- How do we validate audio quality objectively?

---

### 7. Python Bridge Optimization Details
**Enhance Phase 4 (Python Bridge Optimization)** with quantified performance targets:

**Required Elements:**
- **Baseline measurement**: Document expected current performance (3000-15000µs)
- **Per-step targets**: Break down expected improvements:
  - Object caching: 3000µs → 1500µs
  - Zero-copy: 1500µs → 500µs
  - Batch GIL: 500µs → 300µs
- **Benchmark code**: Include complete benchmark for both source AND processor plugins
- **Validation criteria**: Define how to verify each optimization step

**Example Enhancement:**
```markdown
### 3.1 Benchmark Current Performance

**Create:** `tests/benchmark_python_bridge.cpp`

**Baseline Expectations:**
- Source plugin: 3000-15000 µs per buffer (before optimization)
- Processor plugin: Similar overhead expected

**Target After Optimization:**
- Source plugin: < 500 µs per buffer
- Processor plugin: < 500 µs per buffer
- **Improvement: 6-30x faster**

**Benchmark Code Must Test:**
1. Source plugin: `readAudio()` calls
2. Processor plugin: `processAudio()` calls
3. Both with varying buffer sizes (256, 512, 1024 frames)
4. Both with varying channel counts (1, 2, 4 channels)

### 3.2 Implement Object Caching

**Expected Improvement:** 3000 µs → 1500 µs (50% reduction)

**Implementation Checklist:**
- [ ] Cache `AudioBuffer` class object
- [ ] Cache buffer instance (reuse across frames)
- [ ] Cache NumPy array pointer
- [ ] Cache `base_plugin` module import
- [ ] Verify no memory leaks (run 10,000 iterations)

### 3.3 Implement Zero-Copy Data Sharing

**Expected Improvement:** 1500 µs → 500 µs (67% reduction)

**Implementation Checklist:**
- [ ] Replace element-by-element copy with `memcpy()`
- [ ] Verify data integrity (compare before/after)
- [ ] Handle channel interleaving correctly
- [ ] Test with different buffer sizes

### 3.4 Batch GIL Operations

**Expected Improvement:** 500 µs → 300 µs (40% reduction)

**Implementation Checklist:**
- [ ] Consolidate GIL acquisitions to one per frame
- [ ] Verify thread safety (run with multiple threads)
- [ ] Measure GIL contention (use profiling tools)
```

**Questions to Consider:**
- How do we measure GIL contention?
- What's the memory overhead of caching?
- How do we verify zero-copy actually works?

---

### 8. Example Plugins Completeness
**Enhance Phase 7 (Example Crypto Plugins)** with complete, production-ready examples:

**Required Elements:**
- **Passthrough processor**: Add testing utility plugin
- **Full lifecycle methods**: Include initialize, start, stop, shutdown for all examples
- **Parameter handling**: Add setParameter/getParameter examples
- **Plugin exports**: Include `extern "C"` export functions for C++ plugins
- **Error handling**: Show proper error handling patterns

**Example Enhancement:**
```markdown
### C++ AES-256 Encryptor Plugin

**Complete Implementation Checklist:**
- [ ] Constructor with proper initialization
- [ ] Destructor with cleanup (EVP_CIPHER_CTX_free)
- [ ] `initialize()` - Generate/load key
- [ ] `start()` - Set state to Running
- [ ] `stop()` - Set state to Initialized
- [ ] `shutdown()` - Set state to Unloaded
- [ ] `processAudio()` - Full encryption logic
- [ ] `setParameter()` - Key configuration
- [ ] `getParameter()` - Key retrieval
- [ ] `getInfo()` - Plugin metadata
- [ ] Export functions (`createPlugin`, `destroyPlugin`)

**Add Passthrough Processor for Testing:**

**Create:** `plugins_py/examples/passthrough.py`

This plugin does nothing - useful for:
- Testing pipeline integrity
- Measuring overhead
- Validating resampling without processing effects
```

**Questions to Consider:**
- How do plugins handle key exchange securely?
- What's the proper way to handle encryption failures?
- How do we test plugin isolation (crashes don't affect main app)?

---

### 9. UI Implementation Specifications
**Enhance Phase 6 (Dual Pipeline Architecture)** with complete UI widget specifications:

**Required Elements:**
- **All widgets**: List every UI element (combo boxes, buttons, labels, status indicators)
- **Signal/slot connections**: Document all Qt connections
- **State management**: Define UI state transitions
- **Error display**: Specify how errors are shown to users
- **Status indicators**: Define status text/colors/icons

**Example Enhancement:**
```markdown
### Redesign PipelineView for Dual Pipelines

**Complete Widget List:**

**TX Pipeline:**
- `txSourceCombo_` - QComboBox (source selection)
- `txProcessorCombo_` - QComboBox (processor selection, includes "(None - Passthrough)")
- `txSinkCombo_` - QComboBox (sink selection)
- `txStatusLabel_` - QLabel (shows "⚙️ Not configured", "🟢 TX Running", "🔴 TX Error")
- `startTxButton_` - QPushButton ("▶ Start TX")
- `stopTxButton_` - QPushButton ("■ Stop TX", initially disabled)

**RX Pipeline:** (mirror of TX)

**Combined Controls:**
- `startBothButton_` - QPushButton ("▶▶ Start Both Pipelines")
- `stopBothButton_` - QPushButton ("■■ Stop Both Pipelines", initially disabled)

**Signal/Slot Connections:**
```cpp
connect(startTxButton_, &QPushButton::clicked, this, &PipelineView::onStartTxClicked);
connect(stopTxButton_, &QPushButton::clicked, this, &PipelineView::onStopTxClicked);
connect(startRxButton_, &QPushButton::clicked, this, &PipelineView::onStartRxClicked);
connect(stopRxButton_, &QPushButton::clicked, this, &PipelineView::onStopRxClicked);
connect(startBothButton_, &QPushButton::clicked, this, &PipelineView::onStartBothClicked);
connect(stopBothButton_, &QPushButton::clicked, this, &PipelineView::onStopBothClicked);
```

**State Management:**
- Pipeline not configured: Status = "⚙️ Not configured", Start buttons enabled
- Pipeline running: Status = "🟢 TX Running", Start disabled, Stop enabled
- Pipeline error: Status = "🔴 TX Error: [message]", Start enabled, Stop disabled
```

**Questions to Consider:**
- How do we show real-time metrics in the UI?
- What happens if a plugin fails to load?
- How do we handle concurrent pipeline operations?

---

### 10. Metrics Implementation Details
**Enhance Phase 5 (Real-Time Pacing & Accurate Metrics)** with complete metric specifications:

**Required Elements:**
- **All metric getters**: List every getter method
- **Calculation formulas**: Document how each metric is computed
- **Update frequency**: Specify refresh rates
- **Drift warning logic**: Define when warnings are logged
- **Backpressure handling**: Document retry logic

**Example Enhancement:**
```markdown
### Implement Accurate Metrics

**Complete Metric Getters:**
```cpp
public:
    uint64_t getProcessedSamples() const;
    uint64_t getDroppedSamples() const;
    uint64_t getDriftWarnings() const;  // ADD THIS
    uint64_t getBackpressureWaits() const;  // ADD THIS
    double getActualLatency() const;
    float getActualCPULoad() const;
```

**Drift Warning Logic:**
- Log warning when drift > 50ms
- Log every 100th warning (not every single one)
- Format: `[Pipeline] Warning: 125ms behind schedule`

**CPU Load Calculation:**
```cpp
// Formula: (audio_time / wall_time) * 100
// audio_time = (processed_samples * 1000) / sample_rate
// wall_time = elapsed milliseconds since start
```

**Latency Calculation:**
```cpp
// latency = source_buffer_time + processor_latency + sink_buffer_time
// source_buffer_time = workBuffer_.getFrameCount() / targetSampleRate_
// processor_latency = processor_->getProcessingLatency() (if exists)
// sink_buffer_time = sink_->getBufferSize() / sink_->getSampleRate()
```

**Update Frequency:**
- Dashboard refreshes every 100ms (QTimer)
- Metrics calculated on-demand (not cached)
```

**Questions to Consider:**
- How do we measure thread CPU time vs wall time?
- What's the overhead of metric calculation?
- How do we handle metrics during pipeline transitions?

---

### 11. CMakeLists.txt Updates
**Add explicit CMakeLists.txt change specifications:**

**Required Elements:**
- **Crypto removal**: Show exact lines to remove
- **Examples subdirectory**: Show how to add examples build
- **libsamplerate**: Show optional dependency detection
- **Python bridge**: Document any CMake changes needed

**Example Enhancement:**
```markdown
## CMakeLists.txt Updates

### Remove Crypto from Core

**In root CMakeLists.txt:**

```cmake
# REMOVE these lines:
# set(CRYPTO_SOURCES
#     src/crypto/Encryptor.cpp
#     src/crypto/KeyExchange.cpp
# )
# target_sources(${PROJECT_NAME} PRIVATE ${CRYPTO_SOURCES})
# target_link_libraries(${PROJECT_NAME} PRIVATE OpenSSL::SSL OpenSSL::Crypto)

# ADD this instead:
add_subdirectory(plugins_src/examples)
```

### Add Examples Subdirectory

**Create:** `plugins_src/examples/CMakeLists.txt`

```cmake
# Build crypto plugins as separate shared libraries
add_library(AES256EncryptorPlugin SHARED
    AES256EncryptorPlugin.cpp
)
target_link_libraries(AES256EncryptorPlugin PRIVATE OpenSSL::SSL OpenSSL::Crypto)
target_include_directories(AES256EncryptorPlugin PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Install to plugins directory
install(TARGETS AES256EncryptorPlugin
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/plugins
)
```

### Optional libsamplerate Support

**In root CMakeLists.txt:**

```cmake
# Optional high-quality resampling
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(SAMPLERATE QUIET samplerate)
    if(SAMPLERATE_FOUND)
        message(STATUS "Found libsamplerate - high-quality resampling enabled")
        add_definitions(-DHAVE_LIBSAMPLERATE)
        target_link_libraries(${PROJECT_NAME} PRIVATE ${SAMPLERATE_LIBRARIES})
    else()
        message(STATUS "libsamplerate not found - using simple resampling")
    endif()
endif()
```
```

**Questions to Consider:**
- How do we handle optional dependencies gracefully?
- What's the install layout for plugins?
- How do we ensure examples don't break main build?

---

### 14. Code Examples Completeness
**Ensure all code examples are complete and production-ready:**

**Required Elements:**
- **Full class definitions**: Include constructors, destructors, all methods
- **Error handling**: Show proper error handling patterns
- **Memory management**: Document ownership and cleanup
- **Thread safety**: Note thread-safety considerations
- **Comments**: Add explanatory comments for complex logic

**Example Enhancement:**
```markdown
### Complete Code Example Template

**Every code example should include:**

1. **Header guards** (`#ifndef`, `#define`, `#endif`)
2. **Includes** (all necessary headers)
3. **Namespace** (if applicable)
4. **Class declaration** with:
   - Public interface
   - Private members
   - Virtual destructor (if base class)
5. **Implementation** with:
   - Constructor (initialize all members)
   - Destructor (cleanup resources)
   - All virtual method overrides
   - Error handling (check return values)
   - Resource management (RAII where possible)

**Example Checklist for Each Code Block:**
- [ ] Compiles without warnings
- [ ] Handles errors gracefully
- [ ] Manages memory correctly (no leaks)
- [ ] Thread-safe (if applicable)
- [ ] Documented with comments
- [ ] Follows coding style guidelines
```

**Questions to Consider:**
- Are all examples actually compilable?
- Do examples show best practices?
- Are error cases handled?

---

## Enhancement Process

1. **Review each section** of the original plan
2. **Identify gaps** using the criteria above
3. **Add missing details** following the examples provided
4. **Ensure consistency** across all sections
5. **Add validation criteria** for each enhancement
6. **Include testing requirements** where applicable

## Output Format

Enhance the plan file by:
- Adding new sections where missing
- Expanding existing sections with details
- Adding checklists and validation criteria
- Including complete code examples
- Documenting assumptions and decisions

The enhanced plan should be **implementation-ready** - a developer should be able to follow it step-by-step without needing additional clarification.


