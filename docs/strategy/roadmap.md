# NDA v2.0 Implementation Plan
## Tactical Roadmap: v1.x → v2.0 Migration

---

## Overview

This document provides the step-by-step implementation plan to migrate from the current NDA v1.x architecture to the simplified v2.0 architecture outlined in `NDA-SPECS-v2.md`.

**Key Changes:**
- ❌ Delete Bearer plugin type entirely
- ❌ Remove core crypto classes (move to plugin examples)
- ✅ Add AudioProcessorPlugin interface
- ✅ Simplify pipeline to 3 slots (Source → Processor → Sink)
- ✅ Implement dual pipeline architecture (TX + RX)
- ✅ Add sample rate adaptation
- ✅ Optimize Python bridge performance

**Timeline:** 4 weeks  
**Effort:** ~80-100 hours  
**Risk:** Medium (significant refactoring, but well-scoped)

---

## Phase 1: Core Cleanup & Simplification
**Duration:** 3-4 days  
**Status:** 🔴 Not started

### 1.1 Delete Bearer Infrastructure

**Files to Delete:**
```
include/plugins/BearerPlugin.h
src/plugins/BearerPlugin.cpp  (if exists)
examples/UDPBearerPlugin.h
```

**Files to Modify:**
```
include/plugins/PluginTypes.h
  - Remove Bearer from PluginType enum

include/core/ProcessingPipeline.h
  - Remove #include "plugins/BearerPlugin.h"
  - Remove setBearer() method
  - Remove bearer_ member variable

src/core/ProcessingPipeline.cpp
  - Remove bearer initialization
  - Remove bearer start/stop calls
  - Remove packet serialization/send code in processAudioFrame()
  
src/plugins/PluginManager.cpp
  - Remove getBearerPlugin() method
  - Remove bearer-specific loading logic

src/ui/UnifiedPipelineView.cpp (formerly PipelineView.cpp)
  - Remove bearerCombo widget
  - Remove bearer card from UI
  - Remove onBearerChanged() slot
```

**Testing:**
- ✅ Application compiles without errors
- ✅ Pipeline can start/stop without bearer
- ✅ No bearer references remain in codebase

**Commands:**
```bash
# Find remaining bearer references
grep -r "bearer" --include="*.cpp" --include="*.h" src/ include/
grep -r "Bearer" --include="*.cpp" --include="*.h" src/ include/

# Should find 0 results after cleanup
```

### 1.2 Remove Core Crypto Classes

**Files to Delete:**
```
include/crypto/Encryptor.h
include/crypto/KeyExchange.h
src/crypto/Encryptor.cpp
src/crypto/KeyExchange.cpp
```

**Files to Create (Examples):**
```
plugins_src/examples/AES256EncryptorPlugin.h
plugins_src/examples/AES256EncryptorPlugin.cpp
plugins_src/examples/AES256DecryptorPlugin.h
plugins_src/examples/AES256DecryptorPlugin.cpp
```

**CMakeLists.txt Changes:**
```cmake
# Remove crypto sources from core
# OLD:
# set(CRYPTO_SOURCES
#     src/crypto/Encryptor.cpp
#     src/crypto/KeyExchange.cpp
# )

# NEW: Build crypto as example plugins
add_subdirectory(plugins_src/examples)
```

**Testing:**
- ✅ Core compiles without crypto includes
- ✅ Example plugins compile separately
- ✅ Example plugins load via PluginManager

### 1.3 Create AudioProcessorPlugin Interface

**New File:** `include/plugins/AudioProcessorPlugin.h`

```cpp
#ifndef AUDIOPROCESSORPLUGIN_H
#define AUDIOPROCESSORPLUGIN_H

#include "BasePlugin.h"
#include "audio/AudioBuffer.h"

namespace nda {

/**
 * Audio processor plugin interface
 * Processes audio in-place (encryptors, decryptors, effects, etc.)
 */
class AudioProcessorPlugin : public BasePlugin {
public:
    virtual ~AudioProcessorPlugin() = default;

    /**
     * Process audio buffer in-place
     * @param buffer Audio data to process (modified in-place)
     * @return true if processing succeeded
     */
    virtual bool processAudio(AudioBuffer& buffer) = 0;

    /**
     * Get current sample rate
     */
    virtual int getSampleRate() const = 0;

    /**
     * Get current channel count
     */
    virtual int getChannelCount() const = 0;

    /**
     * Set sample rate (called during pipeline initialization)
     */
    virtual void setSampleRate(int rate) = 0;

    /**
     * Set channel count (called during pipeline initialization)
     */
    virtual void setChannelCount(int channels) = 0;

    /**
     * Get processing latency added by this processor (in seconds)
     */
    virtual double getProcessingLatency() const { return 0.0; }
};

} // namespace nda

#endif // AUDIOPROCESSORPLUGIN_H
```

**Update:** `include/plugins/PluginTypes.h`

```cpp
enum class PluginType {
    AudioSource,    // Audio input
    AudioSink,      // Audio output
    Processor       // Audio transformation (was: Bearer, Encryptor)
};
```

**Testing:**
- ✅ Header compiles
- ✅ Can create derived class implementing interface
- ✅ PluginManager recognizes Processor type

### 1.4 Simplify ProcessingPipeline to 3 Slots

**File:** `include/core/ProcessingPipeline.h`

**Changes:**
```cpp
// OLD:
bool setAudioSource(std::shared_ptr<AudioSourcePlugin> source);
bool setBearer(std::shared_ptr<BearerPlugin> bearer);        // DELETE
bool setEncryptor(std::shared_ptr<EncryptorPlugin> encryptor); // DELETE
bool setAudioSink(std::shared_ptr<AudioSinkPlugin> sink);

// NEW:
bool setSource(std::shared_ptr<AudioSourcePlugin> source);
bool setProcessor(std::shared_ptr<AudioProcessorPlugin> processor); // NEW
bool setSink(std::shared_ptr<AudioSinkPlugin> sink);

// OLD members:
std::shared_ptr<AudioSourcePlugin> audioSource_;
std::shared_ptr<BearerPlugin> bearer_;          // DELETE
std::shared_ptr<EncryptorPlugin> encryptor_;    // DELETE
std::shared_ptr<AudioSinkPlugin> audioSink_;

// NEW members:
std::shared_ptr<AudioSourcePlugin> source_;
std::shared_ptr<AudioProcessorPlugin> processor_;  // Optional
std::shared_ptr<AudioSinkPlugin> sink_;
```

**File:** `src/core/ProcessingPipeline.cpp`

**Simplify `processAudioFrame()`:**
```cpp
void ProcessingPipeline::processAudioFrame() {
    // 1. Read from source
    if (!source_->readAudio(workBuffer_)) {
        handleReadFailure();
        return;
    }

    // 2. Apply processor (optional)
    if (processor_ && processor_->getState() == PluginState::Running) {
        processor_->processAudio(workBuffer_);
    }

    // 3. Write to sink
    if (sink_->writeAudio(workBuffer_)) {
        processedSamples_ += workBuffer_.getFrameCount();
    } else {
        droppedSamples_ += workBuffer_.getFrameCount();
    }
}
```

**Testing:**
- ✅ Pipeline initializes with source + sink only
- ✅ Pipeline works with source + processor + sink
- ✅ Processor is truly optional (can be nullptr)

---

## Phase 2: Sample Rate Adaptation
**Duration:** 4-5 days  
**Status:** 🔴 Not started

### 2.1 Implement Basic Resampler

**New File:** `include/audio/Resampler.h`

```cpp
#ifndef RESAMPLER_H
#define RESAMPLER_H

#include "AudioBuffer.h"

namespace nda {

enum class ResampleQuality {
    Simple,    // Linear interpolation (fast, lower quality)
    Medium,    // Windowed sinc (balanced)
    High       // High-quality sinc (slow, best quality)
};

class Resampler {
public:
    Resampler();
    ~Resampler();

    /**
     * Initialize resampler
     * @param inputRate Source sample rate
     * @param outputRate Target sample rate
     * @param channels Number of channels
     * @param quality Resampling quality
     */
    void initialize(int inputRate, int outputRate, int channels, 
                   ResampleQuality quality = ResampleQuality::Simple);

    /**
     * Process buffer (resamples in-place or to new buffer)
     * @param buffer Input/output buffer (may be resized)
     */
    void process(AudioBuffer& buffer);

    /**
     * Check if resampling is needed
     */
    bool isActive() const { return inputRate_ != outputRate_; }

private:
    int inputRate_;
    int outputRate_;
    int channels_;
    ResampleQuality quality_;
    
    // Simple linear interpolation state
    std::vector<float> lastSamples_;  // For continuity between buffers
};

} // namespace nda

#endif // RESAMPLER_H
```

**Implementation:** `src/audio/Resampler.cpp`

Start with **simple linear interpolation**, add higher quality later:

```cpp
void Resampler::process(AudioBuffer& buffer) {
    if (!isActive()) return;  // No resampling needed

    const float ratio = static_cast<float>(outputRate_) / inputRate_;
    const int inputFrames = buffer.getFrameCount();
    const int outputFrames = static_cast<int>(inputFrames * ratio);

    AudioBuffer outputBuffer(buffer.getChannelCount(), outputFrames);

    for (int ch = 0; ch < buffer.getChannelCount(); ++ch) {
        const float* input = buffer.getChannelData(ch);
        float* output = outputBuffer.getChannelData(ch);

        for (int i = 0; i < outputFrames; ++i) {
            float srcPos = i / ratio;
            int srcIndex = static_cast<int>(srcPos);
            float frac = srcPos - srcIndex;

            if (srcIndex + 1 < inputFrames) {
                // Linear interpolation
                output[i] = input[srcIndex] * (1.0f - frac) + 
                           input[srcIndex + 1] * frac;
            } else {
                output[i] = input[srcIndex];
            }
        }
    }

    buffer = std::move(outputBuffer);
}
```

**Testing:**
- ✅ Upsample 44.1kHz → 48kHz (verify duration preserved)
- ✅ Downsample 96kHz → 48kHz (verify no aliasing artifacts)
- ✅ Pass-through when rates match (zero overhead)

### 2.2 Integrate Resampling into Pipeline

**File:** `include/core/ProcessingPipeline.h`

**Add members:**
```cpp
private:
    int targetSampleRate_;           // Pipeline internal rate (48000)
    Resampler sourceResampler_;      // Source rate → target
    Resampler sinkResampler_;        // Target → sink rate
```

**File:** `src/core/ProcessingPipeline.cpp`

**Update `initialize()`:**
```cpp
bool ProcessingPipeline::initialize() {
    targetSampleRate_ = 48000;  // TODO: Make configurable

    // Initialize source resampler if needed
    int sourceRate = source_->getSampleRate();
    if (sourceRate != targetSampleRate_) {
        sourceResampler_.initialize(sourceRate, targetSampleRate_, 
                                   source_->getChannelCount());
    }

    // Initialize sink resampler if needed
    int sinkRate = sink_->getSampleRate();
    if (sinkRate != targetSampleRate_) {
        sinkResampler_.initialize(targetSampleRate_, sinkRate,
                                 sink_->getChannelCount());
    }

    return true;
}
```

**Update `processAudioFrame()`:**
```cpp
void ProcessingPipeline::processAudioFrame() {
    // 1. Read from source
    if (!source_->readAudio(workBuffer_)) {
        handleReadFailure();
        return;
    }

    // 2. Resample source → 48kHz
    if (sourceResampler_.isActive()) {
        sourceResampler_.process(workBuffer_);
    }

    // 3. Apply processor at 48kHz
    if (processor_) {
        processor_->processAudio(workBuffer_);
    }

    // 4. Resample 48kHz → sink rate
    if (sinkResampler_.isActive()) {
        sinkResampler_.process(workBuffer_);
    }

    // 5. Write to sink
    if (sink_->writeAudio(workBuffer_)) {
        processedSamples_ += workBuffer_.getFrameCount();
    }
}
```

**Testing:**
- ✅ 44.1kHz source → 48kHz sink works
- ✅ 48kHz source → 96kHz sink works
- ✅ Audio quality acceptable (compare input/output spectrograms)

### 2.3 Add libsamplerate (Optional High Quality)

**CMakeLists.txt:**
```cmake
find_package(PkgConfig)
if(PkgConfig_FOUND)
    pkg_check_modules(SAMPLERATE samplerate)
    if(SAMPLERATE_FOUND)
        add_definitions(-DHAVE_LIBSAMPLERATE)
        target_link_libraries(${PROJECT_NAME} ${SAMPLERATE_LIBRARIES})
    endif()
endif()
```

**Resampler.cpp (high-quality path):**
```cpp
#ifdef HAVE_LIBSAMPLERATE
#include <samplerate.h>

void Resampler::processHighQuality(AudioBuffer& buffer) {
    SRC_STATE* state = src_new(SRC_SINC_BEST_QUALITY, channels_, nullptr);
    SRC_DATA data;
    // ... use libsamplerate API
    src_delete(state);
}
#endif
```

---

## Phase 3: Python Bridge Optimization
**Duration:** 3-4 days  
**Status:** 🔴 Not started

### 3.1 Benchmark Current Performance

**Create:** `tests/benchmark_python_bridge.cpp`

```cpp
#include "plugins/PythonPluginBridge.h"
#include <chrono>

int main() {
    // Test source plugin
    auto sourcePlugin = std::make_shared<PythonPluginBridge>("plugins_py/sine_wave_source.py");
    sourcePlugin->initialize();
    sourcePlugin->start();

    AudioBuffer buffer(2, 512);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 1000; ++i) {
        sourcePlugin->readAudio(buffer);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Source plugin average: " 
              << (duration.count() / 1000.0) << " µs\n";
    
    // Test processor plugin (NEW in v2.0)
    auto processorPlugin = std::make_shared<PythonPluginBridge>("plugins_py/examples/simple_gain.py");
    processorPlugin->initialize();
    processorPlugin->start();
    
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 1000; ++i) {
        processorPlugin->processAudio(buffer);
    }
    
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Processor plugin average: " 
              << (duration.count() / 1000.0) << " µs\n";
    
    return 0;
}
```

**Baseline:** Measure current overhead (expect 3000-15000 µs for source, similar for processor)

### 3.2 Implement Object Caching

**File:** `include/plugins/PythonPluginBridge.h`

**Add members:**
```cpp
private:
    // Cache Python objects to avoid recreation
    PyObject* cachedAudioBufferClass_;  // base_plugin.AudioBuffer class
    PyObject* cachedBufferInstance_;    // Reusable AudioBuffer object
    PyArrayObject* cachedNumpyArray_;   // Reusable NumPy array
```

**File:** `src/plugins/PythonPluginBridge.cpp`

**Modify `createPythonAudioBuffer()`:**
```cpp
PyObject* PythonPluginBridge::createPythonAudioBuffer(const AudioBuffer& buffer) {
    PyGILState_STATE gilState = PyGILState_Ensure();

    // Cache module and class on first call
    if (!cachedAudioBufferClass_) {
        PyObject* module = PyImport_ImportModule("base_plugin");
        cachedAudioBufferClass_ = PyObject_GetAttrString(module, "AudioBuffer");
        Py_DECREF(module);
    }

    // Reuse or create buffer instance
    if (!cachedBufferInstance_) {
        cachedBufferInstance_ = PyObject_CallFunction(cachedAudioBufferClass_, 
                                                      "ii", 
                                                      buffer.getChannelCount(),
                                                      buffer.getFrameCount());
    }

    // Update NumPy array data (see next section)
    updateCachedBufferData(buffer);

    PyGILState_Release(gilState);
    return cachedBufferInstance_;
}
```

**Expected improvement:** 3000 µs → 1500 µs

### 3.3 Implement Zero-Copy Data Sharing

**Replace element-by-element copy with NumPy array view:**

```cpp
void PythonPluginBridge::updateCachedBufferData(const AudioBuffer& buffer) {
    // Get pointer to NumPy array inside cached buffer
    PyObject* dataAttr = PyObject_GetAttrString(cachedBufferInstance_, "data");
    PyArrayObject* array = reinterpret_cast<PyArrayObject*>(dataAttr);
    
    // Get writable pointer to NumPy data
    float* pyData = static_cast<float*>(PyArray_DATA(array));
    
    // Fast memcpy per channel
    for (int ch = 0; ch < buffer.getChannelCount(); ++ch) {
        const float* cppData = buffer.getChannelData(ch);
        std::memcpy(pyData + (ch * buffer.getFrameCount()), 
                   cppData, 
                   buffer.getFrameCount() * sizeof(float));
    }
    
    Py_DECREF(dataAttr);
}
```

**Expected improvement:** 1500 µs → 500 µs

### 3.4 Batch GIL Operations

**Current (inefficient):**
```cpp
// Acquire GIL multiple times
auto state = PyGILState_Ensure();
int rate = getSampleRate();
PyGILState_Release(state);

state = PyGILState_Ensure();
readAudio(buffer);
PyGILState_Release(state);
```

**Optimized:**
```cpp
// Acquire once per frame
auto state = PyGILState_Ensure();
int rate = getSampleRate();
readAudio(buffer);
PyGILState_Release(state);
```

**Expected improvement:** 500 µs → 300 µs

### 3.5 Final Benchmark

**Target:** <500 µs per buffer (512 frames, 2 channels)

**Metrics:**
- Before: 3000-15000 µs
- After: 300-500 µs
- **Improvement: 6-30x faster**

---

## Phase 4: Real-Time Pacing & Metrics
**Duration:** 3-4 days  
**Status:** 🔴 Not started

### 4.1 Implement Pacing in Processing Loop

**File:** `src/core/ProcessingPipeline.cpp`

**Add members:**
```cpp
private:
    std::chrono::steady_clock::time_point startTime_;
    uint64_t droppedSamples_;
    uint64_t driftWarnings_;
```

**Update `processingThread()`:**
```cpp
void ProcessingPipeline::processingThread() {
    startTime_ = std::chrono::steady_clock::now();
    processedSamples_ = 0;
    droppedSamples_ = 0;

    while (isRunning_) {
        processAudioFrame();

        // Calculate target time for this frame
        auto targetTime = startTime_ + std::chrono::microseconds(
            (processedSamples_ * 1000000) / targetSampleRate_
        );

        auto now = std::chrono::steady_clock::now();

        if (now < targetTime) {
            // We're ahead of schedule, sleep until target time
            std::this_thread::sleep_until(targetTime);
        } else {
            // We're behind schedule
            auto drift = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - targetTime
            );

            if (drift.count() > 50) {
                driftWarnings_++;
                if (driftWarnings_ % 100 == 0) {
                    std::cout << "[Pipeline] Warning: " << drift.count() 
                             << "ms behind schedule\n";
                }
            }
        }
    }
}
```

**Testing:**
- ✅ Sine → Null runs at exactly 1.0x real-time (±5%)
- ✅ Sine → WAV produces correct duration files
- ✅ No drift accumulation over 1 hour

### 4.2 Add Backpressure Handling

**Update `processAudioFrame()`:**
```cpp
void ProcessingPipeline::processAudioFrame() {
    // ... (read, resample, process)

    // Check sink backpressure
    int available = sink_->getAvailableSpace();
    if (available < workBuffer_.getFrameCount()) {
        // Sink queue is full, wait briefly
        backpressureWaits_++;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        
        // Retry once
        available = sink_->getAvailableSpace();
        if (available < workBuffer_.getFrameCount()) {
            // Still no space, drop this buffer
            droppedSamples_ += workBuffer_.getFrameCount();
            return;
        }
    }

    // Write to sink
    if (sink_->writeAudio(workBuffer_)) {
        processedSamples_ += workBuffer_.getFrameCount();
    } else {
        droppedSamples_ += workBuffer_.getFrameCount();
    }
}
```

**Testing:**
- ✅ Sine → SoundDevice speaker doesn't overflow queue
- ✅ Dropped sample counter increments when sink fails
- ✅ Backpressure waits prevent underruns

### 4.3 Implement Accurate Metrics

**CPU Load Measurement:**
```cpp
float ProcessingPipeline::getActualCPULoad() const {
    // Measure thread runtime vs wall time
    auto now = std::chrono::steady_clock::now();
    auto wallTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - startTime_
    ).count();

    // Estimate processing time from samples and rate
    auto audioTime = (processedSamples_ * 1000) / targetSampleRate_;

    if (wallTime == 0) return 0.0f;

    // CPU% = (audio time / wall time) * 100
    // This is approximate; real implementation should track thread CPU time
    return (static_cast<float>(audioTime) / wallTime) * 100.0f;
}
```

**Latency Measurement:**
```cpp
double ProcessingPipeline::getActualLatency() const {
    double latency = 0.0;

    // Source buffer time
    latency += static_cast<double>(workBuffer_.getFrameCount()) / targetSampleRate_;

    // Processor latency (if declared)
    if (processor_) {
        latency += processor_->getProcessingLatency();
    }

    // Sink buffer time
    latency += static_cast<double>(sink_->getBufferSize()) / sink_->getSampleRate();

    return latency;
}
```

**New Getters:**
```cpp
uint64_t getDroppedSamples() const { return droppedSamples_; }
uint64_t getDriftWarnings() const { return driftWarnings_; }
uint64_t getBackpressureWaits() const { return backpressureWaits_; }
```

**Testing:**
- ✅ CPU load shows realistic values (5-15%)
- ✅ Latency matches measured loopback tests
- ✅ Metrics update in real-time (Dashboard)

---

## Phase 5: Dual Pipeline UI
**Duration:** 4-5 days  
**Status:** 🔴 Not started

### 5.1 Create Second Pipeline Instance

**File:** `include/ui/MainWindow.h`

```cpp
private:
    std::shared_ptr<nda::ProcessingPipeline> txPipeline_;  // Transmit
    std::shared_ptr<nda::ProcessingPipeline> rxPipeline_;  // Receive
```

**File:** `src/ui/MainWindow.cpp`

```cpp
MainWindow::MainWindow(QWidget *parent) {
    // Create two independent pipelines
    txPipeline_ = std::make_shared<nda::ProcessingPipeline>();
    rxPipeline_ = std::make_shared<nda::ProcessingPipeline>();

    // Share both with views
    pipelineView->setTxPipeline(txPipeline_);
    pipelineView->setRxPipeline(rxPipeline_);
    dashboard->setTxPipeline(txPipeline_);
    dashboard->setRxPipeline(rxPipeline_);
}
```

### 5.2 Redesign PipelineView for Dual Pipelines

**File:** `include/ui/PipelineView.h`

**Add members:**
```cpp
private:
    // TX Pipeline widgets
    QComboBox* txSourceCombo_;
    QComboBox* txProcessorCombo_;
    QComboBox* txSinkCombo_;
    QPushButton* startTxButton_;
    QPushButton* stopTxButton_;
    QLabel* txStatusLabel_;

    // RX Pipeline widgets
    QComboBox* rxSourceCombo_;
    QComboBox* rxProcessorCombo_;
    QComboBox* rxSinkCombo_;
    QPushButton* startRxButton_;
    QPushButton* stopRxButton_;
    QLabel* rxStatusLabel_;

    // Combined controls
    QPushButton* startBothButton_;
    QPushButton* stopBothButton_;

    // Pipeline instances
    std::shared_ptr<nda::ProcessingPipeline> txPipeline_;
    std::shared_ptr<nda::ProcessingPipeline> rxPipeline_;
```

**File:** `src/ui/UnifiedPipelineView.cpp` (formerly PipelineView.cpp)

**Redesign `setupUI()`:**
```cpp
void PipelineView::setupUI() {
    QVBoxLayout* mainLayout = new QVBoxLayout(this);

    // Title
    QLabel* title = new QLabel("Dual Pipeline Configuration");
    mainLayout->addWidget(title);

    // TX Pipeline Card
    QGroupBox* txGroup = new QGroupBox("TRANSMIT Pipeline (TX)");
    QVBoxLayout* txLayout = new QVBoxLayout(txGroup);

    txSourceCombo_ = new QComboBox();
    txLayout->addWidget(new QLabel("Source (Input):"));
    txLayout->addWidget(txSourceCombo_);

    txProcessorCombo_ = new QComboBox();
    txProcessorCombo_->addItem("(None - Passthrough)");
    txLayout->addWidget(new QLabel("Processor (Optional):"));
    txLayout->addWidget(txProcessorCombo_);

    txSinkCombo_ = new QComboBox();
    txLayout->addWidget(new QLabel("Sink (Output):"));
    txLayout->addWidget(txSinkCombo_);

    txStatusLabel_ = new QLabel("⚙️ Not configured");
    txLayout->addWidget(txStatusLabel_);

    QHBoxLayout* txButtons = new QHBoxLayout();
    startTxButton_ = new QPushButton("▶ Start TX");
    stopTxButton_ = new QPushButton("■ Stop TX");
    stopTxButton_->setEnabled(false);
    txButtons->addWidget(startTxButton_);
    txButtons->addWidget(stopTxButton_);
    txLayout->addLayout(txButtons);

    mainLayout->addWidget(txGroup);

    // RX Pipeline Card (mirror of TX)
    QGroupBox* rxGroup = new QGroupBox("RECEIVE Pipeline (RX)");
    // ... similar structure

    mainLayout->addWidget(rxGroup);

    // Combined controls
    QHBoxLayout* combinedButtons = new QHBoxLayout();
    startBothButton_ = new QPushButton("▶▶ Start Both Pipelines");
    stopBothButton_ = new QPushButton("■■ Stop Both Pipelines");
    stopBothButton_->setEnabled(false);
    combinedButtons->addWidget(startBothButton_);
    combinedButtons->addWidget(stopBothButton_);
    mainLayout->addLayout(combinedButtons);

    // Connect signals
    connect(startTxButton_, &QPushButton::clicked, this, &PipelineView::onStartTxClicked);
    connect(stopTxButton_, &QPushButton::clicked, this, &PipelineView::onStopTxClicked);
    connect(startRxButton_, &QPushButton::clicked, this, &PipelineView::onStartRxClicked);
    connect(stopRxButton_, &QPushButton::clicked, this, &PipelineView::onStopRxClicked);
    connect(startBothButton_, &QPushButton::clicked, this, &PipelineView::onStartBothClicked);
    connect(stopBothButton_, &QPushButton::clicked, this, &PipelineView::onStopBothClicked);
}
```

**Add slot implementations:**
```cpp
void PipelineView::onStartTxClicked() {
    if (!txPipeline_ || !selectedTxSource_ || !selectedTxSink_) return;

    txPipeline_->setSource(selectedTxSource_);
    if (selectedTxProcessor_) txPipeline_->setProcessor(selectedTxProcessor_);
    txPipeline_->setSink(selectedTxSink_);

    if (txPipeline_->initialize() && txPipeline_->start()) {
        txStatusLabel_->setText("🟢 TX Running");
        startTxButton_->setEnabled(false);
        stopTxButton_->setEnabled(true);
        updateCombinedButtons();
    }
}

void PipelineView::onStartBothClicked() {
    onStartTxClicked();
    onStartRxClicked();
}
```

### 5.3 Update Dashboard for Dual Metrics

**File:** `include/ui/Dashboard.h`

```cpp
private:
    std::shared_ptr<nda::ProcessingPipeline> txPipeline_;
    std::shared_ptr<nda::ProcessingPipeline> rxPipeline_;

    // TX metrics
    QLabel* txLatencyLabel_;
    QLabel* txCpuLabel_;
    QLabel* txSamplesLabel_;
    QProgressBar* txLevelMeterL_;
    QProgressBar* txLevelMeterR_;

    // RX metrics  
    QLabel* rxLatencyLabel_;
    QLabel* rxCpuLabel_;
    QLabel* rxSamplesLabel_;
    QProgressBar* rxLevelMeterL_;
    QProgressBar* rxLevelMeterR_;
```

**Update metrics timer:**
```cpp
void Dashboard::updateMetrics() {
    if (txPipeline_ && txPipeline_->isRunning()) {
        txLatencyLabel_->setText(QString::number(txPipeline_->getActualLatency() * 1000, 'f', 1) + " ms");
        txCpuLabel_->setText(QString::number(txPipeline_->getActualCPULoad(), 'f', 1) + "%");
        txSamplesLabel_->setText(QString::number(txPipeline_->getProcessedSamples()));
    }

    if (rxPipeline_ && rxPipeline_->isRunning()) {
        rxLatencyLabel_->setText(QString::number(rxPipeline_->getActualLatency() * 1000, 'f', 1) + " ms");
        rxCpuLabel_->setText(QString::number(rxPipeline_->getActualCPULoad(), 'f', 1) + "%");
        rxSamplesLabel_->setText(QString::number(rxPipeline_->getProcessedSamples()));
    }
}
```

---

## Phase 6: Example Crypto Plugins
**Duration:** 3-4 days  
**Status:** 🔴 Not started

### 6.1 Create AES-256 Encryptor Plugin (C++)

**File:** `plugins_src/examples/AES256EncryptorPlugin.cpp`

```cpp
#include "plugins/AudioProcessorPlugin.h"
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <vector>

class AES256EncryptorPlugin : public nda::AudioProcessorPlugin {
private:
    EVP_CIPHER_CTX* ctx_;
    std::vector<uint8_t> key_;     // 256-bit key
    std::vector<uint8_t> nonce_;   // 96-bit nonce (GCM)
    int sampleRate_;
    int channels_;
    nda::PluginState state_;

public:
    AES256EncryptorPlugin() 
        : ctx_(EVP_CIPHER_CTX_new())
        , key_(32, 0)  // 256 bits
        , nonce_(12, 0)  // 96 bits
        , sampleRate_(48000)
        , channels_(2)
        , state_(nda::PluginState::Unloaded)
    {
    }

    ~AES256EncryptorPlugin() {
        if (ctx_) EVP_CIPHER_CTX_free(ctx_);
    }

    bool initialize() override {
        // Generate random key (in production, load from secure storage)
        if (RAND_bytes(key_.data(), key_.size()) != 1) {
            return false;
        }
        state_ = nda::PluginState::Initialized;
        return true;
    }

    bool start() override {
        state_ = nda::PluginState::Running;
        return true;
    }

    void stop() override {
        state_ = nda::PluginState::Initialized;
    }

    void shutdown() override {
        state_ = nda::PluginState::Unloaded;
    }

    bool processAudio(nda::AudioBuffer& buffer) override {
        if (state_ != nda::PluginState::Running) return false;

        // Generate unique nonce for this buffer
        RAND_bytes(nonce_.data(), nonce_.size());

        // Convert float audio to bytes
        size_t totalSamples = buffer.getFrameCount() * buffer.getChannelCount();
        std::vector<uint8_t> plaintext(totalSamples * sizeof(float));
        
        // Interleave channels into byte array
        float* floatPtr = reinterpret_cast<float*>(plaintext.data());
        for (int f = 0; f < buffer.getFrameCount(); ++f) {
            for (int ch = 0; ch < buffer.getChannelCount(); ++ch) {
                *floatPtr++ = buffer.getChannelData(ch)[f];
            }
        }

        // Encrypt using AES-256-GCM
        std::vector<uint8_t> ciphertext(plaintext.size() + 16);  // +16 for auth tag
        int len;

        EVP_EncryptInit_ex(ctx_, EVP_aes_256_gcm(), nullptr, key_.data(), nonce_.data());
        EVP_EncryptUpdate(ctx_, ciphertext.data(), &len, plaintext.data(), plaintext.size());
        
        int ciphertext_len = len;
        EVP_EncryptFinal_ex(ctx_, ciphertext.data() + len, &len);
        ciphertext_len += len;

        // Get auth tag
        EVP_CIPHER_CTX_ctrl(ctx_, EVP_CTRL_GCM_GET_TAG, 16, ciphertext.data() + ciphertext_len);

        // Convert encrypted bytes back to float buffer
        floatPtr = reinterpret_cast<float*>(ciphertext.data());
        for (int f = 0; f < buffer.getFrameCount(); ++f) {
            for (int ch = 0; ch < buffer.getChannelCount(); ++ch) {
                buffer.getChannelData(ch)[f] = *floatPtr++;
            }
        }

        return true;
    }

    nda::PluginInfo getInfo() const override {
        return {
            "AES-256 Encryptor",
            "1.0.0",
            "NDA Team",
            "AES-256-GCM audio encryption",
            nda::PluginType::Processor,
            1
        };
    }

    nda::PluginState getState() const override { return state_; }
    int getSampleRate() const override { return sampleRate_; }
    int getChannelCount() const override { return channels_; }
    void setSampleRate(int rate) override { sampleRate_ = rate; }
    void setChannelCount(int channels) override { channels_ = channels; }

    bool setParameter(const std::string& key, const std::string& value) override {
        if (key == "key") {
            // Set encryption key from hex string
            // TODO: Implement hex parsing
            return true;
        }
        return false;
    }

    std::string getParameter(const std::string& key) const override {
        if (key == "key") {
            // Return key as hex (for sharing with decryptor)
            // TODO: Implement hex encoding
            return "";
        }
        return "";
    }
};

// Export plugin
extern "C" {
    __declspec(dllexport) nda::BasePlugin* createPlugin() {
        return new AES256EncryptorPlugin();
    }

    __declspec(dllexport) void destroyPlugin(nda::BasePlugin* plugin) {
        delete plugin;
    }
}
```

### 6.2 Create AES-256 Decryptor Plugin

**Similar structure, but uses `EVP_DecryptInit_ex` and validates auth tag.**

### 6.3 Python Processor Examples

#### Example 1: Simple Gain Processor

**File:** `plugins_py/examples/simple_gain.py`

```python
import numpy as np
from base_plugin import AudioProcessorPlugin, PluginInfo, PluginType, PluginState

class SimpleGainPlugin(AudioProcessorPlugin):
    """Simple volume/gain adjustment processor"""
    
    def __init__(self):
        self.state = PluginState.UNLOADED
        self.gain = 1.0  # Unity gain (no change)
        self.sample_rate = 48000
        self.channels = 2

    def initialize(self):
        self.state = PluginState.INITIALIZED
        return True

    def start(self):
        self.state = PluginState.RUNNING
        return True

    def stop(self):
        self.state = PluginState.INITIALIZED

    def shutdown(self):
        self.state = PluginState.UNLOADED

    def process_audio(self, buffer):
        """Apply gain to audio buffer in-place"""
        if self.state != PluginState.RUNNING:
            return False

        # Multiply all samples by gain
        buffer.data *= self.gain
        
        # Clip to prevent overflow
        np.clip(buffer.data, -1.0, 1.0, out=buffer.data)
        
        return True

    def get_info(self):
        return PluginInfo(
            name="Simple Gain",
            version="1.0.0",
            author="NDA Team",
            description="Basic volume/gain adjustment",
            plugin_type=PluginType.PROCESSOR,
            api_version=1
        )

    def get_state(self):
        return self.state

    def get_sample_rate(self):
        return self.sample_rate

    def get_channel_count(self):
        return self.channels

    def set_sample_rate(self, rate):
        self.sample_rate = rate

    def set_channel_count(self, channels):
        self.channels = channels

    def set_parameter(self, key, value):
        if key == "gain":
            try:
                self.gain = float(value)
                return True
            except ValueError:
                return False
        return False

    def get_parameter(self, key):
        if key == "gain":
            return str(self.gain)
        return ""

def create_plugin():
    return SimpleGainPlugin()
```

#### Example 2: Fernet Crypto Encryptor

**File:** `plugins_py/examples/fernet_encryptor.py`

```python
from cryptography.fernet import Fernet
import numpy as np
from base_plugin import AudioProcessorPlugin, PluginInfo, PluginType, PluginState

class FernetEncryptorPlugin(AudioProcessorPlugin):
    """Symmetric encryption using Python's cryptography library"""
    
    def __init__(self):
        self.state = PluginState.UNLOADED
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)
        self.sample_rate = 48000
        self.channels = 2

    def initialize(self):
        self.state = PluginState.INITIALIZED
        return True

    def start(self):
        self.state = PluginState.RUNNING
        return True

    def stop(self):
        self.state = PluginState.INITIALIZED

    def shutdown(self):
        self.state = PluginState.UNLOADED

    def process_audio(self, buffer):
        """Encrypt audio buffer in-place"""
        if self.state != PluginState.RUNNING:
            return False

        # Convert float audio to bytes
        audio_bytes = buffer.data.tobytes()

        # Encrypt
        encrypted = self.cipher.encrypt(audio_bytes)

        # Pad to original size (Fernet adds overhead)
        # In production, use a streaming cipher mode
        # For now, truncate/pad to match buffer size
        encrypted_array = np.frombuffer(encrypted[:len(audio_bytes)], dtype=np.float32)
        encrypted_array = encrypted_array.reshape(buffer.data.shape)

        buffer.data[:] = encrypted_array
        return True

    def get_info(self):
        return PluginInfo(
            name="Fernet Encryptor",
            version="1.0.0",
            author="NDA Team",
            description="Fernet symmetric encryption (Python)",
            plugin_type=PluginType.PROCESSOR,
            api_version=1
        )

    def get_state(self):
        return self.state

    def get_sample_rate(self):
        return self.sample_rate

    def get_channel_count(self):
        return self.channels

    def set_sample_rate(self, rate):
        self.sample_rate = rate

    def set_channel_count(self, channels):
        self.channels = channels

    def set_parameter(self, key, value):
        if key == "key":
            # Set encryption key (base64 encoded)
            try:
                self.key = value.encode()
                self.cipher = Fernet(self.key)
                return True
            except:
                return False
        return False

    def get_parameter(self, key):
        if key == "key":
            return self.key.decode()
        return ""

def create_plugin():
    return FernetEncryptorPlugin()
```

#### Example 3: Passthrough Processor (Testing)

**File:** `plugins_py/examples/passthrough.py`

```python
from base_plugin import AudioProcessorPlugin, PluginInfo, PluginType, PluginState

class PassthroughPlugin(AudioProcessorPlugin):
    """No-op processor for testing pipeline integrity"""
    
    def __init__(self):
        self.state = PluginState.UNLOADED
        self.sample_rate = 48000
        self.channels = 2

    def initialize(self):
        self.state = PluginState.INITIALIZED
        return True

    def start(self):
        self.state = PluginState.RUNNING
        return True

    def stop(self):
        self.state = PluginState.INITIALIZED

    def shutdown(self):
        self.state = PluginState.UNLOADED

    def process_audio(self, buffer):
        """Do nothing - pass audio through unchanged"""
        if self.state != PluginState.RUNNING:
            return False
        
        # Intentionally empty - no processing
        return True

    def get_info(self):
        return PluginInfo(
            name="Passthrough",
            version="1.0.0",
            author="NDA Team",
            description="No-op processor for testing",
            plugin_type=PluginType.PROCESSOR,
            api_version=1
        )

    def get_state(self):
        return self.state

    def get_sample_rate(self):
        return self.sample_rate

    def get_channel_count(self):
        return self.channels

    def set_sample_rate(self, rate):
        self.sample_rate = rate

    def set_channel_count(self, channels):
        self.channels = channels

    def set_parameter(self, key, value):
        return False

    def get_parameter(self, key):
        return ""

def create_plugin():
    return PassthroughPlugin()
```

---

## Testing Strategy

### Unit Tests

**Create:** `tests/test_pipeline.cpp`

```cpp
TEST(PipelineTest, SimplexFlow) {
    auto pipeline = std::make_shared<ProcessingPipeline>();
    auto source = std::make_shared<SineWaveSourcePlugin>();
    auto sink = std::make_shared<NullSinkPlugin>();

    pipeline->setSource(source);
    pipeline->setSink(sink);
    ASSERT_TRUE(pipeline->initialize());
    ASSERT_TRUE(pipeline->start());

    std::this_thread::sleep_for(std::chrono::seconds(2));

    pipeline->stop();
    EXPECT_GT(pipeline->getProcessedSamples(), 0);
}
```

### Integration Tests

**Test Scenarios:**
1. ✅ TX pipeline: Sine → Encryptor → WAV (verify encrypted file)
2. ✅ RX pipeline: WAV → Decryptor → Null (verify decryption)
3. ✅ Dual pipelines: Both running simultaneously
4. ✅ Sample rate mismatch: 44.1 source → 48 sink
5. ✅ Python plugins: Sounddevice mic → Null sink (1 hour)

### Stress Tests

**24-Hour Soak Test:**
```bash
# Run both pipelines for 24 hours
./NDA --tx-config tx_aioc.json --rx-config rx_aioc.json --duration 86400
```

**Memory Leak Test:**
```bash
# Start/stop 1000 times
for i in {1..1000}; do
    ./NDA --start --duration 5 --stop
    sleep 1
done
```

---

## Rollout Plan

### Week 1: Core Cleanup
- Days 1-2: Delete bearer, remove crypto
- Days 3-4: Create processor interface, simplify pipeline
- Day 5: Testing and bug fixes

### Week 2: Performance
- Days 1-2: Implement resampler
- Days 3-4: Optimize Python bridge
- Day 5: Benchmark and validate

### Week 3: Pacing & UI
- Days 1-2: Real-time pacing and metrics
- Days 3-4: Dual pipeline UI
- Day 5: Integration testing

### Week 4: Crypto & Polish
- Days 1-2: Crypto plugin examples
- Days 3-4: Documentation and migration guide
- Day 5: Final testing and v2.0 release

---

## Success Metrics

**Code Quality:**
- ✅ Zero bearer references in codebase
- ✅ Zero crypto includes in core
- ✅ Pipeline.cpp < 500 lines (down from ~800)
- ✅ No compiler warnings

**Performance:**
- ✅ Python bridge < 500µs per buffer
- ✅ CPU usage < 30% (dual pipelines)
- ✅ Latency < 50ms end-to-end
- ✅ Memory < 100MB total

**Stability:**
- ✅ 24-hour test passes (both pipelines)
- ✅ 1000 start/stop cycles (no leaks)
- ✅ Handle disconnects gracefully

**Usability:**
- ✅ Dual pipeline UI intuitive
- ✅ One-click "Start Both" works
- ✅ Metrics update in real-time
- ✅ Clear error messages

---

## Risk Mitigation

**Risk 1: Resampling Quality**
- **Mitigation:** Start with simple linear interpolation, add libsamplerate later
- **Fallback:** Force matching sample rates, reject mismatches

**Risk 2: Python Bridge Performance**
- **Mitigation:** Benchmark early, iterate optimizations
- **Fallback:** Recommend C++ plugins for production

**Risk 3: UI Complexity (Dual Pipelines)**
- **Mitigation:** Prototype UI mockup before implementation
- **Fallback:** Simplify to tabs (one pipeline per tab)

**Risk 4: Crypto Plugin Compatibility**
- **Mitigation:** Document key exchange requirements clearly
- **Fallback:** Provide reference implementations (AES-256, ChaCha20)

---

## Next Steps

1. **Review and approve this plan**
2. **Set up development branch:** `git checkout -b feature/v2-migration`
3. **Begin Phase 1 (Core Cleanup)** following checklist above
4. **Daily standups** to track progress and address blockers
5. **Weekly demos** to stakeholders showing incremental progress

---

*Ready to begin implementation on approval.*

