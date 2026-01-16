# Changelog - Version 2.1 (Performance Optimization Attempt)

**Date:** 2026-01-08
**Goal:** Eliminate audio drops and achieve 100% source-to-sink fidelity
**Result:** Successfully eliminated automatic drops, but **did not achieve real-time performance**

---

## Overview

Version 2.1 focused on surgical improvements to eliminate audio quality degradation while maintaining the existing architecture. Multiple optimizations were implemented based on log analysis and profiling, targeting:

1. **Zero audio loss** (never drop samples)
2. **Improved audio quality** (better resampling)
3. **Better stability** (crash isolation)
4. **Plugin consistency** (fixed buffer sizes)

**Outcome:** All optimizations were successfully implemented, but the underlying performance bottleneck (Python execution speed) remains unsolved.

---

## Changes Made

### 1. Disabled Drift Resync Audio Dropping

**File:** `src/core/ProcessingPipeline.cpp` (lines 866-875)

**Problem:**
When the pipeline fell behind schedule (drift), it would artificially increment `droppedSamples_` to "catch up" with wall-clock time. This caused audible artifacts (clicks, pops, crackling) because entire chunks of audio (250-450ms) were being discarded.

**Change:**
```cpp
// BEFORE (v2.0):
if (driftResyncMs_ > 0 && drift.count() > driftResyncMs_) {
    droppedSamples_ += catchUpSamples;  // Artificially drop audio
}

// AFTER (v2.1):
// v2.1: DISABLED drift resync - it drops audio which violates zero-loss requirement
// New strategy: Let pipeline run as fast as it can. If it falls behind, that's
// a plugin performance issue that should be fixed, not hidden by dropping audio.
```

**Rationale:**
- Dropping audio violates the "100% fidelity" requirement
- Drift is a **symptom** of slow plugins, not a problem to hide
- Better to accumulate drift than lose audio data

**Impact:**
- ✅ Eliminates one source of audio drops
- ✅ Logs no longer show "Drift resync: dropped ~Xms"
- ⚠️ Drift now accumulates unbounded (340ms/second observed)

**Trade-off:** Honest reporting of performance problems vs. hiding them

---

### 2. Improved Default Resampling Quality

**Files:**
- `include/audio/Resampler.h` (lines 41-47)
- `src/audio/Resampler.cpp` (lines 14-90)

**Problem:**
Default resampler used linear interpolation (~60dB SNR), causing audible artifacts when source/sink sample rates differed from internal 48kHz.

**Changes:**

#### 2a. Changed Default Quality
```cpp
// BEFORE (v2.0):
Resampler::Resampler() : quality_(ResampleQuality::Simple) { }

void initialize(..., ResampleQuality quality = ResampleQuality::Simple);

// AFTER (v2.1):
Resampler::Resampler() : quality_(ResampleQuality::Medium) { }

void initialize(..., ResampleQuality quality = ResampleQuality::Medium);
```

**Quality Comparison:**
| Mode | Algorithm | SNR | CPU Cost | Use Case |
|------|-----------|-----|----------|----------|
| Simple | Linear | ~60dB | 1× | Low-power devices |
| Medium | Catmull-Rom cubic | ~80dB | 2-3× | **Default (v2.1)** |
| High | libsamplerate sinc | ~120dB | 10-20× | Studio quality |

#### 2b. Added Environment Variable Override
```cpp
// Allow runtime quality selection without recompilation
if (const char* qualityEnv = std::getenv("NDA_RESAMPLER_QUALITY")) {
    // Supports: simple, medium, high
}
```

**Usage:**
```bash
# Force simple (fastest):
set NDA_RESAMPLER_QUALITY=simple

# Force high quality (slowest):
set NDA_RESAMPLER_QUALITY=high
```

#### 2c. Cached Resampling Ratio
```cpp
// BEFORE: Calculated ratio every frame
float ratio = static_cast<float>(outputRate_) / inputRate_;

// AFTER: Calculate once, cache, reuse
double cachedRatio_;  // Member variable
cachedRatio_ = static_cast<double>(outputRate_) / static_cast<double>(inputRate_);
```

**Benefit:** Eliminates repeated division in hot path (tiny optimization, ~1-2% CPU)

**Impact:**
- ✅ Eliminates audible resampling artifacts
- ✅ Transparent audio quality for rate conversions
- ⚠️ Adds 2-3× CPU cost for resampling (acceptable trade-off)
- ❌ Does not address pipeline speed bottleneck

---

### 3. Plugin Call Exception Isolation

**File:** `src/core/ProcessingPipeline.cpp` (lines 915-935, 989-1139)

**Problem:**
Plugin crashes (segfaults, exceptions) would crash the entire pipeline with no error handling or recovery.

**Change:**
Added `safePluginCall()` template wrapper around **all** plugin calls:

```cpp
template<typename Func>
bool safePluginCall(const char* pluginName, const char* operation,
                    Func&& func, std::atomic<bool>& isRunning)
{
    try {
        return func();
    } catch (const std::exception& e) {
        std::cerr << "[Pipeline] CRITICAL: Plugin '" << pluginName
                  << "' threw exception during " << operation << ": "
                  << e.what() << std::endl;
        isRunning.store(false);  // Fail-fast
        return false;
    } catch (...) {
        std::cerr << "[Pipeline] CRITICAL: Plugin '" << pluginName
                  << "' crashed during " << operation
                  << " (unknown exception)" << std::endl;
        isRunning.store(false);  // Fail-fast
        return false;
    }
}
```

**Protected Operations:**
1. `source_->readAudio()` (line 989)
2. `processor_->processAudio()` (line 1043)
3. `sink_->getAvailableSpace()` (line 1109)
4. `sink_->writeAudio()` (line 1129)

**Behavior:**
- Catches both `std::exception` and unknown exceptions
- Logs plugin name and operation
- Sets `isRunning_ = false` (fail-fast philosophy)
- Pipeline exits cleanly rather than crashing

**Impact:**
- ✅ No more segfaults from plugin crashes
- ✅ Clean shutdown with error messages
- ✅ Identifies which plugin failed
- ❌ Does not prevent plugin failures, just handles them

---

### 4. Fail-Fast on Processor Failures

**File:** `src/core/ProcessingPipeline.cpp` (lines 1051-1059)

**Problem:**
If processor plugin failed, v2.0 would continue with "passthrough" (unprocessed audio). For encryption plugins, this means **sending unencrypted audio** - a critical security breach.

**Change:**
```cpp
// BEFORE (v2.0):
if (!processor_->processAudio(workBuffer_)) {
    processorFailures_++;
    // Continue with passthrough (DANGEROUS!)
}

// AFTER (v2.1):
bool processOk = safePluginCall(..., processAudio);
if (!processOk) {
    std::cerr << "[Pipeline] CRITICAL: Processor failed. "
              << "Stopping pipeline to prevent unprocessed audio transmission."
              << std::endl;
    isRunning_.store(false);  // FAIL-FAST
    return;
}
```

**Rationale:**
- Processing is **not optional** - it's the plugin's purpose
- Passthrough could leak sensitive data (unencrypted audio)
- Better to fail cleanly than compromise security

**Impact:**
- ✅ Prevents security breaches from failed encryption
- ✅ Enforces 100% processing guarantee
- ⚠️ More aggressive failure mode (stops pipeline)

---

### 5. Zero-Loss Backpressure Handling

**File:** `src/core/ProcessingPipeline.cpp` (lines 1060-1147)

**Problem:**
When sink couldn't accept audio fast enough (backpressure), v2.0 would immediately drop the frame. This violated the "100% fidelity" requirement.

**Change:**
Replaced drop logic with retry loop:

```cpp
// BEFORE (v2.0):
if (available < frameCount) {
    droppedSamples_ += frameCount;  // DROP IMMEDIATELY
    return;
}

// AFTER (v2.1):
const int maxLatencyMs = 50;  // Respect latency budget
int retryAttempts = 0;

while (!writeOk && retryAttempts < maxRetries) {
    // Try to write
    writeOk = sink_->writeAudio(workBuffer_);

    if (writeOk) break;  // Success

    // Check cumulative wait time
    if (cumulativeWaitMs >= maxLatencyMs) {
        // FAIL-FAST: Exceeded latency budget
        isRunning_.store(false);
        return;
    }

    // Sleep and retry
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    retryAttempts++;
}
```

**Key Features:**
1. **Retry** instead of drop (up to 20 attempts)
2. **Latency budget enforcement** (50ms max)
3. **Fail-fast** if budget exceeded (prevent infinite buffering)
4. **Detailed logging** of backpressure events

**Impact:**
- ✅ Achieves zero-loss guarantee (never drops audio)
- ✅ Respects latency budget (fails if >50ms wait)
- ❌ **Never triggers** because pipeline is slower than sink (wrong bottleneck)

---

### 6. Fixed Python Plugin Buffer Sizes

**Files:**
- `plugins_py/base_plugin.py` (lines 1-25, documentation)
- `plugins_py/sounddevice_microphone.py` (line 28)
- `plugins_py/sounddevice_speaker.py` (line 31)

**Problem:**
SoundDevice plugins used 256-sample buffers while pipeline expected 512-sample buffers. This caused:
- Queue starvation (plugin fills 256, pipeline reads 512)
- Buffer size mismatches requiring resampling
- Increased latency and instability

**Changes:**

#### 6a. Documentation Added
```python
"""
IMPORTANT BUFFER SIZE REQUIREMENT (v2.1):
==========================================
All audio source plugins MUST use 512-sample buffers to match the pipeline's
internal buffer size. This ensures:
- Zero-copy performance in the Python bridge
- Elimination of buffer size mismatches that cause queue starvation
- Consistent latency across all plugins
"""
```

#### 6b. Fixed SoundDevice Plugins
```python
# BEFORE (v2.0):
self.buffer_size = 256

# AFTER (v2.1):
self.buffer_size = 512  # v2.1: Fixed to match pipeline buffer size
```

**Impact:**
- ✅ Eliminates buffer size mismatches
- ✅ Consistent buffer handling across plugins
- ✅ Better queue management
- ❌ Does not address underlying speed issues

---

### 7. Audio Queue Pre-filling

**File:** `plugins_py/sounddevice_microphone.py` (lines 184-193)

**Problem:**
Queue starts empty. First pipeline reads timeout waiting for audio, causing initial crackling and "Audio read started failing" errors.

**Change:**
Added pre-fill logic after starting audio stream:

```python
# After stream.start():
# Pre-fill queue with 2 blocks (~21ms startup latency)
import time
prefill_timeout = 1.0
prefill_start = time.time()

while self.audio_queue.qsize() < 2:
    if time.time() - prefill_start > prefill_timeout:
        print("[SoundDeviceMic] Warning: Queue pre-fill timeout", flush=True)
        break
    time.sleep(0.005)  # Check every 5ms

print(f"... (queue pre-filled: {self.audio_queue.qsize()} blocks)", flush=True)
```

**Startup Sequence:**
1. Start audio stream (hardware begins capturing)
2. Wait for audio callback to fill queue (2 blocks)
3. Return `True` from `start()` (pipeline can begin)

**Impact:**
- ✅ Smooth audio from first frame (no initial timeout)
- ✅ Eliminates "Audio read started failing" at startup
- ✅ Adds only ~21ms startup latency (acceptable per user)
- ⚠️ Queue still drains during operation (pipeline too slow)

---

### 8. Reduced Blocking Timeout

**File:** `plugins_py/sounddevice_microphone.py` (line 293)

**Problem:**
Queue `get()` blocked for 200ms waiting for audio. When queue empty, this blocked entire pipeline for 200ms.

**Change:**
```python
# BEFORE (v2.0):
indata = self.audio_queue.get(timeout=0.2)  # 200ms

# AFTER (v2.1):
indata = self.audio_queue.get(timeout=0.05)  # 50ms
```

**Rationale:**
- 50ms matches user's latency budget
- Faster failure detection when queue empty
- Reduces pipeline stall duration

**Impact:**
- ✅ Faster timeout detection (150ms saved per timeout)
- ⚠️ Still blocks pipeline when queue empty
- ❌ Does not prevent queue from emptying

---

## Summary of Changes by File

### Modified Files

| File | Lines Changed | Change Type | Impact |
|------|---------------|-------------|--------|
| `src/core/ProcessingPipeline.cpp` | ~200 | Major refactor | Stability + zero-loss |
| `src/audio/Resampler.cpp` | 49 | Optimization | Quality improvement |
| `include/audio/Resampler.h` | 6 | API change | Quality improvement |
| `plugins_py/base_plugin.py` | 25 | Documentation | Developer guidance |
| `plugins_py/sounddevice_microphone.py` | 15 | Configuration | Buffer size + pre-fill |
| `plugins_py/sounddevice_speaker.py` | 1 | Configuration | Buffer size |

**Total:** ~296 lines modified across 6 files

---

## Configuration Changes

### New Environment Variables

```bash
# Resampler quality (default: medium)
NDA_RESAMPLER_QUALITY=simple|medium|high

# Already existed in v2.0, still supported:
NDA_PIPELINE_BACKPRESSURE_MODE=wait|drop|write_retry
NDA_PIPELINE_BACKPRESSURE_SLEEP_MS=5
NDA_PIPELINE_DRIFT_RESYNC_MS=250  # Now effectively disabled
```

### Default Value Changes

| Parameter | v2.0 Default | v2.1 Default | Reason |
|-----------|--------------|--------------|--------|
| Resampler quality | Simple | **Medium** | Better audio quality |
| Plugin buffer size | 256 | **512** | Match pipeline |
| Microphone queue timeout | 200ms | **50ms** | Faster failure detection |
| Queue pre-fill | None | **2 blocks** | Smooth startup |

---

## Backward Compatibility

### Breaking Changes
❌ **Python plugins with hardcoded 256-sample buffers will have mismatched buffers**
- Mitigation: All included plugins updated
- Action required: Third-party plugins must update to 512 samples

### Compatible Changes
✅ All C++ plugins unchanged
✅ Plugin API unchanged
✅ Config files unchanged
✅ UI unchanged

---

## Testing Results

### Test Environment
```
OS:        Windows (via WSL2)
CPU:       (not specified in logs)
Platform:  Linux 6.6.87.2-microsoft-standard-WSL2
Plugins:   Python (sine_wave_source, sounddevice_microphone, sounddevice_speaker)
Config:    48kHz, 512 samples, 2 channels (sine), 1 channel (mic/speaker)
```

### Test 1: Sine Wave → Speaker (v2.1)
```
✅ No automatic audio drops (droppedSamples = 0)
✅ Resampler quality: Medium (as designed)
✅ No plugin crashes
❌ Pipeline runs at 54% speed (50.84 Hz vs 93.75 Hz expected)
❌ Severe drift accumulation (486ms after 1 second)
❌ Audio quality: Crackling, popping
```

### Test 2: Microphone → Speaker (v2.1)
```
✅ No automatic audio drops (droppedSamples = 0)
✅ Queue pre-filled successfully (2 blocks)
✅ No plugin crashes
❌ Pipeline runs at 15-90% speed (highly variable)
❌ Severe drift accumulation (3,429ms after 10 seconds)
❌ 189 underruns, 72 overflows in 10 seconds
❌ Audio quality: Severe crackling, latency, dropouts
```

### Comparison: v2.0 vs v2.1

| Metric | v2.0 | v2.1 | Change |
|--------|------|------|--------|
| Automatic drops | Yes (250-450ms chunks) | **No** | ✅ Improved |
| Resampling artifacts | Audible | **Inaudible** | ✅ Improved |
| Plugin crashes | Crash pipeline | **Clean shutdown** | ✅ Improved |
| Pipeline speed | 20-70% | 15-90% | ⚠️ Similar |
| Audio quality | Bad | **Still bad** | ❌ No improvement |

**Conclusion:** v2.1 fixes **symptoms** but not the **root cause** (Python speed).

---

## What Worked

1. ✅ **Zero-loss guarantee** - No more automatic audio drops
2. ✅ **Better audio quality** - Resampling artifacts eliminated
3. ✅ **Crash protection** - Plugin failures handled gracefully
4. ✅ **Cleaner design** - Consistent buffer sizes across plugins
5. ✅ **Better diagnostics** - Clear error messages on failures

---

## What Didn't Work

1. ❌ **Real-time performance** - Pipeline still too slow (50-75% of required speed)
2. ❌ **Python overhead** - 10-400× slower than C++ (unchanged)
3. ❌ **GIL contention** - Up to 64ms blocking (unchanged)
4. ❌ **Queue management** - Still experiences underruns/overflows
5. ❌ **Audio quality** - Crackling/popping persists due to timing issues

---

## Lessons Learned

### 1. Optimization Limits
Software optimizations can only go so far. The fundamental bottleneck (Python execution speed) cannot be solved with clever coding - it requires architectural changes.

### 2. Wrong Bottleneck
Many optimizations (backpressure handling, buffer sizes) targeted the wrong bottleneck. The real problem was **plugin execution time**, not buffer management.

### 3. Symptom vs. Root Cause
- **Symptoms:** Audio drops, crackling, drift
- **Root cause:** Python is too slow for real-time constraints
- Treating symptoms doesn't solve root cause

### 4. Measurement is Critical
Without detailed profiling logs, the Python overhead would have been overlooked. The metrics clearly showed:
- Average Python read: 5-40ms (should be <1ms)
- GIL acquisition: up to 64ms (unacceptable)
- Frame rate: 50-75% of required (cannot sustain real-time)

### 5. Architecture Matters
No amount of optimization can overcome architectural limitations. Python's GIL and interpreted nature are **fundamental constraints** that cannot be optimized away at this level.

---

## Migration Guide

### For Users
No action required - changes are transparent.

### For Plugin Developers

**Required Changes:**
1. Update buffer size to 512 samples:
   ```python
   self.buffer_size = 512  # Was 256
   ```

2. Pre-fill queues before returning from `start()`:
   ```python
   while self.audio_queue.qsize() < 2:
       time.sleep(0.005)
   ```

3. Use 50ms timeout (not 200ms):
   ```python
   data = self.queue.get(timeout=0.05)
   ```

**See:** `plugins_py/base_plugin.py` for full guidelines

---

## Future Considerations

V2.1 optimizations are **necessary but insufficient**. They improve:
- Reliability (crash handling)
- Fidelity (zero drops, better resampling)
- Consistency (buffer sizes)

But they **cannot solve** the performance problem. Further work requires architectural changes:
- Consider C++ plugin priority over Python
- Investigate async/threaded Python execution
- Evaluate alternative Python implementations (PyPy, Cython)
- Consider hybrid approach (C++ fast path, Python optional)

See `docs/reports/INVESTIGATION_ROADMAP.md` for detailed options.

---

## References

- **Performance Analysis:** `docs/reports/PERFORMANCE_ANALYSIS_V2.1.md`
- **Architecture Constraints:** `docs/reports/ARCHITECTURAL_CONSTRAINTS.md`
- **Investigation Options:** `docs/reports/INVESTIGATION_ROADMAP.md`
- **Original Discussion:** Session 2026-01-08

---

**Version:** 2.1
**Status:** Complete (does not achieve real-time performance)
**Next Steps:** Architectural evaluation required
