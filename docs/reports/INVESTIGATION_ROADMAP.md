# Investigation Roadmap - Achieving Real-Time Performance

**Date:** 2026-01-08
**Context:** V2.1 software optimizations exhausted
**Goal:** Identify architectural changes to achieve real-time audio performance

---

## Problem Statement

The current NDA audio pipeline cannot maintain real-time performance (93.75 Hz) with Python plugins at 48kHz/512-sample configuration. The pipeline runs at **50-75% of required speed**, causing 340ms/second drift accumulation.

**Core Constraint:** Python plugin compatibility is mandatory, but Python execution is 10-400× slower than C++.

---

## Investigation Paths

This document outlines potential solutions, ordered by implementation complexity and risk. Each option includes:
- Feasibility analysis
- Implementation effort estimate
- Performance impact prediction
- Compatibility implications
- Trade-offs

---

## Path 1: Increase Buffer Size (Lowest Risk)

### Concept
Increase frame size from 512 → 2048 or 4096 samples to reduce frame rate and per-frame overhead.

### Analysis

**Current:**
```
Frame size: 512 samples
Frame rate: 93.75 Hz (10.66ms per frame)
Overhead per frame: 5-40ms Python calls
Result: Cannot keep up
```

**Proposed:**
```
Frame size: 2048 samples
Frame rate: 23.44 Hz (42.66ms per frame)
Overhead per frame: 5-40ms Python calls
Result: Likely sustainable
```

### Feasibility: ★★★★★ (Very High)

**Pros:**
- ✅ Simplest change (one config value)
- ✅ No code changes required
- ✅ Reduces frame rate by 4× (more time per frame)
- ✅ Amortizes Python overhead over more samples
- ✅ Backward compatible

**Cons:**
- ❌ Increases latency (10.66ms → 42.66ms minimum)
- ❌ May violate user's 10-50ms latency requirement
- ❌ Still has Python overhead

### Implementation

1. Change pipeline frame size:
   ```cpp
   // ProcessingPipeline.cpp
   frameCount_ = 2048;  // Was 512
   ```

2. Update all Python plugins:
   ```python
   self.buffer_size = 2048  // Was 512
   ```

3. Test and measure actual latency

### Expected Performance

```
Estimated frame rate: 23.44 Hz (target)
Python overhead: 5-40ms (unchanged)
Frame budget: 42.66ms

Calculation:
  If Python overhead avg = 15ms
  Processing time = 42.66ms
  Python overhead = 35% of budget (acceptable)

Expected result: LIKELY ACHIEVES REAL-TIME
```

### Decision Criteria

**✅ RECOMMEND if:**
- User can accept 42-85ms latency (2048-4096 samples)
- Quick win is priority over optimal performance

**❌ AVOID if:**
- User's 10-50ms latency requirement is strict
- Low latency is critical for use case (live monitoring, VoIP)

---

## Path 2: Hybrid Architecture (C++ Fast Path)

### Concept
Create a "fast path" for all-C++ pipelines that bypasses Python bridge entirely. Python plugins only used when explicitly loaded.

### Analysis

**Current Architecture:**
```
Pipeline → PythonPluginBridge → Python Plugin
         (always pays Python overhead)
```

**Proposed Architecture:**
```
IF all plugins are C++:
  Pipeline → C++ Plugin (fast path)

ELSE:
  Pipeline → PythonPluginBridge → Python Plugin (slow path)
```

### Feasibility: ★★★★☆ (High)

**Pros:**
- ✅ C++-only pipelines get full native performance
- ✅ Python plugins still supported (compatibility maintained)
- ✅ Users can choose performance vs. convenience
- ✅ Gradual migration path (convert critical plugins to C++)

**Cons:**
- ⚠️ Requires dual code paths (complexity)
- ⚠️ Plugin developers must choose language
- ⚠️ May fragment plugin ecosystem

### Implementation

#### Phase 1: Detect Plugin Type
```cpp
// PluginManager.cpp
bool isNativePlugin(const std::shared_ptr<BasePlugin>& plugin) {
    // Check if Python bridge or native C++
    return dynamic_cast<PythonPluginBridge*>(plugin.get()) == nullptr;
}
```

#### Phase 2: Direct C++ Calls
```cpp
// ProcessingPipeline.cpp
if (isNativePlugin(source_)) {
    // Fast path: direct call, no bridge
    source_->readAudio(workBuffer_);
} else {
    // Slow path: through Python bridge
    pythonBridge_->readAudio(workBuffer_);
}
```

#### Phase 3: Provide C++ Plugin Templates
```cpp
// Create easy-to-use templates for common plugins
template<typename Algorithm>
class ProcessorPlugin : public AudioProcessorPlugin {
    // Handle boilerplate, user just implements algorithm
};
```

### Expected Performance

```
C++ plugins:    <1ms per call (near-zero overhead)
Python plugins: 5-40ms per call (current behavior)

Mixed pipeline:
  C++ source + C++ processor + Python sink = FAST
  Python source + Python processor + Python sink = SLOW
```

### Decision Criteria

**✅ RECOMMEND if:**
- Long-term solution desired
- Willing to maintain dual paths
- Can migrate critical plugins to C++

**❌ AVOID if:**
- Pure Python pipeline required
- Cannot afford development time
- Plugin ecosystem fragmentation concern

---

## Path 3: Async Python Execution (Threading)

### Concept
Run Python plugins in separate thread, communicate via lock-free queues. Pipeline thread never blocks on Python.

### Analysis

**Current:**
```
Pipeline Thread:
  1. Call Python plugin (BLOCKS)
  2. Wait for Python to finish (BLOCKS on GIL)
  3. Continue processing
```

**Proposed:**
```
Pipeline Thread:
  1. Push work to queue (non-blocking)
  2. Continue processing

Python Thread:
  - Pop work from queue
  - Execute Python plugin
  - Push result to queue

Pipeline Thread:
  3. Pop result from queue (when ready)
```

### Feasibility: ★★★☆☆ (Medium)

**Pros:**
- ✅ Pipeline never blocks on Python
- ✅ Python can run slower without blocking audio
- ✅ GIL contention isolated to Python thread
- ✅ Maintains Python plugin compatibility

**Cons:**
- ❌ Complex implementation (lock-free queues, synchronization)
- ❌ Adds latency (queuing overhead + Python execution time)
- ❌ Requires careful buffer management
- ❌ May not help if Python still can't keep up

### Implementation

#### Phase 1: Work Queue Infrastructure
```cpp
// Lock-free single-producer, single-consumer queue
template<typename T>
class SPSCQueue {
    std::atomic<size_t> readIdx_;
    std::atomic<size_t> writeIdx_;
    std::vector<T> buffer_;

    bool push(const T& item);  // Non-blocking
    bool pop(T& item);         // Non-blocking
};
```

#### Phase 2: Python Worker Thread
```cpp
class PythonWorker {
    SPSCQueue<WorkItem> workQueue_;
    SPSCQueue<Result> resultQueue_;
    std::thread workerThread_;

    void workerLoop() {
        while (running_) {
            WorkItem work;
            if (workQueue_.pop(work)) {
                // Execute Python plugin
                Result result = executePython(work);
                resultQueue_.push(result);
            }
        }
    }
};
```

#### Phase 3: Pipeline Integration
```cpp
// Pipeline thread
void processAudioFrame() {
    // Send work to Python thread
    pythonWorker_.submitWork(workBuffer_);

    // Do other processing while Python runs
    // ...

    // Retrieve result (may block if not ready)
    if (!pythonWorker_.getResult(workBuffer_, timeout)) {
        // Python couldn't keep up
    }
}
```

### Expected Performance

**Best case:** Pipeline runs at full speed, Python catches up asynchronously

**Worst case:** Python still can't keep up, but pipeline doesn't block

**Likely case:** Adds 1-2 frame latency (queuing delay), but more stable

### Decision Criteria

**✅ RECOMMEND if:**
- Need Python compatibility
- Can tolerate added latency
- Have threading expertise

**❌ AVOID if:**
- Low latency critical
- Complexity budget limited
- Python still too slow (doesn't solve root cause)

---

## Path 4: Alternative Python Implementations

### Concept
Replace CPython with faster Python implementation (PyPy, Cython, or Nuitka).

### Option 4a: PyPy (JIT Compilation)

**What:** PyPy uses Just-In-Time compilation for faster execution

**Feasibility:** ★★☆☆☆ (Low)

**Pros:**
- ✅ Can be 5-10× faster than CPython for pure Python code
- ✅ Drop-in replacement (mostly compatible)

**Cons:**
- ❌ NumPy support limited/slower than CPython
- ❌ Ctypes interface may be incompatible
- ❌ GIL still present
- ❌ Startup time overhead

**Expected Performance:** Unlikely to help (NumPy-heavy code won't benefit)

### Option 4b: Cython (Compile Python to C)

**What:** Compile Python plugins to C extensions

**Feasibility:** ★★★☆☆ (Medium)

**Pros:**
- ✅ Near-C performance for compiled code
- ✅ Can release GIL in critical sections
- ✅ Incremental adoption (compile critical plugins only)

**Cons:**
- ⚠️ Requires rewriting plugins with Cython syntax
- ⚠️ Compilation required (not pure Python anymore)
- ⚠️ Debugging harder

**Expected Performance:** 10-100× speedup possible, but requires plugin rewrite

### Option 4c: Nuitka (Python → C++ Compiler)

**What:** Compile entire Python plugins to C++

**Feasibility:** ★★★☆☆ (Medium)

**Pros:**
- ✅ Produces native C++ extensions
- ✅ Faster than CPython
- ✅ No GIL in compiled code

**Cons:**
- ⚠️ Compilation step required
- ⚠️ Not all Python features supported
- ⚠️ Complex build process

**Expected Performance:** 5-20× speedup, but not guaranteed to reach real-time

### Decision Criteria

**✅ RECOMMEND Cython if:**
- Willing to rewrite plugins
- Need maximum Python compatibility
- Have compilation infrastructure

**❌ AVOID PyPy** (NumPy compatibility issues)

**⚠️ INVESTIGATE Nuitka** (worth prototyping)

---

## Path 5: Reduce Sample Rate / Frame Rate

### Concept
Lower internal processing to 24kHz or 32kHz instead of 48kHz.

### Analysis

```
Current: 48kHz, 512 samples = 93.75 Hz
Option 1: 32kHz, 512 samples = 62.5 Hz (-33% frame rate)
Option 2: 24kHz, 512 samples = 46.875 Hz (-50% frame rate)
```

### Feasibility: ★★★★☆ (High)

**Pros:**
- ✅ Reduces frame rate proportionally
- ✅ Simple to implement
- ✅ Reduces overall workload

**Cons:**
- ❌ Lower audio quality (frequency bandwidth reduced)
- ❌ May not meet user requirements
- ❌ Still has Python overhead

### Expected Performance

At 32kHz:
```
Frame rate: 62.5 Hz (16ms per frame)
Python overhead: 5-40ms
Result: May be sustainable
```

At 24kHz:
```
Frame rate: 46.875 Hz (21.3ms per frame)
Python overhead: 5-40ms
Result: Likely sustainable
```

### Decision Criteria

**✅ RECOMMEND if:**
- Audio quality not critical (voice only)
- Quick win needed
- Can accept lower fidelity

**❌ AVOID if:**
- Music/high-fidelity audio required
- User expects professional quality

---

## Path 6: Optimize Python Plugin Implementations

### Concept
Rewrite Python plugins to be more efficient using NumPy vectorization and reducing allocations.

### Analysis

**Current Issues in Plugins:**
- Memory allocations in hot path (`.copy()`, `.T`)
- Queue operations (copy overhead)
- Not leveraging NumPy vectorization

### Feasibility: ★★★☆☆ (Medium)

**Example Optimizations:**

#### 6a. Pre-allocate Buffers
```python
# BAD (allocates every frame):
def read_audio(self, buffer):
    data = self.queue.get()
    buffer.data[:] = data.T  # Allocates

# GOOD (reuse buffers):
def __init__(self):
    self._work_buffer = np.zeros((channels, frames), dtype=np.float32)

def read_audio(self, buffer):
    data = self.queue.get()
    np.transpose(data, out=self._work_buffer)
    buffer.data[:] = self._work_buffer  # No allocation
```

#### 6b. Reduce Queue Copying
```python
# BAD:
self.queue.put(indata.copy())  # Copy 1
data = self.queue.get()        # Copy 2
buffer.data[:] = data.T        # Copy 3

# GOOD: Use view
 instead of copy
self.queue.put(indata)  # No copy (if safe)
```

### Expected Performance

**Optimistic:** 20-30% speedup
**Realistic:** 10-15% speedup
**Pessimistic:** 5% speedup (GIL still dominant)

### Decision Criteria

**✅ RECOMMEND as:**
- Supplementary optimization
- Low risk, incremental improvement

**❌ DON'T EXPECT:**
- To solve root problem
- Achieve real-time without other changes

---

## Path 7: GPU Acceleration (CUDA/OpenCL)

### Concept
Offload audio processing to GPU for parallel computation.

### Feasibility: ★☆☆☆☆ (Very Low)

**Pros:**
- ✅ Massive parallelism available
- ✅ High throughput for certain algorithms

**Cons:**
- ❌ High latency (CPU ↔ GPU transfer)
- ❌ Complex implementation
- ❌ Not suitable for small buffers (512 samples)
- ❌ Requires GPU hardware
- ❌ Overkill for simple operations (sine wave, I/O)

**Verdict:** Not recommended for real-time audio with small buffers

---

## Path 8: Dedicated Audio OS/RTOS

### Concept
Run pipeline on real-time operating system with guaranteed scheduling.

### Feasibility: ★★☆☆☆ (Low)

**Pros:**
- ✅ Guaranteed real-time performance
- ✅ Predictable latency
- ✅ Priority scheduling

**Cons:**
- ❌ Requires different OS (Linux RT-PREEMPT, QNX, VxWorks)
- ❌ May not support Python easily
- ❌ Complex deployment
- ❌ Doesn't solve Python speed issue

**Verdict:** Addresses symptom, not root cause

---

## Recommended Investigation Order

### Phase 1: Quick Wins (Week 1)
1. **Try larger buffer sizes** (Path 1)
   - Test 1024, 2048, 4096 samples
   - Measure latency vs. performance
   - **Easiest to validate**

2. **Optimize Python plugins** (Path 6)
   - Remove allocations
   - Pre-allocate buffers
   - Profile with improved code

### Phase 2: Architecture Evaluation (Week 2-3)
3. **Prototype hybrid C++ fast path** (Path 2)
   - Implement detection mechanism
   - Test performance with C++ plugins
   - Measure overhead of switching

4. **Evaluate Cython compilation** (Path 4b)
   - Port one plugin to Cython
   - Measure speedup
   - Assess feasibility for all plugins

### Phase 3: Advanced Solutions (Week 4+)
5. **Consider async execution** (Path 3)
   - Prototype lock-free queues
   - Test latency impact
   - Evaluate complexity

6. **Investigate Nuitka** (Path 4c)
   - Compile test plugin
   - Measure performance
   - Assess compatibility

### Not Recommended
- ❌ PyPy (NumPy incompatibility)
- ❌ Lower sample rate (quality impact)
- ❌ GPU acceleration (unsuitable for small buffers)
- ❌ RTOS (doesn't solve Python speed)

---

## Decision Matrix

| Path | Complexity | Risk | Time | Performance Gain | Compatibility |
|------|------------|------|------|------------------|---------------|
| 1. Larger buffers | Low | Low | 1 day | ★★★★☆ | ✅ High |
| 2. Hybrid C++ | Medium | Medium | 2 weeks | ★★★★★ | ✅ High |
| 3. Async Python | High | High | 4 weeks | ★★★☆☆ | ✅ High |
| 4a. PyPy | Low | High | 1 week | ★☆☆☆☆ | ❌ Low |
| 4b. Cython | Medium | Medium | 3 weeks | ★★★★☆ | ⚠️ Medium |
| 4c. Nuitka | Medium | Medium | 2 weeks | ★★★☆☆ | ⚠️ Medium |
| 5. Lower sample rate | Low | Medium | 1 day | ★★★☆☆ | ⚠️ Medium |
| 6. Optimize plugins | Low | Low | 1 week | ★★☆☆☆ | ✅ High |
| 7. GPU | Very High | High | 8+ weeks | ★☆☆☆☆ | ❌ Low |
| 8. RTOS | Very High | High | 8+ weeks | ★★☆☆☆ | ❌ Low |

**Legend:**
- ★★★★★ = Likely solves problem
- ★★★☆☆ = May help significantly
- ★★☆☆☆ = Minor improvement
- ★☆☆☆☆ = Unlikely to help

---

## Recommended Path Forward

### Immediate Action (This Week)
```
1. Test buffer size = 2048 samples
   → If acceptable latency: SHIP IT
   → If unacceptable: Continue to Phase 2

2. Optimize Python plugin code
   → Remove allocations, pre-allocate buffers
   → Measure improvement (expect 10-20%)
```

### Short-Term (Next Month)
```
3. Implement hybrid C++ fast path
   → Create native C++ versions of critical plugins
   → Measure performance (expect near-zero overhead)
   → Provides migration path

4. Evaluate Cython for critical plugins
   → Port 1-2 plugins to Cython
   → Measure speedup (expect 10-100×)
   → Assess effort vs. reward
```

### Long-Term (3+ Months)
```
5. Async Python execution
   → Prototype lock-free queue architecture
   → Test in parallel with hybrid approach
   → Use if Python plugins still needed

6. Establish plugin performance guidelines
   → Document C++ vs. Python tradeoffs
   → Provide plugin templates
   → Encourage high-performance plugin development
```

---

## Success Metrics

Track these metrics through each optimization:

```
Target Performance:
  Frame rate:         >90 Hz (>95% of 93.75 Hz)
  Frame time:         <11ms (within budget)
  Drift accumulation: <100ms over 10 seconds
  Plugin call time:   <1ms average
  GIL acquisition:    <1ms

Acceptable Performance:
  Frame rate:         >85 Hz (>90% of 93.75 Hz)
  Frame time:         <12ms
  Drift accumulation: <200ms over 10 seconds
  Plugin call time:   <2ms average
  GIL acquisition:    <2ms

Unacceptable (Current):
  Frame rate:         50-75 Hz (53-80%)
  Frame time:         15-66ms
  Drift accumulation: 3,429ms over 10 seconds
  Plugin call time:   5-40ms average
  GIL acquisition:    0-64ms
```

---

## Risk Assessment

### High-Risk Approaches
- **Async Python execution**: Complex, may not solve problem
- **PyPy**: Poor NumPy support
- **RTOS**: Massive infrastructure change

### Medium-Risk Approaches
- **Cython**: Requires rewriting plugins
- **Nuitka**: Compilation complexity
- **Lower sample rate**: Quality degradation

### Low-Risk Approaches
- **Larger buffers**: Simple, reversible
- **Hybrid C++**: Additive, doesn't break existing code
- **Plugin optimization**: Incremental improvements

---

## Team Assignments (Suggested)

### Team Member A: Quick Wins
- Test buffer size variations (1024, 2048, 4096)
- Optimize existing Python plugin code
- Measure and document results

### Team Member B: Architecture Exploration
- Prototype hybrid C++ fast path
- Implement plugin type detection
- Create C++ plugin templates

### Team Member C: Python Compilation
- Evaluate Cython for SoundDevice plugins
- Test Nuitka compilation
- Compare performance vs. complexity

### Team Member D: Monitoring & Testing
- Implement comprehensive performance tests
- Create automated benchmarking
- Track metrics across optimization attempts

---

## Conclusion

The NDA audio pipeline performance problem has **no silver bullet**. Multiple approaches should be pursued in parallel:

1. **Immediate:** Increase buffer size (fastest validation)
2. **Short-term:** Hybrid C++ architecture (long-term solution)
3. **Medium-term:** Optimize Python or compile to native
4. **Continuous:** Monitor metrics and iterate

**Key Insight:** Python is fundamentally unsuitable for 48kHz/512-sample real-time audio. The solution requires either:
- Accepting higher latency (larger buffers)
- Moving critical plugins to C++
- Compiling Python to native code

---

## Questions for Team Discussion

1. **Can we accept 40-80ms latency?** (Enables larger buffers)
2. **Which plugins are most critical?** (Prioritize C++ conversion)
3. **What's our Python commitment?** (Determines path forward)
4. **What's our timeline?** (Affects scope of changes)
5. **What are our use cases?** (Dictates latency requirements)

---

## References

- **Performance Analysis:** `docs/reports/PERFORMANCE_ANALYSIS_V2.1.md`
- **Changelog:** `docs/CHANGELOG_V2.1.md`
- **Test Logs:** Session 2026-01-08
- **Python GIL Documentation:** https://docs.python.org/3/c-api/init.html#thread-state-and-the-global-interpreter-lock
- **Real-Time Audio Best Practices:** [Ross Bencina - Real-Time Audio Programming](http://www.rossbencina.com/code/real-time-audio-programming-101-time-waits-for-nothing)
