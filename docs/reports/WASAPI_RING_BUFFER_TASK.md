# WASAPI Ring Buffer Implementation - Task Document

**Date:** 2026-01-11
**Priority:** CRITICAL
**Complexity:** Medium
**Estimated Effort:** 1-2 days
**Risk Level:** LOW (incremental change, proven pattern)

---

## Executive Summary

Implement internal ring buffers in Windows WASAPI audio plugins to eliminate packet timing mismatches between WASAPI's asynchronous packet delivery and the pipeline's synchronous polling model. This will resolve the "cutted/glitched" audio issue observed in microphone capture while maintaining 100% audio fidelity and quick deployment timeline.

**Status**: READY FOR IMPLEMENTATION
**Alignment**: Fully compliant with one-week-release.md quick-deployment and 100% fidelity policies

---

## Problem Statement

### Current Behavior

**Observed Symptoms:**
- Microphone audio sounds "cutted/glitched" when recorded to WAV file
- 2 underruns consistently throughout test duration
- Audio has small gaps/discontinuities causing quality degradation

**From Test Logs (2026-01-11 01:00:42):**
```
[Pipeline] Read #2 readOk=0 isRunning=1
[Pipeline] Audio read started failing
[Pipeline] Audio read recovered after 1 failures
[WindowsMicrophone] Stats: 48000 frames (1s), 2 underruns
[WindowsMicrophone] Stats: 98880 frames (2.06s), 2 underruns
[WindowsMicrophone] Stats: 150240 frames (3.13s), 2 underruns
```

**Pattern**: Consistently 2 underruns total, occurring at startup and occasionally during steady-state operation.

### Root Cause Analysis

**Timing Mismatch Between WASAPI and Pipeline:**

1. **WASAPI's Asynchronous Model:**
   - WASAPI delivers audio packets asynchronously at hardware's pace
   - Typical packet size: 480 frames (~10ms at 48kHz)
   - Packets arrive irregularly based on Windows scheduling and device timing
   - No guarantee packets are available exactly when pipeline polls

2. **Pipeline's Synchronous Model:**
   - Pipeline polls `readAudio()` at fixed intervals
   - Current frame size: 512 frames (~10.67ms at 48kHz)
   - Real-time pacing: sleeps to maintain exact 1.0× playback rate
   - Expects data immediately on every poll

3. **The Mismatch:**
   ```
   Time:        0ms      10ms     20ms     30ms     40ms
   WASAPI:   [packet1]         [packet2]        [packet3]
   Pipeline:    [poll]           [poll]           [poll]
                  ↓               ↓                ↓
              SUCCESS          UNDERRUN         SUCCESS
   ```

   When pipeline polls at 10.67ms but WASAPI packet hasn't arrived yet → `GetNextPacketSize()` returns 0 → `readAudio()` returns false → underrun → audio gap.

### Why This Happens

**From WindowsMicrophoneSourcePlugin.cpp (line 361-376):**
```cpp
// Get next packet from WASAPI
UINT32 packetFrames = 0;
HRESULT hr = captureClient_->GetNextPacketSize(&packetFrames);

if (packetFrames == 0) {
    // No data available - return silence
    buffer.clear();
    underruns_++;
    return false;  // ← This causes the gap in audio
}
```

**Problem**: No buffering between WASAPI's async delivery and pipeline's sync polling.

---

## Proposed Solution: Ring Buffer Implementation

### Architecture Overview

Add an internal **thread-safe ring buffer** to WASAPI plugins that bridges the timing gap:

```
WASAPI Device → [Ring Buffer] → Pipeline
    (async)       (decouples)      (sync)
```

**How It Works:**

1. **WASAPI Capture Thread** (background):
   - Continuously polls WASAPI for new packets
   - Writes incoming packets to ring buffer
   - Runs independently of pipeline timing

2. **Ring Buffer** (thread-safe):
   - Fixed-size circular buffer (e.g., 9600 frames = 200ms at 48kHz)
   - Stores audio samples in FIFO order
   - Thread-safe read/write operations

3. **Pipeline's readAudio()** (foreground):
   - Reads requested frames from ring buffer
   - Always succeeds if buffer has enough data
   - Returns silence if buffer underruns (rare with proper sizing)

### Benefits

✅ **Eliminates timing mismatch** - WASAPI packets accumulate in buffer, pipeline reads smoothly
✅ **100% audio fidelity** - no dropped packets, no gaps
✅ **Low latency** - buffer adds ~50-100ms (configurable)
✅ **Proven pattern** - used by PortAudio, RtAudio, JUCE, Audacity
✅ **Low risk** - isolated to WASAPI plugins, no pipeline changes
✅ **Quick implementation** - 1-2 days work

---

## Technical Requirements

### 1. Ring Buffer Implementation

**Core Functionality:**
- **Thread-safe circular buffer** for audio samples
- **Atomic read/write pointers** with proper memory ordering
- **Configurable size** (default: ~100-200ms of audio)
- **Overflow/underflow detection** and reporting
- **Support for planar audio** (separate buffers per channel)

**API Requirements:**
```cpp
class RingBuffer {
public:
    // Initialize with capacity in frames
    bool initialize(int channels, int capacityFrames);

    // Write audio to buffer (called by WASAPI thread)
    // Returns number of frames actually written
    int write(const float* const* channelData, int frameCount);

    // Read audio from buffer (called by pipeline)
    // Returns number of frames actually read
    int read(float** channelData, int frameCount);

    // Query buffer state
    int getAvailableRead() const;   // Frames available to read
    int getAvailableWrite() const;  // Frames available to write
    int getCapacity() const;

    // Metrics
    uint64_t getOverruns() const;   // Write when full
    uint64_t getUnderruns() const;  // Read when empty
};
```

### 2. WASAPI Microphone Plugin Changes

**File**: `plugins_src/WindowsMicrophoneSourcePlugin.cpp`

**Required Changes:**

1. **Add Ring Buffer Member:**
   ```cpp
   class WindowsMicrophoneSourcePlugin {
   private:
       RingBuffer ringBuffer_;
       std::thread captureThread_;
       std::atomic<bool> captureThreadRunning_;
   };
   ```

2. **Initialize Ring Buffer in initialize():**
   ```cpp
   // After WASAPI initialization, before returning
   int bufferCapacity = (sampleRate_ * 200) / 1000;  // 200ms
   if (!ringBuffer_.initialize(channels_, bufferCapacity)) {
       std::cerr << "[WindowsMicrophone] Ring buffer init failed\n";
       return false;
   }
   ```

3. **Start Capture Thread in start():**
   ```cpp
   bool start() override {
       // ... existing WASAPI Start() code ...

       // Start background capture thread
       captureThreadRunning_ = true;
       captureThread_ = std::thread([this]() {
           this->captureThreadFunc();
       });

       return true;
   }
   ```

4. **Stop Capture Thread in stop():**
   ```cpp
   void stop() override {
       captureThreadRunning_ = false;
       if (captureThread_.joinable()) {
           captureThread_.join();
       }

       // ... existing WASAPI Stop() code ...
   }
   ```

5. **Implement Capture Thread:**
   ```cpp
   void captureThreadFunc() {
       std::vector<std::vector<float>> tempBuffer;
       tempBuffer.resize(channels_);

       while (captureThreadRunning_) {
           // Poll WASAPI for packet
           UINT32 packetFrames = 0;
           captureClient_->GetNextPacketSize(&packetFrames);

           if (packetFrames == 0) {
               // No data yet - sleep briefly and retry
               std::this_thread::sleep_for(std::chrono::milliseconds(1));
               continue;
           }

           // Get packet from WASAPI
           BYTE* captureData = nullptr;
           DWORD flags = 0;
           captureClient_->GetBuffer(&captureData, &packetFrames, &flags, nullptr, nullptr);

           // Convert interleaved → planar
           // Resize temp buffer if needed
           for (int ch = 0; ch < channels_; ++ch) {
               tempBuffer[ch].resize(packetFrames);
           }

           float* src = reinterpret_cast<float*>(captureData);
           for (UINT32 frame = 0; frame < packetFrames; ++frame) {
               for (int ch = 0; ch < channels_; ++ch) {
                   tempBuffer[ch][frame] = *src++;
               }
           }

           // Write to ring buffer
           float* channelPtrs[8];  // Max 8 channels
           for (int ch = 0; ch < channels_; ++ch) {
               channelPtrs[ch] = tempBuffer[ch].data();
           }

           int written = ringBuffer_.write(channelPtrs, packetFrames);
           if (written < (int)packetFrames) {
               // Buffer overflow - some data lost
               overruns_++;
           }

           // Release WASAPI buffer
           captureClient_->ReleaseBuffer(packetFrames);
       }
   }
   ```

6. **Simplify readAudio():**
   ```cpp
   bool readAudio(AudioBuffer& buffer) override {
       if (state_ != PluginState::Running) {
           buffer.clear();
           return false;
       }

       int requestedFrames = buffer.getFrameCount();

       // Read from ring buffer
       float* channelPtrs[8];
       for (int ch = 0; ch < channels_; ++ch) {
           channelPtrs[ch] = buffer.getChannelData(ch);
       }

       int framesRead = ringBuffer_.read(channelPtrs, requestedFrames);

       if (framesRead < requestedFrames) {
           // Buffer underrun - pad with silence
           for (int ch = 0; ch < channels_; ++ch) {
               float* data = buffer.getChannelData(ch);
               for (int i = framesRead; i < requestedFrames; ++i) {
                   data[i] = 0.0f;
               }
           }
           underruns_++;
       }

       framesCaptured_ += framesRead;
       readCalls_++;

       return true;  // Always return true (silence on underrun, not failure)
   }
   ```

### 3. WASAPI Speaker Plugin Changes

**File**: `plugins_src/WindowsSpeakerSinkPlugin.cpp`

**Similar pattern** as microphone but inverted:
- **Playback thread** continuously reads from ring buffer and feeds WASAPI
- **writeAudio()** writes to ring buffer (always succeeds if space available)

**Implementation:**
1. Add ring buffer member
2. Start playback thread in start()
3. Playback thread: reads from ring buffer, writes to WASAPI
4. writeAudio(): writes to ring buffer, returns false only if full

---

## Implementation Details

### Ring Buffer Sizing

**Formula**: `bufferSizeFrames = (sampleRate × bufferMs) / 1000`

**Recommendations:**
- **Minimum**: 50ms (2400 frames at 48kHz) - tight but low latency
- **Default**: 100ms (4800 frames at 48kHz) - balanced
- **Conservative**: 200ms (9600 frames at 48kHz) - maximum stability
- **WASAPI buffer size**: Already 200ms (9600 frames) - ring buffer should match or exceed

**Tradeoff**:
- Smaller buffer = lower latency, higher risk of underruns
- Larger buffer = higher latency, more tolerance for timing jitter

**Recommendation for Bridge Mode**: Start with 200ms (matches WASAPI buffer size), can reduce later if testing shows stable performance.

### Thread Safety Considerations

**Critical Sections:**
1. Ring buffer read/write pointers
2. Overflow/underrun counters
3. Plugin state transitions (start/stop)

**Synchronization Strategy:**
- Use `std::atomic<int>` for read/write pointers
- Use `std::atomic<uint64_t>` for counters
- Use mutex only for state transitions (start/stop)
- NO mutex in audio hot path (read/write operations)

**Memory Ordering:**
- Write pointer: `memory_order_release` when advancing
- Read pointer: `memory_order_acquire` when reading
- Ensures proper visibility across threads

### Platform Considerations

**Windows-Specific:**
- Use `SetThreadPriority(THREAD_PRIORITY_TIME_CRITICAL)` for capture/playback threads
- Consider `AvSetMmThreadCharacteristics("Pro Audio")` for reduced latency
- Handle COM thread affinity (capture thread needs COM initialized)

**Cross-Platform Future:**
- Ring buffer implementation should be platform-agnostic
- Only WASAPI-specific code in plugin classes
- Easy to adapt for PulseAudio/ALSA/CoreAudio later

---

## Testing Requirements

### Unit Tests

**Ring Buffer Correctness:**
1. Write N frames, read N frames → verify data matches
2. Write until full → verify overflow detection
3. Read when empty → verify underrun detection
4. Wrap-around test → write/read across buffer boundary
5. Multi-threaded stress test → simultaneous read/write

### Integration Tests

**Microphone Capture:**
1. **Test 1**: Microphone → WAV File (5 minutes)
   - **Expected**: Clean audio, no gaps, no glitches
   - **Metrics**: <5 underruns total, <1 overrun
   - **Verification**: Play back WAV file, listen for artifacts

2. **Test 2**: Microphone → Speaker (Loopback)
   - **Expected**: Hear yourself clearly with consistent latency
   - **Metrics**: <10 underruns in 5 minutes
   - **Verification**: No audible dropouts or clicks

3. **Test 3**: Microphone → WAV File (20 minutes soak test)
   - **Expected**: Stable operation, no drift accumulation
   - **Metrics**: <20 underruns total, <5 overruns
   - **Verification**: No pipeline failures, consistent audio quality

**Speaker Playback:**
1. **Test 1**: SineWave → Speaker (5 minutes)
   - **Expected**: Clean 440Hz tone, no glitches
   - **Metrics**: <5 underruns, <5 overruns
   - **Verification**: Consistent pitch, no crackling

2. **Test 2**: SineWave → Speaker (20 minutes soak test)
   - **Expected**: Stable continuous playback
   - **Metrics**: <20 total errors
   - **Verification**: No pitch drift, no failures

### Performance Tests

**Latency Measurement:**
- Measure round-trip latency: Microphone → Speaker
- **Target**: <150ms total latency (50ms source + 50ms processing + 50ms sink)
- **Method**: Impulse response test (play click, measure time to hear it)

**CPU Usage:**
- Monitor CPU usage during 5-minute test
- **Target**: <10% CPU for audio threads
- **Method**: Task Manager / Performance Monitor

**Drift Analysis:**
- Run 20-minute test, measure drift accumulation
- **Target**: <1 second total drift
- **Method**: Compare processed samples vs wall time

---

## Success Criteria

### Functional Requirements

✅ **Zero audio gaps** - no "cutted/glitched" audio in recordings
✅ **<5 underruns per 5-minute test** - exceptional stability
✅ **<1 overrun per 5-minute test** - no buffer overflows
✅ **Clean WAV recordings** - playback sounds natural, no artifacts
✅ **Loopback works** - Microphone → Speaker is intelligible

### Performance Requirements

✅ **Latency <150ms** - acceptable for Bridge Mode communication
✅ **CPU <10%** - efficient, low system impact
✅ **No drift accumulation** - <1 second drift over 20 minutes
✅ **No pipeline failures** - 20-minute soak test completes successfully

### Code Quality Requirements

✅ **Thread-safe implementation** - no race conditions
✅ **Proper error handling** - logs overflow/underflow events
✅ **Clear diagnostics** - metrics show buffer health
✅ **Maintainable code** - well-commented, follows project style

---

## Risk Assessment

### Low Risk Items

✅ Ring buffer pattern is well-understood and proven
✅ Isolated to WASAPI plugins - no pipeline changes
✅ Easy to test incrementally (microphone first, then speaker)
✅ Simple rollback if issues arise (revert to current version)

### Medium Risk Items

⚠️ Thread synchronization bugs (mitigated by atomic operations)
⚠️ Buffer sizing (mitigated by conservative 200ms default)
⚠️ COM threading in capture thread (mitigated by proper initialization)

### Mitigation Strategies

1. **Thorough testing** - unit tests + integration tests + soak tests
2. **Conservative defaults** - 200ms buffers for maximum stability
3. **Clear logging** - diagnose issues quickly if they occur
4. **Incremental rollout** - test microphone first, then speaker
5. **Rollback plan** - keep current version for quick revert

---

## Implementation Checklist

### Phase 1: Ring Buffer Class (4-6 hours)

- [ ] Create `include/audio/RingBuffer.h` header
- [ ] Implement thread-safe circular buffer with atomic pointers
- [ ] Add overflow/underflow detection
- [ ] Write unit tests for correctness
- [ ] Verify thread safety with stress test

### Phase 2: Microphone Plugin (6-8 hours)

- [ ] Add RingBuffer member to WindowsMicrophoneSourcePlugin
- [ ] Implement captureThreadFunc() with WASAPI polling
- [ ] Update initialize() to create ring buffer
- [ ] Update start() to launch capture thread
- [ ] Update stop() to join capture thread
- [ ] Simplify readAudio() to read from ring buffer
- [ ] Add COM initialization to capture thread
- [ ] Test: Microphone → WAV File (verify no gaps)
- [ ] Test: Microphone → Speaker (verify loopback works)
- [ ] 20-minute soak test

### Phase 3: Speaker Plugin (6-8 hours)

- [ ] Add RingBuffer member to WindowsSpeakerSinkPlugin
- [ ] Implement playbackThreadFunc() with WASAPI feeding
- [ ] Update initialize() to create ring buffer
- [ ] Update start() to launch playback thread
- [ ] Update stop() to join playback thread
- [ ] Simplify writeAudio() to write to ring buffer
- [ ] Test: SineWave → Speaker (verify clean tone)
- [ ] Test: Microphone → Speaker (full pipeline)
- [ ] 20-minute soak test

### Phase 4: Polish and Documentation (2-4 hours)

- [ ] Add diagnostic logging (buffer fill level, overruns, underruns)
- [ ] Update plugin documentation
- [ ] Create user-facing guide for buffer size tuning
- [ ] Final validation tests
- [ ] Update CHANGELOG

---

## Technical References

### Industry Examples

**PortAudio**: https://github.com/PortAudio/portaudio/blob/master/src/common/pa_ringbuffer.c
- Lock-free ring buffer implementation
- Memory barrier usage for thread safety
- Proven in millions of deployments

**RtAudio**: https://github.com/thestk/rtaudio/blob/master/RtAudio.cpp
- WASAPI buffering strategy (lines 2500-3000)
- Background thread feeding WASAPI
- Error handling patterns

**JUCE**: https://github.com/juce-framework/JUCE/blob/master/modules/juce_audio_basics/buffers/juce_AbstractFifo.h
- High-performance FIFO buffer
- Lock-free design
- Used in thousands of commercial audio applications

### Microsoft Documentation

**WASAPI Streaming**: https://learn.microsoft.com/en-us/windows/win32/coreaudio/stream-management
- Packet-based capture model
- Event-driven vs polling tradeoffs
- Buffer management best practices

**Thread Priorities**: https://learn.microsoft.com/en-us/windows/win32/procthread/scheduling-priorities
- THREAD_PRIORITY_TIME_CRITICAL usage
- Pro Audio thread characteristics
- Latency optimization

---

## Alignment with Project Policies

### One-Week-Release Strategy

✅ **Quick deployment** - 1-2 days implementation vs weeks for async rewrite
✅ **Low risk** - incremental change, easy rollback
✅ **Proven pattern** - industry standard solution

### 100% Audio Fidelity

✅ **Zero packet loss** - ring buffer captures all WASAPI packets
✅ **Smooth playback** - no gaps or glitches
✅ **High quality** - maintains full 48kHz/32-bit float precision

### Reliability and Stability

✅ **Robust error handling** - graceful overflow/underflow recovery
✅ **Clear diagnostics** - metrics reveal buffer health
✅ **Well-tested pattern** - used by professional audio software worldwide

---

## Next Steps

**For Implementation Agent:**

1. Read this document thoroughly
2. Create detailed implementation plan with milestones
3. Start with Ring Buffer class (test thoroughly)
4. Implement Microphone plugin (test before proceeding)
5. Implement Speaker plugin (full pipeline testing)
6. Run all success criteria tests
7. Document any deviations or issues encountered

**For User:**

- Review this document
- Approve approach
- Schedule implementation session
- Prepare testing environment (microphone, speakers, quiet room)

---

**Document Status**: ✅ READY FOR IMPLEMENTATION
**Next Action**: Create implementation plan and begin Phase 1 (Ring Buffer Class)
**Expected Completion**: 2026-01-12 (1-2 day timeline)
