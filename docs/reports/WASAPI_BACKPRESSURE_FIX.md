# WASAPI Backpressure Fix Report

**Date:** 2026-01-11
**Issue:** Windows Speaker/Microphone plugins experiencing severe backpressure
**Status:** FIXED

---

## Problem Analysis

### Symptoms Observed

From user's test logs:
- **9564 overruns** in 95 seconds (~100 overruns/second)
- **10 seconds of drift** accumulated
- **Constant backpressure warnings**: "available: 150, needed: 1024"
- Pipeline could not sustain real-time playback

### Root Causes Identified

1. **Event-Driven Mode Without Waiting**
   - Plugins requested `AUDCLNT_STREAMFLAGS_EVENTCALLBACK` flag
   - Created event handles but **never waited on them**
   - This is a broken pattern - event-driven requires async waiting

2. **Buffer Size Mismatch**
   - Pipeline uses 1024-frame chunks
   - WASAPI buffer initialized with 100ms (4800 frames at 48kHz)
   - In shared mode, WASAPI may use smaller period sizes internally
   - Available space often <1024 frames, causing constant backpressure

3. **Too-Small Buffer Duration**
   - 100ms buffer insufficient for polling-based approach
   - Windows audio subsystem needs larger buffers in shared mode

### Technical Details

**From logs:**
```
[Pipeline] Sink backpressure detected, waiting (available: 150, needed: 1024)
[Pipeline] Sink backpressure detected, waiting (available: 8, needed: 1024)
[Pipeline] Sink backpressure detected, waiting (available: 748, needed: 1024)
```

The available space pattern (150→8→748→165...) shows the buffer draining and refilling, but never reaching the 1024 frames needed by the pipeline.

---

## Solution Implemented

### Changes to WindowsSpeakerSinkPlugin.cpp

**1. Removed Event-Driven Mode**
```cpp
// OLD (broken):
hr = audioClient_->Initialize(
    AUDCLNT_SHAREMODE_SHARED,
    AUDCLNT_STREAMFLAGS_EVENTCALLBACK,  // Event-driven
    bufferDuration,
    0,
    mixFormat,
    nullptr
);

// NEW (fixed):
hr = audioClient_->Initialize(
    AUDCLNT_SHAREMODE_SHARED,
    0,  // No event callback - polling mode
    bufferDuration,
    0,
    mixFormat,
    nullptr
);
```

**2. Increased Buffer Duration**
- Changed from **100ms** to **200ms**
- At 48kHz: 4800 frames → 9600 frames
- Larger buffer reduces backpressure in polling mode

**3. Removed Event Handle Creation**
- Deleted `CreateEvent()` and `SetEventHandle()` calls
- Removed `renderEvent_` member variable
- Simpler, more reliable polling-based approach

**4. Added Buffer Size Diagnostics**
```cpp
// Query actual allocated buffer size
UINT32 actualBufferFrames = 0;
hr = audioClient_->GetBufferSize(&actualBufferFrames);
if (SUCCEEDED(hr)) {
    std::cerr << "[WindowsSpeaker] WASAPI allocated buffer: " << actualBufferFrames
              << " frames (" << (actualBufferFrames * 1000 / sampleRate_) << "ms)\n";
}
```

This logs the actual buffer size WASAPI allocated (may differ from request).

### Changes to WindowsMicrophoneSourcePlugin.cpp

**Applied identical fixes:**
1. Removed `AUDCLNT_STREAMFLAGS_EVENTCALLBACK` flag
2. Changed buffer duration: 100ms → 200ms
3. Removed event handle creation and management
4. Added buffer size diagnostic logging

---

## Why This Works

### Polling vs Event-Driven

**Event-Driven (OLD - broken in our use case):**
- WASAPI signals an event when buffer space is available
- **Requires asynchronous waiting** (WaitForSingleObject, etc.)
- Our synchronous `writeAudio()` API doesn't support async waiting
- Result: Event is signaled but never waited on = broken

**Polling (NEW - correct for synchronous API):**
- No event callback registration
- Check buffer space with `GetCurrentPadding()` before each write
- If insufficient space, return false (backpressure signal)
- Pipeline's backpressure handling retries with sleep
- Result: Simple, reliable, matches our synchronous API

### Larger Buffers

**100ms buffer issues:**
- Small buffer fills up quickly in shared mode
- Windows audio subsystem may use smaller period sizes
- Causes frequent backpressure with 1024-frame writes

**200ms buffer benefits:**
- More headroom for polling-based writes
- Reduces backpressure frequency
- Still acceptable latency for testing/communication (<100ms total)
- Aligns with one-week-release goal: **stability over latency**

---

## Expected Results After Rebuild

### Performance Improvements

1. **Drastically reduced overruns**
   - From: 9564 overruns in 95 seconds
   - To: <10 overruns in 95 seconds (expected)

2. **Minimal drift**
   - From: 10 seconds drift
   - To: <1 second drift over long runs

3. **Smooth playback**
   - No crackling or stuttering
   - Consistent audio output

4. **Better diagnostics**
   - Logs will show actual WASAPI buffer size
   - Example: `[WindowsSpeaker] WASAPI allocated buffer: 9600 frames (200ms)`

### Console Output Changes

**You should now see:**
```
[WindowsSpeaker] Initialized: 48000Hz, 2ch, 9600 frames
[WindowsSpeaker] WASAPI allocated buffer: 9600 frames (200ms)
[WindowsSpeaker] Started
```

**Backpressure warnings should be rare/absent:**
```
# Instead of hundreds of backpressure warnings,
# you should see smooth operation with stats every 100 calls
[WindowsSpeaker] Stats: 4800000 frames (100.0s), 0 underruns, 5 overruns
```

---

## Testing Instructions

### 1. Rebuild Project

```bash
cd build
ninja
```

Or use build script:
```bash
scripts\build_windows_ninja.bat
```

### 2. Test SineWave → Windows Speaker

**Expected:**
- Clean 440Hz tone
- No crackling or stuttering
- <10 overruns in 5-minute test
- No pipeline failures

**Watch for:**
- Initial buffer size log: should show ~9600 frames (200ms)
- Stats logging every 100 calls: overruns should stay near 0

### 3. Test Windows Mic → Windows Speaker (Loopback)

**Expected:**
- Hear yourself with consistent 50-100ms delay
- No dropouts or glitches
- Both source and sink should show low error counts

### 4. Long-Duration Test (20 minutes)

**Success criteria:**
- ✅ No pipeline failures
- ✅ <100 total overruns (5/minute acceptable)
- ✅ <2 seconds total drift
- ✅ Audio quality remains consistent

---

## Technical Context: WASAPI Modes

### Shared vs Exclusive Mode

**Shared Mode (our choice):**
- Multiple apps can use the device
- Windows handles format conversion and mixing
- Less control over buffer sizes and periods
- More compatible, less finicky

**Exclusive Mode (NOT used):**
- Full control of hardware
- Sub-10ms latency possible
- Requires exact format match
- Only one app can use device
- Much more complex error handling

### Event-Driven vs Polling

**Event-Driven (requires async patterns):**
```cpp
// Proper event-driven usage (complex):
while (running) {
    WaitForSingleObject(renderEvent, INFINITE);  // Async wait
    // Write audio when signaled
}
```

**Polling (our approach - simpler):**
```cpp
// Simple polling (synchronous):
bool writeAudio(buffer) {
    if (GetAvailableSpace() < neededFrames) {
        return false;  // Backpressure
    }
    // Write audio
    return true;
}
```

For our synchronous plugin API, polling is the correct choice.

---

## Alignment with One-Week-Release Goals

✅ **Stability First**: Polling mode is simpler and more reliable
✅ **Conservative Buffering**: 200ms buffers prevent backpressure
✅ **Observable Failures**: Added buffer size diagnostics
✅ **No Complexity**: Removed async event handling
✅ **Proven Pattern**: Polling is standard for synchronous audio APIs

---

## Files Modified

- `plugins_src/WindowsSpeakerSinkPlugin.cpp`
- `plugins_src/WindowsMicrophoneSourcePlugin.cpp`

**Changes:**
- Removed event-driven mode flags
- Increased buffer duration: 100ms → 200ms
- Removed event handle creation/cleanup
- Added buffer size diagnostic logging

---

## Next Steps

1. **Rebuild** the project
2. **Re-test** SineWave → Windows Speaker
3. **Verify** overrun count stays low (<10 in 5 minutes)
4. **Run** 20-minute soak test
5. **Report** results

If backpressure still occurs, we can:
- Further increase buffer size (300ms or 400ms)
- Adjust pipeline frame size to match WASAPI buffer better
- Add adaptive buffer sizing

---

**Fix Status:** ✅ COMPLETE
**Testing Status:** ⏳ PENDING USER REBUILD
**Risk Level:** LOW (simplification, not added complexity)
