# Linux Audio Plugins Implementation Plan

**Status:** Planning
**Target:** Linux compatibility via PulseAudio/PipeWire
**Date:** January 2026

---

## Executive Summary

This document specifies the implementation of native C++ audio source and sink plugins for Linux, enabling NDA to run on Linux desktops with proper audio device support.

### Goals

1. **Linux audio support** via PulseAudio API (compatible with PipeWire)
2. **Device enumeration** with UI selection (microphones and speakers)
3. **Latency target** of 20-50ms (desktop standard)
4. **Maintain Windows support** (existing WASAPI code unchanged)

### Non-Goals

- Cross-platform unified codebase (keep platform-specific implementations)
- Professional audio latency (<10ms, would require JACK/ALSA direct)
- macOS support (out of scope for this iteration)

---

## Technical Design

### Audio Backend: PulseAudio API (`libpulse`)

**Rationale:**
- Works on both **PulseAudio** and **PipeWire** systems (via `pipewire-pulse`)
- Modern Linux distributions (Fedora, Ubuntu 22.04+, Arch) use PipeWire with PulseAudio compatibility
- Async callback model integrates well with NDA's pipeline architecture
- Full device enumeration support
- Well-documented, stable C API

**Dependencies:**
```
libpulse-dev (Debian/Ubuntu)
pulseaudio-libs-devel (Fedora/RHEL)
libpulse (Arch)
```

### Threading Model: Dedicated Thread per Plugin

Each `PulseAudioSource` and `PulseAudioSink` instance runs its own:
- PulseAudio threaded mainloop (`pa_threaded_mainloop`)
- PulseAudio context (`pa_context`)
- PulseAudio stream (`pa_stream`)

**Advantages:**
- Complete isolation between plugins
- Simpler lifecycle management
- No shared state between TX and RX pipelines
- Easier debugging

**Trade-offs:**
- Slightly higher resource usage (acceptable for 2 pipelines)
- Multiple connections to PulseAudio server (not a problem in practice)

---

## Architecture

### File Structure

```
NDA/
├── include/
│   └── audio/
│       └── PulseDeviceEnum.h          # Device enumeration (Linux)
│
├── src/
│   └── audio/
│       └── PulseDeviceEnum.cpp        # Device enumeration impl
│
├── plugins_src/
│   └── linux/
│       ├── PulseAudioSource.h         # Microphone capture plugin
│       ├── PulseAudioSource.cpp
│       ├── PulseAudioSink.h           # Speaker playback plugin
│       └── PulseAudioSink.cpp
│
└── CMakeLists.txt                      # Build configuration
```

### Class Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      AudioSourcePlugin                       │
│                         (interface)                          │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ implements
┌─────────────────────────────────────────────────────────────┐
│                     PulseAudioSource                         │
├─────────────────────────────────────────────────────────────┤
│ - mainloop_: pa_threaded_mainloop*                          │
│ - context_: pa_context*                                      │
│ - stream_: pa_stream*                                        │
│ - ringBuffer_: RingBuffer<float>                            │
│ - sampleRate_: int                                           │
│ - channels_: int                                             │
│ - bufferSize_: int                                           │
│ - deviceName_: std::string                                   │
│ - state_: PluginState                                        │
├─────────────────────────────────────────────────────────────┤
│ + initialize() → bool                                        │
│ + start() → bool                                             │
│ + stop()                                                     │
│ + shutdown()                                                 │
│ + readAudio(buffer) → bool                                   │
│ + setParameter(key, value)                                   │
│ + getParameter(key) → string                                 │
│ - streamReadCallback(stream, nbytes)   [static]             │
│ - contextStateCallback(context)        [static]             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      AudioSinkPlugin                         │
│                        (interface)                           │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ implements
┌─────────────────────────────────────────────────────────────┐
│                      PulseAudioSink                          │
├─────────────────────────────────────────────────────────────┤
│ - mainloop_: pa_threaded_mainloop*                          │
│ - context_: pa_context*                                      │
│ - stream_: pa_stream*                                        │
│ - ringBuffer_: RingBuffer<float>                            │
│ - sampleRate_: int                                           │
│ - channels_: int                                             │
│ - bufferSize_: int                                           │
│ - deviceName_: std::string                                   │
│ - state_: PluginState                                        │
├─────────────────────────────────────────────────────────────┤
│ + initialize() → bool                                        │
│ + start() → bool                                             │
│ + stop()                                                     │
│ + shutdown()                                                 │
│ + writeAudio(buffer) → bool                                  │
│ + getAvailableSpace() → int                                  │
│ + setParameter(key, value)                                   │
│ + getParameter(key) → string                                 │
│ - streamWriteCallback(stream, nbytes)  [static]             │
│ - contextStateCallback(context)        [static]             │
└─────────────────────────────────────────────────────────────┘
```

---

## Detailed Implementation

### 1. Device Enumeration (`PulseDeviceEnum`)

**Header: `include/audio/PulseDeviceEnum.h`**

```cpp
#ifndef PULSEDEVICEENUM_H
#define PULSEDEVICEENUM_H

#include <string>
#include <vector>

namespace nda {

struct PulseDeviceInfo {
    std::string name;           // Internal PulseAudio name
    std::string description;    // Human-readable name
    uint32_t index;             // PulseAudio device index
    bool isDefault;             // Is this the default device?
    int sampleRate;             // Native sample rate
    int channels;               // Channel count
};

/**
 * Enumerate available audio capture devices (microphones).
 * Works with both PulseAudio and PipeWire (via pipewire-pulse).
 */
std::vector<PulseDeviceInfo> enumeratePulseSources();

/**
 * Enumerate available audio playback devices (speakers).
 * Works with both PulseAudio and PipeWire (via pipewire-pulse).
 */
std::vector<PulseDeviceInfo> enumeratePulseSinks();

/**
 * Get the default source (microphone) name.
 */
std::string getDefaultPulseSource();

/**
 * Get the default sink (speaker) name.
 */
std::string getDefaultPulseSink();

} // namespace nda

#endif // PULSEDEVICEENUM_H
```

**Implementation Strategy:**

The enumeration uses PulseAudio's async API with a temporary context:

```cpp
// Pseudocode for enumeration
1. Create pa_mainloop (simple, not threaded - enumeration is one-shot)
2. Create pa_context, connect to server
3. Wait for PA_CONTEXT_READY state
4. Call pa_context_get_source_info_list() or pa_context_get_sink_info_list()
5. Collect results in callback
6. Clean up and return vector
```

### 2. PulseAudioSource (Microphone Capture)

**Key Design Decisions:**

1. **Ring Buffer Size:** 8 buffers worth of samples (~85ms at 48kHz/512 frames)
   - Absorbs jitter from PulseAudio callbacks
   - Prevents underruns during brief pipeline stalls

2. **Callback Model:** PulseAudio pushes data via `pa_stream_set_read_callback`
   - Callback copies data into ring buffer
   - `readAudio()` pulls from ring buffer (non-blocking)

3. **Stream Parameters:**
   ```cpp
   pa_sample_spec spec = {
       .format = PA_SAMPLE_FLOAT32LE,  // Match NDA's internal format
       .rate = 48000,                   // Pipeline internal rate
       .channels = 1 or 2              // Configurable
   };

   pa_buffer_attr attr = {
       .maxlength = (uint32_t)-1,       // Server decides
       .fragsize = 512 * sizeof(float) * channels  // Request ~10ms fragments
   };
   ```

**Lifecycle:**

```
┌──────────────────────────────────────────────────────────────┐
│  initialize()                                                 │
│  ├─ Create pa_threaded_mainloop                              │
│  ├─ Create pa_context with state callback                    │
│  ├─ Start mainloop thread                                    │
│  ├─ Connect context (pa_context_connect)                     │
│  └─ Wait for PA_CONTEXT_READY                                │
├──────────────────────────────────────────────────────────────┤
│  start()                                                      │
│  ├─ Create pa_stream with sample spec                        │
│  ├─ Set read callback (pa_stream_set_read_callback)          │
│  ├─ Connect stream for recording (pa_stream_connect_record)  │
│  └─ Wait for PA_STREAM_READY                                 │
├──────────────────────────────────────────────────────────────┤
│  [Running: callbacks push to ring buffer]                    │
│  readAudio() → pull from ring buffer                         │
├──────────────────────────────────────────────────────────────┤
│  stop()                                                       │
│  ├─ Disconnect stream (pa_stream_disconnect)                 │
│  └─ Destroy stream (pa_stream_unref)                         │
├──────────────────────────────────────────────────────────────┤
│  shutdown()                                                   │
│  ├─ Disconnect context (pa_context_disconnect)               │
│  ├─ Stop mainloop (pa_threaded_mainloop_stop)                │
│  └─ Free all resources                                       │
└──────────────────────────────────────────────────────────────┘
```

**Read Callback Implementation:**

```cpp
// Called by PulseAudio when audio data is available
static void streamReadCallback(pa_stream* stream, size_t nbytes, void* userdata) {
    auto* self = static_cast<PulseAudioSource*>(userdata);

    const void* data;
    size_t actualBytes;

    // Peek at available data (zero-copy)
    if (pa_stream_peek(stream, &data, &actualBytes) < 0) {
        return;  // Error
    }

    if (data == nullptr && actualBytes > 0) {
        // Hole in stream (underrun), drop it
        pa_stream_drop(stream);
        return;
    }

    if (data != nullptr && actualBytes > 0) {
        // Copy to ring buffer
        const float* floatData = static_cast<const float*>(data);
        size_t frames = actualBytes / (sizeof(float) * self->channels_);
        self->ringBuffer_.write(floatData, frames * self->channels_);
    }

    // Release the data
    pa_stream_drop(stream);
}
```

**readAudio() Implementation:**

```cpp
bool PulseAudioSource::readAudio(AudioBuffer& buffer) {
    if (state_ != PluginState::Running) {
        buffer.clear();
        return false;
    }

    size_t framesNeeded = buffer.getFrameCount();
    size_t samplesNeeded = framesNeeded * channels_;

    // Check if enough data is available
    if (ringBuffer_.availableRead() < samplesNeeded) {
        // Underrun - not enough data yet
        underrunCount_++;
        buffer.clear();
        return false;
    }

    // Read interleaved data from ring buffer
    std::vector<float> interleaved(samplesNeeded);
    ringBuffer_.read(interleaved.data(), samplesNeeded);

    // De-interleave into AudioBuffer channels
    for (int frame = 0; frame < framesNeeded; ++frame) {
        for (int ch = 0; ch < channels_; ++ch) {
            buffer.getChannelData(ch)[frame] = interleaved[frame * channels_ + ch];
        }
    }

    return true;
}
```

### 3. PulseAudioSink (Speaker Playback)

**Key Design Decisions:**

1. **Ring Buffer Size:** 8 buffers (~85ms)
   - Absorbs timing jitter
   - Prevents underruns during brief stalls

2. **Callback Model:** PulseAudio requests data via `pa_stream_set_write_callback`
   - Callback pulls from ring buffer
   - `writeAudio()` pushes to ring buffer (non-blocking)

3. **Backpressure:** `getAvailableSpace()` returns ring buffer free space
   - Pipeline can throttle if sink is backed up

**Stream Parameters:**

```cpp
pa_sample_spec spec = {
    .format = PA_SAMPLE_FLOAT32LE,
    .rate = 48000,
    .channels = 1 or 2
};

pa_buffer_attr attr = {
    .maxlength = (uint32_t)-1,
    .tlength = 512 * sizeof(float) * channels * 4,  // ~40ms target latency
    .prebuf = 512 * sizeof(float) * channels * 2,   // Start after 2 buffers
    .minreq = 512 * sizeof(float) * channels        // Request ~10ms chunks
};
```

**Write Callback Implementation:**

```cpp
// Called by PulseAudio when it needs more audio data
static void streamWriteCallback(pa_stream* stream, size_t nbytes, void* userdata) {
    auto* self = static_cast<PulseAudioSink*>(userdata);

    size_t samplesRequested = nbytes / sizeof(float);
    size_t samplesAvailable = self->ringBuffer_.availableRead();
    size_t samplesToWrite = std::min(samplesRequested, samplesAvailable);

    if (samplesToWrite > 0) {
        // Read from ring buffer
        std::vector<float> data(samplesToWrite);
        self->ringBuffer_.read(data.data(), samplesToWrite);

        // Write to PulseAudio stream
        pa_stream_write(stream, data.data(), samplesToWrite * sizeof(float),
                        nullptr, 0, PA_SEEK_RELATIVE);
    }

    // If we couldn't provide enough, write silence for the rest
    if (samplesToWrite < samplesRequested) {
        size_t silenceSamples = samplesRequested - samplesToWrite;
        std::vector<float> silence(silenceSamples, 0.0f);
        pa_stream_write(stream, silence.data(), silenceSamples * sizeof(float),
                        nullptr, 0, PA_SEEK_RELATIVE);
        self->underrunCount_++;
    }
}
```

**writeAudio() Implementation:**

```cpp
bool PulseAudioSink::writeAudio(const AudioBuffer& buffer) {
    if (state_ != PluginState::Running) {
        return false;
    }

    size_t frames = buffer.getFrameCount();
    size_t samples = frames * channels_;

    // Interleave channels
    std::vector<float> interleaved(samples);
    for (int frame = 0; frame < frames; ++frame) {
        for (int ch = 0; ch < channels_; ++ch) {
            interleaved[frame * channels_ + ch] = buffer.getChannelData(ch)[frame];
        }
    }

    // Write to ring buffer
    size_t written = ringBuffer_.write(interleaved.data(), samples);

    if (written < samples) {
        overrunCount_++;
        return false;  // Buffer full, data dropped
    }

    return true;
}
```

### 4. Ring Buffer Implementation

Use the existing `include/audio/RingBuffer.h` or implement a lock-free version:

```cpp
template<typename T>
class RingBuffer {
public:
    explicit RingBuffer(size_t capacity);

    // Returns number of samples written (may be less than count if full)
    size_t write(const T* data, size_t count);

    // Returns number of samples read (may be less than count if empty)
    size_t read(T* data, size_t count);

    size_t availableRead() const;
    size_t availableWrite() const;

    void clear();

private:
    std::vector<T> buffer_;
    std::atomic<size_t> readPos_{0};
    std::atomic<size_t> writePos_{0};
    size_t capacity_;
};
```

---

## Build System Integration

### CMakeLists.txt Changes

```cmake
# At project level
option(BUILD_LINUX_PLUGINS "Build Linux audio plugins" ON)

# In platform detection section
if(UNIX AND NOT APPLE AND BUILD_LINUX_PLUGINS)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(PULSEAUDIO REQUIRED libpulse)

    message(STATUS "PulseAudio found: ${PULSEAUDIO_VERSION}")
    message(STATUS "  Include dirs: ${PULSEAUDIO_INCLUDE_DIRS}")
    message(STATUS "  Libraries: ${PULSEAUDIO_LIBRARIES}")

    set(LINUX_AUDIO_AVAILABLE TRUE)
endif()

# Plugin targets
if(LINUX_AUDIO_AVAILABLE)
    # Device enumeration (built into main app)
    target_sources(NDA PRIVATE
        src/audio/PulseDeviceEnum.cpp
    )
    target_include_directories(NDA PRIVATE ${PULSEAUDIO_INCLUDE_DIRS})
    target_link_libraries(NDA ${PULSEAUDIO_LIBRARIES})

    # Source plugin (shared library)
    add_library(PulseAudioSource SHARED
        plugins_src/linux/PulseAudioSource.cpp
    )
    target_include_directories(PulseAudioSource PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${PULSEAUDIO_INCLUDE_DIRS}
    )
    target_link_libraries(PulseAudioSource ${PULSEAUDIO_LIBRARIES})
    set_target_properties(PulseAudioSource PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugins
    )

    # Sink plugin (shared library)
    add_library(PulseAudioSink SHARED
        plugins_src/linux/PulseAudioSink.cpp
    )
    target_include_directories(PulseAudioSink PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${PULSEAUDIO_INCLUDE_DIRS}
    )
    target_link_libraries(PulseAudioSink ${PULSEAUDIO_LIBRARIES})
    set_target_properties(PulseAudioSink PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugins
    )
endif()
```

### Package Dependencies

**Fedora:**
```bash
sudo dnf install pulseaudio-libs-devel
```

**Ubuntu/Debian:**
```bash
sudo apt install libpulse-dev
```

**Arch:**
```bash
sudo pacman -S libpulse
```

---

## UI Integration

### Device Enumeration in UI

The existing device dropdown code in `UnifiedPipelineView` or `PluginSidebar` needs to:

1. Detect platform at runtime
2. Call appropriate enumeration:
   - Windows: `enumerateWASAPIDevices()`
   - Linux: `enumeratePulseSources()` / `enumeratePulseSinks()`

**Conditional Compilation:**

```cpp
#ifdef __linux__
    #include "audio/PulseDeviceEnum.h"
    #define LINUX_AUDIO 1
#endif

#ifdef _WIN32
    #include "audio/WasapiDeviceEnum.h"
    #define WINDOWS_AUDIO 1
#endif

void populateDeviceDropdown(QComboBox* combo, bool isInput) {
    combo->clear();

#ifdef LINUX_AUDIO
    auto devices = isInput ? nda::enumeratePulseSources()
                           : nda::enumeratePulseSinks();
    for (const auto& dev : devices) {
        QString label = QString::fromStdString(dev.description);
        if (dev.isDefault) label += " (Default)";
        combo->addItem(label, QString::fromStdString(dev.name));
    }
#endif

#ifdef WINDOWS_AUDIO
    auto devices = nda::enumerateWASAPIDevices(isInput ? 0 : 1);
    for (const auto& dev : devices) {
        combo->addItem(QString::fromStdString(dev.friendlyName),
                       QString::fromStdString(dev.id));
    }
#endif
}
```

### Plugin Selection

When user selects device in UI:
1. Get device name from combo box data
2. Call `plugin->setParameter("device", deviceName)`
3. Plugin uses that device on next `start()`

---

## Parameters

### PulseAudioSource Parameters

| Key | Type | Description | Default |
|-----|------|-------------|---------|
| `device` | string | PulseAudio source name (empty = default) | "" |
| `sampleRate` | int | Sample rate in Hz | 48000 |
| `channels` | int | Channel count (1 or 2) | 1 |
| `bufferSize` | int | Frames per buffer | 512 |
| `latencyMode` | string | "low", "high", "auto" | "auto" |

### PulseAudioSink Parameters

| Key | Type | Description | Default |
|-----|------|-------------|---------|
| `device` | string | PulseAudio sink name (empty = default) | "" |
| `sampleRate` | int | Sample rate in Hz | 48000 |
| `channels` | int | Channel count (1 or 2) | 1 |
| `bufferSize` | int | Frames per buffer | 512 |
| `latencyMode` | string | "low", "high", "auto" | "auto" |

---

## Error Handling

### Connection Failures

```cpp
// In context state callback
case PA_CONTEXT_FAILED:
    const char* error = pa_strerror(pa_context_errno(context));
    qWarning() << "PulseAudio connection failed:" << error;
    state_ = PluginState::Error;
    // Emit error signal if available
    break;
```

### Stream Errors

```cpp
// In stream state callback
case PA_STREAM_FAILED:
    const char* error = pa_strerror(pa_context_errno(pa_stream_get_context(stream)));
    qWarning() << "PulseAudio stream failed:" << error;
    state_ = PluginState::Error;
    break;
```

### Graceful Degradation

If PulseAudio connection fails:
1. Log error with helpful message
2. Set plugin state to Error
3. `readAudio()` / `writeAudio()` return false
4. Pipeline continues with passthrough or stops

---

## Testing Plan

### Unit Tests

1. **Ring buffer tests**
   - Write/read correctness
   - Overflow handling
   - Underflow handling
   - Concurrent access

2. **Device enumeration tests**
   - Returns at least one device (default)
   - Handles no PulseAudio server gracefully

### Integration Tests

1. **Source plugin test**
   - Initialize → Start → Read 1000 buffers → Stop → Shutdown
   - Verify no crashes, reasonable latency

2. **Sink plugin test**
   - Initialize → Start → Write 1000 buffers → Stop → Shutdown
   - Verify audio output (manual)

3. **Full pipeline test**
   - Source → Processor (passthrough) → Sink
   - Run for 10 minutes, check for glitches

### Manual Testing Checklist

- [ ] Enumerate devices, verify all system devices appear
- [ ] Select non-default device, verify it's used
- [ ] Unplug device during operation, verify graceful handling
- [ ] Run with PulseAudio (legacy system)
- [ ] Run with PipeWire (modern system)
- [ ] Test mono and stereo modes
- [ ] Verify latency is within 50ms target

---

## Implementation Phases

### Phase 1: Foundation (Core infrastructure)

**Tasks:**
1. Create `include/audio/PulseDeviceEnum.h`
2. Implement `src/audio/PulseDeviceEnum.cpp`
3. Add CMake detection for libpulse
4. Test device enumeration standalone

**Deliverable:** Device list visible in debug output

### Phase 2: Source Plugin (Microphone capture)

**Tasks:**
1. Create `plugins_src/linux/PulseAudioSource.h`
2. Implement `plugins_src/linux/PulseAudioSource.cpp`
3. Add to CMake build
4. Test with Null Sink (verify data flow)

**Deliverable:** Microphone capture working

### Phase 3: Sink Plugin (Speaker playback)

**Tasks:**
1. Create `plugins_src/linux/PulseAudioSink.h`
2. Implement `plugins_src/linux/PulseAudioSink.cpp`
3. Add to CMake build
4. Test with Sine Source (verify audio output)

**Deliverable:** Speaker playback working

### Phase 4: UI Integration

**Tasks:**
1. Add device dropdowns to UI (conditional for Linux)
2. Wire device selection to plugin parameters
3. Test full workflow: select device → start pipeline

**Deliverable:** Full Linux audio support with UI

### Phase 5: Polish and Testing

**Tasks:**
1. Error handling improvements
2. Latency optimization
3. Extended testing (stability, edge cases)
4. Documentation updates

**Deliverable:** Production-ready Linux support

---

## Appendix: PulseAudio API Reference

### Key Functions Used

**Context:**
- `pa_threaded_mainloop_new()` - Create threaded event loop
- `pa_threaded_mainloop_start()` - Start loop thread
- `pa_context_new()` - Create context
- `pa_context_connect()` - Connect to server
- `pa_context_set_state_callback()` - Monitor connection state

**Device Enumeration:**
- `pa_context_get_source_info_list()` - List capture devices
- `pa_context_get_sink_info_list()` - List playback devices
- `pa_context_get_server_info()` - Get default devices

**Streams:**
- `pa_stream_new()` - Create stream
- `pa_stream_connect_record()` - Start capture
- `pa_stream_connect_playback()` - Start playback
- `pa_stream_set_read_callback()` - Set capture callback
- `pa_stream_set_write_callback()` - Set playback callback
- `pa_stream_peek()` / `pa_stream_drop()` - Read captured audio
- `pa_stream_write()` - Write audio for playback

### Sample Formats

| PulseAudio | NDA | Notes |
|------------|-----|-------|
| `PA_SAMPLE_FLOAT32LE` | `float` | Best match, no conversion needed |
| `PA_SAMPLE_S16LE` | - | Would need conversion |
| `PA_SAMPLE_S32LE` | - | Would need conversion |

We use `PA_SAMPLE_FLOAT32LE` exclusively for zero-copy compatibility with NDA's internal format.

---

## Conclusion

This implementation provides native Linux audio support through the PulseAudio API, which is compatible with both legacy PulseAudio and modern PipeWire systems. The dedicated-thread-per-plugin model simplifies the design while providing adequate performance for the 20-50ms latency target.

**Key benefits:**
- Works on all modern Linux distributions
- Full device enumeration and selection
- Clean integration with existing NDA architecture
- No changes to Windows code path

**Next step:** Begin Phase 1 implementation after plan approval.
