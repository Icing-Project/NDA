# NDA v2.0 - Plugin Development Guide

**Last Updated:** December 2025  
**API Version:** 2.0

---

## Overview

NDA v2.0 uses a simplified **3-type plugin architecture**:

1. **AudioSource** - Input providers (microphone, file, network receiver)
2. **AudioSink** - Output consumers (speaker, file, network sender)
3. **Processor** - Audio transformations (encryption, effects, resampling)

**Removed in v2.0:** Bearer (deleted), Encryptor (merged into Processor)

---

## Plugin Types

### AudioSourcePlugin

Provides audio input to the pipeline.

```cpp
class AudioSourcePlugin : public BasePlugin {
public:
    virtual bool readAudio(AudioBuffer& buffer) = 0;
    virtual int getSampleRate() const = 0;
    virtual int getChannelCount() const = 0;
    virtual void setSampleRate(int rate) = 0;
    virtual void setChannelCount(int channels) = 0;
    virtual int getBufferSize() const = 0;
    virtual void setBufferSize(int frames) = 0;
};
```

**Examples:**
- `SineWaveSourcePlugin.cpp` - Test signal generator
- `sounddevice_microphone.py` - System microphone (Python)

---

### AudioSinkPlugin

Consumes audio output from the pipeline.

```cpp
class AudioSinkPlugin : public BasePlugin {
public:
    virtual bool writeAudio(const AudioBuffer& buffer) = 0;
    virtual int getSampleRate() const = 0;
    virtual int getChannelCount() const = 0;
    virtual void setSampleRate(int rate) = 0;
    virtual void setChannelCount(int channels) = 0;
    virtual int getBufferSize() const = 0;
    virtual void setBufferSize(int frames) = 0;
    virtual int getAvailableSpace() const = 0;  // For backpressure
};
```

**Examples:**
- `NullSinkPlugin.cpp` - Silent sink for testing
- `WavFileSinkPlugin.cpp` - Record to WAV file
- `sounddevice_speaker.py` - System speaker (Python)

---

### AudioProcessorPlugin (NEW in v2.0)

Transforms audio in-place (encryption, effects, etc.)

```cpp
class AudioProcessorPlugin : public BasePlugin {
public:
    /**
     * Process audio buffer in-place.
     * @param buffer Audio data (modified in-place)
     * @return true on success, false on error (pipeline passthroughs on false)
     */
    virtual bool processAudio(AudioBuffer& buffer) = 0;
    
    virtual int getSampleRate() const = 0;
    virtual int getChannelCount() const = 0;
    virtual void setSampleRate(int rate) = 0;
    virtual void setChannelCount(int channels) = 0;
    
    /**
     * Declare added latency in seconds (for reporting).
     * Default: 0.0
     */
    virtual double getProcessingLatency() const { return 0.0; }
};
```

**Key Points:**
- Buffer is always at 48kHz (pipeline handles resampling)
- Process in-place (modify buffer.data directly)
- Return false on error (pipeline will passthrough)
- No buffer size changes (process fixed-size buffers)

**Examples:**
- `plugins_src/examples/AES256EncryptorPlugin.cpp` - Production encryption
- `plugins_src/examples/AES256DecryptorPlugin.cpp` - Production decryption
- `plugins_py/examples/passthrough.py` - No-op processor
- `plugins_py/examples/simple_gain.py` - Volume control
- `plugins_py/examples/fernet_encryptor.py` - Python encryption

---

## C++ Plugin Development

### Minimal Processor Example

```cpp
#include "plugins/AudioProcessorPlugin.h"
#include <cmath>

class GainProcessor : public nda::AudioProcessorPlugin {
private:
    float gain_;
    int sampleRate_;
    int channels_;
    nda::PluginState state_;

public:
    GainProcessor() : gain_(1.0f), sampleRate_(48000), channels_(2), 
                     state_(nda::PluginState::Unloaded) {}
    
    bool initialize() override {
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
        
        // Apply gain to all channels
        for (int ch = 0; ch < buffer.getChannelCount(); ++ch) {
            float* data = buffer.getChannelData(ch);
            for (int f = 0; f < buffer.getFrameCount(); ++f) {
                data[f] *= gain_;
                // Clip
                if (data[f] > 1.0f) data[f] = 1.0f;
                if (data[f] < -1.0f) data[f] = -1.0f;
            }
        }
        
        return true;
    }
    
    nda::PluginInfo getInfo() const override {
        return {
            "Gain Processor",
            "1.0.0",
            "Your Name",
            "Simple volume adjustment",
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
        if (key == "gain") {
            gain_ = std::stof(value);
            return true;
        }
        return false;
    }
    
    std::string getParameter(const std::string& key) const override {
        if (key == "gain") return std::to_string(gain_);
        return "";
    }
};

// Export functions
extern "C" {
    __declspec(dllexport) nda::BasePlugin* createPlugin() {
        return new GainProcessor();
    }
    
    __declspec(dllexport) void destroyPlugin(nda::BasePlugin* plugin) {
        delete plugin;
    }
}
```

### Build Configuration

**CMakeLists.txt:**
```cmake
add_library(MyProcessorPlugin SHARED
    MyProcessorPlugin.cpp
)

target_include_directories(MyProcessorPlugin PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(MyProcessorPlugin PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PLUGIN_OUTPUT_DIR}
)
```

**Build:**
```bash
cd build
cmake ..
make MyProcessorPlugin
# Output: build/plugins/libMyProcessorPlugin.so
```

---

## Python Plugin Development

### Minimal Processor Example

```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from base_plugin import AudioProcessorPlugin, PluginInfo, PluginType, PluginState, AudioBuffer

class MyProcessor(AudioProcessorPlugin):
    def __init__(self):
        super().__init__()
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
    
    def process_audio(self, buffer: AudioBuffer) -> bool:
        """Process audio in-place"""
        if self.state != PluginState.RUNNING:
            return False
        
        # Example: Apply gain
        buffer.data *= 0.5
        np.clip(buffer.data, -1.0, 1.0, out=buffer.data)
        
        return True
    
    def get_info(self):
        return PluginInfo(
            name="My Processor",
            version="1.0.0",
            author="Your Name",
            description="Example processor",
            plugin_type=PluginType.PROCESSOR,
            api_version=1
        )
    
    def get_type(self):
        return PluginType.PROCESSOR
    
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
        return False  # No parameters
    
    def get_parameter(self, key):
        return ""

def create_plugin():
    return MyProcessor()
```

### Testing Python Plugin

```bash
# Test standalone
python3 my_processor.py

# Test with NDA
./NDA
# Click "Auto-Load Python Plugins"
# Select "My Processor" in Processor dropdown
```

---

## Common Patterns

### 1. Encryption Processor

See `plugins_src/examples/AES256EncryptorPlugin.cpp` for production example.

**Key requirements:**
- Generate unique nonce/IV per buffer
- Handle key via setParameter("key", hex_string)
- Return key via getParameter("key") for sharing
- Authenticate (GCM mode recommended)
- Handle errors gracefully

### 2. Effects Processor

```cpp
bool processAudio(AudioBuffer& buffer) override {
    for (int ch = 0; ch < buffer.getChannelCount(); ++ch) {
        float* data = buffer.getChannelData(ch);
        for (int f = 0; f < buffer.getFrameCount(); ++f) {
            // Apply effect (e.g., distortion, reverb, etc.)
            data[f] = applyEffect(data[f]);
        }
    }
    return true;
}
```

### 3. Analysis Processor (Passthrough)

```cpp
bool processAudio(AudioBuffer& buffer) override {
    // Analyze audio (e.g., spectrum, RMS, peaks)
    float rms = calculateRMS(buffer);
    logMetric("RMS", rms);
    
    // Don't modify buffer - passthrough
    return true;
}
```

---

## Best Practices

### Performance

1. **Minimize allocations** - Reuse buffers, don't allocate per frame
2. **SIMD when possible** - Use vectorized operations
3. **Profile first** - Measure before optimizing
4. **Declare latency** - If you add algorithmic latency, declare it

### Thread Safety

1. **No shared state** - Each pipeline has its own processor instance
2. **No UI calls** - processAudio() runs on pipeline thread
3. **GIL handling** - Python: automatically managed by bridge

### Error Handling

1. **Return false on failure** - Pipeline will passthrough
2. **Log errors** - Help users debug issues
3. **Don't crash** - Handle exceptions gracefully
4. **Validate inputs** - Check buffer size, channel count

---

## FAQ

### Q: Can I chain multiple processors?

**A:** Not in v2.0 - single processor slot. For v2.1+, or create a composite processor that chains internally.

### Q: Can processors change buffer size?

**A:** No - process fixed-size buffers. Pipeline handles resampling.

### Q: Can processors change sample rate?

**A:** No - processors operate at 48kHz (pipeline resamples before/after).

### Q: Python vs C++ performance?

**A:** 
- C++: <0.1ms overhead
- Python (unoptimized): 3-15ms overhead
- Python (optimized): <500µs overhead (planned)

Use C++ for production if latency critical.

### Q: How do I share keys between encryptor/decryptor?

**A:** Out-of-band via parameters:
```cpp
std::string key = encryptor->getParameter("key");
decryptor->setParameter("key", key);
```

---

## Migration from v1.x

### Encryptor → Processor

**v1.x EncryptorPlugin:**
```cpp
class MyEncryptor : public EncryptorPlugin {
    bool encrypt(uint8_t* in, size_t inSize, 
                uint8_t* out, size_t& outSize,
                const uint8_t* nonce, size_t nonceSize) override;
};
```

**v2.0 Processor:**
```cpp
class MyEncryptor : public AudioProcessorPlugin {
    bool processAudio(AudioBuffer& buffer) override {
        // Encrypt buffer.data in-place
        // Convert float → bytes, encrypt, convert back
        return true;
    }
};
```

**Key changes:**
- In-place modification (not in/out buffers)
- Works on AudioBuffer (not raw bytes)
- Simpler API (no nonce parameter - handle internally)

---

## Testing Your Plugin

### Unit Test Template

```cpp
// test_my_plugin.cpp
#include "MyPlugin.h"
#include <cassert>

int main() {
    MyPlugin plugin;
    assert(plugin.initialize());
    assert(plugin.start());
    
    AudioBuffer buffer(2, 512);
    // Fill with test data
    
    assert(plugin.processAudio(buffer));
    // Validate output
    
    plugin.stop();
    plugin.shutdown();
    
    std::cout << "✓ Plugin test passed\n";
    return 0;
}
```

### Integration Test

```bash
# Load in NDA
./NDA

# Steps:
# 1. Load your plugin
# 2. Configure pipeline: Sine → MyProcessor → Null
# 3. Start pipeline
# 4. Verify no errors in console
# 5. Check metrics (latency, CPU)
```

---

## Reference Implementations

**Simple/Testing:**
- `plugins_py/examples/passthrough.py` - Minimal processor template
- `plugins_py/examples/simple_gain.py` - Parameter handling example

**Production/Crypto:**
- `plugins_src/examples/AES256EncryptorPlugin.cpp` - OpenSSL encryption
- `plugins_src/examples/AES256DecryptorPlugin.cpp` - OpenSSL decryption with auth
- `plugins_py/examples/fernet_encryptor.py` - Python crypto (demo only)

---

## Advanced Topics

### Stateful Processing

```cpp
class EchoProcessor : public AudioProcessorPlugin {
private:
    std::vector<float> delayBuffer_;  // Circular buffer
    int writePos_;
    
public:
    bool processAudio(AudioBuffer& buffer) override {
        for (int f = 0; f < buffer.getFrameCount(); ++f) {
            // Read from delay buffer
            float delayed = delayBuffer_[writePos_];
            
            // Mix with current sample
            for (int ch = 0; ch < buffer.getChannelCount(); ++ch) {
                float* data = buffer.getChannelData(ch);
                data[f] = data[f] * 0.7f + delayed * 0.3f;  // 30% echo
            }
            
            // Write to delay buffer
            delayBuffer_[writePos_] = buffer.getChannelData(0)[f];
            writePos_ = (writePos_ + 1) % delayBuffer_.size();
        }
        return true;
    }
};
```

### Multi-Stage Processing

```cpp
class CompressorEQProcessor : public AudioProcessorPlugin {
private:
    Compressor compressor_;
    Equalizer eq_;
    
public:
    bool processAudio(AudioBuffer& buffer) override {
        // Stage 1: Compress
        if (!compressor_.process(buffer)) return false;
        
        // Stage 2: EQ
        if (!eq_.process(buffer)) return false;
        
        return true;
    }
};
```

---

## Debugging Tips

1. **Console logging** - Processor output goes to stdout/stderr
2. **Test with passthrough** - Verify pipeline works without your processor
3. **Test with sine** - Use SineWaveSource for predictable input
4. **Check sample rate** - Processors always receive 48kHz
5. **Monitor metrics** - Watch CPU/latency in dashboard

---

## Performance Guidelines

### Target Performance

| Plugin Type | Target Latency | Maximum CPU |
|-------------|----------------|-------------|
| C++ Processor | <100µs | <5% |
| Python Processor | <500µs | <10% |

### Optimization Tips

**C++:**
- Use SIMD (SSE/AVX) for bulk operations
- Minimize allocations (reuse buffers)
- Profile with perf/gprof

**Python:**
- Use NumPy operations (vectorized)
- Avoid loops (use numpy.multiply, etc.)
- Cache expensive computations

---

## Conclusion

**NDA v2.0 makes processor development simple:**
- Clean interface (just processAudio())
- Python and C++ equal support
- Automatic sample rate handling
- Graceful error recovery

**Start with examples, modify to your needs, test thoroughly.**

For more details, see:
- `docs/NDA-SPECS-v2.md` - Complete v2.0 specification
- `docs/MIGRATION_GUIDE.md` - v1 → v2 migration guide  
- Example plugins in `plugins_src/examples/` and `plugins_py/examples/`

---

*NDA v2.0 Plugin Development Guide*  
*Happy plugin development!*

