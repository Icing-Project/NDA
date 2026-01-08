# NDA v1.x → v2.0 Migration Guide

**Last Updated:** December 2025  
**Target Audience:** Developers migrating existing NDA v1.x code to v2.0

---

## Overview

NDA v2.0 represents a significant simplification and refactoring of the audio processing architecture. This guide provides step-by-step instructions for migrating from v1.x to v2.0.

**Key Changes:**
- ❌ Bearer plugin type **removed entirely**
- ❌ Encryptor plugin type **merged into Processor**
- ❌ Core crypto classes **deleted** (moved to plugin examples)
- ✅ 3-slot pipeline model (Source → Processor → Sink)
- ✅ Dual independent pipelines (TX + RX)
- ✅ Automatic sample rate adaptation (48kHz internal)
- ✅ Python processor support (equal to C++)

---

## Breaking Changes

### 1. Plugin Types Reduced (4 → 3)

**v1.x:**
```cpp
enum class PluginType {
    AudioSource,
    AudioSink,
    Bearer,      // REMOVED
    Encryptor,   // REMOVED (merged into Processor)
    Processor
};
```

**v2.0:**
```cpp
enum class PluginType {
    AudioSource,    // Audio input
    AudioSink,      // Audio output
    Processor       // Audio transformation (encryption, effects, etc.)
};
```

**Migration:**
- `Bearer` → Delete or convert to Source/Sink pair
- `Encryptor` → Rename to `Processor`

---

### 2. Pipeline API Simplified (4 slots → 3 slots)

**v1.x:**
```cpp
pipeline->setAudioSource(source);
pipeline->setBearer(bearer);      // REMOVED
pipeline->setEncryptor(encryptor); // RENAMED
pipeline->setAudioSink(sink);
```

**v2.0:**
```cpp
pipeline->setSource(source);
pipeline->setProcessor(processor);  // Encryptor is now a processor
pipeline->setSink(sink);
// Bearer deleted - use AIOC/VB-Cable as source/sink instead
```

**Migration Steps:**
1. Remove all `setBearer()` calls
2. Rename `setEncryptor()` → `setProcessor()`
3. Rename `setAudioSource()` → `setSource()`
4. Rename `setAudioSink()` → `setSink()`

---

### 3. Crypto Moved to Plugin Examples

**v1.x:**
```cpp
#include "crypto/Encryptor.h"
#include "crypto/KeyExchange.h"

// Crypto was part of core
```

**v2.0:**
```cpp
// Crypto is now plugin-only
// Use example plugins:

#include "plugins/AudioProcessorPlugin.h"

// C++ examples:
// - plugins_src/examples/AES256EncryptorPlugin.cpp
// - plugins_src/examples/AES256DecryptorPlugin.cpp

// Python examples:
// - plugins_py/examples/fernet_encryptor.py
// - plugins_py/examples/fernet_decryptor.py
```

**Migration:**
- Remove `#include "crypto/..."` from core code
- Use AES256 example plugins or create your own processor plugins
- Key exchange remains out-of-band (setParameter("key", ...))

---

### 4. Single Pipeline → Dual Pipelines

**v1.x:**
```cpp
// MainWindow.h
std::shared_ptr<ProcessingPipeline> pipeline_;  // Single pipeline
```

**v2.0:**
```cpp
// MainWindow.h  
std::shared_ptr<ProcessingPipeline> txPipeline_;  // Transmit
std::shared_ptr<ProcessingPipeline> rxPipeline_;  // Receive

// Each runs independently in its own thread
```

**Migration:**
- Create two pipeline instances instead of one
- Configure TX pipeline: Mic → Encryptor → AIOC/VB-Cable output
- Configure RX pipeline: AIOC/VB-Cable input → Decryptor → Speaker
- Update UI to show both pipelines

---

### 5. UI Changes

**v1.x UI (4 combo boxes):**
```
┌───────────────────┐
│ Audio Source   ▼ │
│ Encryptor      ▼ │
│ Bearer         ▼ │  ← REMOVED
│ Audio Sink     ▼ │
└───────────────────┘
[Start] [Stop]
```

**v2.0 UI (Dual pipeline, 3 combo boxes each):**
```
┌────────────────────┐
│ TX Pipeline        │
│ Source      ▼     │
│ Processor   ▼     │
│ Sink        ▼     │
│ [Start TX] [Stop]  │
└────────────────────┘

┌────────────────────┐
│ RX Pipeline        │
│ Source      ▼     │
│ Processor   ▼     │
│ Sink        ▼     │
│ [Start RX] [Stop]  │
└────────────────────┘

[Start Both] [Stop Both]
```

**Migration:**
- Remove bearer combo box and card
- Rename "Encryptor" → "Processor"
- Duplicate pipeline config for TX and RX
- Add combined start/stop buttons

---

## Migration Checklist

### Code Changes

- [ ] Update `#include` statements
  - Remove: `#include "plugins/BearerPlugin.h"`
  - Remove: `#include "plugins/EncryptorPlugin.h"`
  - Remove: `#include "crypto/..."`
  - Add: `#include "plugins/AudioProcessorPlugin.h"`
  
- [ ] Update method calls
  - `setAudioSource()` → `setSource()`
  - `setEncryptor()` → `setProcessor()`
  - `setAudioSink()` → `setSink()`
  - Remove: `setBearer()` calls
  
- [ ] Update variable names
  - `audioSource_` → `source_`
  - `encryptor_` → `processor_`
  - `audioSink_` → `sink_`
  - Remove: `bearer_`
  
- [ ] Update PluginManager calls
  - `getBearerPlugin()` → deleted
  - `getEncryptorPlugin()` → `getAudioProcessorPlugin()`

### Build Changes

- [ ] Update CMakeLists.txt
  - Remove: `src/crypto/Encryptor.cpp`
  - Remove: `src/crypto/KeyExchange.cpp`
  - Remove: `include/crypto/Encryptor.h`
  - Remove: `include/crypto/KeyExchange.h`
  - Add: `src/audio/Resampler.cpp`
  - Add: `add_subdirectory(plugins_src/examples)`
  
- [ ] Rebuild
  ```bash
  cd build
  cmake ..
  make clean
  make -j
  ```

### Plugin Migration

**If you wrote custom plugins:**

#### Encryptor → Processor

**v1.x:**
```cpp
class MyEncryptor : public EncryptorPlugin {
    bool encrypt(uint8_t* input, size_t inputSize, 
                uint8_t* output, size_t& outputSize,
                const uint8_t* nonce, size_t nonceSize) override;
};
```

**v2.0:**
```cpp
class MyEncryptor : public AudioProcessorPlugin {
    bool processAudio(AudioBuffer& buffer) override {
        // Process buffer in-place
        // Encryption logic here
        return true;
    }
    
    int getSampleRate() const override { return 48000; }
    int getChannelCount() const override { return 2; }
    void setSampleRate(int rate) override { /* ... */ }
    void setChannelCount(int channels) override { /* ... */ }
};
```

#### Bearer → Source/Sink Pair

**v1.x:**
```cpp
class UDPBearer : public BearerPlugin {
    bool sendPacket(const Packet& packet) override;
    bool receivePacket(Packet& packet) override;
};
```

**v2.0:**
```cpp
// Split into two plugins

class UDPSink : public AudioSinkPlugin {
    bool writeAudio(const AudioBuffer& buffer) override {
        // Send buffer over UDP
    }
};

class UDPSource : public AudioSourcePlugin {
    bool readAudio(AudioBuffer& buffer) override {
        // Receive buffer from UDP
    }
};
```

### UI Migration

**v1.x PipelineView:**
```cpp
QComboBox* audioSourceCombo;
QComboBox* bearerCombo;     // REMOVE
QComboBox* encryptorCombo;  // RENAME to processorCombo
QComboBox* audioSinkCombo;
```

**v2.0 PipelineView:**
```cpp
// TX Pipeline widgets
QComboBox* txSourceCombo;
QComboBox* txProcessorCombo;
QComboBox* txSinkCombo;

// RX Pipeline widgets
QComboBox* rxSourceCombo;
QComboBox* rxProcessorCombo;
QComboBox* rxSinkCombo;

// Combined controls
QPushButton* startBothButton;
QPushButton* stopBothButton;
```

---

## Example Migration

### Before (v1.x)

```cpp
// Configure single pipeline
auto source = pluginManager->getAudioSourcePlugin("Microphone");
auto encryptor = pluginManager->getEncryptorPlugin("AES-256");
auto bearer = pluginManager->getBearerPlugin("UDP");
auto sink = pluginManager->getAudioSinkPlugin("Speaker");

pipeline->setAudioSource(source);
pipeline->setEncryptor(encryptor);
pipeline->setBearer(bearer);
pipeline->setAudioSink(sink);

pipeline->initialize();
pipeline->start();
```

### After (v2.0)

```cpp
// Configure TX pipeline (microphone → encrypted → AIOC)
auto txSource = pluginManager->getAudioSourcePlugin("Microphone");
auto txProcessor = pluginManager->getAudioProcessorPlugin("AES-256 Encryptor");
auto txSink = pluginManager->getAudioSinkPlugin("AIOC Output");

txPipeline->setSource(txSource);
txPipeline->setProcessor(txProcessor);
txPipeline->setSink(txSink);

txPipeline->initialize();
txPipeline->start();

// Configure RX pipeline (AIOC → decrypted → speaker)
auto rxSource = pluginManager->getAudioSourcePlugin("AIOC Input");
auto rxProcessor = pluginManager->getAudioProcessorPlugin("AES-256 Decryptor");
auto rxSink = pluginManager->getAudioSinkPlugin("Speaker");

rxPipeline->setSource(rxSource);
rxPipeline->setProcessor(rxProcessor);
rxPipeline->setSink(rxSink);

rxPipeline->initialize();
rxPipeline->start();

// Both pipelines now running simultaneously
```

---

## New Features in v2.0

### 1. Automatic Sample Rate Adaptation

**No code changes required!** The pipeline automatically resamples:

```cpp
// v2.0: This just works, even with mismatched rates
auto source441 = ...; // 44.1kHz source
auto sink48 = ...;    // 48kHz sink

pipeline->setSource(source441);
pipeline->setSink(sink48);
pipeline->initialize();  // Auto-enables 44.1→48kHz resampling

// Output: "Auto-resampling enabled: 44100Hz → 48000Hz (source)"
```

### 2. Python Processor Plugins

**New in v2.0:** Python plugins can now be processors!

```python
from base_plugin import AudioProcessorPlugin

class MyEffectPlugin(AudioProcessorPlugin):
    def process_audio(self, buffer):
        # Process buffer.data (numpy array)
        buffer.data *= 0.5  # Example: reduce volume
        return True
```

### 3. Improved Error Handling

**v2.0 handles failures gracefully:**

```cpp
// Processor failure doesn't crash pipeline
if (!processor_->processAudio(buffer)) {
    // Logs error, passes audio through unprocessed
}

// Sink failure doesn't increment processed samples
if (sink_->writeAudio(buffer)) {
    processedSamples_ += buffer.getFrameCount();  // Only on success
}
```

---

## Troubleshooting

### Q: Build fails with "BearerPlugin.h not found"

**A:** Remove all bearer includes:
```bash
# Find remaining bearer references
grep -r "BearerPlugin" include/ src/

# Remove the includes
```

### Q: Plugins don't load after migration

**A:** Check plugin type returned by `getInfo()`:
- Old: `PluginType::Encryptor` → New: `PluginType::Processor`
- Old: `PluginType::Bearer` → Delete plugin or convert to Source/Sink

### Q: Sample rate mismatch errors

**A:** v2.0 auto-fixes this! Just initialize the pipeline:
```cpp
pipeline->initialize();  // Automatically enables resampling
```

### Q: Where did crypto classes go?

**A:** Moved to plugin examples:
- C++: `plugins_src/examples/AES256EncryptorPlugin.cpp`
- Python: `plugins_py/examples/fernet_encryptor.py`

Copy these as templates for your own crypto plugins.

### Q: How do I share keys between encryptor/decryptor?

**A:** Use parameters (out-of-band key exchange):
```cpp
auto encryptor = ...; // AES256EncryptorPlugin
auto decryptor = ...; // AES256DecryptorPlugin

// Get key from encryptor
std::string key = encryptor->getParameter("key");

// Set key on decryptor
decryptor->setParameter("key", key);
```

---

## Testing After Migration

### 1. Verify Compilation

```bash
mkdir build && cd build
cmake ..
make -j
# Should compile without warnings
```

### 2. Test Basic Pipeline

```bash
./NDA

# In UI:
# 1. Load plugins
# 2. Configure TX: Source → Processor → Sink
# 3. Start TX
# 4. Verify audio flows
```

### 3. Test Dual Pipelines

```bash
# Configure both TX and RX
# Click "Start Both"
# Verify both show as running in dashboard
```

### 4. Test Sample Rate Adaptation

```bash
# Use 44.1kHz source with 48kHz sink
# Should see: "Auto-resampling enabled: 44100Hz → 48000Hz"
```

---

## Common Migration Scenarios

### Scenario 1: Simple Audio Router (No Encryption)

**v1.x:**
```cpp
pipeline->setAudioSource(mic);
pipeline->setAudioSink(speaker);
pipeline->initialize();
```

**v2.0:**
```cpp
pipeline->setSource(mic);
pipeline->setSink(speaker);
// Processor is optional - omit for passthrough
pipeline->initialize();
```

**No changes needed!** (API renamed but functionality identical)

---

### Scenario 2: Encrypted Audio to File

**v1.x:**
```cpp
pipeline->setAudioSource(mic);
pipeline->setEncryptor(aes256);
pipeline->setAudioSink(wavFile);
```

**v2.0:**
```cpp
pipeline->setSource(mic);
pipeline->setProcessor(aes256encryptor);  // Now a processor
pipeline->setSink(wavFile);
```

**Change:** `setEncryptor()` → `setProcessor()`

---

### Scenario 3: Network Communication (Bearer Removed)

**v1.x:**
```cpp
// TX side
pipeline->setAudioSource(mic);
pipeline->setEncryptor(aes256);
pipeline->setBearer(udpBearer);  // Sent over network

// RX side: No receive path existed in v1.x!
```

**v2.0:**
```cpp
// TX Pipeline
txPipeline->setSource(mic);
txPipeline->setProcessor(aes256encryptor);
txPipeline->setSink(udpSink);  // Network sink

// RX Pipeline (NEW in v2.0!)
rxPipeline->setSource(udpSource);  // Network source
rxPipeline->setProcessor(aes256decryptor);
rxPipeline->setSink(speaker);

// Both run simultaneously
```

**Change:** Bearer split into UDPSource/UDPSink pair, full duplex support

---

### Scenario 4: AIOC Radio Encryption

**v1.x:**
```cpp
// Only simplex - couldn't do TX+RX simultaneously
```

**v2.0:**
```cpp
// TX: Device Mic → Encryptor → AIOC Output
txPipeline->setSource(deviceMic);
txPipeline->setProcessor(aes256encryptor);
txPipeline->setSink(aiocOutput);

// RX: AIOC Input → Decryptor → Device Speaker
rxPipeline->setSource(aiocInput);
rxPipeline->setProcessor(aes256decryptor);
rxPipeline->setSink(deviceSpeaker);

// Full duplex radio communication with encryption
```

**Change:** Dual pipeline architecture enables true full-duplex

---

## Python Plugin Migration

### Encryptor Plugin

**v1.x:**
```python
# Encryptor plugins didn't exist in Python
```

**v2.0:**
```python
from base_plugin import AudioProcessorPlugin, PluginType

class MyEncryptor(AudioProcessorPlugin):
    def get_type(self):
        return PluginType.PROCESSOR  # Not ENCRYPTOR
    
    def process_audio(self, buffer):
        # Encrypt buffer.data (numpy array) in-place
        return True
```

**Change:** Python processors now fully supported!

---

## Performance Improvements

### v1.x Performance Issues

- Python bridge: 3-15ms overhead per buffer
- No real-time pacing (pipeline ran 0.36× or 1.77× speed)
- Hardcoded CPU metrics (always returned 5%)
- Sample rate mismatches caused glitches

### v2.0 Performance Fixes

- Python bridge: **<500µs overhead** (6-30× faster via caching/zero-copy)
- Real-time pacing: Runs at exactly 1.0× speed
- Measured CPU metrics: Actual thread runtime
- Auto-resampling: Handles any sample rate transparently

---

## Timeline

**Recommended migration approach:**

1. **Week 1:** Update core code (API renaming, remove bearer)
2. **Week 2:** Test single pipeline with new API
3. **Week 3:** Implement dual pipeline UI
4. **Week 4:** Migrate plugins, full testing

**For small projects:** 2-4 hours  
**For complex projects:** 1-2 weeks

---

## Getting Help

**Issues during migration?**

1. Check this guide's troubleshooting section
2. Review example plugins:
   - `plugins_src/examples/` (C++)
   - `plugins_py/examples/` (Python)
3. Read `docs/NDA-SPECS-v2.md` for complete v2.0 spec
4. Read `docs/V2_IMPLEMENTATION_PLAN.md` for detailed changes

**Common issues:**
- "Bearer not found" → Remove bearer references, use Source/Sink
- "Encryptor not found" → Rename to Processor
- "Sample rate mismatch" → v2.0 auto-fixes this, just initialize
- "Plugin won't load" → Check PluginType returned by getInfo()

---

## Success Validation

After migration, verify:

```bash
# 1. Zero bearer references
grep -r "bearer\|Bearer" src/ include/ | wc -l
# Expected: 0

# 2. Code compiles
cmake --build build 2>&1 | grep "warning:" | wc -l
# Expected: 0

# 3. Dual pipelines work
./NDA
# Start both TX and RX, verify both run simultaneously

# 4. Sample rate adaptation works
# Use mismatched source/sink rates, verify no errors

# 5. Python processors load
# Load Python example plugins, verify they appear in Processor dropdown
```

---

## Rollback Plan

**If migration fails or issues arise:**

```bash
# Revert to v1.x
git checkout v1.x-stable

# Or keep v2.0 but use compatibility mode:
# - Use only single pipeline (don't use dual)
# - Force all plugins to 48kHz (disable auto-resampling)
# - Use C++ plugins only (skip Python if issues)
```

---

## Summary

**What you gain in v2.0:**
- ✅ Simpler architecture (3 slots vs 4)
- ✅ Dual pipelines (TX + RX simultaneously)
- ✅ Auto-resampling (works with any device)
- ✅ Python processors (equal to C++)
- ✅ Faster Python bridge (6-30× improvement)
- ✅ Cleaner codebase (35% code reduction)

**What you must change:**
- ❌ Remove bearer references
- ❌ Rename encryptor → processor
- ❌ Update pipeline API calls
- ❌ Move to dual pipeline model (if needed)

**Total effort:** 2 hours - 2 weeks depending on project complexity

---

*NDA v2.0 Migration Guide*  
*For questions or issues, see docs/NDA-SPECS-v2.md*

