# NDA v2.0 Strategic Summary
## Executive Decision Record

---

## Strategic Direction (Approved)

Based on our deep analysis and your strategic guidance, NDA v2.0 will be a **focused, achievable real-time audio encryption bridge** with the following core decisions:

### 1. ✅ Latency: Aspiration, Not Requirement

**Decision:** Target <50ms end-to-end, not <5ms

**Rationale:**
- Sub-5ms requires ASIO exclusive mode, real-time OS scheduling, zero-copy everything
- For encrypted communication, 20-50ms is perfectly acceptable
- Allows Python plugins to remain first-class citizens
- Removes pressure for extreme optimizations that compromise stability

**Impact:**
- Python bridge overhead (10-20ms) is acceptable
- Can use standard OS audio APIs (no ASIO requirement)
- Threading can use normal priorities (no RT kernel needed)
- Focus shifts to correctness and stability over raw speed

---

### 2. ✅ Encryption: Plugin Responsibility, Not Core

**Decision:** Remove all crypto from core; encryption is 100% plugin-provided

**Rationale:**
- Current `include/crypto/` and `src/crypto/` mix concerns (audio processing + cryptography)
- Plugins allow flexibility: AES, ChaCha, custom algorithms, or no encryption
- Core becomes truly generic audio processor
- Easier to test, maintain, and extend

**What Gets Deleted:**
```
❌ include/crypto/Encryptor.h
❌ include/crypto/KeyExchange.h
❌ src/crypto/Encryptor.cpp
❌ src/crypto/KeyExchange.cpp
❌ include/plugins/EncryptorPlugin.h (merged into Processor)
```

**What Gets Created:**
```
✅ plugins_src/examples/AES256EncryptorPlugin.cpp
✅ plugins_src/examples/AES256DecryptorPlugin.cpp
✅ plugins_py/examples/fernet_encryptor.py
✅ plugins_py/examples/fernet_decryptor.py
```

**Impact:**
- Core codebase shrinks ~15%
- ProcessingPipeline.cpp becomes ~40% simpler
- OpenSSL is plugin dependency, not core dependency
- Easier to add new encryption algorithms (just write a plugin)

---

### 3. ✅ Sample Rate: 48kHz Default, Flexible & Modular

**Decision:** Pipeline operates at 48kHz internally, auto-resamples source/sink

**Rationale:**
- 48kHz is standard for communication (Discord, VoIP, radio)
- Hardware devices vary (44.1, 48, 96 kHz) — forcing one rate is fragile
- Automatic resampling provides seamless compatibility
- Processors always see consistent 48kHz buffers (simplifies plugin development)

**How It Works:**
```
Source (any rate) → [Resampler] → 48kHz → [Processor] → 48kHz → [Resampler] → Sink (any rate)
```

**Resampling Strategy:**
- **Simple (default):** Linear interpolation (fast, acceptable quality)
- **High (optional):** libsamplerate (slower, better quality)
- **Plugin:** User can insert explicit resampler processor for control

**Impact:**
- Plugins no longer need to handle sample rate mismatches
- Users can mix-and-match any audio devices
- Small CPU overhead (1-3%) for resampling
- Quality vs. speed tradeoff is configurable

---

### 4. ✅ Bearer: Deleted Entirely

**Decision:** Remove bearer abstraction; network is out of scope for NDA core

**Rationale:**
- Bearer tried to abstract "network transport" but NDA's job is audio processing
- Mixing audio and network semantics created premature abstraction
- External tools already handle transport (AIOC hardware, VB-Cable, Discord)
- Duplex audio communication = two independent pipelines, not one bidirectional one

**What "Bearer" Was Trying to Do:**
```
OLD (broken):
Mic → Encryptor → [Bearer sends over network] → Decryptor → Speaker
                       ↑
                 Mixed concerns: audio + network
```

**New Clean Model:**
```
TX Pipeline: Mic → Encryptor → AIOC Sink (hardware handles RF)
RX Pipeline: AIOC Source → Decryptor → Speaker

Network transport is external to NDA.
```

**What Gets Deleted:**
```
❌ include/plugins/BearerPlugin.h
❌ src/plugins/BearerPlugin.cpp (if exists)
❌ examples/UDPBearerPlugin.h
❌ Bearer from PluginType enum
❌ setBearer() from ProcessingPipeline
❌ Bearer card from UI
```

**Impact:**
- Core codebase shrinks ~20%
- Pipeline logic becomes dramatically simpler
- No more packet serialization/deserialization in core
- Network plugins (if needed) become source/sink pairs (UDPSource, UDPSink)

---

## New Architecture

### Simplified 3-Slot Pipeline

**Old (4 slots, confusing):**
```
Source → Encryptor → Bearer → Sink
         (special)   (special)
```

**New (3 slots, clean):**
```
Source → Processor → Sink
         (optional,
          just another
          audio transform)
```

**Processor Examples:**
- AES-256 Encryptor
- AES-256 Decryptor
- Equalizer
- Compressor
- Resampler (manual override)
- Passthrough (empty slot = no processing)

**Key Insight:** Encryption is not special — it's just another audio transformation.

---

### Dual Independent Pipelines

**NDA runs TWO pipelines simultaneously:**

**TX Pipeline (Transmit):**
```
Local Microphone → [Encryptor] → AIOC Output / VB-Cable
                                      ↓
                          External tool handles transmission
```

**RX Pipeline (Receive):**
```
AIOC Input / VB-Cable → [Decryptor] → Local Speaker
        ↑
External tool delivers audio
```

**Why Dual vs. Single Bidirectional?**
- **Simpler:** Two simple chains vs. one complex graph
- **Independent:** TX failure doesn't crash RX
- **Clear UI:** Two side-by-side configs vs. complex routing diagram
- **Threading:** Two independent threads vs. complex coordination

**UI Layout:**
```
┌───────────────────────────────────┐
│ TX Pipeline                       │
│ Source:    [Microphone       ▼]  │
│ Processor: [AES Encryptor    ▼]  │
│ Sink:      [AIOC Output      ▼]  │
│ [▶ Start TX]  [■ Stop TX]        │
└───────────────────────────────────┘

┌───────────────────────────────────┐
│ RX Pipeline                       │
│ Source:    [AIOC Input       ▼]  │
│ Processor: [AES Decryptor    ▼]  │
│ Sink:      [Speaker          ▼]  │
│ [▶ Start RX]  [■ Stop RX]        │
└───────────────────────────────────┘

[▶▶ Start Both]  [■■ Stop Both]
```

---

## Critical Fixes from Audit

### Problem 1: No Real-Time Pacing ❌ → ✅ Fixed

**Old (broken):**
```cpp
while (isRunning_) {
    processAudioFrame();  // Runs as fast as possible
}
```

**New (fixed):**
```cpp
while (isRunning_) {
    processAudioFrame();
    
    // Sleep to maintain real-time cadence
    auto targetTime = startTime + (processedSamples / sampleRate);
    std::this_thread::sleep_until(targetTime);
}
```

**Impact:** Pipeline runs at exactly 1.0× real-time (not 0.36× or 1.77×)

---

### Problem 2: Python Bridge Overhead (3-15ms) ❌ → ✅ Fixed

**Optimizations:**
1. **Cache Python objects** (no recreation per buffer)
2. **Zero-copy data** (NumPy array views, not element-by-element)
3. **Batch GIL operations** (acquire once per frame, not multiple times)
4. **Cache imports** (don't re-import `base_plugin` every call)

**Expected Improvement:**
- Before: 3,000-15,000 µs per buffer
- After: 300-500 µs per buffer
- **Speedup: 6-30×**

**Impact:** Python plugins become viable for production (not just prototyping)

---

### Problem 3: No Backpressure ❌ → ✅ Fixed

**Old (broken):**
```cpp
sink->writeAudio(buffer);
processedSamples_ += buffer.frames;  // Even if write failed!
```

**New (fixed):**
```cpp
// Check sink queue space
if (sink->getAvailableSpace() < buffer.frames) {
    std::this_thread::sleep_for(5ms);  // Wait for space
}

if (sink->writeAudio(buffer)) {
    processedSamples_ += buffer.frames;  // Only on success
} else {
    droppedSamples_ += buffer.frames;   // Track failures
}
```

**Impact:** No more silent audio drops; metrics are accurate

---

### Problem 4: Metrics Lie ❌ → ✅ Fixed

**Old (broken):**
```cpp
float getCPULoad() const { return 5.0f; }  // Hardcoded!
```

**New (fixed):**
```cpp
float getCPULoad() const {
    auto audioTime = processedSamples_ / sampleRate_;
    auto wallTime = (now - startTime).seconds();
    return (audioTime / wallTime) * 100.0f;  // Measured
}
```

**New Metrics:**
- ✅ `getProcessedSamples()` — only counts successful writes
- ✅ `getDroppedSamples()` — tracks failures
- ✅ `getActualLatency()` — measured, not estimated
- ✅ `getActualCPULoad()` — measured, not hardcoded

**Impact:** Dashboard shows reality, not fantasy

---

## Implementation Phases

### Phase 1: Core Cleanup (Week 1)
- Delete bearer infrastructure
- Remove crypto from core
- Create AudioProcessorPlugin interface
- Simplify pipeline to 3 slots

**Deliverable:** Clean core that compiles

---

### Phase 2: Performance (Week 2)
- Implement sample rate resampler
- Optimize Python bridge (caching, zero-copy)

**Deliverable:** Pipeline handles any sample rate; Python is fast

---

### Phase 3: Stability (Week 3)
- Real-time pacing
- Backpressure handling
- Accurate metrics

**Deliverable:** Pipeline runs at 1.0× real-time with accurate counters

---

### Phase 4: Dual Pipeline UI (Week 3)
- Two pipeline instances
- Dual configuration UI
- Combined controls

**Deliverable:** Full TX/RX support

---

### Phase 5: Crypto Examples (Week 4)
- AES-256 encryptor/decryptor plugins
- Python crypto examples
- Key exchange documentation

**Deliverable:** Working encryption end-to-end

---

### Phase 6: Testing & Release (Week 4)
- 24-hour soak test
- 1000 start/stop cycles
- Cross-platform validation
- Documentation

**Deliverable:** v2.0 release

---

## Success Criteria

### Functional
- ✅ Dual pipelines run simultaneously
- ✅ Sample rate mismatches handled automatically
- ✅ Encryption/decryption works end-to-end
- ✅ Python and C++ plugins equal status
- ✅ Plugins hot-load without restart

### Performance
- ✅ Latency <50ms (dual pipelines, encrypted)
- ✅ CPU <30% on quad-core laptop
- ✅ Memory <100MB total
- ✅ No dropouts for 1 hour

### Stability
- ✅ 24-hour soak test passes
- ✅ 1000 start/stop cycles (no leaks)
- ✅ Source disconnect handled gracefully
- ✅ Plugin crash isolated (doesn't crash core)

---

## Migration Path (v1 → v2)

### Breaking Changes

**1. API Changes:**
```cpp
// OLD:
pipeline->setAudioSource(source);
pipeline->setEncryptor(encryptor);
pipeline->setBearer(bearer);
pipeline->setAudioSink(sink);

// NEW:
pipeline->setSource(source);
pipeline->setProcessor(processor);  // Encryptor is now a processor
// Bearer deleted entirely
pipeline->setSink(sink);
```

**2. Plugin Types:**
```cpp
// OLD:
enum class PluginType {
    AudioSource,
    AudioSink,
    Bearer,      // DELETED
    Encryptor,   // DELETED (now Processor)
    Processor
};

// NEW:
enum class PluginType {
    AudioSource,
    AudioSink,
    Processor    // Handles encryption, effects, etc.
};
```

**3. UI Changes:**
- Bearer combo box removed
- Encryptor renamed to Processor
- Two pipeline configs instead of one

### Migration Script

**Provided:** `scripts/migrate_v1_to_v2.py`

**Usage:**
```bash
python scripts/migrate_v1_to_v2.py --input src/ --dry-run
python scripts/migrate_v1_to_v2.py --input src/ --apply
```

**What it does:**
- Updates `#include` statements
- Renames `setEncryptor` → `setProcessor`
- Removes `setBearer` calls
- Adds dual pipeline boilerplate

---

## What You Get

### Before (v1.x):
- ❌ Bearer abstraction nobody uses
- ❌ Crypto hardcoded in core
- ❌ Pipeline runs too fast or too slow
- ❌ Metrics lie (hardcoded CPU, wrong sample counts)
- ❌ Python bridge too slow (15ms overhead)
- ❌ Single pipeline (can't do TX+RX)
- ❌ Sample rate mismatches crash or glitch

### After (v2.0):
- ✅ Clean 3-slot pipeline (Source → Processor → Sink)
- ✅ Encryption is a plugin (flexible, testable)
- ✅ Runs at exactly 1.0× real-time (paced)
- ✅ Metrics are accurate (measured, not guessed)
- ✅ Python bridge optimized (0.5ms overhead)
- ✅ Dual pipelines (simultaneous TX+RX)
- ✅ Automatic resampling (works with any device)

---

## Example Configurations

### Use Case 1: Encrypted AIOC Radio

**TX:**
```
Device Mic → AES-256 Encryptor → AIOC USB Output
                                       ↓
                              (AIOC transmits over RF)
```

**RX:**
```
AIOC USB Input → AES-256 Decryptor → Device Speaker
      ↑
(AIOC receives from RF)
```

**Result:** Secure two-way radio communication

---

### Use Case 2: Encrypted Discord

**TX:**
```
Device Mic → AES-256 Encryptor → VB-Cable Input
                                       ↓
                              (Discord reads from VB-Cable)
```

**RX:**
```
VB-Cable Output → AES-256 Decryptor → Device Speaker
      ↑
(Discord writes to VB-Cable)
```

**Result:** Add encryption to any voice app

---

### Use Case 3: Secure File Recording

**TX:**
```
Microphone → AES-256 Encryptor → WAV File Sink
```

**RX:**
```
WAV File Source → AES-256 Decryptor → Speaker
```

**Result:** Encrypted audio archive

---

## Key Decisions Still Open

### ✅ RESOLVED

All major decisions are locked in based on your guidance:
1. ✅ Latency target: <50ms (not <5ms)
2. ✅ Encryption: Plugin responsibility
3. ✅ Sample rate: 48kHz internal, flexible
4. ✅ Bearer: Deleted
5. ✅ Dual pipelines: Yes
6. ✅ 3-slot model: Yes
7. ✅ Python optimization: High priority
8. ✅ Network: Out of scope

### ✅ MINOR DETAILS (CONFIRMED)

**1. Resampling Quality Default: SIMPLE ✓**
- Use linear interpolation by default (fast, acceptable quality)
- Add libsamplerate high-quality mode as optional config later

**2. Error Handling on Mismatch: AUTO-FIX ✓**
- Auto-enable resampling when rates mismatch (forgiving UX)
- Log warning but don't fail initialization
- Users can override via settings if needed

**3. Processor Chaining: v2.1+ ✓**
- Keep single processor slot for v2.0 (simplicity)
- Add multi-processor chaining in v2.1 if demand exists
- For now, users can create composite processor plugins

**4. Python Processor Plugins: REQUIRED ✓**
- Python bridge MUST support AudioProcessorPlugin interface
- Python and C++ processors are equal (same capabilities)
- Examples: Python Fernet encryptor, simple gain plugin, etc.

---

## Next Steps

### Immediate Actions

1. **Review Documents:**
   - ✅ Read `NDA-SPECS-v2.md` (full specification)
   - ✅ Read `V2_IMPLEMENTATION_PLAN.md` (tactical roadmap)
   - ✅ Read this summary

2. **Approve Direction:**
   - Confirm strategic decisions are correct
   - Flag any concerns or changes

3. **Begin Development:**
   - Create branch: `git checkout -b feature/v2-migration`
   - Start Phase 1 (Core Cleanup)
   - Daily commits with progress

4. **Track Progress:**
   - Use implementation plan as checklist
   - Update status (🔴 Not started → 🟡 In progress → 🟢 Complete)
   - Weekly demos to stakeholders

---

## Timeline

**Total Duration:** 4 weeks  
**Effort:** ~80-100 hours  
**Parallel Work:** UI can proceed while core is refactored  
**Risk:** Medium (significant refactoring, but scope is clear)

**Milestones:**
- Week 1 End: Core compiles without bearer/crypto
- Week 2 End: Python bridge optimized, resampling works
- Week 3 End: Dual pipeline UI functional
- Week 4 End: v2.0 released with crypto examples

---

## Questions?

If anything is unclear or needs adjustment:
1. Flag it now before implementation begins
2. We can refine the spec without breaking momentum
3. Small tactical decisions can be made during development

**Otherwise, we're ready to begin Phase 1.**

---

*Strategic alignment confirmed. Ready to execute.*

