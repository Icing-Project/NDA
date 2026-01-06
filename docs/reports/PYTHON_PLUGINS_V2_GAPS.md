# Python Plugins V2 - Gap Analysis & Action Items

**Quick Reference:** Issues found during V2 compliance audit  
**See Full Report:** `PYTHON_PLUGINS_V2_AUDIT.md`

---

## Critical Gaps (Breaking Changes)

**None** ✅

---

## Major Gaps (Should Fix Before Release)

### 1. Method Naming Inconsistency

**Issue:** `get_channels()` should be `get_channel_count()` to match C++ API and docs

**Affected Files:**
```
plugins_py/base_plugin.py (lines 142, 152)
plugins_py/sounddevice_microphone.py (uses get_channels)
plugins_py/sounddevice_speaker.py (uses get_channels)
plugins_py/pulseaudio_microphone.py (uses get_channels)
plugins_py/pulseaudio_speaker.py (uses get_channels)
plugins_py/sine_wave_source.py (uses get_channels)
plugins_py/wav_file_sink.py (uses get_channels)
plugins_py/null_sink.py (uses get_channels)
```

**Fix Required:**
- [ ] Update `base_plugin.py`: `get_channels()` → `get_channel_count()`
- [ ] Update `base_plugin.py`: `set_channels()` → `set_channel_count()`
- [ ] Update all 7 source/sink plugins to use new method names
- [ ] Update C++ `PythonPluginBridge.cpp` to call correct Python methods
- [ ] Test all plugins after rename

**Estimated Effort:** 2-3 hours  
**Priority:** HIGH

---

### 2. Documentation Mismatch - Crypto Examples

**Issue:** Docs reference files that don't exist

**PLUGIN_DEVELOPMENT_v2.md says:**
```
Line 109: plugins_py/examples/fernet_encryptor.py - Python encryption  ✅ EXISTS
Line 682-684:
  └── examples/
      ├── aes256_encryptor.py     ❌ DOES NOT EXIST
      └── aes256_decryptor.py     ❌ DOES NOT EXIST
```

**Actual Files:**
```
plugins_py/examples/fernet_encryptor.py   ✅ EXISTS
plugins_py/examples/fernet_decryptor.py   ✅ EXISTS
```

**Fix Options:**

**Option A: Fix Documentation (Quick)**
- [ ] Update `docs/PLUGIN_DEVELOPMENT_v2.md` line 682-684
- [ ] Change `aes256_*.py` → `fernet_*.py`
- [ ] Estimated Effort: 30 minutes

**Option B: Implement AES-256 Examples (Better)**
- [ ] Create `plugins_py/examples/aes256_encryptor.py`
- [ ] Create `plugins_py/examples/aes256_decryptor.py`
- [ ] Use `cryptography` library's AES-GCM mode
- [ ] Match C++ AES256 plugin functionality
- [ ] Estimated Effort: 6-8 hours

**Priority:** MEDIUM

---

## Minor Gaps (Good to Have)

### 3. Naming Convention Not Documented

**Issue:** Python uses snake_case, C++ uses camelCase - not explained

**Current State:**
- Python: `process_audio()`, `read_audio()`, `write_audio()` ✅ Correct per PEP 8
- C++ Spec: `processAudio()`, `readAudio()`, `writeAudio()` ✅ Correct per project style
- Bridge: Must translate between conventions (status unknown)

**Fix Required:**
- [ ] Add section to `PLUGIN_DEVELOPMENT_v2.md` explaining naming difference
- [ ] Verify `PythonPluginBridge.cpp` handles translation correctly
- [ ] Document in Python plugin development section

**Estimated Effort:** 1 hour  
**Priority:** LOW

---

### 4. Push vs Pull Model Not Documented

**Issue:** Python has `set_audio_callback()` for push model, not in C++ spec

**Current State:**
- Python AudioSourcePlugin has push model: `set_audio_callback(callback)`
- C++ AudioSourcePlugin only shows pull model: `readAudio(buffer)`
- Both models work but difference not explained

**Fix Required:**
- [ ] Add section to `PLUGIN_DEVELOPMENT_v2.md` explaining:
  - Pull model: Pipeline calls `read_audio()` to get data
  - Push model: Plugin calls callback when data ready
  - Python supports both, C++ typically pull only

**Estimated Effort:** 30 minutes  
**Priority:** LOW

---

## Verification Needed

### 5. Python Bridge Optimization Status

**Issue:** V2 specs describe optimizations, but implementation status unknown

**V2 Claims (NDA-SPECS-v2.md Phase 3):**
- Object caching (reuse AudioBuffer objects)
- Zero-copy data sharing (NumPy direct memory access)
- Batch GIL operations (hold GIL for entire frame)
- Module import caching (import once, reuse)
- Target: <500µs overhead per buffer

**Current Status:** Unknown ❓

**Verification Needed:**
- [ ] Audit `src/plugins/PythonPluginBridge.cpp` implementation
- [ ] Check if optimizations are implemented
- [ ] Run `tests/benchmark_python_bridge.cpp` if exists
- [ ] Document actual measured overhead

**Estimated Effort:** 4-6 hours (audit + benchmarking)  
**Priority:** MEDIUM

---

## Implementation Quality Assessment

### What's Working Well ✅

1. **Base Architecture** - 3-type system correctly implemented
2. **AudioProcessorPlugin** - Perfect V2 compliance
3. **Example Processors** - All 4 examples present and working:
   - `passthrough.py` - Minimal template
   - `simple_gain.py` - Parameter handling
   - `fernet_encryptor.py` - Encryption example
   - `fernet_decryptor.py` - Decryption example
4. **Standalone Tests** - All examples have `if __name__ == "__main__"` tests
5. **AudioBuffer Class** - Well-designed NumPy-based buffer
6. **State Management** - Proper lifecycle handling
7. **Error Handling** - Graceful failure modes

### What Needs Work ⚠️

1. Method naming: `get_channels` → `get_channel_count`
2. Documentation: crypto example filenames
3. Documentation: naming convention explanation
4. Documentation: push/pull model explanation
5. Verification: bridge optimization status

---

## Action Plan

### Phase 1: Critical Fixes (Before Next Release)

**Total Time: 2-3 hours**

- [ ] **Fix Method Naming** (2 hours)
  - Update base_plugin.py
  - Update all 7 source/sink plugins
  - Update C++ bridge calls
  - Test all plugins

- [ ] **Fix Documentation** (30 minutes)
  - Update PLUGIN_DEVELOPMENT_v2.md
  - Change aes256 → fernet references

### Phase 2: Nice-to-Have Improvements

**Total Time: 2-3 hours**

- [ ] **Document Naming Conventions** (1 hour)
  - Explain snake_case vs camelCase
  - Update plugin development guide

- [ ] **Document Push/Pull Models** (30 minutes)
  - Explain both models
  - Show examples

- [ ] **Verify Bridge Optimizations** (1-2 hours)
  - Quick audit of PythonPluginBridge.cpp
  - Note optimization status

### Phase 3: Optional Enhancements

**Total Time: 6-8 hours**

- [ ] **Implement AES-256 Python Examples** (6-8 hours)
  - Create aes256_encryptor.py
  - Create aes256_decryptor.py
  - Match C++ functionality
  - Add tests

---

## File Change Checklist

### Files to Modify (Phase 1)

```
plugins_py/
  ✏️ base_plugin.py                      (2 method renames)
  ✏️ sounddevice_microphone.py          (method call updates)
  ✏️ sounddevice_speaker.py             (method call updates)
  ✏️ pulseaudio_microphone.py           (method call updates)
  ✏️ pulseaudio_speaker.py              (method call updates)
  ✏️ sine_wave_source.py                (method call updates)
  ✏️ wav_file_sink.py                   (method call updates)
  ✏️ null_sink.py                       (method call updates)

docs/
  ✏️ PLUGIN_DEVELOPMENT_v2.md           (fix crypto example names)
  ✏️ NDA-SPECS-v2.md                    (fix crypto example names)

src/plugins/
  ✏️ PythonPluginBridge.cpp             (update Python method calls)
```

### Files to Create (Optional Phase 3)

```
plugins_py/examples/
  ➕ aes256_encryptor.py               (new file)
  ➕ aes256_decryptor.py               (new file)
```

---

## Testing Checklist

After applying fixes:

- [ ] Test passthrough.py standalone
- [ ] Test simple_gain.py standalone
- [ ] Test fernet_encryptor.py standalone
- [ ] Test fernet_decryptor.py roundtrip
- [ ] Test sounddevice_microphone.py in pipeline
- [ ] Test sounddevice_speaker.py in pipeline
- [ ] Test sine_wave_source.py in pipeline
- [ ] Test null_sink.py in pipeline
- [ ] Test TX pipeline: Mic → Processor → Sink
- [ ] Test RX pipeline: Source → Processor → Speaker
- [ ] Verify no Python exceptions in console
- [ ] Verify C++ bridge loads plugins correctly

---

## Success Criteria

**Phase 1 Complete When:**
- ✅ All method names match V2 spec
- ✅ All documentation references correct filenames
- ✅ All plugins load and run without errors
- ✅ C++ bridge calls correct Python methods

**Phase 2 Complete When:**
- ✅ Naming conventions documented
- ✅ Push/pull models explained
- ✅ Bridge optimization status known

**Phase 3 Complete When:**
- ✅ AES-256 Python examples implemented (optional)
- ✅ Examples tested and documented

---

**Last Updated:** December 26, 2025  
**Status:** Gaps identified, action plan ready  
**Next Step:** Apply Phase 1 fixes

