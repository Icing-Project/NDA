# NDA v2.0 Documentation Index

Welcome to the NDA v2.0 documentation! This index guides you through all specification and planning documents.

---

## 📋 Quick Reference

| Document | Purpose | Audience |
|----------|---------|----------|
| **[V2 Decisions](../strategy/v2-decisions-locked.md)** | Final approved decisions | Everyone |
| **[Technical Specs](../technical/specifications.md)** | Complete technical spec | Developers |
| **[Implementation Plan](../strategy/implementation-plan.md)** | Step-by-step roadmap | Developers |
| **[Strategic Summary](../strategy/v2-strategic-summary.md)** | Strategic rationale | Leadership/PM |
| **[Python Processor Guide](../development/python-processor-guide.md)** | Python plugin development | Plugin authors |

---

## 🎯 Start Here

### If you want to understand the vision:
→ **Read: [V2 Decisions](../strategy/v2-decisions-locked.md)** (5 min read)

### If you're implementing v2.0:
→ **Read: [Implementation Plan](../strategy/implementation-plan.md)** (detailed roadmap)

### If you're writing Python plugins:
→ **Read: [Python Processor Guide](../development/python-processor-guide.md)** (comprehensive guide)

### If you need the full technical spec:
→ **Read: [Technical Specifications](../technical/specifications.md)** (complete specification)

---

## 📚 Document Summaries

### 1. [V2 Decisions Locked](../strategy/v2-decisions-locked.md) ✅

**Status:** APPROVED — Ready for implementation  
**Length:** ~500 lines  

**What it covers:**
- All 8 strategic decisions (confirmed and locked)
- Rationale for each decision
- Files to create/delete/modify
- Success criteria
- Timeline overview

**Key Decisions:**
1. Latency: <50ms (not <5ms)
2. Encryption: Plugin-only (not core)
3. Sample rate: 48kHz, auto-adapt, simple resampling
4. Bearer: **DELETED**
5. Dual pipelines: TX + RX
6. 3-slot model: Source → Processor → Sink
7. Python processors: **REQUIRED**
8. Python optimization: <500µs overhead

---

### 2. [Technical Specifications](../technical/specifications.md) 📖

**Length:** ~960 lines  
**Comprehensive technical specification**

**Contents:**
1. Executive summary
2. Architecture overview
3. Plugin interfaces (Source, Sink, Processor)
4. Processing pipeline design
5. Python bridge optimization
6. User interface (dual pipeline layout)
7. Example configurations
8. Performance targets
9. Directory structure
10. Implementation roadmap
11. Migration guide
12. FAQ

**Use this for:**
- Complete system design reference
- API specifications
- Performance requirements
- Success criteria

---

### 3. [Implementation Plan](../strategy/implementation-plan.md) 🛠️

**Length:** ~1100 lines  
**Tactical step-by-step implementation guide**

**Contents:**
- 6 phases over 4 weeks
- Phase 1: Core cleanup (delete bearer, remove crypto)
- Phase 2: Sample rate adaptation (resampler)
- Phase 3: Python bridge optimization (6-30× faster)
- Phase 4: Real-time pacing & metrics
- Phase 5: Dual pipeline UI
- Phase 6: Crypto plugin examples

**Each phase includes:**
- Specific files to modify
- Code examples for changes
- Testing requirements
- Success criteria

**Use this for:**
- Day-to-day development work
- Task tracking
- Code review checklist

---

### 4. [Strategic Summary](../strategy/v2-strategic-summary.md) 📊

**Length:** ~450 lines  
**Executive decision record**

**Contents:**
- Strategic rationale for each change
- Before/after comparisons
- Architecture diagrams
- Example use cases
- Migration path from v1.x
- Timeline and milestones

**Use this for:**
- Understanding why decisions were made
- Explaining v2.0 to stakeholders
- Architecture presentations

---

### 5. [Python Processor Guide](../development/python-processor-guide.md) 🐍

**Length:** ~550 lines  
**Comprehensive Python plugin development guide**

**Contents:**
- AudioProcessorPlugin interface
- Example 1: Simple gain processor
- Example 2: XOR encryptor
- Example 3: High-pass filter
- Buffer format and sample range
- Performance tips
- Testing strategies
- Common patterns
- Troubleshooting
- Best practices

**Use this for:**
- Writing Python processor plugins
- Understanding buffer format
- Performance optimization
- Debugging plugin issues

---

## 🚀 What Changed from v1.x

### Deleted ❌
```
include/crypto/Encryptor.h
include/crypto/KeyExchange.h
src/crypto/Encryptor.cpp
src/crypto/KeyExchange.cpp
include/plugins/BearerPlugin.h
include/plugins/EncryptorPlugin.h
examples/UDPBearerPlugin.h
```

### Created ✅
```
include/plugins/AudioProcessorPlugin.h
include/audio/Resampler.h
src/audio/Resampler.cpp
plugins_src/examples/AES256EncryptorPlugin.cpp
plugins_src/examples/AES256DecryptorPlugin.cpp
plugins_py/examples/simple_gain.py
plugins_py/examples/fernet_encryptor.py
plugins_py/examples/fernet_decryptor.py
plugins_py/examples/passthrough.py
docs/NDA-SPECS-v2.md
docs/V2_IMPLEMENTATION_PLAN.md
docs/V2_STRATEGIC_SUMMARY.md
docs/V2_DECISIONS_LOCKED.md
docs/PYTHON_PROCESSOR_GUIDE.md
```

### Simplified 📉
```
ProcessingPipeline.cpp:  ~800 → ~500 lines (-37%)
PluginTypes.h:           4 types → 3 types (-25%)
Overall codebase:        -20% lines of code
```

---

## 🎯 Architecture Changes

### Old (v1.x): Confusing 4-Slot Model
```
Source → Encryptor → Bearer → Sink
         (special)   (special)
         ^           ^
         Hardcoded crypto + network
```

### New (v2.0): Clean 3-Slot Model
```
Source → Processor → Sink
         (optional)
         ^
         Just another transform
         (encryption, effects, etc.)
```

### Dual Pipelines (NEW)
```
TX: Mic → Encryptor → AIOC Sink
RX: AIOC Source → Decryptor → Speaker

Both run simultaneously, independently
```

---

## 📅 Timeline

**Total:** 4 weeks  
**Effort:** ~80-100 hours  
**Risk:** Medium (significant refactor, clear scope)

```
Week 1: Core Cleanup
├─ Delete bearer
├─ Remove crypto
├─ Add processor interface
└─ Simplify pipeline

Week 2: Performance
├─ Sample rate resampler
├─ Python bridge optimization
└─ Benchmarking

Week 3: Pacing & UI
├─ Real-time pacing
├─ Accurate metrics
└─ Dual pipeline UI

Week 4: Polish
├─ Crypto plugin examples
├─ Documentation
├─ 24-hour soak test
└─ v2.0 release
```

---

## ✅ Success Criteria

### Functional
- [ ] Bearer removed (0 references)
- [ ] Crypto in plugins only
- [ ] AudioProcessorPlugin interface works
- [ ] Python processors work
- [ ] Dual pipelines run simultaneously
- [ ] Sample rates auto-adapt
- [ ] Encryption/decryption end-to-end

### Performance
- [ ] Latency <50ms (dual, encrypted, Python)
- [ ] Python bridge <500µs
- [ ] CPU <30%
- [ ] Memory <100MB
- [ ] No dropouts for 1 hour

### Stability
- [ ] 24-hour test passes
- [ ] 1000 start/stop cycles
- [ ] Graceful error handling
- [ ] No memory leaks

---

## 📖 Reading Order

### For Developers (Full Implementation):
1. [V2 Decisions Locked](../strategy/v2-decisions-locked.md) (understand decisions)
2. [Technical Specifications](../technical/specifications.md) (complete spec)
3. [Implementation Plan](../strategy/implementation-plan.md) (follow step-by-step)
4. [Python Processor Guide](../development/python-processor-guide.md) (if writing Python plugins)

### For Plugin Authors:
1. [Python Processor Guide](../development/python-processor-guide.md) (primary guide)
2. [Technical Specifications](../technical/specifications.md) § Plugin Architecture
3. plugins_py/examples/ (reference implementations)

### For Leadership/PM:
1. [V2 Decisions Locked](../strategy/v2-decisions-locked.md) (approved decisions)
2. [Strategic Summary](../strategy/v2-strategic-summary.md) (rationale)
3. [Technical Specifications](../technical/specifications.md) § Executive Summary

### For Code Review:
1. [Implementation Plan](../strategy/implementation-plan.md) (checklist per phase)
2. [V2 Decisions Locked](../strategy/v2-decisions-locked.md) (validate against approved decisions)

---

## 🔗 Related Documents

### Legacy v1.x Documentation
- [`legacy/v1-specs.md`](../legacy/v1-specs.md) — Original v1.x specification (superseded)
- [`reports/audio_pipeline_audit.md`](../reports/audio_pipeline_audit.md) — Detailed v1.x performance audit
- [`reports/ARCHITECTURE_REPORT.md`](../reports/ARCHITECTURE_REPORT.md) — v1.x architecture report

### Build & Deployment
- [`reports/DEPLOYMENT_COMPLETE.md`](../reports/DEPLOYMENT_COMPLETE.md) — Deployment procedures
- [`PLATFORM_PACKAGES_READY.md`](../PLATFORM_PACKAGES_READY.md) — Platform packaging guide

### Plugin Development
- `plugins_py/QUICKSTART.md` — Python plugin quick start
- `plugins_py/README.md` — Python plugin overview

---

## 🆘 Support

### Questions about v2.0 direction?
→ Read **[Strategic Summary](../strategy/v2-strategic-summary.md)** § "Why This Decision"

### Stuck during implementation?
→ Check **[Implementation Plan](../strategy/implementation-plan.md)** for your current phase

### Python plugin not working?
→ Debug with **[Python Processor Guide](../development/python-processor-guide.md)** § "Troubleshooting"

### Need full API reference?
→ Consult **[Technical Specifications](../technical/specifications.md)** § Plugin Interfaces

---

## 🚦 Current Status

**Phase:** Planning Complete ✅  
**Next:** Begin Phase 1 (Core Cleanup)

**Branch:** `feature/v2-migration` (to be created)

**First Steps:**
```bash
# Create branch
git checkout -b feature/v2-migration

# Commit planning docs
git add docs/
git commit -m "[v2.0] Add v2.0 specifications and planning"

# Begin Phase 1
# Follow V2_IMPLEMENTATION_PLAN.md § Phase 1
```

---

## 📝 Change Log

### December 25, 2025
- ✅ Created NDA-SPECS-v2.md
- ✅ Created V2_IMPLEMENTATION_PLAN.md
- ✅ Created V2_STRATEGIC_SUMMARY.md
- ✅ Created V2_DECISIONS_LOCKED.md
- ✅ Created PYTHON_PROCESSOR_GUIDE.md
- ✅ Created README_V2.md (this file)
- ✅ All strategic decisions confirmed
- ✅ Ready for implementation

---

**All planning complete. Ready to build.** 🎉

*For questions or clarifications, refer to the appropriate document above.*


