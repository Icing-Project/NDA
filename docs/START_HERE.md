# 🎯 NDA Documentation - START HERE

Welcome! This guide helps you find exactly what you need.

---

## What is NDA?

**NDA** (Nade Desktop Application) is a **real-time audio encryption bridge** for secure communication. It processes audio through a clean 3-slot pipeline (Source → Processor → Sink), supports dual independent TX/RX pipelines, and provides plugin-based encryption.

**Key Features:**
- ✅ Dual independent pipelines (simultaneous TX + RX)
- ✅ Plugin-based encryption (not hardcoded)
- ✅ Automatic sample rate adaptation (44.1/48/96 kHz)
- ✅ Python & C++ plugin support
- ✅ <50ms latency, <30% CPU

---

## 🚀 Choose Your Path

### **I'm a User** – I just want to use NDA
→ **Start here:** [`getting-started/README.md`](./getting-started/README.md)
→ **Then:** [`getting-started/use-cases.md`](./getting-started/use-cases.md) (find your scenario)
→ **Finally:** [`getting-started/installation.md`](./getting-started/installation.md) (setup instructions)
→ **Need help?** [`examples/encrypted-aioc-radio.md`](./examples/encrypted-aioc-radio.md) or [`examples/discord-voip-encryption.md`](./examples/discord-voip-encryption.md)

---

### **I'm an AI/Coding Bot** – I need to implement features
→ **Start here:** [`ai-instructions/AGENTS.md`](../AGENTS.md) (your instructions)
→ **Then:** [`technical/ARCHITECTURE.md`](./technical/ARCHITECTURE.md) (understand the design)
→ **Reference:** [`technical/specifications.md`](./technical/specifications.md) (complete API)
→ **If writing plugins:** [`development/plugin-development.md`](./development/plugin-development.md)
→ **All docs index:** [`ai-instructions/ai-documentation-index.md`](./ai-instructions/ai-documentation-index.md)

---

### **I'm a Developer** – I need to understand everything
→ **Start here:** [`technical/ARCHITECTURE.md`](./technical/ARCHITECTURE.md) (core design)
→ **Complete spec:** [`technical/specifications.md`](./technical/specifications.md)
→ **For plugins:** [`development/plugin-development.md`](./development/plugin-development.md)
→ **Migration from v1.x?** [`development/migration-v1-to-v2.md`](./development/migration-v1-to-v2.md)
→ **Troubleshooting:** [`development/troubleshooting.md`](./development/troubleshooting.md)
→ **Status:** [`reports/v2-implementation-report.md`](./reports/v2-implementation-report.md)

---

### **I'm a Specialist** – Deep performance/security work
→ **Start here:** [`technical/ARCHITECTURE.md`](./technical/ARCHITECTURE.md)
→ **Decisions:** [`strategy/v2-decisions-locked.md`](./strategy/v2-decisions-locked.md) (why these choices?)
→ **Performance:** [`reports/v2.1-performance-analysis.md`](./reports/v2.1-performance-analysis.md)
→ **Bridge internals:** [`technical/python-bridge.md`](./technical/python-bridge.md)
→ **Resampling:** [`technical/resampling.md`](./technical/resampling.md)

---

### **I'm Leadership/PM** – Strategic overview
→ **Quick decision summary:** [`strategy/v2-decisions-locked.md`](./strategy/v2-decisions-locked.md)
→ **Strategic rationale:** [`strategy/v2-strategic-summary.md`](./strategy/v2-strategic-summary.md)
→ **Implementation timeline:** [`strategy/implementation-plan.md`](./strategy/implementation-plan.md)
→ **Build status:** [`reports/v2-implementation-report.md`](./reports/v2-implementation-report.md)

---

## 📂 Folder Organization

```
docs/
├── getting-started/          ← Start here if you're new
│   ├── README.md             "What is NDA?"
│   ├── use-cases.md          "Can I use it for my scenario?"
│   └── installation.md       "How do I install & build?"
│
├── technical/               ← Deep technical reference
│   ├── ARCHITECTURE.md       "Core design & components"
│   ├── specifications.md     "Complete API reference"
│   ├── python-bridge.md      "Python plugin optimization"
│   └── resampling.md         "Sample rate adaptation details"
│
├── development/            ← For developers & plugin authors
│   ├── plugin-development.md "How to write plugins"
│   ├── python-processor-guide.md "Python plugin reference"
│   ├── migration-v1-to-v2.md "v1.x → v2.0 migration"
│   └── troubleshooting.md    "Common issues & solutions"
│
├── strategy/              ← Decisions & planning
│   ├── v2-decisions-locked.md "Final approved decisions"
│   ├── v2-strategic-summary.md "Executive summary"
│   └── implementation-plan.md "Detailed roadmap"
│
├── reports/              ← Analysis & findings
│   ├── v2-implementation-report.md "Build status"
│   ├── v2.1-performance-analysis.md "Performance limits"
│   └── v2.1-executive-summary.md "Optimization findings"
│
├── ai-instructions/      ← For coding AIs/bots
│   ├── AGENTS.md           "Quick AI reference"
│   └── ai-documentation-index.md "AI reading paths"
│
├── examples/            ← Real-world walkthroughs
│   ├── encrypted-aioc-radio.md "Step-by-step AIOC setup"
│   └── discord-voip-encryption.md "Step-by-step Discord setup"
│
├── legacy/             ← v1.x documentation (deprecated)
│   ├── v1-specs.md      "Original v1.x specification"
│   └── README_LEGACY.md "v1.x archive"
│
└── START_HERE.md       ← YOU ARE HERE
```

---

## 🎯 Quick Reference by Task

| Task | Read This |
|------|-----------|
| **I'm new, what is NDA?** | [`getting-started/README.md`](./getting-started/README.md) |
| **How do I install it?** | [`getting-started/installation.md`](./getting-started/installation.md) |
| **Can I use it for [my use case]?** | [`getting-started/use-cases.md`](./getting-started/use-cases.md) |
| **How do I build from source?** | [`getting-started/installation.md`](./getting-started/installation.md) |
| **How do I write a plugin?** | [`development/plugin-development.md`](./development/plugin-development.md) |
| **I'm migrating from v1.x** | [`development/migration-v1-to-v2.md`](./development/migration-v1-to-v2.md) |
| **Something's not working** | [`development/troubleshooting.md`](./development/troubleshooting.md) |
| **I need the complete API spec** | [`technical/specifications.md`](./technical/specifications.md) |
| **How does the architecture work?** | [`technical/ARCHITECTURE.md`](./technical/ARCHITECTURE.md) |
| **Why was [decision] made?** | [`strategy/v2-decisions-locked.md`](./strategy/v2-decisions-locked.md) |
| **What's the timeline?** | [`strategy/v2-strategic-summary.md`](./strategy/v2-strategic-summary.md) |
| **What's been implemented?** | [`reports/v2-implementation-report.md`](./reports/v2-implementation-report.md) |
| **Performance & optimization details** | [`reports/v2.1-performance-analysis.md`](./reports/v2.1-performance-analysis.md) |
| **I'm an AI, where do I start?** | [`ai-instructions/AGENTS.md`](../AGENTS.md) |

---

## 🏗️ Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────┐
│  TRANSMIT (TX) Pipeline                                     │
│  Device Mic → [Encryptor Plugin] → AIOC/Network Output     │
│                                                             │
│  RECEIVE (RX) Pipeline                                      │
│  AIOC/Network Input → [Decryptor Plugin] → Device Speaker   │
└─────────────────────────────────────────────────────────────┘
```

**Key Points:**
- **3-Slot Pipeline:** Source → Processor (optional) → Sink
- **Dual Independent:** TX and RX run simultaneously
- **Plugin Architecture:** Encryption/effects are plugins, not core
- **Sample Rate Agnostic:** Automatically converts 44.1/48/96 kHz
- **Python-Friendly:** Python plugins = C++ plugins in terms of support

---

## 📊 v2.0 Major Changes

| Aspect | v1.x | v2.0 |
|--------|------|------|
| **Pipeline slots** | 4 | 3 (Source → Processor → Sink) |
| **Encryption** | Core + plugins | Plugins only |
| **Network/Bearer** | Bearer plugin | Removed (use external tools) |
| **Dual pipelines** | No | Yes (independent TX + RX) |
| **Sample rates** | Manual | Automatic 48kHz internal |
| **Python support** | Limited | Equal to C++ |
| **Code size** | ~800 lines pipeline | ~430 lines (-46%) |

**→ Want migration guide?** See [`development/migration-v1-to-v2.md`](./development/migration-v1-to-v2.md)

---

## 🚦 Current Status

- ✅ **Architecture:** v2.0 complete
- ✅ **Code:** Implementation complete (49/59 tasks)
- ✅ **Documentation:** Fully reorganized
- ⚠️ **Performance:** Optimization complete, real-time validation pending
- 📊 **Testing:** Build & stability tests required

**Full status:** [`reports/v2-implementation-report.md`](./reports/v2-implementation-report.md)

---

## 🤔 Common Questions

**Q: What's the difference between v1.x and v2.0?**
A: See [`strategy/v2-decisions-locked.md`](./strategy/v2-decisions-locked.md) for decisions, or [`development/migration-v1-to-v2.md`](./development/migration-v1-to-v2.md) for technical changes.

**Q: Is it really real-time?**
A: Target is <50ms latency. Performance analysis: [`reports/v2.1-performance-analysis.md`](./reports/v2.1-performance-analysis.md)

**Q: Can I use Python plugins?**
A: Yes! Equal support to C++. Guide: [`development/plugin-development.md`](./development/plugin-development.md)

**Q: How do I set up encrypted AIOC?**
A: Walkthrough: [`examples/encrypted-aioc-radio.md`](./examples/encrypted-aioc-radio.md)

**Q: Something's broken, help!**
A: Check [`development/troubleshooting.md`](./development/troubleshooting.md)

---

## 📞 Need Help?

1. **Quick question?** Check the relevant section above
2. **Can't find what you need?** Check [`development/troubleshooting.md`](./development/troubleshooting.md)
3. **Specific use case?** Browse [`examples/`](./examples/)
4. **Deep technical issue?** Check [`technical/specifications.md`](./technical/specifications.md)

---

**Last Updated:** January 2026
**Version:** 2.0

→ **Now go read the docs for your role!** ↑
