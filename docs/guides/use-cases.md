# NDA Use Cases & Scenarios

This document describes common use cases for NDA and how to set them up.

---

## Common Use Cases

### 1. Encrypted AIOC Radio Communication

**Scenario:** Secure two-way radio communication using AIOC hardware

**Configuration:**
- **TX Pipeline:** Device Microphone → AES-256 Encryptor → AIOC USB Output
- **RX Pipeline:** AIOC USB Input → AES-256 Decryptor → Device Speaker

**Why:** AIOC hardware handles RF transmission; NDA adds encryption layer

**See:** `examples/encrypted-aioc-radio.md` for step-by-step setup

---

### 2. Encrypted VoIP (Discord, Skype, Teams)

**Scenario:** Add encryption to any voice chat application

**Configuration:**
- **TX Pipeline:** Device Microphone → AES-256 Encryptor → VB-Cable Input
  - (Discord reads from VB-Cable)
- **RX Pipeline:** VB-Cable Output → AES-256 Decryptor → Device Speaker
  - (Discord writes to VB-Cable)

**Why:** Virtual audio cables let NDA sit between apps transparently

**See:** `examples/discord-voip-encryption.md` for step-by-step setup

---

### 3. Encrypted File Recording

**Scenario:** Record encrypted audio for later playback

**Configuration:**
- **TX Pipeline:** Microphone → AES-256 Encryptor → WAV File Sink
- **RX Pipeline:** WAV File Source → AES-256 Decryptor → Speaker

**Why:** Encrypt sensitive recordings at capture time

---

### 4. Passthrough Testing (No Encryption)

**Scenario:** Test audio quality and latency without encryption overhead

**Configuration:**
- **TX Pipeline:** Sine Generator → (None) → Null Sink
- **RX Pipeline:** File Source → (None) → Speaker

**Why:** Baseline performance testing

---

### 5. Audio Effects Chain (Not Encrypted)

**Scenario:** Apply audio effects (gain, EQ, compression)

**Configuration:**
- **Source:** Microphone
- **Processor:** Custom Effect Plugin
- **Sink:** Speaker

**Why:** NDA's plugin architecture supports any audio transformation

---

## Audience-Specific Guides

### For End Users
1. Read `README.md` (what is NDA?)
2. Find your use case above
3. Follow the step-by-step guide in `examples/`
4. Configure plugins in UI and start

### For Security Engineers
1. Review `examples/encrypted-aioc-radio.md` for encryption setup
2. Check `technical/specifications.md` § Plugin Interfaces
3. Consider custom processor plugins for specialized encryption
4. See `development/plugin-development.md` for plugin creation

### For Developers
1. Understand architecture: `technical/ARCHITECTURE.md`
2. Review plugin interfaces: `technical/specifications.md`
3. Study example plugins: `plugins_src/examples/` and `plugins_py/examples/`
4. Implement custom processor: `development/plugin-development.md`

---

## Not Supported

- ❌ **Sub-5ms latency** (target: <50ms, due to encryption/resampling overhead)
- ❌ **DAW-grade audio quality** (focus: secure communication, not music production)
- ❌ **Network transport in core** (use AIOC, VB-Cable, Discord as transport)
- ❌ **Built-in key exchange** (share keys out-of-band via Signal, in-person, etc.)

---

## Next Steps

Pick your use case, then:

1. Go to `examples/[use-case].md` for detailed walkthrough
2. Review `getting-started/installation.md` for build instructions
3. Load plugins and configure pipelines in the UI
4. Start both TX and RX pipelines
5. Verify audio flows correctly

---

For detailed examples, see:
- `examples/encrypted-aioc-radio.md`
- `examples/discord-voip-encryption.md`

For custom solutions, see:
- `development/plugin-development.md`
