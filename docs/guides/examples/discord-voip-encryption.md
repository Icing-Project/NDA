# Step-by-Step: Encrypted Discord/VoIP Calls

This guide walks through setting up end-to-end encrypted voice calls in Discord, Skype, Teams, or any VoIP app using NDA.

---

## How It Works

NDA sits between your microphone and Discord using virtual audio cables:

```
Your Voice:
  Microphone → NDA Encryptor → VB-Cable Input → Discord sends encrypted

Their Voice:
  Discord → VB-Cable Output → NDA Decryptor → Speaker
```

Discord thinks it's talking to a regular microphone/speaker, but it's actually encrypted.

---

## What You Need

- NDA installed and built (see `getting-started/installation.md`)
- Discord, Skype, Teams, or similar VoIP app
- Virtual audio cable software:
  - **Windows:** VB-Cable (https://vb-audio.com/Cable/)
  - **Linux:** PulseAudio (usually pre-installed)
  - **macOS:** BlackHole (https://github.com/ExistentialAudio/BlackHole)
- Microphone and speaker (or headset)

---

## Step 1: Install Virtual Audio Cable

### Windows (VB-Cable)
1. Download: https://vb-audio.com/Cable/
2. Run installer, follow instructions
3. Restart Windows
4. Verify in Sound Settings: "CABLE Input" and "CABLE Output" appear

### Linux (PulseAudio)
Already installed on most distributions. Check:
```bash
pactl list short sinks | grep -i loopback
```

### macOS (BlackHole)
1. Download: https://github.com/ExistentialAudio/BlackHole/releases
2. Run installer
3. Restart macOS

---

## Step 2: Build & Run NDA

```bash
# Build
mkdir build && cd build
cmake .. && make -j

# Run
./NDA
```

---

## Step 3: Load Plugins

In NDA UI:
1. Click **"Load Plugins"** or **"Auto-Load Python Plugins"**
2. Verify you see available plugins

---

## Step 4: Configure TX Pipeline (Your Voice)

**Goal:** Encrypt microphone audio and send to Discord via virtual cable

1. **Source:** Select "Microphone"
2. **Processor:** Select "AES-256 Encryptor"
3. **Sink:** Select "VB-Cable Input" (or "Loopback" on Linux/macOS)

**In Discord Settings:**
- Go to Voice & Video
- Input Device: Select "CABLE Output" (Windows) or "Loopback Monitor" (Linux/macOS)
- Click Apply

**Now Discord reads encrypted audio from NDA!**

---

## Step 5: Configure RX Pipeline (Their Voice)

**Goal:** Receive Discord audio from virtual cable and decrypt to speaker

1. **Source:** Select "VB-Cable Output" (or "Loopback" on Linux/macOS)
2. **Processor:** Select "AES-256 Decryptor"
3. **Sink:** Select "Speaker"

**In Discord Settings:**
- Go to Voice & Video
- Output Device: Select "CABLE Input" (Windows) or "Loopback" (Linux/macOS)
- Click Apply

**Now NDA decrypts Discord audio to your speaker!**

---

## Step 6: Share Encryption Key

Both users must use the SAME encryption key!

### Generate a Key
```bash
# Linux/macOS
openssl rand -hex 32

# Windows PowerShell
$bytes = New-Object byte[] 32
(New-Object System.Security.Cryptography.RNGCryptoServiceProvider).GetBytes($bytes)
-join ($bytes | ForEach-Object { $_.ToString("x2") })
```

### Share Securely
Send the key to the other person via:
- Signal (encrypted chat)
- In-person
- Secure email
- **NOT** Discord (defeats the purpose)

### Set Key in NDA
Both users:
1. Click settings for AES-256 Encryptor (TX pipeline)
2. Set `key` parameter: paste the hex string
3. Click settings for AES-256 Decryptor (RX pipeline)
4. Set `key` parameter: paste the same hex string

---

## Step 7: Start NDA & Discord

1. In NDA: Click **"Start Both"**
   - Both TX and RX should show **"🟢 Running"**

2. In Discord:
   - Join a voice channel
   - Test mic: Speak, should see input level
   - Listen to someone else, should hear decrypted audio

3. Verify:
   - Check NDA Dashboard for latency <50ms, CPU <30%
   - Listen for audio quality (should be clear)

---

## Troubleshooting

### "Discord can't hear me"
1. Check TX pipeline shows "Running" status
2. In Discord Voice Settings: Input Device = VB-Cable Output (Windows) or Loopback (Linux/macOS)
3. Test with default microphone first (no encryption) to verify cable setup
4. Increase TX buffer size in Settings if needed

### "I can't hear Discord"
1. Check RX pipeline shows "Running" status
2. In Discord Voice Settings: Output Device = VB-Cable Input (Windows) or Loopback (Linux/macOS)
3. Test with default speaker first (no decryption)
4. Check system volume isn't muted

### "Audio is glitchy or delayed"
1. Check NDA Dashboard: Latency should be <50ms
2. Increase buffer size in NDA Settings
3. Close other audio applications
4. Lower Discord bandwidth settings
5. Check CPU load (should be <30%)

### "Encryption keys don't match"
1. Make sure both users have exact same key
2. No spaces or typos in hex string
3. Use `openssl rand -hex 32` to generate new shared key
4. Share key outside of Discord

### "Plugin won't load"
1. For Python plugins: ensure Python 3.8+ installed
2. Run: `pip install -r plugins_py/requirements.txt`
3. Check plugins_py/ folder has example plugins
4. Check console for specific error

---

## Advanced: Custom Encryption

Want different encryption (AES, ChaCha20, etc.)?

Create custom processor plugin:
```python
from base_plugin import AudioProcessorPlugin
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os

class CustomEncryptor(AudioProcessorPlugin):
    def __init__(self):
        self.key = b'...'  # 32 bytes
        
    def process_audio(self, buffer):
        # Use custom encryption algorithm
        iv = os.urandom(16)
        cipher = Cipher(
            algorithms.AES(self.key),
            modes.CBC(iv),
            backend=default_backend()
        )
        # ... encrypt buffer.data ...
        return True
```

See `development/python-processor-guide.md` for details.

---

## Network Topology

```
┌─────────────────┐                    ┌─────────────────┐
│   User A        │                    │   User B        │
├─────────────────┤                    ├─────────────────┤
│ Mic → Encrypt → │ (encrypted audio)  │ → Decrypt → Speaker
│    VB-Cable     ├────────────────────┤   VB-Cable      │
│ Speaker ← D.... │ (encrypted audio)  │ ← ...Encrypt ← Mic
└─────────────────┘                    └─────────────────┘
        ↓                                      ↓
    Discord ↔─────────────────────────→ Discord
   (encrypted stream only)
```

**Key insight:** Discord never sees unencrypted audio. It only transmits encrypted bytes.

---

## Security Notes

- ✅ **Encryption strong:** AES-256 is military-grade
- ✅ **End-to-end:** Only you and recipient can decrypt
- ✅ **Key exchange:** Do this securely (not via Discord!)
- ⚠️ **Discord metadata:** Timestamps, connection info still visible to Discord
- ⚠️ **Endpoint security:** Your device security is critical

For maximum security:
1. Share keys in person or via Signal
2. Use unique keys for each conversation
3. Change keys periodically
4. Don't share keys via Discord, email, etc.

---

## Works With

Any VoIP application that reads from microphone/speaker:
- ✅ Discord
- ✅ Skype
- ✅ Teams
- ✅ Zoom
- ✅ Mumble
- ✅ Phone softphone apps
- ✅ Custom VoIP software

Just configure the virtual audio cable inputs/outputs in the app.

---

## Next Steps

- **Want encrypted file recording?** See `examples/encrypted-file-recording.md` (if exists)
- **Want to use with AIOC radio?** See `examples/encrypted-aioc-radio.md`
- **Custom encryption?** See `development/plugin-development.md`
- **Troubleshooting?** See `development/troubleshooting.md`

---

**Key Points:**
- ✅ NDA sits between you and VoIP app via virtual cable
- ✅ Both TX and RX must run (full-duplex)
- ✅ Encryption key must match on both devices
- ✅ <50ms latency target
- ✅ Works with any VoIP application

