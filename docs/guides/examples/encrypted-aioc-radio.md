# Step-by-Step: Encrypted AIOC Radio Communication

This guide walks through setting up encrypted two-way radio communication using AIOC hardware.

---

## What You Need

- NDA installed and built (see `getting-started/installation.md`)
- AIOC USB radio hardware
- Microphone and speaker (or headset)
- Two users (sender and receiver) with matching encryption keys

---

## Step 1: Build & Run NDA

```bash
# Build
mkdir build && cd build
cmake .. && make -j

# Run
./NDA
```

The UI window appears with TX and RX pipeline sections.

---

## Step 2: Load Plugins

In the NDA UI:
1. Click **"Load Plugins"** (or **"Auto-Load Python Plugins"**)
2. Wait for plugin discovery to complete
3. You should see available plugins listed in the dropdown menus

**Available plugins you should see:**
- Sources: Microphone, AIOC Input, Sine Wave Generator
- Processors: AES-256 Encryptor, AES-256 Decryptor
- Sinks: Speaker, AIOC Output, WAV File, Null Sink

---

## Step 3: Configure TX Pipeline (Transmit)

This encrypts your microphone for transmission over radio:

1. **Source:** Select "Microphone"
2. **Processor:** Select "AES-256 Encryptor"
3. **Sink:** Select "AIOC Output"
4. Set encryption key (see **Step 5**)

**Configuration:**
```
Device Microphone → AES-256 Encryptor → AIOC USB Output
                                              ↓
                                         Radio transmission
```

---

## Step 4: Configure RX Pipeline (Receive)

This receives encrypted radio signal and decrypts for local speaker:

1. **Source:** Select "AIOC Input"
2. **Processor:** Select "AES-256 Decryptor"
3. **Sink:** Select "Speaker"
4. Set matching encryption key (same as TX!)

**Configuration:**
```
AIOC USB Input → AES-256 Decryptor → Device Speaker
      ↑
  Radio reception
```

---

## Step 5: Share Encryption Key

**Important:** Both TX and RX must use the SAME encryption key!

### Option A: Manual Key Exchange (Secure)

1. Generate a random key on one device:
   ```bash
   # On Linux/Mac:
   openssl rand -hex 32
   # Output: a1b2c3d4e5f6...  (64 hex characters)
   
   # On Windows (PowerShell):
   $bytes = New-Object byte[] 32
   (New-Object System.Security.Cryptography.RNGCryptoServiceProvider).GetBytes($bytes)
   -join ($bytes | ForEach-Object { $_.ToString("x2") })
   ```

2. Share key securely with other user (Signal, in-person, etc.)

3. In NDA (TX side):
   - Click settings for AES-256 Encryptor
   - Set `key` parameter: paste the hex string

4. In NDA (RX side):
   - Click settings for AES-256 Decryptor
   - Set `key` parameter: paste the same hex string

### Option B: Auto-Generated Key (Quick Testing)

Both devices automatically generate the same key (not recommended for production):
- Just start both TX and RX pipelines
- Audio will be encrypted/decrypted with generated key

---

## Step 6: Start Both Pipelines

1. Click **"Start Both"** button
2. Both TX and RX should show **"🟢 Running"**
3. Check Dashboard for:
   - TX: Latency <50ms, CPU <20%
   - RX: Latency <50ms, CPU <20%

---

## Step 7: Test Encrypted Audio

1. **Speak into microphone**
   - TX pipeline reads from your microphone
   - Encryptor encrypts the audio
   - AIOC output sends encrypted stream to radio

2. **Receive from radio**
   - RX pipeline reads encrypted stream from AIOC
   - Decryptor decrypts the audio
   - Speaker outputs decrypted audio locally

3. **On other radio user's device:**
   - Same process in reverse
   - Their encrypted transmission → Your decryption

---

## Troubleshooting

### "No audio coming through"
1. Check AIOC hardware is connected
2. Verify plugins loaded successfully (Dashboard shows plugin names)
3. Check processor is not skipped (Processor dropdown shows AES-256)
4. Try with passthrough (no processor) to test audio path

### "Audio is glitchy / crackling"
1. Check latency in Dashboard (<50ms target)
2. Increase buffer size in Settings
3. Close other applications using audio
4. Check CPU load (should be <30% for dual pipelines)

### "Encryption keys don't match"
1. Use same key on both devices
2. Copy hex string exactly (no spaces)
3. Use `openssl rand -hex 32` for key generation
4. Share key securely (Signal, in-person, etc.)

### "Plugin won't load"
1. Check plugin files exist in `plugins_src/` or `plugins_py/`
2. For C++ plugins: verify build succeeded
3. For Python plugins: check Python is installed and requirements met
4. Check console for specific error messages

---

## Advanced: Custom Encryption

Want to use a different encryption algorithm?

1. Create custom processor plugin:
   - C++: See `development/plugin-development.md`
   - Python: See `development/python-processor-guide.md`

2. Example Python encryptor:
   ```python
   from base_plugin import AudioProcessorPlugin
   from cryptography.fernet import Fernet
   
   class CustomEncryptor(AudioProcessorPlugin):
       def __init__(self):
           self.cipher = Fernet(key)
       
       def process_audio(self, buffer):
           # Encrypt buffer.data
           encrypted = self.cipher.encrypt(buffer.data.tobytes())
           buffer.data = np.frombuffer(encrypted, dtype=np.float32)
           return True
   ```

3. Load custom plugin in NDA and use as Processor

See `development/plugin-development.md` for details.

---

## Next Steps

- **Want different encryption?** See `development/plugin-development.md`
- **Want to record encrypted audio?** Use `examples/encrypted-file-recording.md`
- **Want to use with Discord?** See `examples/discord-voip-encryption.md`
- **Something not working?** See `development/troubleshooting.md`

---

**Key Points:**
- ✅ Both TX and RX must run (full-duplex)
- ✅ Encryption key must match on both devices
- ✅ Audio is encrypted in NDA, not on radio
- ✅ AIOC hardware handles RF transmission
- ✅ <50ms latency target with dual pipelines

