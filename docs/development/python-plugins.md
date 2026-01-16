# Python Audio Processor Plugin Development Guide

## Overview

In NDA v2.0, **processors are first-class plugins** that transform audio in-place. Python processor plugins have **equal status** to C++ processors and can implement any audio transformation: encryption, decryption, effects, filters, etc.

---

## What is a Processor?

A **processor** sits between the source and sink in the pipeline:

```
Source → [PROCESSOR] → Sink
         ↑
    Transforms audio
    (encryption, effects, etc.)
```

**Key characteristics:**
- Receives audio buffer from source
- Modifies buffer **in-place**
- Returns modified buffer to continue pipeline
- Optional (can be None for direct passthrough)

---

## AudioProcessorPlugin Interface

### Required Methods

```python
from base_plugin import AudioProcessorPlugin, PluginInfo, PluginType, PluginState

class MyProcessorPlugin(AudioProcessorPlugin):
    
    def initialize(self) -> bool:
        """Initialize the processor (load resources, setup state)"""
        pass
    
    def start(self) -> bool:
        """Start processing (called when pipeline starts)"""
        pass
    
    def stop(self):
        """Stop processing (called when pipeline stops)"""
        pass
    
    def shutdown(self):
        """Cleanup resources (called before unload)"""
        pass
    
    def process_audio(self, buffer: AudioBuffer) -> bool:
        """
        Process audio buffer IN-PLACE
        
        Args:
            buffer: AudioBuffer with shape (channels, frames)
                   Buffer data is NumPy array of float32
        
        Returns:
            True if processing succeeded, False on error
        """
        pass
    
    def get_info(self) -> PluginInfo:
        """Return plugin metadata"""
        pass
    
    def get_state(self) -> PluginState:
        """Return current plugin state"""
        pass
    
    def get_sample_rate(self) -> int:
        """Return current sample rate"""
        pass
    
    def get_channel_count(self) -> int:
        """Return current channel count"""
        pass
    
    def set_sample_rate(self, rate: int):
        """Set sample rate (called during initialization)"""
        pass
    
    def set_channel_count(self, channels: int):
        """Set channel count (called during initialization)"""
        pass
    
    def set_parameter(self, key: str, value: str) -> bool:
        """Set plugin parameter (optional)"""
        pass
    
    def get_parameter(self, key: str) -> str:
        """Get plugin parameter (optional)"""
        pass
```

### Factory Function

**Every processor plugin must export a factory:**

```python
def create_plugin():
    return MyProcessorPlugin()
```

---

## Example 1: Simple Gain Processor

**File:** `plugins_py/examples/simple_gain.py`

```python
import numpy as np
from base_plugin import AudioProcessorPlugin, PluginInfo, PluginType, PluginState

class SimpleGainPlugin(AudioProcessorPlugin):
    def __init__(self):
        self.state = PluginState.UNLOADED
        self.gain = 1.0  # Unity gain
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

    def process_audio(self, buffer):
        if self.state != PluginState.RUNNING:
            return False

        # Apply gain (multiply all samples)
        buffer.data *= self.gain
        
        # Prevent clipping
        np.clip(buffer.data, -1.0, 1.0, out=buffer.data)
        
        return True

    def get_info(self):
        return PluginInfo(
            name="Simple Gain",
            version="1.0.0",
            author="NDA Team",
            description="Volume adjustment",
            plugin_type=PluginType.PROCESSOR,
            api_version=1
        )

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
        if key == "gain":
            self.gain = float(value)
            return True
        return False

    def get_parameter(self, key):
        if key == "gain":
            return str(self.gain)
        return ""

def create_plugin():
    return SimpleGainPlugin()
```

**Usage:**
```python
# In pipeline configuration
processor = plugin_manager.load_plugin("plugins_py/examples/simple_gain.py")
processor.set_parameter("gain", "0.5")  # 50% volume
pipeline.set_processor(processor)
```

---

## Example 2: Audio Encryptor

**File:** `plugins_py/examples/simple_xor_encryptor.py`

```python
import numpy as np
from base_plugin import AudioProcessorPlugin, PluginInfo, PluginType, PluginState

class XOREncryptorPlugin(AudioProcessorPlugin):
    """Simple XOR encryption for demonstration (NOT secure for production!)"""
    
    def __init__(self):
        self.state = PluginState.UNLOADED
        self.key = 0x5A  # Single-byte XOR key
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

    def process_audio(self, buffer):
        if self.state != PluginState.RUNNING:
            return False

        # Convert float32 to bytes
        audio_bytes = buffer.data.tobytes()
        byte_array = np.frombuffer(audio_bytes, dtype=np.uint8)
        
        # XOR encrypt
        encrypted = byte_array ^ self.key
        
        # Convert back to float32
        encrypted_float = np.frombuffer(encrypted.tobytes(), dtype=np.float32)
        encrypted_float = encrypted_float.reshape(buffer.data.shape)
        
        buffer.data[:] = encrypted_float
        return True

    def get_info(self):
        return PluginInfo(
            name="XOR Encryptor",
            version="1.0.0",
            author="NDA Team",
            description="Simple XOR encryption (demo only)",
            plugin_type=PluginType.PROCESSOR,
            api_version=1
        )

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
        if key == "key":
            self.key = int(value) & 0xFF
            return True
        return False

    def get_parameter(self, key):
        if key == "key":
            return str(self.key)
        return ""

def create_plugin():
    return XOREncryptorPlugin()
```

---

## Example 3: High-Pass Filter

**File:** `plugins_py/examples/highpass_filter.py`

```python
import numpy as np
from scipy import signal
from base_plugin import AudioProcessorPlugin, PluginInfo, PluginType, PluginState

class HighPassFilterPlugin(AudioProcessorPlugin):
    """Simple high-pass filter using scipy"""
    
    def __init__(self):
        self.state = PluginState.UNLOADED
        self.cutoff_freq = 100.0  # Hz
        self.sample_rate = 48000
        self.channels = 2
        self.filter_state = None

    def initialize(self):
        self._update_filter()
        self.state = PluginState.INITIALIZED
        return True

    def start(self):
        # Reset filter state
        self.filter_state = [None] * self.channels
        self.state = PluginState.RUNNING
        return True

    def stop(self):
        self.state = PluginState.INITIALIZED

    def shutdown(self):
        self.state = PluginState.UNLOADED

    def _update_filter(self):
        """Design Butterworth high-pass filter"""
        nyquist = self.sample_rate / 2.0
        normalized_cutoff = self.cutoff_freq / nyquist
        self.b, self.a = signal.butter(4, normalized_cutoff, btype='high')

    def process_audio(self, buffer):
        if self.state != PluginState.RUNNING:
            return False

        # Process each channel
        for ch in range(buffer.data.shape[0]):
            if self.filter_state[ch] is None:
                # Initialize filter state
                filtered, self.filter_state[ch] = signal.lfilter(
                    self.b, self.a, buffer.data[ch], zi=signal.lfilter_zi(self.b, self.a) * buffer.data[ch][0]
                )
            else:
                # Continue with previous state
                filtered, self.filter_state[ch] = signal.lfilter(
                    self.b, self.a, buffer.data[ch], zi=self.filter_state[ch]
                )
            
            buffer.data[ch] = filtered

        return True

    def get_info(self):
        return PluginInfo(
            name="High-Pass Filter",
            version="1.0.0",
            author="NDA Team",
            description="Butterworth high-pass filter",
            plugin_type=PluginType.PROCESSOR,
            api_version=1
        )

    def get_state(self):
        return self.state

    def get_sample_rate(self):
        return self.sample_rate

    def get_channel_count(self):
        return self.channels

    def set_sample_rate(self, rate):
        if self.sample_rate != rate:
            self.sample_rate = rate
            self._update_filter()

    def set_channel_count(self, channels):
        self.channels = channels

    def set_parameter(self, key, value):
        if key == "cutoff":
            self.cutoff_freq = float(value)
            self._update_filter()
            return True
        return False

    def get_parameter(self, key):
        if key == "cutoff":
            return str(self.cutoff_freq)
        return ""

def create_plugin():
    return HighPassFilterPlugin()
```

---

## Buffer Format

### AudioBuffer Structure

```python
buffer.data         # NumPy array, dtype=float32
buffer.data.shape   # (channels, frames)

# Example: 2 channels, 512 frames
buffer.data.shape == (2, 512)

# Access channels
left_channel = buffer.data[0]   # First channel
right_channel = buffer.data[1]  # Second channel

# Access individual samples
sample = buffer.data[ch][frame]

# Modify in-place
buffer.data *= 0.5              # Reduce volume by 50%
buffer.data[0] += buffer.data[1] # Mix channels
```

### Sample Range

- **Audio samples are float32 in range [-1.0, 1.0]**
- Values outside this range will clip when output to hardware
- Always clip after processing:
  ```python
  np.clip(buffer.data, -1.0, 1.0, out=buffer.data)
  ```

---

## Performance Tips

### 1. Use NumPy Vectorization

**BAD (slow):**
```python
for ch in range(channels):
    for frame in range(frames):
        buffer.data[ch][frame] *= gain  # Slow!
```

**GOOD (fast):**
```python
buffer.data *= gain  # NumPy vectorized
```

### 2. Modify In-Place

**BAD (allocates new array):**
```python
buffer.data = buffer.data * gain  # New allocation
```

**GOOD (in-place, no allocation):**
```python
buffer.data *= gain  # In-place operation
```

### 3. Avoid Repeated Allocations

**BAD:**
```python
def process_audio(self, buffer):
    temp = np.zeros_like(buffer.data)  # New allocation every call!
    # ... use temp
```

**GOOD:**
```python
def initialize(self):
    self.temp_buffer = None  # Allocate once

def process_audio(self, buffer):
    if self.temp_buffer is None:
        self.temp_buffer = np.zeros_like(buffer.data)
    # ... reuse temp_buffer
```

### 4. Profile Your Code

```python
import time

def process_audio(self, buffer):
    start = time.perf_counter()
    
    # ... processing
    
    elapsed = (time.perf_counter() - start) * 1000
    if elapsed > 10:  # Warn if > 10ms
        print(f"[Warning] Processing took {elapsed:.2f}ms")
    
    return True
```

---

## Testing Your Processor

### Standalone Test

```python
# test_my_processor.py
from plugins_py.examples.my_processor import create_plugin
from base_plugin import AudioBuffer
import numpy as np

# Create plugin
processor = create_plugin()
processor.initialize()
processor.start()

# Create test buffer
buffer = AudioBuffer(channels=2, frames=512)
buffer.data[:] = np.random.randn(2, 512) * 0.1  # Random noise

# Process
success = processor.process_audio(buffer)
print(f"Processing success: {success}")
print(f"Output range: [{buffer.data.min():.3f}, {buffer.data.max():.3f}]")

processor.stop()
processor.shutdown()
```

### Integration Test

```bash
# Test your plugin from command line
cd plugins_py
python test_plugins.py

# Or use in full pipeline via the NDA GUI:
# Configure TX pipeline:
#   Source: Sine Wave
#   Processor: My Processor
#   Sink: Null Sink
# Start and check logs
```

---

## Common Patterns

### Pattern 1: Stateless Processing

```python
def process_audio(self, buffer):
    # No internal state, pure function
    buffer.data *= self.gain
    return True
```

### Pattern 2: Stateful Processing (Filter)

```python
def start(self):
    self.filter_state = initialize_state()
    return True

def process_audio(self, buffer):
    buffer.data, self.filter_state = apply_filter(buffer.data, self.filter_state)
    return True
```

### Pattern 3: Per-Channel Processing

```python
def process_audio(self, buffer):
    for ch in range(buffer.data.shape[0]):
        buffer.data[ch] = process_channel(buffer.data[ch])
    return True
```

### Pattern 4: Encryption/Decryption

```python
def process_audio(self, buffer):
    # Convert to bytes
    audio_bytes = buffer.data.tobytes()
    
    # Encrypt
    encrypted_bytes = self.cipher.encrypt(audio_bytes)
    
    # Convert back (handle size changes)
    encrypted_array = np.frombuffer(encrypted_bytes[:len(audio_bytes)], dtype=np.float32)
    buffer.data[:] = encrypted_array.reshape(buffer.data.shape)
    
    return True
```

---

## Deployment

### 1. Install Dependencies

```bash
cd plugins_py
pip install -r requirements.txt

# Add your processor's dependencies
echo "scipy" >> requirements.txt
echo "cryptography" >> requirements.txt
```

### 2. Place in Plugin Directory

```
plugins_py/
├── examples/
│   ├── my_processor.py          # Your plugin
│   ├── simple_gain.py
│   └── fernet_encryptor.py
├── base_plugin.py
└── requirements.txt
```

### 3. Load in NDA

```python
# Via UI: Load Plugins → select plugins_py/examples/my_processor.py
# Via code:
plugin_manager.load_plugin("plugins_py/examples/my_processor.py")
```

---

## Best Practices

### ✅ DO

- **Validate state** before processing
- **Clip output** to [-1.0, 1.0]
- **Use NumPy vectorization** for performance
- **Document parameters** in docstrings
- **Test standalone** before integration
- **Profile** to ensure <10ms processing time
- **Handle errors gracefully** (return False, don't crash)

### ❌ DON'T

- **Don't allocate** large arrays every call
- **Don't use Python loops** for sample-by-sample processing
- **Don't assume** buffer size (it may change)
- **Don't crash** on invalid input (validate and return False)
- **Don't block** (no sleep, no waiting for I/O)
- **Don't modify** buffer shape (only values)

---

## Troubleshooting

### "Plugin failed to load"
- Check `create_plugin()` function exists
- Verify imports (base_plugin, numpy, etc.)
- Run standalone to catch import errors

### "Processing returns False"
- Check plugin state is RUNNING
- Validate buffer shape matches expectations
- Add debug prints to identify failure point

### "Audio is distorted"
- Check for clipping (values > 1.0 or < -1.0)
- Verify in-place modifications don't corrupt data
- Test with known input (sine wave)

### "Performance is poor"
- Profile with `time.perf_counter()`
- Replace loops with NumPy operations
- Avoid repeated allocations

---

## Further Reading

- **NDA-SPECS-v2.md** — Full v2.0 specification
- **base_plugin.py** — Python plugin interface definitions
- **QUICKSTART.md** — Getting started with Python plugins
- **V2_IMPLEMENTATION_PLAN.md** — Development roadmap

---

*Happy processing! 🎵*


