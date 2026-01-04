# Quick Start Guide - NDA Python Plugins

Get started with NDA Python plugins in 5 minutes!

## 1. Install Dependencies

### Windows
```cmd
cd plugins_py
setup_windows.bat
```

### Linux
```bash
cd plugins_py
./setup_linux.sh
```

Or manually:
```bash
pip install -r requirements.txt
```

## 2. Test Plugins

```bash
python test_plugins.py
```

You should see:
- List of available plugins
- Sine wave generator test (~2 seconds)
- WAV file recording test (~1 second)
- A new WAV file: `test_recording.wav`

## 3. Use from Python

```python
from plugin_loader import PluginLoader
from base_plugin import AudioBuffer

# Load plugin
loader = PluginLoader(".")
plugin = loader.load_plugin("sine_wave_source")

# Initialize and start
plugin.initialize()
plugin.start()

# Process audio
buffer = AudioBuffer(2, 512)
plugin.read_audio(buffer)

# Stop
plugin.stop()
loader.unload_all()
```

## 4. Use from C++ Application

### Build NDA with Python Support

```bash
# Linux
mkdir build && cd build
cmake .. -DNDA_ENABLE_PYTHON=ON
make

# Windows
mkdir build && cd build
cmake .. -DNDA_ENABLE_PYTHON=ON -G "Visual Studio 17 2022"
cmake --build . --config Release
```

### Load Python Plugin from C++

```cpp
#include "plugins/PythonPluginBridge.h"

// Create plugin bridge
auto* plugin = PythonPluginFactory::createPlugin("sine_wave_source");

// Use plugin
plugin->initialize();
plugin->start();

AudioBuffer buffer(2, 512);
plugin->readAudio(buffer);

// Cleanup
plugin->stop();
delete plugin;
```

## 5. Available Plugins

| Plugin | Type | Description |
|--------|------|-------------|
| `sine_wave_source` | Source | 440Hz test tone |
| `null_sink` | Sink | Console monitor |
| `wav_file_sink` | Sink | WAV file recorder |
| `pulseaudio_microphone` | Source | Mic input |
| `pulseaudio_speaker` | Sink | Speaker output |
| `sounddevice_microphone` | Source | Mic input via sounddevice (PortAudio) |
| `sounddevice_speaker` | Sink | Speaker output via sounddevice (PortAudio) |
| `soundcard_microphone` | Source | Mic input via soundcard (WASAPI/PulseAudio) |
| `soundcard_speaker` | Sink | Speaker output via soundcard (WASAPI/PulseAudio) |

## 6. Create Your Own Plugin

```python
# my_plugin.py
from base_plugin import AudioSourcePlugin, AudioBuffer, PluginInfo, PluginType, PluginState
import numpy as np

class MyPlugin(AudioSourcePlugin):
    def __init__(self):
        super().__init__()
        self.sample_rate = 48000
        self.channels = 2

    def initialize(self) -> bool:
        self.state = PluginState.INITIALIZED
        return True

    def start(self) -> bool:
        self.state = PluginState.RUNNING
        return True

    def stop(self):
        self.state = PluginState.INITIALIZED

    def shutdown(self):
        self.state = PluginState.UNLOADED

    def get_info(self) -> PluginInfo:
        return PluginInfo(
            name="My Plugin",
            version="1.0.0",
            author="Me",
            description="My custom plugin",
            plugin_type=PluginType.AUDIO_SOURCE,
            api_version=1
        )

    def read_audio(self, buffer: AudioBuffer) -> bool:
        # Generate audio here
        buffer.data.fill(0)  # silence
        return True

    def set_parameter(self, key: str, value: str):
        pass

    def get_parameter(self, key: str) -> str:
        return ""

    def set_audio_callback(self, callback):
        self.callback = callback

    def get_sample_rate(self) -> int:
        return self.sample_rate

    def get_channel_count(self) -> int:
        return self.channels

    def set_sample_rate(self, sr: int):
        self.sample_rate = sr

    def set_channel_count(self, ch: int):
        self.channels = ch

def create_plugin():
    return MyPlugin()
```

Save as `my_plugin.py` in `plugins_py/` directory, then:

```python
plugin = loader.load_plugin("my_plugin")
```

## Troubleshooting

### "Module not found: numpy"
```bash
pip install numpy
```

### "Module not found: pyaudio" (Windows)
```cmd
pip install pipwin
pipwin install pyaudio
```

### "Module not found: pyaudio" (Linux)
```bash
# Fedora
sudo dnf install python3-pyaudio

# Ubuntu
sudo apt install python3-pyaudio
```

### "Module not found: soundcard"
```bash
pip install soundcard
```

Linux note: soundcard needs PulseAudio, or PipeWire with `pipewire-pulse` enabled.

### Plugins not loading from C++
- Make sure `plugins_py` directory is in your working directory
- Check that Python is in PATH
- Verify CMake found Python: look for "Python support enabled" in build output

## Next Steps

- Read [PYTHON_PLUGINS.md](../PYTHON_PLUGINS.md) for detailed documentation
- Check [examples/python_plugin_example.cpp](../examples/python_plugin_example.cpp)
- Explore the plugin source code
- Create your own custom plugins!
