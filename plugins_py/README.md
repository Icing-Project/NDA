# NADE Python Plugins

Python implementation of NADE audio plugins. These plugins provide the same functionality as the C++ versions but are easier to develop and modify.

## Available Plugins

### Audio Sources
- **sine_wave_source.py** - Generates a sine wave tone for testing (default 440Hz A4 note)
- **pulseaudio_microphone.py** - Captures audio from system microphone using PulseAudio

### Audio Sinks
- **null_sink.py** - Discards audio but shows metrics in console for debugging
- **wav_file_sink.py** - Records audio to WAV file (32-bit float PCM)
- **pulseaudio_speaker.py** - Plays audio through system speakers using PulseAudio

## Installation

### Requirements
```bash
pip install numpy pyaudio
```

Note: `pyaudio` is optional and only needed for microphone and speaker plugins.

### Linux (Fedora/RHEL)
```bash
sudo dnf install python3-numpy python3-pyaudio portaudio-devel
```

### Linux (Ubuntu/Debian)
```bash
sudo apt install python3-numpy python3-pyaudio portaudio19-dev
```

## Usage

### Basic Plugin Loading

```python
from plugin_loader import PluginLoader
from base_plugin import AudioBuffer

# Create loader
loader = PluginLoader("plugins_py")

# Load a plugin
plugin = loader.load_plugin("sine_wave_source")

# Initialize and configure
plugin.initialize()
plugin.set_sample_rate(48000)
plugin.set_channels(2)

# Start the plugin
plugin.start()

# Use the plugin
buffer = AudioBuffer(2, 512)
plugin.read_audio(buffer)

# Stop and cleanup
plugin.stop()
loader.unload_all()
```

### Test Script

Run the included test script to verify all plugins work:

```bash
cd plugins_py
python3 test_plugins.py
```

This will:
1. List all available plugins
2. Test sine wave generator with null sink (3 seconds)
3. Test sine wave generator recording to WAV file (2 seconds)

## Plugin Architecture

### Base Classes

- **BasePlugin** - Base interface for all plugins
- **AudioSourcePlugin** - Base for audio input plugins
- **AudioSinkPlugin** - Base for audio output plugins

### Plugin States

1. **Unloaded** - Plugin not initialized
2. **Loaded** - Plugin loaded into memory
3. **Initialized** - Plugin initialized and ready
4. **Running** - Plugin actively processing audio
5. **Error** - Plugin encountered an error

### Creating Custom Plugins

Create a new Python file in `plugins_py/`:

```python
from base_plugin import AudioSourcePlugin, AudioBuffer, PluginInfo, PluginType, PluginState

class MyCustomPlugin(AudioSourcePlugin):
    def __init__(self):
        super().__init__()
        self.sample_rate = 48000
        self.channels = 2

    def initialize(self) -> bool:
        self.state = PluginState.INITIALIZED
        return True

    def shutdown(self):
        self.state = PluginState.UNLOADED

    def start(self) -> bool:
        if self.state != PluginState.INITIALIZED:
            return False
        self.state = PluginState.RUNNING
        return True

    def stop(self):
        if self.state == PluginState.RUNNING:
            self.state = PluginState.INITIALIZED

    def get_info(self) -> PluginInfo:
        return PluginInfo(
            name="My Custom Plugin",
            version="1.0.0",
            author="Your Name",
            description="My custom audio plugin",
            plugin_type=PluginType.AUDIO_SOURCE,
            api_version=1
        )

    def read_audio(self, buffer: AudioBuffer) -> bool:
        # Implement your audio processing here
        return True

    # Implement other required methods...

# Factory function
def create_plugin():
    return MyCustomPlugin()
```

## Advantages over C++

1. **Easier Development** - No compilation required
2. **Rapid Prototyping** - Quick iteration and testing
3. **Rich Ecosystem** - Access to Python libraries (numpy, scipy, etc.)
4. **Cross-Platform** - Works on Linux, Windows, macOS without changes
5. **Dynamic Loading** - Hot-swap plugins without restarting application

## Performance Considerations

While Python plugins are easier to develop, C++ plugins offer better performance for:
- Real-time audio processing with ultra-low latency (<5ms)
- CPU-intensive DSP operations
- Direct hardware access

Use Python plugins for:
- Prototyping and testing
- Non-real-time processing
- Network/file I/O operations
- Complex business logic

## Integration with C++ Application

The C++ NADE application can load Python plugins using:
1. **Python C API** - Embed Python interpreter
2. **pybind11** - Modern C++/Python binding
3. **IPC** - Inter-process communication via sockets

Example using Python C API in C++:

```cpp
#include <Python.h>

class PythonPluginLoader {
    PyObject* loadPlugin(const std::string& path) {
        PyObject* pName = PyUnicode_DecodeFSDefault(path.c_str());
        PyObject* pModule = PyImport_Import(pName);
        PyObject* pFunc = PyObject_GetAttrString(pModule, "create_plugin");
        PyObject* pPlugin = PyObject_CallObject(pFunc, NULL);
        return pPlugin;
    }
};
```

## License

Same as NADE project license.
