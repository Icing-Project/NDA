"""
NDA Plugin Base Classes for Python
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Callable
import numpy as np


class PluginType(Enum):
    """Plugin type enumeration"""
    AUDIO_SOURCE = "AudioSource"
    AUDIO_SINK = "AudioSink"
    BEARER = "Bearer"
    ENCRYPTOR = "Encryptor"
    PROCESSOR = "Processor"


class PluginState(Enum):
    """Plugin state enumeration"""
    UNLOADED = "Unloaded"
    LOADED = "Loaded"
    INITIALIZED = "Initialized"
    RUNNING = "Running"
    ERROR = "Error"


class PluginInfo:
    """Plugin information structure"""
    def __init__(self, name: str, version: str, author: str,
                 description: str, plugin_type: PluginType, api_version: int = 1):
        self.name = name
        self.version = version
        self.author = author
        self.description = description
        self.type = plugin_type
        self.api_version = api_version


class AudioBuffer:
    """Audio buffer wrapper for multi-channel audio data"""
    def __init__(self, channels: int, frame_count: int):
        self.data = np.zeros((channels, frame_count), dtype=np.float32)

    def get_channel_data(self, channel: int) -> np.ndarray:
        """Get data for specific channel"""
        return self.data[channel]

    def get_frame_count(self) -> int:
        """Get number of frames"""
        return self.data.shape[1]

    def get_channel_count(self) -> int:
        """Get number of channels"""
        return self.data.shape[0]

    def clear(self):
        """Clear buffer to zeros"""
        self.data.fill(0)

    def copy_from(self, other: 'AudioBuffer'):
        """Copy data from another buffer"""
        np.copyto(self.data, other.data)


class BasePlugin(ABC):
    """Base plugin interface"""

    def __init__(self):
        self.state = PluginState.UNLOADED

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize plugin"""
        pass

    @abstractmethod
    def shutdown(self):
        """Shutdown plugin"""
        pass

    @abstractmethod
    def start(self) -> bool:
        """Start plugin"""
        pass

    @abstractmethod
    def stop(self):
        """Stop plugin"""
        pass

    @abstractmethod
    def get_info(self) -> PluginInfo:
        """Get plugin information"""
        pass

    @abstractmethod
    def get_type(self) -> PluginType:
        """Get plugin type"""
        pass

    @abstractmethod
    def set_parameter(self, key: str, value: str):
        """Set plugin parameter"""
        pass

    @abstractmethod
    def get_parameter(self, key: str) -> str:
        """Get plugin parameter"""
        pass

    def get_state(self) -> PluginState:
        """Get current plugin state"""
        return self.state


# Type alias for audio source callback
AudioSourceCallback = Callable[[AudioBuffer], None]


class AudioSourcePlugin(BasePlugin):
    """Base class for audio source plugins"""

    def get_type(self) -> PluginType:
        return PluginType.AUDIO_SOURCE

    @abstractmethod
    def set_audio_callback(self, callback: AudioSourceCallback):
        """Set audio callback for push model"""
        pass

    @abstractmethod
    def read_audio(self, buffer: AudioBuffer) -> bool:
        """Read audio data (pull model)"""
        pass

    @abstractmethod
    def get_sample_rate(self) -> int:
        """Get sample rate"""
        pass

    @abstractmethod
    def get_channels(self) -> int:
        """Get number of channels"""
        pass

    @abstractmethod
    def set_sample_rate(self, sample_rate: int):
        """Set sample rate"""
        pass

    @abstractmethod
    def set_channels(self, channels: int):
        """Set number of channels"""
        pass


class AudioSinkPlugin(BasePlugin):
    """Base class for audio sink plugins"""

    def get_type(self) -> PluginType:
        return PluginType.AUDIO_SINK

    @abstractmethod
    def write_audio(self, buffer: AudioBuffer) -> bool:
        """Write audio data"""
        pass

    @abstractmethod
    def get_sample_rate(self) -> int:
        """Get sample rate"""
        pass

    @abstractmethod
    def get_channels(self) -> int:
        """Get number of channels"""
        pass

    @abstractmethod
    def set_sample_rate(self, sample_rate: int):
        """Set sample rate"""
        pass

    @abstractmethod
    def set_channels(self, channels: int):
        """Set number of channels"""
        pass

    @abstractmethod
    def get_buffer_size(self) -> int:
        """Get buffer size"""
        pass

    @abstractmethod
    def set_buffer_size(self, samples: int):
        """Set buffer size"""
        pass

    @abstractmethod
    def get_available_space(self) -> int:
        """Get available buffer space"""
        pass
