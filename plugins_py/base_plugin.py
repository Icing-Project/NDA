"""
NDA Plugin Base Classes for Python

IMPORTANT BUFFER SIZE REQUIREMENT (v2.1):
==========================================
All audio source plugins MUST use 512-sample buffers to match the pipeline's
internal buffer size. This ensures:
- Zero-copy performance in the Python bridge
- Elimination of buffer size mismatches that cause queue starvation
- Consistent latency across all plugins

When implementing AudioSourcePlugin:
1. Set self.buffer_size = 512 in __init__()
2. Configure your audio library to use 512-sample blocks
3. If your library produces different sizes, rebuffer to 512 samples

When implementing AudioSinkPlugin:
1. Accept any buffer size the pipeline provides (typically 512)
2. Internally rebuffer if your audio library requires different sizes

For optimal performance:
- Pre-fill your queue with 2-3 blocks before returning True from start()
- Use reasonable timeouts (50ms recommended, not 200ms+)
- Handle queue.Empty gracefully by returning False, not blocking indefinitely
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Callable
import numpy as np


class PluginType(Enum):
    """Plugin type enumeration"""
    AUDIO_SOURCE = "AudioSource"
    AUDIO_SINK = "AudioSink"
    PROCESSOR = "Processor"  # Handles encryption, effects, resampling, etc.


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
    def get_channel_count(self) -> int:
        """Get number of channels"""
        pass

    # Compatibility aliases: the C++ Python bridge calls get_channels/set_channels.
    def get_channels(self) -> int:
        return self.get_channel_count()

    @abstractmethod
    def set_sample_rate(self, sample_rate: int):
        """Set sample rate"""
        pass

    @abstractmethod
    def set_channel_count(self, channels: int):
        """Set number of channels"""
        pass

    def set_channels(self, channels: int):
        self.set_channel_count(channels)

    # Optional buffer sizing (frames per buffer). Sources can override if they
    # support configurable frame sizes.
    def get_buffer_size(self) -> int:
        return 512

    def set_buffer_size(self, samples: int):
        _ = samples


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
    def get_channel_count(self) -> int:
        """Get number of channels"""
        pass

    # Compatibility aliases: the C++ Python bridge calls get_channels/set_channels.
    def get_channels(self) -> int:
        return self.get_channel_count()

    @abstractmethod
    def set_sample_rate(self, sample_rate: int):
        """Set sample rate"""
        pass

    @abstractmethod
    def set_channel_count(self, channels: int):
        """Set number of channels"""
        pass

    def set_channels(self, channels: int):
        self.set_channel_count(channels)

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


class AudioProcessorPlugin(BasePlugin):
    """
    Base class for audio processor plugins.
    
    Processes audio in-place (encryption, decryption, effects, resampling, etc.)
    Python processors have equal status to C++ processors in v2.0.
    
    Example:
        class GainProcessor(AudioProcessorPlugin):
            def process_audio(self, buffer):
                buffer.data *= 0.5  # Reduce volume by half
                return True
    """

    def get_type(self) -> PluginType:
        return PluginType.PROCESSOR

    @abstractmethod
    def process_audio(self, buffer: AudioBuffer) -> bool:
        """
        Process audio buffer in-place.
        
        Args:
            buffer: AudioBuffer with .data (numpy array, shape [channels, frames])
                    Modify buffer.data in-place to process audio.
        
        Returns:
            True on success, False on error (pipeline will passthrough on failure)
        
        Note:
            - Buffer is guaranteed to be at 48kHz sample rate (pipeline handles resampling)
            - Errors should be logged but handled gracefully
            - Return False to skip processing for this frame
        """
        pass
    
    @abstractmethod
    def get_sample_rate(self) -> int:
        """
        Get current sample rate.
        
        Returns:
            Sample rate in Hz (typically 48000)
        """
        pass
    
    @abstractmethod
    def get_channel_count(self) -> int:
        """
        Get current channel count.
        
        Returns:
            Number of channels (typically 2 for stereo)
        """
        pass

    # Compatibility aliases: the C++ Python bridge calls get_channels/set_channels.
    def get_channels(self) -> int:
        return self.get_channel_count()
    
    @abstractmethod
    def set_sample_rate(self, rate: int) -> None:
        """
        Set sample rate (called by pipeline during initialization).
        
        Args:
            rate: Sample rate in Hz
        """
        pass
    
    @abstractmethod
    def set_channel_count(self, channels: int) -> None:
        """
        Set channel count (called by pipeline during initialization).
        
        Args:
            channels: Number of channels
        """
        pass

    def set_channels(self, channels: int) -> None:
        self.set_channel_count(channels)
    
    def get_processing_latency(self) -> float:
        """
        Get processing latency added by this processor.
        
        Returns:
            Latency in seconds (default: 0.0 for zero-latency processors)
        
        Note:
            This is algorithmic latency (e.g., lookahead buffers), not computational time.
            Used for latency reporting in dashboard.
        """
        return 0.0
