"""
Sine Wave Generator Plugin - Python Implementation
Generates a sine wave tone for testing
"""

import math
import numpy as np
from base_plugin import (
    AudioSourcePlugin, AudioBuffer, PluginInfo,
    PluginType, PluginState, AudioSourceCallback
)


class SineWaveSourcePlugin(AudioSourcePlugin):
    """Generates a sine wave (default 440Hz A4 note)"""

    def __init__(self):
        super().__init__()
        self.sample_rate = 48000
        self.channel_count = 2
        self.buffer_size = 512
        self.frequency = 440.0  # A4 note
        self.phase = 0.0
        self.callback = None

        self._offsets = None
        self._phase_work = None
        self._samples = None

    def _ensure_work_buffers(self, frames: int) -> None:
        frames = int(frames)
        if frames <= 0:
            self._offsets = None
            self._phase_work = None
            self._samples = None
            return
        if self._offsets is not None and int(self._offsets.shape[0]) == frames:
            return

        self._offsets = np.arange(frames, dtype=np.float32)
        self._phase_work = np.empty(frames, dtype=np.float32)
        self._samples = np.empty(frames, dtype=np.float32)

    def initialize(self) -> bool:
        """Initialize the plugin"""
        self.state = PluginState.INITIALIZED
        return True

    def shutdown(self):
        """Shutdown the plugin"""
        self.state = PluginState.UNLOADED

    def start(self) -> bool:
        """Start generating audio"""
        if self.state != PluginState.INITIALIZED:
            return False
        self.state = PluginState.RUNNING
        self.phase = 0.0
        return True

    def stop(self):
        """Stop generating audio"""
        if self.state == PluginState.RUNNING:
            self.state = PluginState.INITIALIZED

    def get_info(self) -> PluginInfo:
        """Get plugin information"""
        return PluginInfo(
            name="Sine Wave Generator",
            version="1.0.0",
            author="Icing Project",
            description="Generates a 440Hz sine wave (A4 note) for testing",
            plugin_type=PluginType.AUDIO_SOURCE,
            api_version=1
        )

    def set_parameter(self, key: str, value: str):
        """Set plugin parameter"""
        if key == "frequency":
            self.frequency = float(value)

    def get_parameter(self, key: str) -> str:
        """Get plugin parameter"""
        if key == "frequency":
            return str(self.frequency)
        elif key == "sampleRate":
            return str(self.sample_rate)
        elif key == "channels":
            return str(self.channel_count)
        return ""

    def set_audio_callback(self, callback: AudioSourceCallback):
        """Set audio callback"""
        self.callback = callback

    def read_audio(self, buffer: AudioBuffer) -> bool:
        """Read audio data - generates sine wave"""
        if self.state != PluginState.RUNNING:
            buffer.clear()
            return False

        frame_count = int(buffer.get_frame_count())
        if frame_count <= 0:
            return True

        self._ensure_work_buffers(frame_count)

        phase_increment = (math.tau * float(self.frequency)) / float(self.sample_rate)
        phase0 = float(self.phase)

        phase_increment_f32 = np.float32(phase_increment)
        phase0_f32 = np.float32(phase0)

        np.multiply(self._offsets, phase_increment_f32, out=self._phase_work)
        self._phase_work += phase0_f32
        np.sin(self._phase_work, out=self._samples)
        self._samples *= np.float32(0.5)

        channels = int(buffer.get_channel_count())
        buffer.data[:channels, :frame_count] = self._samples

        # Advance phase and wrap robustly to keep arguments bounded (performance + accuracy).
        self.phase = (phase0 + (frame_count * phase_increment)) % math.tau

        return True

    def get_sample_rate(self) -> int:
        """Get sample rate"""
        return self.sample_rate

    def get_channel_count(self) -> int:
        """Get number of channels"""
        return self.channel_count

    def set_sample_rate(self, sample_rate: int):
        """Set sample rate"""
        if self.state in (PluginState.UNLOADED, PluginState.INITIALIZED):
            self.sample_rate = sample_rate

    def set_channel_count(self, channels: int):
        """Set number of channels"""
        if self.state in (PluginState.UNLOADED, PluginState.INITIALIZED):
            self.channel_count = channels

    def get_buffer_size(self) -> int:
        return self.buffer_size

    def set_buffer_size(self, samples: int):
        if self.state in (PluginState.UNLOADED, PluginState.INITIALIZED):
            self.buffer_size = max(64, int(samples))
            self._ensure_work_buffers(self.buffer_size)


# Plugin factory function
def create_plugin():
    """Factory function to create plugin instance"""
    return SineWaveSourcePlugin()
