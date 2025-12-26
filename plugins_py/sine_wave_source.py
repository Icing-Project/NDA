"""
Sine Wave Generator Plugin - Python Implementation
Generates a sine wave tone for testing
"""

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
        self.frequency = 440.0  # A4 note
        self.phase = 0.0
        self.callback = None

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

        frame_count = buffer.get_frame_count()
        phase_increment = 2.0 * np.pi * self.frequency / self.sample_rate

        # Generate phase values for all frames
        phases = self.phase + np.arange(frame_count) * phase_increment

        # Generate sine wave samples
        samples = 0.5 * np.sin(phases).astype(np.float32)

        # Write to all channels
        for ch in range(self.channel_count):
            buffer.data[ch] = samples

        # Update phase for next call
        self.phase = phases[-1] + phase_increment
        if self.phase >= 2.0 * np.pi:
            self.phase -= 2.0 * np.pi

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


# Plugin factory function
def create_plugin():
    """Factory function to create plugin instance"""
    return SineWaveSourcePlugin()
