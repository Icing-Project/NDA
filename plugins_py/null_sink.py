"""
Null Sink Plugin - Python Implementation
Discards audio but shows metrics in console
"""

import numpy as np
from base_plugin import (
    AudioSinkPlugin, AudioBuffer, PluginInfo,
    PluginType, PluginState
)


class NullSinkPlugin(AudioSinkPlugin):
    """Null sink that discards audio but shows metrics"""

    def __init__(self):
        super().__init__()
        self.sample_rate = 48000
        self.channel_count = 2
        self.buffer_size = 512
        self.frames_processed = 0
        self.show_metrics = True

    def initialize(self) -> bool:
        """Initialize the plugin"""
        self.state = PluginState.INITIALIZED
        self.frames_processed = 0
        return True

    def shutdown(self):
        """Shutdown the plugin"""
        self.state = PluginState.UNLOADED

    def start(self) -> bool:
        """Start consuming audio"""
        if self.state != PluginState.INITIALIZED:
            return False
        self.state = PluginState.RUNNING
        self.frames_processed = 0
        print("[NullSink] Started - consuming audio data")
        return True

    def stop(self):
        """Stop consuming audio"""
        if self.state == PluginState.RUNNING:
            seconds = self.frames_processed / self.sample_rate
            print(f"[NullSink] Stopped - processed {self.frames_processed} frames ({seconds:.2f} seconds)")
            self.state = PluginState.INITIALIZED

    def get_info(self) -> PluginInfo:
        """Get plugin information"""
        return PluginInfo(
            name="Null Sink (Console Monitor)",
            version="1.0.0",
            author="Icing Project",
            description="Discards audio but shows metrics in console",
            plugin_type=PluginType.AUDIO_SINK,
            api_version=1
        )

    def set_parameter(self, key: str, value: str):
        """Set plugin parameter"""
        if key == "showMetrics":
            self.show_metrics = value.lower() in ("true", "1", "yes")

    def get_parameter(self, key: str) -> str:
        """Get plugin parameter"""
        if key == "sampleRate":
            return str(self.sample_rate)
        elif key == "channels":
            return str(self.channel_count)
        elif key == "bufferSize":
            return str(self.buffer_size)
        elif key == "showMetrics":
            return "true" if self.show_metrics else "false"
        return ""

    def write_audio(self, buffer: AudioBuffer) -> bool:
        """Write audio data (discard but calculate metrics)"""
        if self.state != PluginState.RUNNING:
            return False

        frame_count = buffer.get_frame_count()
        self.frames_processed += frame_count

        # Calculate RMS level for monitoring (every 0.1 seconds)
        if self.show_metrics and self.frames_processed % (self.sample_rate // 10) == 0:
            # Calculate RMS per channel, handle mono safely
            left_channel = buffer.get_channel_data(0)
            rms_l = np.sqrt(np.mean(left_channel ** 2))

            if buffer.get_channel_count() > 1:
                right_channel = buffer.get_channel_data(1)
                rms_r = np.sqrt(np.mean(right_channel ** 2))
            else:
                rms_r = rms_l

            # Convert to dB
            db_l = 20.0 * np.log10(rms_l + 1e-10)
            db_r = 20.0 * np.log10(rms_r + 1e-10)

            seconds = self.frames_processed / self.sample_rate
            print(f"[NullSink] L: {db_l:6.1f} dB  R: {db_r:6.1f} dB  ({seconds:.0f}s)")

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
        """Get buffer size"""
        return self.buffer_size

    def set_buffer_size(self, samples: int):
        """Set buffer size"""
        self.buffer_size = samples

    def get_available_space(self) -> int:
        """Get available space (always has space)"""
        return self.buffer_size


# Plugin factory function
def create_plugin():
    """Factory function to create plugin instance"""
    return NullSinkPlugin()
