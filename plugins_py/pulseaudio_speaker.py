"""
PulseAudio Speaker Plugin - Python Implementation
Plays audio through system speakers using PulseAudio
"""

import sys

PLATFORM_SUPPORTED = sys.platform.startswith("linux")

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

import numpy as np
from base_plugin import (
    AudioSinkPlugin, AudioBuffer, PluginInfo,
    PluginType, PluginState
)


class PulseAudioSpeakerPlugin(AudioSinkPlugin):
    """Plays audio through speakers via PulseAudio/PyAudio"""

    def __init__(self):
        super().__init__()
        self.sample_rate = 48000
        self.channel_count = 2
        self.buffer_size = 512
        self.pyaudio_instance = None
        self.stream = None

    def initialize(self) -> bool:
        """Initialize the plugin"""
        if not PLATFORM_SUPPORTED:
            print(
                "[PulseAudioSpeaker] PulseAudio plugin is Linux-only. "
                "Use sounddevice_speaker on Windows/macOS.",
                flush=True,
            )
            self.state = PluginState.ERROR
            return False

        if not PYAUDIO_AVAILABLE:
            print("[PulseAudioSpeaker] PyAudio not available. Install with: pip install pyaudio", flush=True)
            self.state = PluginState.ERROR
            return False

        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            self.state = PluginState.INITIALIZED
            return True
        except Exception as e:
            print(f"[PulseAudioSpeaker] Failed to initialize PyAudio: {e}")
            self.state = PluginState.ERROR
            return False

    def shutdown(self):
        """Shutdown the plugin"""
        self.stop()
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
            self.pyaudio_instance = None
        self.state = PluginState.UNLOADED

    def start(self) -> bool:
        """Start playing audio"""
        if self.state != PluginState.INITIALIZED:
            return False

        try:
            self.stream = self.pyaudio_instance.open(
                format=pyaudio.paFloat32,
                channels=self.channel_count,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.buffer_size,
                stream_callback=None  # Use blocking mode for simplicity
            )

            self.state = PluginState.RUNNING
            print(f"[PulseAudioSpeaker] Started - {self.sample_rate}Hz, {self.channel_count} channels", flush=True)
            return True
        except Exception as e:
            print(f"[PulseAudioSpeaker] Failed to start stream: {e}", flush=True)
            import traceback
            traceback.print_exc()
            self.state = PluginState.ERROR
            return False

    def stop(self):
        """Stop playing audio"""
        try:
            if self.stream:
                try:
                    if self.stream.is_active():
                        self.stream.stop_stream()
                except Exception as e:
                    print(f"[PulseAudioSpeaker] Error stopping stream: {e}", flush=True)

                try:
                    self.stream.close()
                except Exception as e:
                    print(f"[PulseAudioSpeaker] Error closing stream: {e}", flush=True)
                finally:
                    self.stream = None

            if self.state == PluginState.RUNNING:
                self.state = PluginState.INITIALIZED
                print("[PulseAudioSpeaker] Stopped", flush=True)
        except Exception as e:
            print(f"[PulseAudioSpeaker] Error in stop(): {e}", flush=True)
            self.stream = None
            self.state = PluginState.INITIALIZED

    def get_info(self) -> PluginInfo:
        """Get plugin information"""
        return PluginInfo(
            name="PulseAudio Speaker Output",
            version="2.0.0",
            author="Icing Project",
            description="Plays audio through system speakers using PulseAudio",
            plugin_type=PluginType.AUDIO_SINK,
            api_version=1
        )

    def set_parameter(self, key: str, value: str):
        """Set plugin parameter"""
        if key == "bufferSize":
            try:
                self.set_buffer_size(int(value))
            except ValueError:
                pass
        elif key == "sampleRate":
            try:
                self.set_sample_rate(int(value))
            except ValueError:
                pass
        elif key == "channels":
            try:
                self.set_channel_count(int(value))
            except ValueError:
                pass

    def get_parameter(self, key: str) -> str:
        """Get plugin parameter"""
        if key == "sampleRate":
            return str(self.sample_rate)
        elif key == "channels":
            return str(self.channel_count)
        elif key == "bufferSize":
            return str(self.buffer_size)
        return ""

    def write_audio(self, buffer: AudioBuffer) -> bool:
        """Write audio data to speakers"""
        if self.state != PluginState.RUNNING or not self.stream:
            return False

        try:
            # Interleave channels efficiently using numpy
            # buffer.data has shape (channels, frames)
            # Transpose to (frames, channels) and flatten to interleave
            interleaved = buffer.data.T.flatten().astype(np.float32)

            # Write to stream - use exception_on_underflow=False for robustness
            self.stream.write(interleaved.tobytes(), exception_on_underflow=False)

            return True
        except Exception as e:
            print(f"[PulseAudioSpeaker] Write error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return False

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
            self.channel_count = max(1, channels)

    def get_buffer_size(self) -> int:
        """Get buffer size"""
        return self.buffer_size

    def set_buffer_size(self, samples: int):
        """Set buffer size"""
        if self.state in (PluginState.UNLOADED, PluginState.INITIALIZED):
            self.buffer_size = max(64, samples)

    def get_available_space(self) -> int:
        """Get available space"""
        return self.buffer_size


# Plugin factory function
def create_plugin():
    """Factory function to create plugin instance"""
    return PulseAudioSpeakerPlugin()
