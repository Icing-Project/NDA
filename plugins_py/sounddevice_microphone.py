"""
SoundDevice Microphone Plugin - Python Implementation
Captures audio from system microphone using sounddevice (better than PyAudio)
"""

try:
    import sounddevice as sd
    import queue
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

import numpy as np
from base_plugin import (
    AudioSourcePlugin, AudioBuffer, PluginInfo,
    PluginType, PluginState, AudioSourceCallback
)


class SoundDeviceMicrophonePlugin(AudioSourcePlugin):
    """Captures audio from microphone via sounddevice"""

    def __init__(self):
        super().__init__()
        self.sample_rate = 48000
        self.channels = 1  # mono by default to match AIOC path
        self.buffer_size = 512
        self.callback = None
        self.stream = None
        self.audio_queue = queue.Queue()

    def initialize(self) -> bool:
        """Initialize the plugin"""
        if not SOUNDDEVICE_AVAILABLE:
            print("[SoundDeviceMic] sounddevice not available. Install with: pip install sounddevice", flush=True)
            self.state = PluginState.ERROR
            return False

        try:
            # Just check that sounddevice is working
            sd.query_devices()
            self.state = PluginState.INITIALIZED
            return True
        except Exception as e:
            print(f"[SoundDeviceMic] Failed to initialize: {e}", flush=True)
            self.state = PluginState.ERROR
            return False

    def shutdown(self):
        """Shutdown the plugin"""
        self.stop()
        self.state = PluginState.UNLOADED

    def _audio_callback(self, indata, frames, time, status):
        """Callback from sounddevice - runs on audio thread"""
        if status:
            print(f"[SoundDeviceMic] Status: {status}", flush=True)
        # Put a copy in the queue
        self.audio_queue.put(indata.copy())

    def start(self) -> bool:
        """Start capturing audio"""
        if self.state != PluginState.INITIALIZED:
            return False

        try:
            # Clear the queue
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except:
                    pass

            # Open input stream with callback
            def _open(channels: int):
                return sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=channels,
                    blocksize=self.buffer_size,
                    dtype=np.float32,
                    callback=self._audio_callback
                )

            try_channels = self.channels or 1
            try:
                self.stream = _open(try_channels)
                self.channels = try_channels
            except Exception as e:
                if try_channels != 1:
                    print(f"[SoundDeviceMic] Falling back to mono due to: {e}", flush=True)
                    self.stream = _open(1)
                    self.channels = 1
                else:
                    raise

            self.stream.start()
            self.state = PluginState.RUNNING
            print(f"[SoundDeviceMic] Started - {self.sample_rate}Hz, {self.channels} channels", flush=True)
            return True
        except Exception as e:
            print(f"[SoundDeviceMic] Failed to start: {e}", flush=True)
            import traceback
            traceback.print_exc()
            self.state = PluginState.ERROR
            return False

    def stop(self):
        """Stop capturing audio"""
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None

            if self.state == PluginState.RUNNING:
                self.state = PluginState.INITIALIZED
                print("[SoundDeviceMic] Stopped", flush=True)
        except Exception as e:
            print(f"[SoundDeviceMic] Error in stop(): {e}", flush=True)
            self.stream = None
            self.state = PluginState.INITIALIZED

    def get_info(self) -> PluginInfo:
        """Get plugin information"""
        return PluginInfo(
            name="SoundDevice Microphone",
            version="1.0.0",
            author="Icing Project",
            description="Captures audio from microphone using sounddevice library",
            plugin_type=PluginType.AUDIO_SOURCE,
            api_version=1
        )

    def set_parameter(self, key: str, value: str):
        """Set plugin parameter"""
        pass

    def get_parameter(self, key: str) -> str:
        """Get plugin parameter"""
        if key == "sampleRate":
            return str(self.sample_rate)
        elif key == "channels":
            return str(self.channels)
        return ""

    def set_audio_callback(self, callback: AudioSourceCallback):
        """Set audio callback"""
        self.callback = callback

    def read_audio(self, buffer: AudioBuffer) -> bool:
        """Read audio data from queue - BLOCKS until data available"""
        if self.state != PluginState.RUNNING:
            buffer.clear()
            return False

        try:
            # Block with timeout to allow checking state
            # This is the KEY difference - proper blocking with timeout
            indata = self.audio_queue.get(timeout=0.1)  # 100ms timeout

            frame_count = buffer.get_frame_count()

            # indata shape is (frames, channels) - we need (channels, frames)
            if indata.shape[0] != frame_count:
                print(f"[SoundDeviceMic] Frame count mismatch: got {indata.shape[0]}, expected {frame_count}", flush=True)
                buffer.clear()
                return False

            # Transpose from (frames, channels) to (channels, frames)
            buffer.data[:] = indata.T

            return True
        except queue.Empty:
            # Timeout - no data available
            buffer.clear()
            return False
        except Exception as e:
            print(f"[SoundDeviceMic] Read error: {e}", flush=True)
            buffer.clear()
            return False

    def get_sample_rate(self) -> int:
        """Get sample rate"""
        return self.sample_rate

    def get_channels(self) -> int:
        """Get number of channels"""
        return self.channels

    def set_sample_rate(self, sample_rate: int):
        """Set sample rate"""
        if self.state in (PluginState.UNLOADED, PluginState.INITIALIZED):
            self.sample_rate = sample_rate

    def set_channels(self, channels: int):
        """Set number of channels"""
        if self.state in (PluginState.UNLOADED, PluginState.INITIALIZED):
            # Clamp to mono to avoid device errors with multi-channel input.
            self.channels = max(1, min(1, channels))


# Plugin factory function
def create_plugin():
    """Factory function to create plugin instance"""
    return SoundDeviceMicrophonePlugin()
