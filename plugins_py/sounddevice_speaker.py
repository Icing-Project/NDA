"""
SoundDevice Speaker Plugin - Python Implementation
Plays audio through system speakers using sounddevice
"""

try:
    import sounddevice as sd
    import queue
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

import numpy as np
from base_plugin import (
    AudioSinkPlugin, AudioBuffer, PluginInfo,
    PluginType, PluginState
)


class SoundDeviceSpeakerPlugin(AudioSinkPlugin):
    """Plays audio through speakers via sounddevice"""

    def __init__(self):
        super().__init__()
        self.sample_rate = 48000
        self.channels = 1  # mono by default to align with AIOC
        self.buffer_size = 512
        self.stream = None
        self.audio_queue = queue.Queue(maxsize=10)  # Limit queue size

    def initialize(self) -> bool:
        """Initialize the plugin"""
        if not SOUNDDEVICE_AVAILABLE:
            print("[SoundDeviceSpeaker] sounddevice not available. Install with: pip install sounddevice", flush=True)
            self.state = PluginState.ERROR
            return False

        try:
            sd.query_devices()
            self.state = PluginState.INITIALIZED
            return True
        except Exception as e:
            print(f"[SoundDeviceSpeaker] Failed to initialize: {e}", flush=True)
            self.state = PluginState.ERROR
            return False

    def shutdown(self):
        """Shutdown the plugin"""
        self.stop()
        self.state = PluginState.UNLOADED

    def _audio_callback(self, outdata, frames, time, status):
        """Callback from sounddevice - runs on audio thread"""
        if status:
            print(f"[SoundDeviceSpeaker] Status: {status}", flush=True)

        try:
            # Get data from queue (non-blocking)
            data = self.audio_queue.get_nowait()
            outdata[:] = data
        except queue.Empty:
            # No data - output silence
            outdata.fill(0)

    def start(self) -> bool:
        """Start playing audio"""
        if self.state != PluginState.INITIALIZED:
            return False

        try:
            # Clear the queue
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except:
                    pass

            # Open output stream with callback
            def _open(channels: int):
                return sd.OutputStream(
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
                    print(f"[SoundDeviceSpeaker] Falling back to mono due to: {e}", flush=True)
                    self.stream = _open(1)
                    self.channels = 1
                else:
                    raise

            self.stream.start()
            self.state = PluginState.RUNNING
            print(f"[SoundDeviceSpeaker] Started - {self.sample_rate}Hz, {self.channels} channels", flush=True)
            return True
        except Exception as e:
            print(f"[SoundDeviceSpeaker] Failed to start: {e}", flush=True)
            import traceback
            traceback.print_exc()
            self.state = PluginState.ERROR
            return False

    def stop(self):
        """Stop playing audio"""
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None

            if self.state == PluginState.RUNNING:
                self.state = PluginState.INITIALIZED
                print("[SoundDeviceSpeaker] Stopped", flush=True)
        except Exception as e:
            print(f"[SoundDeviceSpeaker] Error in stop(): {e}", flush=True)
            self.stream = None
            self.state = PluginState.INITIALIZED

    def get_info(self) -> PluginInfo:
        """Get plugin information"""
        return PluginInfo(
            name="SoundDevice Speaker",
            version="1.0.0",
            author="Icing Project",
            description="Plays audio through system speakers using sounddevice",
            plugin_type=PluginType.AUDIO_SINK,
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

    def write_audio(self, buffer: AudioBuffer) -> bool:
        """Write audio data to speakers"""
        if self.state != PluginState.RUNNING:
            return False

        try:
            in_channels = buffer.data.shape[0]

            if in_channels == self.channels:
                # Convert from (channels, frames) to (frames, channels)
                outdata = buffer.data.T.copy()
            elif in_channels == 1 and self.channels > 1:
                # Mono source, multichannel output -> duplicate across channels
                mono = buffer.data[0]
                outdata = np.repeat(mono[np.newaxis, :].T, self.channels, axis=1)
            elif in_channels > 1 and self.channels == 1:
                # Multichannel source, mono output -> downmix
                mixed = buffer.data.mean(axis=0, keepdims=True)
                outdata = mixed.T
            else:
                # Fallback: simple average then tile to requested channels
                mixed = buffer.data.mean(axis=0, keepdims=True)
                outdata = np.repeat(mixed.T, self.channels, axis=1)

            outdata = outdata.astype(np.float32, copy=False)

            # Put in queue - block if full
            self.audio_queue.put(outdata, timeout=0.1)
            return True
        except queue.Full:
            print("[SoundDeviceSpeaker] Queue full, dropping frame", flush=True)
            return False
        except Exception as e:
            print(f"[SoundDeviceSpeaker] Write error: {e}", flush=True)
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
            # Allow requested channel count but never below mono.
            self.channels = max(1, channels)

    def get_buffer_size(self) -> int:
        """Get buffer size"""
        return self.buffer_size

    def set_buffer_size(self, samples: int):
        """Set buffer size"""
        if self.state in (PluginState.UNLOADED, PluginState.INITIALIZED):
            self.buffer_size = samples

    def get_available_space(self) -> int:
        """Get available space"""
        return self.buffer_size


# Plugin factory function
def create_plugin():
    """Factory function to create plugin instance"""
    return SoundDeviceSpeakerPlugin()
