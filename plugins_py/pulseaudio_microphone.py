"""
PulseAudio Microphone Plugin - Python Implementation
Captures audio from system microphone using PulseAudio
Uses callback mode with ring buffer for reliable, non-blocking operation
"""

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

import numpy as np
import threading
from collections import deque
from base_plugin import (
    AudioSourcePlugin, AudioBuffer, PluginInfo,
    PluginType, PluginState, AudioSourceCallback
)


class PulseAudioMicrophonePlugin(AudioSourcePlugin):
    """Captures audio from microphone via PulseAudio/PyAudio using callback mode"""

    def __init__(self):
        super().__init__()
        self.sample_rate = 48000
        self.channels = 2
        self.buffer_size = 512
        self.callback = None
        self.pyaudio_instance = None
        self.stream = None

        # Ring buffer for audio data (stores up to 2 seconds of audio)
        self.ring_buffer = deque(maxlen=200)  # ~200 chunks = 2 seconds
        self.buffer_lock = threading.Lock()
        self.underrun_count = 0

    def initialize(self) -> bool:
        """Initialize the plugin"""
        if not PYAUDIO_AVAILABLE:
            print("[PulseAudioMic] PyAudio not available. Install with: pip install pyaudio", flush=True)
            self.state = PluginState.ERROR
            return False

        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            self.state = PluginState.INITIALIZED
            return True
        except Exception as e:
            print(f"[PulseAudioMic] Failed to initialize PyAudio: {e}", flush=True)
            self.state = PluginState.ERROR
            return False

    def shutdown(self):
        """Shutdown the plugin"""
        self.stop()
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
            self.pyaudio_instance = None
        self.state = PluginState.UNLOADED

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback - called on separate thread"""
        if status:
            print(f"[PulseAudioMic] Callback status: {status}", flush=True)

        # Convert bytes to numpy array
        try:
            samples = np.frombuffer(in_data, dtype=np.float32)

            # Store in ring buffer
            with self.buffer_lock:
                self.ring_buffer.append(samples.copy())
        except Exception as e:
            print(f"[PulseAudioMic] Callback error: {e}", flush=True)

        return (None, pyaudio.paContinue)

    def start(self) -> bool:
        """Start capturing audio"""
        if self.state != PluginState.INITIALIZED:
            return False

        try:
            # Clear the ring buffer
            with self.buffer_lock:
                self.ring_buffer.clear()
                self.underrun_count = 0

            # Open stream with callback mode
            self.stream = self.pyaudio_instance.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.buffer_size,
                stream_callback=self._audio_callback
            )

            self.state = PluginState.RUNNING
            print(f"[PulseAudioMic] Started - {self.sample_rate}Hz, {self.channels} channels (callback mode)", flush=True)
            return True
        except Exception as e:
            print(f"[PulseAudioMic] Failed to start stream: {e}", flush=True)
            import traceback
            traceback.print_exc()
            self.state = PluginState.ERROR
            return False

    def stop(self):
        """Stop capturing audio"""
        try:
            if self.stream:
                try:
                    if self.stream.is_active():
                        self.stream.stop_stream()
                except Exception as e:
                    print(f"[PulseAudioMic] Error stopping stream: {e}", flush=True)

                try:
                    self.stream.close()
                except Exception as e:
                    print(f"[PulseAudioMic] Error closing stream: {e}", flush=True)
                finally:
                    self.stream = None

            if self.state == PluginState.RUNNING:
                self.state = PluginState.INITIALIZED

                if self.underrun_count > 0:
                    print(f"[PulseAudioMic] Stopped (had {self.underrun_count} buffer underruns)", flush=True)
                else:
                    print("[PulseAudioMic] Stopped", flush=True)
        except Exception as e:
            print(f"[PulseAudioMic] Error in stop(): {e}", flush=True)
            self.stream = None
            self.state = PluginState.INITIALIZED

    def get_info(self) -> PluginInfo:
        """Get plugin information"""
        return PluginInfo(
            name="PulseAudio Microphone",
            version="2.0.0",
            author="NADE Team",
            description="Captures audio from system microphone using PulseAudio (callback mode)",
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
        """Read audio data from ring buffer (non-blocking)"""
        if self.state != PluginState.RUNNING:
            buffer.clear()
            return False

        try:
            # Try to get data from ring buffer
            with self.buffer_lock:
                if len(self.ring_buffer) == 0:
                    # Buffer underrun
                    self.underrun_count += 1
                    if self.underrun_count % 100 == 1:
                        print(f"[PulseAudioMic] Buffer underrun ({self.underrun_count} total)", flush=True)
                    buffer.clear()
                    return False

                # Pop oldest chunk from buffer
                samples = self.ring_buffer.popleft()

            frame_count = buffer.get_frame_count()
            expected_samples = frame_count * self.channels

            # Verify size
            if len(samples) != expected_samples:
                print(f"[PulseAudioMic] Size mismatch: got {len(samples)}, expected {expected_samples}", flush=True)
                # Pad or truncate
                if len(samples) < expected_samples:
                    padding = np.zeros(expected_samples - len(samples), dtype=np.float32)
                    samples = np.concatenate([samples, padding])
                else:
                    samples = samples[:expected_samples]

            # De-interleave channels efficiently
            if self.channels == 1:
                buffer.data[0] = samples
            else:
                # Reshape interleaved data to (frames, channels) then transpose to (channels, frames)
                buffer.data[:] = samples.reshape(frame_count, self.channels).T

            return True
        except Exception as e:
            print(f"[PulseAudioMic] Read error: {e}", flush=True)
            import traceback
            traceback.print_exc()
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
            self.channels = channels


# Plugin factory function
def create_plugin():
    """Factory function to create plugin instance"""
    return PulseAudioMicrophonePlugin()
