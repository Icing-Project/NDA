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
        self.buffer_size = 256  # low latency default, pipeline can override
        self.callback = None
        self.stream = None
        self.audio_queue = queue.Queue()
        self.auto_scale_enabled = True
        self.latency_mode = "auto"  # start low, bump on overflow
        self._needs_latency_bump = False
        self._latency_upscaled = False
        self._overflow_count = 0

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
            if getattr(status, "input_overflow", False) or "overflow" in str(status).lower():
                self._handle_overflow()
        # Put a copy in the queue
        self.audio_queue.put(indata.copy())

    def _handle_overflow(self):
        """Request higher latency when capture overflows."""
        self._overflow_count += 1
        if not self.auto_scale_enabled:
            return
        if (self.latency_mode == "auto"
                and not self._latency_upscaled
                and self._overflow_count >= 1):
            self._needs_latency_bump = True
            print("[SoundDeviceMic] Requesting higher latency to avoid overflows", flush=True)

    def _effective_latency(self):
        """Resolve latency hint based on mode and scaling."""
        mode = (self.latency_mode or "auto").lower()
        if mode == "high":
            return "high"
        if mode == "low":
            return "low"
        if mode == "default":
            return None
        # auto
        return "high" if self._latency_upscaled else "low"

    def _maybe_bump_latency(self):
        """Reopen stream with higher latency if requested (called from main thread)."""
        if not self.auto_scale_enabled or not self._needs_latency_bump or self._latency_upscaled:
            return
        self._needs_latency_bump = False
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None

            # Clear queue
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break

            def _open(channels: int):
                return sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=channels,
                    blocksize=self.buffer_size,
                    dtype=np.float32,
                    callback=self._audio_callback,
                    latency="high"
                )

            self.stream = _open(self.channels)
            self.stream.start()
            self._latency_upscaled = True
            print("[SoundDeviceMic] Latency bumped to high after overflow", flush=True)
        except Exception as e:
            print(f"[SoundDeviceMic] Failed to bump latency: {e}", flush=True)
            import traceback
            traceback.print_exc()

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
            self._needs_latency_bump = False
            self._latency_upscaled = False
            self._overflow_count = 0

            # Open input stream with callback
            def _open(channels: int):
                return sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=channels,
                    blocksize=self.buffer_size,
                    dtype=np.float32,
                    callback=self._audio_callback,
                    latency=self._effective_latency()
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
        if key == "bufferSize":
            try:
                self.set_buffer_size(int(value))
            except ValueError:
                pass
        elif key == "autoScale":
            self.auto_scale_enabled = value.lower() in ("1", "true", "yes", "on")
        elif key == "latencyMode":
            mode = value.lower()
            if mode in ("low", "high", "default", "auto"):
                self.latency_mode = mode

    def get_parameter(self, key: str) -> str:
        """Get plugin parameter"""
        if key == "sampleRate":
            return str(self.sample_rate)
        elif key == "channels":
            return str(self.channels)
        elif key == "bufferSize":
            return str(self.buffer_size)
        elif key == "autoScale":
            return "true" if self.auto_scale_enabled else "false"
        elif key == "latencyMode":
            return self.latency_mode
        return ""

    def set_audio_callback(self, callback: AudioSourceCallback):
        """Set audio callback"""
        self.callback = callback

    def read_audio(self, buffer: AudioBuffer) -> bool:
        """Read audio data from queue - BLOCKS until data available"""
        if self.state != PluginState.RUNNING:
            buffer.clear()
            return False

        # If previous overflows asked for more latency, apply it now (main thread).
        self._maybe_bump_latency()

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

    def get_buffer_size(self) -> int:
        """Get buffer size in frames"""
        return self.buffer_size

    def set_sample_rate(self, sample_rate: int):
        """Set sample rate"""
        if self.state in (PluginState.UNLOADED, PluginState.INITIALIZED):
            self.sample_rate = sample_rate

    def set_channels(self, channels: int):
        """Set number of channels"""
        if self.state in (PluginState.UNLOADED, PluginState.INITIALIZED):
            # Clamp to mono to avoid device errors with multi-channel input.
            self.channels = max(1, min(1, channels))

    def set_buffer_size(self, samples: int):
        """Set buffer size"""
        if self.state in (PluginState.UNLOADED, PluginState.INITIALIZED):
            self.buffer_size = max(64, samples)


# Plugin factory function
def create_plugin():
    """Factory function to create plugin instance"""
    return SoundDeviceMicrophonePlugin()
