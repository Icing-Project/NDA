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
        self.buffer_size = 256  # low latency default, pipeline can override
        self.stream = None
        self.audio_queue = queue.Queue(maxsize=8)  # Start with small queue for low latency
        self.auto_scale_enabled = True
        self.latency_mode = "auto"  # low by default, can auto-bump to high
        self._underflow_count = 0
        self._scaled_up = False
        self._latency_upscaled = False
        self._needs_latency_bump = False

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
            if getattr(status, "output_underflow", False) or "underflow" in str(status).lower():
                self._handle_underflow()

        try:
            # Get data from queue (non-blocking)
            data = self.audio_queue.get_nowait()
            outdata[:] = data
        except queue.Empty:
            # No data - output silence
            outdata.fill(0)
            # Treat empty queue as potential underrun
            self._handle_underflow()

    def _handle_underflow(self):
        """Scale buffering when the output starves."""
        self._underflow_count += 1
        if not self.auto_scale_enabled:
            return
        # Increase queue depth progressively up to 64 buffers
        current_max = self.audio_queue.maxsize
        if current_max < 64:
            new_max = min(64, max(current_max * 2, current_max + 4))
            self.audio_queue.maxsize = new_max
            print(f"[SoundDeviceSpeaker] Output underrun detected, increasing queue depth to {new_max} buffers", flush=True)
            self._scaled_up = True
            # Prefill a little silence to immediately cover next callback
            try:
                silence = np.zeros((self.buffer_size, self.channels), dtype=np.float32)
                self.audio_queue.put_nowait(silence)
            except queue.Full:
                pass

        # After multiple underruns, request a latency bump if allowed
        if (self.latency_mode == "auto"
                and not self._latency_upscaled
                and self._underflow_count >= 2):
            self._needs_latency_bump = True
            print("[SoundDeviceSpeaker] Requesting higher latency to avoid underruns", flush=True)

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
            self._scaled_up = False
            self._underflow_count = 0
            self._latency_upscaled = False
            self._needs_latency_bump = False

            # Prefill a small amount of silence to avoid immediate underruns
            try:
                silence = np.zeros((self.buffer_size, self.channels), dtype=np.float32)
                self.audio_queue.put_nowait(silence)
                self.audio_queue.put_nowait(silence.copy())
            except queue.Full:
                pass

            # Open output stream with callback
            def _open(channels: int):
                return sd.OutputStream(
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
        """Reopen stream with higher latency if requested."""
        if not self.auto_scale_enabled or not self._needs_latency_bump or self._latency_upscaled:
            return
        self._needs_latency_bump = False
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None

            # Clear and rebuild queue with same maxsize
            maxsize = self.audio_queue.maxsize or 8
            self.audio_queue = queue.Queue(maxsize=maxsize)
            silence = np.zeros((self.buffer_size, self.channels), dtype=np.float32)
            try:
                self.audio_queue.put_nowait(silence)
                self.audio_queue.put_nowait(silence.copy())
            except queue.Full:
                pass

            def _open(channels: int):
                return sd.OutputStream(
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
            print("[SoundDeviceSpeaker] Latency bumped to high after underruns", flush=True)
        except Exception as e:
            print(f"[SoundDeviceSpeaker] Failed to bump latency: {e}", flush=True)
            import traceback
            traceback.print_exc()

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
        if key == "autoScale":
            self.auto_scale_enabled = value.lower() in ("1", "true", "yes", "on")
        elif key == "bufferSize":
            try:
                self.set_buffer_size(int(value))
            except ValueError:
                pass
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

    def write_audio(self, buffer: AudioBuffer) -> bool:
        """Write audio data to speakers"""
        if self.state != PluginState.RUNNING:
            return False

        try:
            # If previous underruns asked for more latency, apply it now.
            self._maybe_bump_latency()

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
            # Keep a sensible floor to avoid zero/negative sizes.
            self.buffer_size = max(64, samples)
            # Reset queue with the new sizing goal
            self.audio_queue = queue.Queue(maxsize=self.audio_queue.maxsize or 8)

    def get_available_space(self) -> int:
        """Get available space"""
        return self.buffer_size


# Plugin factory function
def create_plugin():
    """Factory function to create plugin instance"""
    return SoundDeviceSpeakerPlugin()
