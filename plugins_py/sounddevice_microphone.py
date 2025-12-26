"""
SoundDevice Microphone Plugin - Python Implementation
Captures audio from system microphone using sounddevice (better than PyAudio)
"""

import queue

try:
    import sounddevice as sd
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
        self.channel_count = 1
        self.buffer_size = 256
        self.device = None
        self.device_name = ""
        self.callback = None
        self.stream = None
        self.max_queue_buffers = 8  # bounds latency to ~8 blocks for jitter absorption
        self.audio_queue = queue.Queue(maxsize=self.max_queue_buffers)
        self.auto_scale_enabled = True
        self.latency_mode = "auto"
        self._needs_latency_bump = False
        self._latency_upscaled = False
        self._overflow_count = 0
        self._latency_bumps = 0
        self._log_overflow_notice = True

    def initialize(self) -> bool:
        """Initialize the plugin"""
        if not SOUNDDEVICE_AVAILABLE:
            print("[SoundDeviceMic] sounddevice not available. Install with: pip install sounddevice", flush=True)
            self.state = PluginState.ERROR
            return False

        try:
            dev_info = sd.query_devices(self.device, "input")
            self.device_name = dev_info.get("name", "")
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
        """Audio thread callback"""
        if status:
            if getattr(status, "input_overflow", False) or "overflow" in str(status).lower():
                self._handle_overflow()
        try:
            self.audio_queue.put_nowait(indata.copy())
        except queue.Full:
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(indata.copy())
            except queue.Empty:
                pass
            except queue.Full:
                pass
            self._handle_overflow()

    def _handle_overflow(self):
        """Request higher latency when capture overflows."""
        self._overflow_count += 1
        if not self.auto_scale_enabled:
            return
        if (self.latency_mode == "auto"
                and not self._latency_upscaled
                and self._overflow_count >= 3):
            self._needs_latency_bump = True
            if self._log_overflow_notice:
                print("[SoundDeviceMic] Requesting higher latency to avoid overflows", flush=True)
                self._log_overflow_notice = False

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

            self.audio_queue = queue.Queue(maxsize=self.max_queue_buffers)

            def _open(channels: int):
                return sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=channels,
                    blocksize=self.buffer_size,
                    dtype=np.float32,
                    callback=self._audio_callback,
                    latency="high",
                    device=self.device
                )

            self.stream = _open(self.channel_count)
            self.stream.start()
            self._latency_upscaled = True
            self._latency_bumps += 1
            print("[SoundDeviceMic] Latency bumped to high after overflow", flush=True)
        except Exception as e:
            print(f"[SoundDeviceMic] Failed to bump latency: {e}", flush=True)
            import traceback
            traceback.print_exc()

    def _maybe_bump_block_size(self):
        """Increase block size progressively (256 -> 512 -> 1024) when still overflowing."""
        # Disabled for now to avoid mid-run blocksize mismatch with pipeline.
        return

    def start(self) -> bool:
        """Start capturing audio"""
        if self.state != PluginState.INITIALIZED:
            return False

        try:
            self.audio_queue = queue.Queue(maxsize=self.max_queue_buffers)
            self._needs_latency_bump = False
            self._latency_upscaled = False
            self._overflow_count = 0
            self._latency_bumps = 0
            self._log_overflow_notice = True

            def _open(channels: int):
                return sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=channels,
                    blocksize=self.buffer_size,
                    dtype=np.float32,
                    callback=self._audio_callback,
                    latency=self._effective_latency(),
                    device=self.device
                )

            try_channels = self.channel_count or 1
            try:
                self.stream = _open(try_channels)
                self.channel_count = try_channels
            except Exception as e:
                if try_channels != 1:
                    print(f"[SoundDeviceMic] Falling back to mono due to: {e}", flush=True)
                    self.stream = _open(1)
                    self.channel_count = 1
                else:
                    raise

            self.stream.start()
            self.state = PluginState.RUNNING
            print(f"[SoundDeviceMic] Started - {self.sample_rate}Hz, {self.channel_count} channels, block {self.buffer_size}, device '{self.device_name or self.device}'", flush=True)
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
        elif key == "device":
            self.set_device(value)
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
        elif key == "autoScale":
            return "true" if self.auto_scale_enabled else "false"
        elif key == "latencyMode":
            return self.latency_mode
        elif key == "deviceName":
            return self.device_name or ""
        elif key == "overflows":
            return str(self._overflow_count)
        elif key == "latencyBumps":
            return str(self._latency_bumps)
        return ""

    def set_audio_callback(self, callback: AudioSourceCallback):
        """Set audio callback"""
        self.callback = callback

    def read_audio(self, buffer: AudioBuffer) -> bool:
        """Read audio data from queue - BLOCKS until data available"""
        if self.state != PluginState.RUNNING:
            buffer.clear()
            return False

        self._maybe_bump_latency()

        try:
            indata = self.audio_queue.get(timeout=0.2)  # allow time for real-time pacing

            frame_count = buffer.get_frame_count()

            if indata.shape[0] != frame_count:
                if indata.shape[0] > frame_count:
                    indata = indata[:frame_count, :]
                else:
                    pad = np.zeros((frame_count - indata.shape[0], indata.shape[1]), dtype=indata.dtype)
                    indata = np.vstack((indata, pad))

            buffer.data[:] = indata.T
            return True
        except queue.Empty:
            buffer.clear()
            return False
        except Exception as e:
            print(f"[SoundDeviceMic] Read error: {e}", flush=True)
            buffer.clear()
            return False

    def get_sample_rate(self) -> int:
        """Get sample rate"""
        return self.sample_rate

    def get_channel_count(self) -> int:
        """Get number of channels"""
        return self.channel_count

    def get_buffer_size(self) -> int:
        """Get buffer size in frames"""
        return self.buffer_size

    def set_sample_rate(self, sample_rate: int):
        """Set sample rate"""
        if self.state in (PluginState.UNLOADED, PluginState.INITIALIZED):
            self.sample_rate = sample_rate

    def set_channel_count(self, channels: int):
        """Set number of channels"""
        if self.state in (PluginState.UNLOADED, PluginState.INITIALIZED):
            self.channel_count = max(1, channels)

    def set_buffer_size(self, samples: int):
        """Set buffer size"""
        if self.state in (PluginState.UNLOADED, PluginState.INITIALIZED):
            self.buffer_size = max(64, samples)
            self.audio_queue = queue.Queue(maxsize=self.max_queue_buffers)

    def set_device(self, device):
        if self.state in (PluginState.UNLOADED, PluginState.INITIALIZED):
            self.device = device
            try:
                dev_info = sd.query_devices(self.device, "input")
                self.device_name = dev_info.get("name", "")
            except Exception:
                self.device_name = ""


# Plugin factory function
def create_plugin():
    """Factory function to create plugin instance"""
    return SoundDeviceMicrophonePlugin()
