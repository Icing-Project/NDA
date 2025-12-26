"""
SoundDevice Speaker Plugin - Python Implementation
Plays audio through system speakers using sounddevice
"""

import queue

try:
    import sounddevice as sd
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
        self.channel_count = 1
        self.buffer_size = 256
        self.device = None
        self.device_name = ""
        self.stream = None
        self.max_queue_buffers = 8  # modest headroom for jitter without runaway latency
        self.audio_queue = queue.Queue(maxsize=self.max_queue_buffers)
        self.auto_scale_enabled = True
        self.latency_mode = "auto"
        self._underflow_count = 0
        self._overflow_count = 0
        self._latency_upscaled = False
        self._needs_latency_bump = False
        self._latency_bumps = 0
        self._log_under_notice = True

    def initialize(self) -> bool:
        """Initialize the plugin"""
        if not SOUNDDEVICE_AVAILABLE:
            print("[SoundDeviceSpeaker] sounddevice not available. Install with: pip install sounddevice", flush=True)
            self.state = PluginState.ERROR
            return False

        try:
            dev_info = sd.query_devices(self.device, "output")
            self.device_name = dev_info.get("name", "")
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
        """Audio thread callback"""
        if status:
            if getattr(status, "output_underflow", False) or "underflow" in str(status).lower():
                self._handle_underflow()

        try:
            data = self.audio_queue.get_nowait()
            if data.shape[0] == outdata.shape[0]:
                outdata[:] = data
            elif data.shape[0] < outdata.shape[0]:
                padded = np.zeros_like(outdata)
                padded[:data.shape[0], :] = data
                outdata[:] = padded
            else:
                outdata[:] = data[:outdata.shape[0], :]
        except queue.Empty:
            outdata.fill(0)
            self._handle_underflow()

    def _handle_underflow(self):
        """Track underruns and request latency bump if needed."""
        self._underflow_count += 1
        if not self.auto_scale_enabled:
            return
        if (self.latency_mode == "auto"
                and not self._latency_upscaled
                and self._underflow_count >= 3):
            self._needs_latency_bump = True
            if self._log_under_notice:
                print("[SoundDeviceSpeaker] Requesting higher latency to avoid underruns", flush=True)
                self._log_under_notice = False

    def start(self) -> bool:
        """Start playing audio"""
        if self.state != PluginState.INITIALIZED:
            return False

        try:
            self.audio_queue = queue.Queue(maxsize=self.max_queue_buffers)
            self._underflow_count = 0
            self._overflow_count = 0
            self._latency_upscaled = False
            self._needs_latency_bump = False
            self._latency_bumps = 0
            self._log_under_notice = True

            silence = np.zeros((self.buffer_size, self.channel_count), dtype=np.float32)
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
                    latency=self._effective_latency(),
                    device=self.device
                )

            try_channels = self.channel_count or 1
            try:
                self.stream = _open(try_channels)
                self.channel_count = try_channels
            except Exception as e:
                if try_channels != 1:
                    print(f"[SoundDeviceSpeaker] Falling back to mono due to: {e}", flush=True)
                    self.stream = _open(1)
                    self.channel_count = 1
                else:
                    raise

            self.stream.start()
            self.state = PluginState.RUNNING
            print(f"[SoundDeviceSpeaker] Started - {self.sample_rate}Hz, {self.channel_count} channels, block {self.buffer_size}, device '{self.device_name or self.device}'", flush=True)
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

            maxsize = self.audio_queue.maxsize or self.max_queue_buffers
            self.audio_queue = queue.Queue(maxsize=maxsize)
            silence = np.zeros((self.buffer_size, self.channel_count), dtype=np.float32)
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
                    latency="high",
                    device=self.device
                )

            self.stream = _open(self.channel_count)
            self.stream.start()
            self._latency_upscaled = True
            self._latency_bumps += 1
            print("[SoundDeviceSpeaker] Latency bumped to high after underruns", flush=True)
        except Exception as e:
            print(f"[SoundDeviceSpeaker] Failed to bump latency: {e}", flush=True)
            import traceback
            traceback.print_exc()

    def _maybe_bump_block_size(self):
        """Increase block size progressively (256 -> 512 -> 1024) when still underrunning."""
        # Disabled for now to avoid mid-run blocksize mismatch with pipeline.
        return

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
        elif key == "underruns":
            return str(self._underflow_count)
        elif key == "overflows":
            return str(self._overflow_count)
        elif key == "latencyBumps":
            return str(self._latency_bumps)
        return ""

    def write_audio(self, buffer: AudioBuffer) -> bool:
        """Write audio data to speakers"""
        if self.state != PluginState.RUNNING:
            return False

        try:
            self._maybe_bump_latency()

            in_channels = buffer.data.shape[0]

            if in_channels == self.channel_count:
                outdata = buffer.data.T.copy()
            elif self.channel_count == 1:
                mixed = buffer.data.mean(axis=0)
                outdata = mixed[np.newaxis, :].T
            else:
                mono = buffer.data.mean(axis=0)
                outdata = np.repeat(mono[np.newaxis, :].T, self.channel_count, axis=1)

            outdata = outdata.astype(np.float32, copy=False)

            try:
                self.audio_queue.put_nowait(outdata)
            except queue.Full:
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put_nowait(outdata)
                except queue.Empty:
                    pass
                except queue.Full:
                    pass
                self._overflow_count += 1
                return False
            return True
        except Exception as e:
            print(f"[SoundDeviceSpeaker] Write error: {e}", flush=True)
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
            self.audio_queue = queue.Queue(maxsize=self.max_queue_buffers)

    def set_device(self, device):
        if self.state in (PluginState.UNLOADED, PluginState.INITIALIZED):
            self.device = device
            try:
                dev_info = sd.query_devices(self.device, "output")
                self.device_name = dev_info.get("name", "")
            except Exception:
                self.device_name = ""

    def get_available_space(self) -> int:
        """Get available space"""
        try:
            free_slots = self.audio_queue.maxsize - self.audio_queue.qsize()
            return max(0, free_slots * self.buffer_size)
        except Exception:
            return self.buffer_size


# Plugin factory function
def create_plugin():
    """Factory function to create plugin instance"""
    return SoundDeviceSpeakerPlugin()
