"""
SoundDevice Speaker Plugin - Python Implementation
Plays audio through system speakers using sounddevice
"""

import os
import queue
import threading
import time

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
        self.buffer_size = 512  # v2.1: Fixed to 512 to match pipeline buffer size
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

        self._control_thread = None
        self._control_stop_event = threading.Event()

        self._profile_enabled = self._is_truthy_env("NDA_PROFILE") or self._is_truthy_env("NDA_PROFILE_PYPLUGINS")
        self._profile_interval_s = max(0.1, self._read_env_int("NDA_PROFILE_PYPLUGINS_INTERVAL_MS", 1000) / 1000.0)
        self._profile_last_log = time.monotonic()

        self._profile_write_calls = 0
        self._profile_write_us_total = 0
        self._profile_write_us_max = 0
        self._profile_queue_full_events = 0

        self._profile_cb_calls = 0
        self._profile_cb_us_total = 0
        self._profile_cb_us_max = 0
        self._profile_cb_frames_total = 0
        self._profile_cb_frames_min = None
        self._profile_cb_frames_max = 0
        self._profile_cb_queue_empty = 0
        self._profile_cb_queue_qsize_min = None
        self._profile_cb_queue_qsize_max = 0
        self._profile_cb_last_t = None
        self._profile_cb_dt_us_total = 0
        self._profile_cb_dt_us_count = 0
        self._profile_cb_dt_us_min = None
        self._profile_cb_dt_us_max = 0

    @staticmethod
    def _is_truthy_env(name: str) -> bool:
        value = os.getenv(name)
        if not value:
            return False
        return value.strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _read_env_int(name: str, default: int) -> int:
        value = os.getenv(name)
        if not value:
            return default
        try:
            return int(value)
        except ValueError:
            return default

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

    def _audio_callback(self, outdata, frames, time_info, status):
        """Audio thread callback"""
        cb_start = time.perf_counter() if self._profile_enabled else 0.0

        if status:
            if getattr(status, "output_underflow", False) or "underflow" in str(status).lower():
                self._handle_underflow()

        if self._profile_enabled:
            self._profile_cb_calls += 1
            if self._profile_cb_last_t is not None:
                dt_us = int((cb_start - self._profile_cb_last_t) * 1_000_000.0)
                self._profile_cb_dt_us_total += dt_us
                self._profile_cb_dt_us_count += 1
                if self._profile_cb_dt_us_min is None:
                    self._profile_cb_dt_us_min = dt_us
                else:
                    self._profile_cb_dt_us_min = min(self._profile_cb_dt_us_min, dt_us)
                self._profile_cb_dt_us_max = max(self._profile_cb_dt_us_max, dt_us)
            self._profile_cb_last_t = cb_start
            self._profile_cb_frames_total += int(frames)
            if self._profile_cb_frames_min is None:
                self._profile_cb_frames_min = int(frames)
            else:
                self._profile_cb_frames_min = min(self._profile_cb_frames_min, int(frames))
            self._profile_cb_frames_max = max(self._profile_cb_frames_max, int(frames))

            try:
                qsize = int(self.audio_queue.qsize())
                if self._profile_cb_queue_qsize_min is None:
                    self._profile_cb_queue_qsize_min = qsize
                else:
                    self._profile_cb_queue_qsize_min = min(self._profile_cb_queue_qsize_min, qsize)
                self._profile_cb_queue_qsize_max = max(self._profile_cb_queue_qsize_max, qsize)
            except Exception:
                pass

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
            if self._profile_enabled:
                self._profile_cb_queue_empty += 1

        if self._profile_enabled:
            cb_us = int((time.perf_counter() - cb_start) * 1_000_000.0)
            self._profile_cb_us_total += cb_us
            self._profile_cb_us_max = max(self._profile_cb_us_max, cb_us)

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
            self._control_stop_event.clear()
            self.audio_queue = queue.Queue(maxsize=self.max_queue_buffers)
            self._underflow_count = 0
            self._overflow_count = 0
            self._latency_upscaled = False
            self._needs_latency_bump = False
            self._latency_bumps = 0
            self._log_under_notice = True

            if self._profile_enabled:
                self._profile_last_log = time.monotonic()
                self._profile_write_calls = 0
                self._profile_write_us_total = 0
                self._profile_write_us_max = 0
                self._profile_queue_full_events = 0

                self._profile_cb_calls = 0
                self._profile_cb_us_total = 0
                self._profile_cb_us_max = 0
                self._profile_cb_frames_total = 0
                self._profile_cb_frames_min = None
                self._profile_cb_frames_max = 0
                self._profile_cb_queue_empty = 0
                self._profile_cb_queue_qsize_min = None
                self._profile_cb_queue_qsize_max = 0
                self._profile_cb_last_t = None
                self._profile_cb_dt_us_total = 0
                self._profile_cb_dt_us_count = 0
                self._profile_cb_dt_us_min = None
                self._profile_cb_dt_us_max = 0

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

            self._control_thread = threading.Thread(
                target=self._control_loop,
                name="SoundDeviceSpeakerControl",
                daemon=True,
            )
            self._control_thread.start()

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

    def _control_loop(self) -> None:
        try:
            while not self._control_stop_event.is_set():
                if self._needs_latency_bump:
                    self._maybe_bump_latency()
                if self._control_stop_event.wait(timeout=0.05):
                    break
        except Exception:
            return

    def _drain_queue(self) -> None:
        try:
            while True:
                self.audio_queue.get_nowait()
        except queue.Empty:
            return

    def _maybe_bump_latency(self):
        """Reopen stream with higher latency if requested."""
        if self._control_stop_event.is_set():
            return
        if not self.auto_scale_enabled or not self._needs_latency_bump or self._latency_upscaled:
            return
        self._needs_latency_bump = False
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None

            self._drain_queue()
            silence = np.zeros((self.buffer_size, self.channel_count), dtype=np.float32)
            try:
                self.audio_queue.put_nowait(silence)
                self.audio_queue.put_nowait(silence.copy())
            except queue.Full:
                pass

            if self._control_stop_event.is_set():
                return

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
            self._control_stop_event.set()
            if self._control_thread and self._control_thread.is_alive():
                self._control_thread.join(timeout=2.0)
            self._control_thread = None

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

        t0 = time.perf_counter() if self._profile_enabled else 0.0

        try:
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
                replaced = False
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put_nowait(outdata)
                    replaced = True
                except queue.Empty:
                    pass
                except queue.Full:
                    pass
                self._overflow_count += 1
                if self._profile_enabled:
                    self._profile_queue_full_events += 1
                if not replaced:
                    return False

            if self._profile_enabled:
                dt_us = int((time.perf_counter() - t0) * 1_000_000.0)
                self._profile_write_calls += 1
                self._profile_write_us_total += dt_us
                self._profile_write_us_max = max(self._profile_write_us_max, dt_us)
                self._maybe_log_profile()
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

    def _maybe_log_profile(self) -> None:
        if not self._profile_enabled:
            return

        now = time.monotonic()
        dt_s = now - self._profile_last_log
        if dt_s < self._profile_interval_s:
            return

        write_avg_us = (self._profile_write_us_total / self._profile_write_calls) if self._profile_write_calls else 0.0
        write_hz = (self._profile_write_calls / dt_s) if dt_s > 0 else 0.0
        cb_avg_us = (self._profile_cb_us_total / self._profile_cb_calls) if self._profile_cb_calls else 0.0
        cb_hz = (self._profile_cb_calls / dt_s) if dt_s > 0 else 0.0
        cb_dt_avg_us = (
            (self._profile_cb_dt_us_total / self._profile_cb_dt_us_count) if self._profile_cb_dt_us_count else 0.0
        )
        cb_frames_avg = (
            (self._profile_cb_frames_total / self._profile_cb_calls) if self._profile_cb_calls else 0.0
        )

        expected_cb_hz = (float(self.sample_rate) / float(self.buffer_size)) if self.buffer_size > 0 else 0.0

        print(
            "[SoundDeviceSpeakerProfile]"
            f" dt={dt_s * 1000.0:.1f}ms"
            f" sr={self.sample_rate}"
            f" ch={self.channel_count}"
            f" block={self.buffer_size}"
            f" hz(write/cb/exp)={write_hz:.2f}/{cb_hz:.2f}/{expected_cb_hz:.2f}"
            f" write(avgUs/maxUs)={write_avg_us:.0f}/{self._profile_write_us_max}"
            f" qFull={self._profile_queue_full_events}"
            f" cb(frames min/avg/max)={self._profile_cb_frames_min or 0}/{cb_frames_avg:.1f}/{self._profile_cb_frames_max}"
            f" cbDt(min/avg/maxUs)={self._profile_cb_dt_us_min or 0}/{cb_dt_avg_us:.0f}/{self._profile_cb_dt_us_max}"
            f" cb(avgUs/maxUs)={cb_avg_us:.0f}/{self._profile_cb_us_max}"
            f" cbQueueEmpty={self._profile_cb_queue_empty}"
            f" qsize(min/max)={self._profile_cb_queue_qsize_min or 0}/{self._profile_cb_queue_qsize_max}"
            f" underruns={self._underflow_count}"
            f" overflows={self._overflow_count}"
            f" latencyMode={self.latency_mode}"
            f" latencyBumps={self._latency_bumps}",
            flush=True,
        )

        self._profile_last_log = now
        self._profile_write_calls = 0
        self._profile_write_us_total = 0
        self._profile_write_us_max = 0
        self._profile_queue_full_events = 0

        self._profile_cb_calls = 0
        self._profile_cb_us_total = 0
        self._profile_cb_us_max = 0
        self._profile_cb_frames_total = 0
        self._profile_cb_frames_min = None
        self._profile_cb_frames_max = 0
        self._profile_cb_queue_empty = 0
        self._profile_cb_queue_qsize_min = None
        self._profile_cb_queue_qsize_max = 0
        self._profile_cb_last_t = None
        self._profile_cb_dt_us_total = 0
        self._profile_cb_dt_us_count = 0
        self._profile_cb_dt_us_min = None
        self._profile_cb_dt_us_max = 0


# Plugin factory function
def create_plugin():
    """Factory function to create plugin instance"""
    return SoundDeviceSpeakerPlugin()
