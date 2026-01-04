"""
SoundCard Microphone Plugin - Python Implementation
Captures audio from system microphone using the cross-platform `soundcard` library.

Design:
- A capture thread continuously pulls audio from soundcard and fills a bounded ring buffer.
- `read_audio()` is non-blocking and returns silence on underrun to keep NDA's cadence stable.
"""

from __future__ import annotations

import threading
from typing import Optional

try:
    import soundcard as sc

    SOUNDCARD_AVAILABLE = True
except ImportError:
    SOUNDCARD_AVAILABLE = False

import numpy as np

from base_plugin import (
    AudioSourcePlugin,
    AudioBuffer,
    AudioSourceCallback,
    PluginInfo,
    PluginState,
    PluginType,
)

from nda_py_utils.audio_ring_buffer import AudioRingBuffer


class SoundCardMicrophonePlugin(AudioSourcePlugin):
    """Captures audio from microphone via soundcard (WASAPI/PulseAudio)."""

    def __init__(self):
        super().__init__()
        self.sample_rate = 48000
        self.channel_count = 1
        self.buffer_size = 512

        self.device_query: str = ""
        self.device_index: Optional[int] = None
        self.device_name: str = ""

        self.max_buffer_ms = 200

        self.callback = None

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread_error: Optional[str] = None

        self._ring: Optional[AudioRingBuffer] = None

        self._underrun_events = 0
        self._underrun_frames = 0
        self._device_open_failures = 0

        self._log_underrun_notice = True

    def initialize(self) -> bool:
        if not SOUNDCARD_AVAILABLE:
            print(
                "[SoundCardMic] soundcard not available. Install with: pip install soundcard",
                flush=True,
            )
            self.state = PluginState.ERROR
            return False

        try:
            # Resolve a default device name early for UI/diagnostics.
            mic = self._select_microphone()
            self.device_name = self._device_display_name(mic)
            self.state = PluginState.INITIALIZED
            return True
        except Exception as e:
            print(f"[SoundCardMic] Failed to initialize: {e}", flush=True)
            self.state = PluginState.ERROR
            return False

    def shutdown(self):
        self.stop()
        self.state = PluginState.UNLOADED

    def start(self) -> bool:
        if self.state != PluginState.INITIALIZED:
            return False

        self._stop_event.clear()
        self._ready_event.clear()
        self._thread_error = None
        self._underrun_events = 0
        self._underrun_frames = 0
        self._device_open_failures = 0
        self._log_underrun_notice = True

        capacity_frames = max(1, int(self.sample_rate * (self.max_buffer_ms / 1000.0)))
        self._ring = AudioRingBuffer(channels=self.channel_count, capacity_frames=capacity_frames)

        self._thread = threading.Thread(target=self._capture_loop, name="SoundCardMicCapture", daemon=True)
        self._thread.start()

        if not self._ready_event.wait(timeout=2.0):
            print("[SoundCardMic] Capture thread did not become ready", flush=True)
            self._stop_event.set()
            self.state = PluginState.ERROR
            return False

        if self._thread_error:
            print(f"[SoundCardMic] Failed to start: {self._thread_error}", flush=True)
            self.state = PluginState.ERROR
            return False

        self.state = PluginState.RUNNING
        print(
            f"[SoundCardMic] Started - {self.sample_rate}Hz, {self.channel_count}ch, "
            f"buffer={self.buffer_size} frames, max={self.max_buffer_ms}ms, device='{self.device_name}'",
            flush=True,
        )
        return True

    def stop(self):
        try:
            self._stop_event.set()
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=2.0)
            self._thread = None
            self._ready_event.clear()

            if self.state == PluginState.RUNNING:
                self.state = PluginState.INITIALIZED

            if self._ring:
                self._ring.clear()
        except Exception as e:
            print(f"[SoundCardMic] Error in stop(): {e}", flush=True)
            self.state = PluginState.INITIALIZED

    def get_info(self) -> PluginInfo:
        return PluginInfo(
            name="SoundCard Microphone",
            version="1.0.0",
            author="Icing Project",
            description="Captures audio using the soundcard library (WASAPI/PulseAudio)",
            plugin_type=PluginType.AUDIO_SOURCE,
            api_version=1,
        )

    def set_parameter(self, key: str, value: str):
        if key == "device":
            self.set_device(value)
        elif key == "deviceIndex":
            try:
                self.set_device_index(int(value))
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
        elif key == "bufferSize":
            try:
                self.set_buffer_size(int(value))
            except ValueError:
                pass
        elif key == "maxBufferMs":
            try:
                self.max_buffer_ms = max(50, int(value))
            except ValueError:
                pass

    def get_parameter(self, key: str) -> str:
        if key == "sampleRate":
            return str(self.sample_rate)
        if key == "channels":
            return str(self.channel_count)
        if key == "bufferSize":
            return str(self.buffer_size)
        if key == "maxBufferMs":
            return str(self.max_buffer_ms)
        if key == "deviceName":
            return self.device_name or ""
        if key == "deviceQuery":
            return self.device_query or ""
        if key == "deviceIndex":
            return "" if self.device_index is None else str(self.device_index)
        if key == "underruns":
            return str(self._underrun_events)
        if key == "underrunFrames":
            return str(self._underrun_frames)
        if key == "deviceOpenFailures":
            return str(self._device_open_failures)
        if key == "overflows":
            if not self._ring:
                return "0"
            return str(self._ring.get_stats().overflow_events)
        if key == "droppedFrames":
            if not self._ring:
                return "0"
            return str(self._ring.get_stats().dropped_frames)
        return ""

    def set_audio_callback(self, callback: AudioSourceCallback):
        self.callback = callback

    def read_audio(self, buffer: AudioBuffer) -> bool:
        if self.state != PluginState.RUNNING or not self._ring:
            buffer.clear()
            return False

        frames = buffer.get_frame_count()
        channels = buffer.get_channel_count()

        if channels != self.channel_count:
            temp = np.zeros((self.channel_count, frames), dtype=np.float32)
            filled = self._ring.read_into(temp)
            if filled < frames:
                temp[:, filled:] = 0
                self._track_underrun(frames - filled)
            self._mix_channels(temp, buffer.data)
            return True

        filled = self._ring.read_into(buffer.data)
        if filled < frames:
            buffer.data[:, filled:] = 0
            self._track_underrun(frames - filled)
        return True

    def _track_underrun(self, missing_frames: int) -> None:
        self._underrun_events += 1
        self._underrun_frames += int(max(0, missing_frames))
        if self._log_underrun_notice:
            print("[SoundCardMic] Underrun detected (returning silence)", flush=True)
            self._log_underrun_notice = False

    @staticmethod
    def _mix_channels(src: np.ndarray, dst: np.ndarray) -> None:
        src_ch = int(src.shape[0])
        dst_ch = int(dst.shape[0])

        if src_ch == dst_ch:
            np.copyto(dst, src)
            return

        if dst_ch == 1:
            dst[0] = src.mean(axis=0)
            return

        if src_ch == 1:
            dst[:] = np.repeat(src, dst_ch, axis=0)
            return

        mono = src.mean(axis=0)
        dst[:] = np.repeat(mono[np.newaxis, :], dst_ch, axis=0)

    def get_sample_rate(self) -> int:
        return self.sample_rate

    def get_channel_count(self) -> int:
        return self.channel_count

    def get_buffer_size(self) -> int:
        return self.buffer_size

    def set_sample_rate(self, sample_rate: int):
        if self.state in (PluginState.UNLOADED, PluginState.INITIALIZED):
            self.sample_rate = int(sample_rate)

    def set_channel_count(self, channels: int):
        if self.state in (PluginState.UNLOADED, PluginState.INITIALIZED):
            self.channel_count = max(1, int(channels))

    def set_buffer_size(self, samples: int):
        if self.state in (PluginState.UNLOADED, PluginState.INITIALIZED):
            self.buffer_size = max(64, int(samples))

    def set_device(self, query: str):
        if self.state in (PluginState.UNLOADED, PluginState.INITIALIZED):
            self.device_query = (query or "").strip()
            self.device_index = None
            try:
                mic = self._select_microphone()
                self.device_name = self._device_display_name(mic)
            except Exception:
                self.device_name = ""

    def set_device_index(self, index: int):
        if self.state in (PluginState.UNLOADED, PluginState.INITIALIZED):
            self.device_index = int(index)
            self.device_query = ""
            try:
                mic = self._select_microphone()
                self.device_name = self._device_display_name(mic)
            except Exception:
                self.device_name = ""

    def _capture_loop(self):
        try:
            mic = self._select_microphone()
            self.device_name = self._device_display_name(mic)

            try:
                recorder_cm = mic.recorder(samplerate=self.sample_rate, channels=self.channel_count)
            except TypeError:
                recorder_cm = mic.recorder(samplerate=self.sample_rate)

            with recorder_cm as recorder:
                self._ready_event.set()
                while not self._stop_event.is_set():
                    data = recorder.record(self.buffer_size)
                    if data is None:
                        continue

                    data = np.asarray(data, dtype=np.float32)
                    if data.ndim == 1:
                        data = data.reshape(-1, 1)

                    frames, in_ch = int(data.shape[0]), int(data.shape[1])
                    if frames <= 0:
                        continue

                    block = self._adapt_capture_block(data, in_ch=in_ch)
                    if self._ring:
                        self._ring.write(block)
        except Exception as e:
            self._device_open_failures += 1
            self._thread_error = str(e)
            self._ready_event.set()

    def _adapt_capture_block(self, data_frames_channels: np.ndarray, in_ch: int) -> np.ndarray:
        frames = int(data_frames_channels.shape[0])
        out_ch = int(self.channel_count)

        if in_ch == out_ch:
            return data_frames_channels.T.copy()

        if out_ch == 1:
            mono = data_frames_channels.mean(axis=1, dtype=np.float32)
            return mono.reshape(1, frames)

        if in_ch == 1:
            mono = data_frames_channels[:, 0].astype(np.float32, copy=False)
            return np.repeat(mono.reshape(1, frames), out_ch, axis=0)

        mono = data_frames_channels.mean(axis=1, dtype=np.float32)
        return np.repeat(mono.reshape(1, frames), out_ch, axis=0)

    @staticmethod
    def _device_display_name(device) -> str:
        name = getattr(device, "name", None)
        if isinstance(name, str) and name.strip():
            return name
        return str(device)

    def _select_microphone(self):
        if not SOUNDCARD_AVAILABLE:
            raise RuntimeError("soundcard library not available")

        microphones = list(sc.all_microphones())

        if self.device_index is not None:
            if 0 <= self.device_index < len(microphones):
                return microphones[self.device_index]
            raise ValueError(f"deviceIndex out of range (0..{max(0, len(microphones)-1)})")

        query = (self.device_query or "").strip().lower()
        if query:
            for mic in microphones:
                if query in self._device_display_name(mic).lower():
                    return mic
            raise ValueError(f"No microphone matched device='{self.device_query}'")

        try:
            return sc.default_microphone()
        except Exception:
            if microphones:
                return microphones[0]
            raise RuntimeError("No microphones found")


def create_plugin():
    return SoundCardMicrophonePlugin()
