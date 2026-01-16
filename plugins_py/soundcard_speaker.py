"""
SoundCard Speaker Plugin - Python Implementation
Plays audio through system speakers using the cross-platform `soundcard` library.

Design:
- `write_audio()` queues audio into a bounded ring buffer (non-blocking).
- A playback thread drains the ring buffer and calls soundcard's blocking playback API.
- Backpressure is exposed via `get_available_space()` so NDA can avoid overruns.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional

try:
    import soundcard as sc

    SOUNDCARD_AVAILABLE = True
except ImportError:
    SOUNDCARD_AVAILABLE = False

import numpy as np

from base_plugin import AudioBuffer, AudioSinkPlugin, PluginInfo, PluginState, PluginType

from nda_py_utils.audio_ring_buffer import AudioRingBuffer


def _is_truthy_env(name: str) -> bool:
    value = os.getenv(name)
    if not value:
        return False
    return value.strip().lower() in ("1", "true", "yes", "on")


def _read_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


class SoundCardSpeakerPlugin(AudioSinkPlugin):
    """Plays audio through speakers via soundcard (WASAPI/PulseAudio)."""

    def __init__(self):
        super().__init__()
        self.sample_rate = 48000
        self.channel_count = 2
        self.buffer_size = 512

        self.device_query: str = ""
        self.device_index: Optional[int] = None
        self.device_name: str = ""

        self.max_buffer_ms = 200
        # WASAPI/PulseAudio client buffer (via soundcard's `blocksize`).
        # Defaulting to a modest value avoids `player.play()` blocking on tiny device periods.
        self.player_buffer_ms = 50

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread_error: Optional[str] = None

        self._ring: Optional[AudioRingBuffer] = None

        self._underrun_events = 0
        self._underrun_frames = 0
        self._device_open_failures = 0
        self._log_underrun_notice = True

        self._profile_enabled = _is_truthy_env("NDA_PROFILE") or _is_truthy_env("NDA_PROFILE_PYPLUGINS")
        self._profile_interval_s = max(0.1, _read_env_int("NDA_PROFILE_PYPLUGINS_INTERVAL_MS", 1000) / 1000.0)
        self._profile_last_log = time.monotonic()

        self._profile_write_calls = 0
        self._profile_write_us_total = 0
        self._profile_write_us_max = 0
        self._profile_frames_written = 0

        self._profile_play_calls = 0
        self._profile_play_us_total = 0
        self._profile_play_us_max = 0
        self._profile_play_last_t = None
        self._profile_play_dt_us_total = 0
        self._profile_play_dt_us_count = 0
        self._profile_play_dt_us_min = None
        self._profile_play_dt_us_max = 0
        self._profile_play_stall_events = 0
        self._profile_play_resync_events = 0
        self._profile_frames_played = 0
        self._profile_ring_fill_samples = 0
        self._profile_ring_fill_total = 0
        self._profile_ring_fill_min = None
        self._profile_ring_fill_max = 0

        self._active_player_blocksize = None
        self._active_player_deviceperiod = None
        self._active_player_buffersize = None
        self._active_play_frames = None

    def initialize(self) -> bool:
        if not SOUNDCARD_AVAILABLE:
            print(
                "[SoundCardSpeaker] soundcard not available. Install with: pip install soundcard",
                flush=True,
            )
            self.state = PluginState.ERROR
            return False

        try:
            spk = self._select_speaker()
            self.device_name = self._device_display_name(spk)
            self.state = PluginState.INITIALIZED
            return True
        except Exception as e:
            print(f"[SoundCardSpeaker] Failed to initialize: {e}", flush=True)
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
        self._profile_last_log = time.monotonic()

        self._profile_write_calls = 0
        self._profile_write_us_total = 0
        self._profile_write_us_max = 0
        self._profile_frames_written = 0
        self._profile_play_calls = 0
        self._profile_play_us_total = 0
        self._profile_play_us_max = 0
        self._profile_play_last_t = None
        self._profile_play_dt_us_total = 0
        self._profile_play_dt_us_count = 0
        self._profile_play_dt_us_min = None
        self._profile_play_dt_us_max = 0
        self._profile_play_stall_events = 0
        self._profile_play_resync_events = 0
        self._profile_frames_played = 0
        self._profile_ring_fill_samples = 0
        self._profile_ring_fill_total = 0
        self._profile_ring_fill_min = None
        self._profile_ring_fill_max = 0

        self._active_player_blocksize = None
        self._active_player_deviceperiod = None
        self._active_player_buffersize = None
        self._active_play_frames = None

        capacity_frames = max(1, int(self.sample_rate * (self.max_buffer_ms / 1000.0)))
        self._ring = AudioRingBuffer(channels=self.channel_count, capacity_frames=capacity_frames)

        self._thread = threading.Thread(target=self._playback_loop, name="SoundCardSpeakerPlayback", daemon=True)
        self._thread.start()

        if not self._ready_event.wait(timeout=2.0):
            print("[SoundCardSpeaker] Playback thread did not become ready", flush=True)
            self._stop_event.set()
            self.state = PluginState.ERROR
            return False

        if self._thread_error:
            print(f"[SoundCardSpeaker] Failed to start: {self._thread_error}", flush=True)
            self.state = PluginState.ERROR
            return False

        self.state = PluginState.RUNNING
        print(
            f"[SoundCardSpeaker] Started - {self.sample_rate}Hz, {self.channel_count}ch, "
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
            print(f"[SoundCardSpeaker] Error in stop(): {e}", flush=True)
            self.state = PluginState.INITIALIZED

    def get_info(self) -> PluginInfo:
        return PluginInfo(
            name="SoundCard Speaker Output",
            version="1.0.0",
            author="Icing Project",
            description="Plays audio using the soundcard library (WASAPI/PulseAudio)",
            plugin_type=PluginType.AUDIO_SINK,
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
        elif key == "playerBufferMs":
            try:
                self.player_buffer_ms = max(0, int(value))
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
        if key == "playerBufferMs":
            return str(self.player_buffer_ms)
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

    def write_audio(self, buffer: AudioBuffer) -> bool:
        if self.state != PluginState.RUNNING or not self._ring:
            return False

        t0 = time.perf_counter() if self._profile_enabled else 0.0

        frames = buffer.get_frame_count()
        if frames <= 0:
            return True

        in_ch = buffer.get_channel_count()

        block = self._adapt_output_block(buffer.data, in_ch=in_ch)
        try:
            self._ring.write(block)
            if self._profile_enabled:
                dt_us = int((time.perf_counter() - t0) * 1_000_000.0)
                self._profile_write_calls += 1
                self._profile_write_us_total += dt_us
                self._profile_write_us_max = max(self._profile_write_us_max, dt_us)
                self._profile_frames_written += int(frames)
            return True
        except Exception as e:
            print(f"[SoundCardSpeaker] Write error: {e}", flush=True)
            return False

    def _track_underrun(self, missing_frames: int) -> None:
        self._underrun_events += 1
        self._underrun_frames += int(max(0, missing_frames))
        if self._log_underrun_notice:
            print("[SoundCardSpeaker] Underrun detected (playing silence)", flush=True)
            self._log_underrun_notice = False

    def get_sample_rate(self) -> int:
        return self.sample_rate

    def get_channel_count(self) -> int:
        return self.channel_count

    def set_sample_rate(self, sample_rate: int):
        if self.state in (PluginState.UNLOADED, PluginState.INITIALIZED):
            self.sample_rate = int(sample_rate)

    def set_channel_count(self, channels: int):
        if self.state in (PluginState.UNLOADED, PluginState.INITIALIZED):
            self.channel_count = max(1, int(channels))

    def get_buffer_size(self) -> int:
        return self.buffer_size

    def set_buffer_size(self, samples: int):
        if self.state in (PluginState.UNLOADED, PluginState.INITIALIZED):
            self.buffer_size = max(64, int(samples))

    def get_available_space(self) -> int:
        if self.state != PluginState.RUNNING or not self._ring:
            return self.buffer_size
        return self._ring.get_free_frames()

    def _maybe_log_profile(self) -> None:
        if not self._profile_enabled:
            return

        now = time.monotonic()
        dt_s = now - self._profile_last_log
        if dt_s < self._profile_interval_s:
            return

        ring_fill_avg = (
            (self._profile_ring_fill_total / self._profile_ring_fill_samples)
            if self._profile_ring_fill_samples
            else 0.0
        )

        play_frames = int(self._active_play_frames or self.buffer_size)
        expected_play_us = int(max(1.0, (play_frames / max(1, self.sample_rate)) * 1_000_000.0))

        ring_stats = self._ring.get_stats() if self._ring else None
        ring_capacity = self._ring.get_capacity_frames() if self._ring else 0

        write_avg_us = (self._profile_write_us_total / self._profile_write_calls) if self._profile_write_calls else 0.0
        play_avg_us = (self._profile_play_us_total / self._profile_play_calls) if self._profile_play_calls else 0.0
        play_hz = (self._profile_play_calls / dt_s) if dt_s > 0 else 0.0
        play_dt_avg_us = (
            (self._profile_play_dt_us_total / self._profile_play_dt_us_count) if self._profile_play_dt_us_count else 0.0
        )
        expected_play_hz = (float(self.sample_rate) / float(play_frames)) if play_frames > 0 else 0.0

        print(
            "[SoundCardSpeakerProfile]"
            f" dt={dt_s * 1000.0:.1f}ms"
            f" sr={self.sample_rate}"
            f" ch={self.channel_count}"
            f" buf={self.buffer_size}"
            f" playerBufMs={self.player_buffer_ms}"
            f" playerBlock={self._active_player_blocksize or 0}"
            f" playFrames={play_frames}"
            f" devPeriod={self._active_player_deviceperiod or ''}"
            f" devBuf={self._active_player_buffersize or ''}"
            f" ringFill(min/avg/max)={self._profile_ring_fill_min or 0}/{ring_fill_avg:.1f}/{self._profile_ring_fill_max}"
            f" ringCap={ring_capacity}"
            f" hz(play/exp)={play_hz:.2f}/{expected_play_hz:.2f}"
            f" playDt(min/avg/maxUs)={self._profile_play_dt_us_min or 0}/{play_dt_avg_us:.0f}/{self._profile_play_dt_us_max}"
            f" play(avgUs/maxUs/expectedUs)={play_avg_us:.0f}/{self._profile_play_us_max}/{expected_play_us}"
            f" playDiag(stall/resync)={self._profile_play_stall_events}/{self._profile_play_resync_events}"
            f" write(avgUs/maxUs)={write_avg_us:.0f}/{self._profile_write_us_max}"
            f" frames(written/played)={self._profile_frames_written}/{self._profile_frames_played}"
            f" underruns(events/frames)={self._underrun_events}/{self._underrun_frames}"
            f" overflows(events/frames)={getattr(ring_stats, 'overflow_events', 0)}/{getattr(ring_stats, 'dropped_frames', 0)}",
            flush=True,
        )

        self._profile_last_log = now
        self._profile_write_calls = 0
        self._profile_write_us_total = 0
        self._profile_write_us_max = 0
        self._profile_frames_written = 0

        self._profile_play_calls = 0
        self._profile_play_us_total = 0
        self._profile_play_us_max = 0
        self._profile_play_last_t = None
        self._profile_play_dt_us_total = 0
        self._profile_play_dt_us_count = 0
        self._profile_play_dt_us_min = None
        self._profile_play_dt_us_max = 0
        self._profile_play_stall_events = 0
        self._profile_play_resync_events = 0
        self._profile_frames_played = 0
        self._profile_ring_fill_samples = 0
        self._profile_ring_fill_total = 0
        self._profile_ring_fill_min = None
        self._profile_ring_fill_max = 0

    def set_device(self, query: str):
        if self.state in (PluginState.UNLOADED, PluginState.INITIALIZED):
            self.device_query = (query or "").strip()
            self.device_index = None
            try:
                spk = self._select_speaker()
                self.device_name = self._device_display_name(spk)
            except Exception:
                self.device_name = ""

    def set_device_index(self, index: int):
        if self.state in (PluginState.UNLOADED, PluginState.INITIALIZED):
            self.device_index = int(index)
            self.device_query = ""
            try:
                spk = self._select_speaker()
                self.device_name = self._device_display_name(spk)
            except Exception:
                self.device_name = ""

    def _playback_loop(self):
        try:
            spk = self._select_speaker()
            self.device_name = self._device_display_name(spk)

            player_blocksize = None
            if self.player_buffer_ms > 0:
                player_blocksize = max(
                    int(self.buffer_size),
                    int(self.sample_rate * (self.player_buffer_ms / 1000.0)),
                )

            player_cm = None
            open_error = None
            try:
                if player_blocksize is not None:
                    player_cm = spk.player(
                        samplerate=self.sample_rate,
                        channels=self.channel_count,
                        blocksize=player_blocksize,
                    )
                else:
                    player_cm = spk.player(samplerate=self.sample_rate, channels=self.channel_count)
            except TypeError:
                try:
                    if player_blocksize is not None:
                        player_cm = spk.player(samplerate=self.sample_rate, blocksize=player_blocksize)
                    else:
                        player_cm = spk.player(samplerate=self.sample_rate)
                except TypeError:
                    player_cm = spk.player(samplerate=self.sample_rate)
            except Exception as e:
                open_error = e

            if player_cm is None and self.channel_count != 1:
                try:
                    self.channel_count = 1
                    if self._ring:
                        capacity_frames = max(1, int(self.sample_rate * (self.max_buffer_ms / 1000.0)))
                        self._ring.reset(channels=1, capacity_frames=capacity_frames)
                    if player_blocksize is not None:
                        player_cm = spk.player(samplerate=self.sample_rate, channels=1, blocksize=player_blocksize)
                    else:
                        player_cm = spk.player(samplerate=self.sample_rate, channels=1)
                except TypeError:
                    try:
                        if player_blocksize is not None:
                            player_cm = spk.player(samplerate=self.sample_rate, blocksize=player_blocksize)
                        else:
                            player_cm = spk.player(samplerate=self.sample_rate)
                    except TypeError:
                        player_cm = spk.player(samplerate=self.sample_rate)
                except Exception as e:
                    if open_error is not None:
                        raise RuntimeError(f"{open_error}; fallback failed: {e}") from e
                    raise

            with player_cm as player:
                self._active_player_blocksize = player_blocksize
                self._active_player_deviceperiod = (
                    getattr(player, "deviceperiod", None)
                    or getattr(player, "device_period", None)
                    or getattr(player, "_device_period", None)
                )
                self._active_player_buffersize = (
                    getattr(player, "buffersize", None)
                    or getattr(player, "buffer_size", None)
                    or getattr(player, "_buffer_size", None)
                )
                play_frames = int(player_blocksize or self.buffer_size)
                play_frames = max(int(self.buffer_size), play_frames)
                self._active_play_frames = play_frames
                self._ready_event.set()

                play_ch = int(self.channel_count)
                out_channels_frames = np.zeros((play_ch, play_frames), dtype=np.float32)
                out_frames_channels = np.zeros((play_frames, play_ch), dtype=np.float32)
                silence = out_frames_channels

                block_s = float(play_frames) / float(max(1, self.sample_rate))
                next_play_t = time.monotonic()

                while not self._stop_event.is_set():
                    now_t = time.monotonic()
                    sleep_s = next_play_t - now_t
                    if sleep_s > 0:
                        if self._stop_event.wait(timeout=sleep_s):
                            break
                    else:
                        # If we fell far behind (e.g. device stall), resync to avoid bursty catch-up.
                        if (-sleep_s) > 0.25:
                            next_play_t = now_t
                            if self._profile_enabled:
                                self._profile_play_resync_events += 1

                    if not self._ring:
                        player.play(silence)
                        next_play_t += block_s
                        continue

                    if self._profile_enabled:
                        try:
                            ring_fill = self._ring.get_available_frames()
                            self._profile_ring_fill_samples += 1
                            self._profile_ring_fill_total += ring_fill
                            if self._profile_ring_fill_min is None:
                                self._profile_ring_fill_min = ring_fill
                            else:
                                self._profile_ring_fill_min = min(self._profile_ring_fill_min, ring_fill)
                            self._profile_ring_fill_max = max(self._profile_ring_fill_max, ring_fill)
                        except Exception:
                            pass

                    out_channels_frames.fill(0)
                    filled = self._ring.read_into(out_channels_frames)
                    if filled < play_frames:
                        self._track_underrun(play_frames - filled)
                    if self._profile_enabled:
                        self._profile_frames_played += int(filled)

                    # soundcard expects (frames, channels) contiguous
                    out_frames_channels[:, :] = out_channels_frames.T
                    play_start = time.perf_counter() if self._profile_enabled else 0.0
                    player.play(out_frames_channels)
                    if self._profile_enabled:
                        if self._profile_play_last_t is not None:
                            dt_us = int((play_start - self._profile_play_last_t) * 1_000_000.0)
                            self._profile_play_dt_us_total += dt_us
                            self._profile_play_dt_us_count += 1
                            if self._profile_play_dt_us_min is None:
                                self._profile_play_dt_us_min = dt_us
                            else:
                                self._profile_play_dt_us_min = min(self._profile_play_dt_us_min, dt_us)
                            self._profile_play_dt_us_max = max(self._profile_play_dt_us_max, dt_us)
                        self._profile_play_last_t = play_start

                        play_us = int((time.perf_counter() - play_start) * 1_000_000.0)
                        expected_play_us = int(
                            max(1.0, (play_frames / max(1, self.sample_rate)) * 1_000_000.0)
                        )
                        if play_us > (expected_play_us * 3):
                            self._profile_play_stall_events += 1
                        self._profile_play_calls += 1
                        self._profile_play_us_total += play_us
                        self._profile_play_us_max = max(self._profile_play_us_max, play_us)
                        self._maybe_log_profile()

                    next_play_t += block_s
        except Exception as e:
            self._device_open_failures += 1
            self._thread_error = str(e)
            self._ready_event.set()

    def _adapt_output_block(self, in_channels_frames: np.ndarray, in_ch: int) -> np.ndarray:
        frames = int(in_channels_frames.shape[1])
        out_ch = int(self.channel_count)

        if in_ch == out_ch:
            if in_channels_frames.dtype == np.float32:
                return in_channels_frames
            return in_channels_frames.astype(np.float32, copy=False)

        if out_ch == 1:
            mono = in_channels_frames.mean(axis=0, dtype=np.float32)
            return mono.reshape(1, frames)

        if in_ch == 1:
            mono = in_channels_frames[0].astype(np.float32, copy=False)
            return np.repeat(mono.reshape(1, frames), out_ch, axis=0)

        mono = in_channels_frames.mean(axis=0, dtype=np.float32)
        return np.repeat(mono.reshape(1, frames), out_ch, axis=0)

    @staticmethod
    def _device_display_name(device) -> str:
        name = getattr(device, "name", None)
        if isinstance(name, str) and name.strip():
            return name
        return str(device)

    def _select_speaker(self):
        if not SOUNDCARD_AVAILABLE:
            raise RuntimeError("soundcard library not available")

        speakers = list(sc.all_speakers())

        if self.device_index is not None:
            if 0 <= self.device_index < len(speakers):
                return speakers[self.device_index]
            raise ValueError(f"deviceIndex out of range (0..{max(0, len(speakers)-1)})")

        query = (self.device_query or "").strip().lower()
        if query:
            for spk in speakers:
                if query in self._device_display_name(spk).lower():
                    return spk
            raise ValueError(f"No speaker matched device='{self.device_query}'")

        try:
            return sc.default_speaker()
        except Exception:
            if speakers:
                return speakers[0]
            raise RuntimeError("No speakers found")


def create_plugin():
    return SoundCardSpeakerPlugin()
