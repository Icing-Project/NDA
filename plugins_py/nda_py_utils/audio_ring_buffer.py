from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np


@dataclass
class RingBufferStats:
    dropped_frames: int = 0
    overflow_events: int = 0


class AudioRingBuffer:
    """
    Thread-safe circular buffer for float32 audio.

    Storage shape is (channels, capacity_frames).
    """

    def __init__(self, channels: int, capacity_frames: int):
        self._lock = threading.Lock()
        self._stats = RingBufferStats()
        self._configure(channels=channels, capacity_frames=capacity_frames)

    def _configure(self, channels: int, capacity_frames: int) -> None:
        channels = int(max(1, channels))
        capacity_frames = int(max(1, capacity_frames))

        self._channels = channels
        self._capacity = capacity_frames
        self._buffer = np.zeros((channels, capacity_frames), dtype=np.float32)
        self._read_pos = 0
        self._write_pos = 0
        self._size = 0

    def reset(self, channels: int, capacity_frames: int) -> None:
        with self._lock:
            self._stats = RingBufferStats()
            self._configure(channels=channels, capacity_frames=capacity_frames)

    def clear(self) -> None:
        with self._lock:
            self._read_pos = 0
            self._write_pos = 0
            self._size = 0

    def get_channels(self) -> int:
        return self._channels

    def get_capacity_frames(self) -> int:
        return self._capacity

    def get_available_frames(self) -> int:
        with self._lock:
            return self._size

    def get_free_frames(self) -> int:
        with self._lock:
            return self._capacity - self._size

    def get_stats(self) -> RingBufferStats:
        with self._lock:
            return RingBufferStats(
                dropped_frames=self._stats.dropped_frames,
                overflow_events=self._stats.overflow_events,
            )

    def write(self, block: np.ndarray) -> None:
        """
        Write a block of shape (channels, frames) into the buffer.
        Drops oldest frames if needed to keep buffer bounded.
        """
        if block.ndim != 2:
            raise ValueError("block must be 2D (channels, frames)")

        channels, frames = int(block.shape[0]), int(block.shape[1])
        if channels != self._channels:
            raise ValueError(f"channel mismatch: buffer={self._channels} block={channels}")
        if frames <= 0:
            return

        if block.dtype != np.float32:
            block = block.astype(np.float32, copy=False)

        with self._lock:
            if frames >= self._capacity:
                self._buffer[:, :] = block[:, -self._capacity :]
                self._read_pos = 0
                self._write_pos = 0
                self._size = self._capacity
                self._stats.dropped_frames += frames - self._capacity
                self._stats.overflow_events += 1
                return

            overflow = max(0, self._size + frames - self._capacity)
            if overflow:
                self._read_pos = (self._read_pos + overflow) % self._capacity
                self._size -= overflow
                self._stats.dropped_frames += overflow
                self._stats.overflow_events += 1

            end_space = self._capacity - self._write_pos
            if frames <= end_space:
                self._buffer[:, self._write_pos : self._write_pos + frames] = block
            else:
                self._buffer[:, self._write_pos :] = block[:, :end_space]
                self._buffer[:, : frames - end_space] = block[:, end_space:]

            self._write_pos = (self._write_pos + frames) % self._capacity
            self._size += frames

    def read_into(self, out: np.ndarray) -> int:
        """
        Read up to out.shape[1] frames into out (shape (channels, frames)).
        Returns number of frames actually read. Caller can zero-pad remainder.
        """
        if out.ndim != 2:
            raise ValueError("out must be 2D (channels, frames)")
        channels, frames = int(out.shape[0]), int(out.shape[1])
        if channels != self._channels:
            raise ValueError(f"channel mismatch: buffer={self._channels} out={channels}")
        if frames <= 0:
            return 0

        with self._lock:
            available = min(frames, self._size)
            if available <= 0:
                return 0

            end_space = self._capacity - self._read_pos
            if available <= end_space:
                out[:, :available] = self._buffer[:, self._read_pos : self._read_pos + available]
            else:
                out[:, :end_space] = self._buffer[:, self._read_pos :]
                out[:, end_space:available] = self._buffer[:, : available - end_space]

            self._read_pos = (self._read_pos + available) % self._capacity
            self._size -= available
            return available
