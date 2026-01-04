#!/usr/bin/env python3
"""
Minimal sanity runner for NDA Python plugins.

Default behavior (no args):
- Discover plugin modules in this folder
- Run a short sine_wave_source -> null_sink loop
- Run a short sine_wave_source -> wav_file_sink recording

Optional:
- --list-devices: list soundcard microphones/speakers
- --soundcard-sine: play a short sine burst via SoundCardSpeakerPlugin
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def _ensure_plugin_dir_on_path() -> Path:
    plugin_dir = Path(__file__).resolve().parent
    if str(plugin_dir) not in sys.path:
        sys.path.insert(0, str(plugin_dir))
    return plugin_dir


def discover_plugins(plugin_dir: Path) -> list[str]:
    plugins: list[str] = []
    for p in plugin_dir.glob("*.py"):
        name = p.name
        if name in {"base_plugin.py", "plugin_loader.py", "test_plugins.py", "__init__.py"}:
            continue
        if name.startswith("setup_"):
            continue
        plugins.append(p.stem)
    return sorted(plugins)


def list_soundcard_devices() -> int:
    try:
        import soundcard as sc  # type: ignore
    except Exception as e:
        print(f"[test_plugins] soundcard not available: {e}", flush=True)
        return 1

    try:
        mics = list(sc.all_microphones())
        spks = list(sc.all_speakers())
        print("[test_plugins] soundcard devices:", flush=True)
        print("  Microphones:", flush=True)
        for i, mic in enumerate(mics):
            name = getattr(mic, "name", str(mic))
            print(f"    [{i}] {name}", flush=True)
        print("  Speakers:", flush=True)
        for i, spk in enumerate(spks):
            name = getattr(spk, "name", str(spk))
            print(f"    [{i}] {name}", flush=True)

        try:
            print(f"  Default microphone: {getattr(sc.default_microphone(), 'name', sc.default_microphone())}", flush=True)
        except Exception:
            pass
        try:
            print(f"  Default speaker: {getattr(sc.default_speaker(), 'name', sc.default_speaker())}", flush=True)
        except Exception:
            pass

        return 0
    except Exception as e:
        print(f"[test_plugins] Failed to list devices: {e}", flush=True)
        return 2


def run_loop(source, sink, seconds: float, frames_per_buffer: int) -> bool:
    from base_plugin import AudioBuffer  # local import for path safety

    sr = int(source.get_sample_rate())
    channels = int(source.get_channel_count())

    sink.set_sample_rate(sr)
    sink.set_channel_count(channels)
    sink.set_buffer_size(frames_per_buffer)
    if hasattr(source, "set_buffer_size"):
        try:
            source.set_buffer_size(frames_per_buffer)
        except Exception:
            pass

    if not source.initialize():
        print("[test_plugins] Source initialize() failed", flush=True)
        return False
    if not sink.initialize():
        print("[test_plugins] Sink initialize() failed", flush=True)
        return False
    if not source.start():
        print("[test_plugins] Source start() failed", flush=True)
        return False
    if not sink.start():
        print("[test_plugins] Sink start() failed", flush=True)
        return False

    buf = AudioBuffer(channels, frames_per_buffer)
    frame_duration = frames_per_buffer / float(sr)
    end = time.time() + float(seconds)

    ok = True
    while time.time() < end:
        if not source.read_audio(buf):
            buf.clear()
        if not sink.write_audio(buf):
            ok = False
        time.sleep(frame_duration)

    try:
        source.stop()
    finally:
        sink.stop()

    try:
        source.shutdown()
    finally:
        sink.shutdown()

    return ok


def play_soundcard_sine(seconds: float, freq: float, frames_per_buffer: int) -> int:
    try:
        from soundcard_speaker import SoundCardSpeakerPlugin
    except Exception as e:
        print(f"[test_plugins] Failed to import soundcard_speaker: {e}", flush=True)
        return 2

    from base_plugin import AudioBuffer

    sink = SoundCardSpeakerPlugin()
    sink.set_sample_rate(48000)
    sink.set_channel_count(2)
    sink.set_buffer_size(frames_per_buffer)

    if not sink.initialize():
        return 3
    if not sink.start():
        return 4

    sr = sink.get_sample_rate()
    ch = sink.get_channel_count()
    buf = AudioBuffer(ch, frames_per_buffer)
    phase = 0.0
    phase_inc = 2.0 * np.pi * float(freq) / float(sr)
    frame_duration = frames_per_buffer / float(sr)
    end = time.time() + float(seconds)

    while time.time() < end:
        phases = phase + np.arange(frames_per_buffer) * phase_inc
        samples = 0.2 * np.sin(phases).astype(np.float32)
        for c in range(ch):
            buf.data[c] = samples
        phase = float(phases[-1] + phase_inc)
        if phase >= 2.0 * np.pi:
            phase -= 2.0 * np.pi

        sink.write_audio(buf)
        time.sleep(frame_duration)

    sink.stop()
    sink.shutdown()
    return 0


def main() -> int:
    plugin_dir = _ensure_plugin_dir_on_path()

    parser = argparse.ArgumentParser()
    parser.add_argument("--list-devices", action="store_true", help="List soundcard devices")
    parser.add_argument("--soundcard-sine", action="store_true", help="Play sine burst via SoundCardSpeakerPlugin")
    parser.add_argument("--seconds", type=float, default=2.0, help="Duration for optional audio tests")
    parser.add_argument("--frames", type=int, default=512, help="Frames per buffer for tests")
    args = parser.parse_args()

    plugins = discover_plugins(plugin_dir)
    print("[test_plugins] Discovered plugins:", flush=True)
    for name in plugins:
        print(f"  - {name}", flush=True)

    if args.list_devices:
        rc = list_soundcard_devices()
        if rc != 0:
            return rc

    # Always run safe, non-hardware tests by default.
    try:
        from sine_wave_source import SineWaveSourcePlugin
        from null_sink import NullSinkPlugin
        from wav_file_sink import WavFileSinkPlugin
    except Exception as e:
        print(f"[test_plugins] Failed to import core test plugins: {e}", flush=True)
        return 10

    print("[test_plugins] Running sine_wave_source -> null_sink ...", flush=True)
    if not run_loop(SineWaveSourcePlugin(), NullSinkPlugin(), seconds=2.0, frames_per_buffer=args.frames):
        print("[test_plugins] null sink test FAILED", flush=True)
        return 11

    print("[test_plugins] Running sine_wave_source -> wav_file_sink ...", flush=True)
    wav_path = plugin_dir / "test_recording.wav"
    wav = WavFileSinkPlugin()
    wav.set_parameter("filename", str(wav_path))
    if not run_loop(SineWaveSourcePlugin(), wav, seconds=1.0, frames_per_buffer=args.frames):
        print("[test_plugins] wav file test FAILED", flush=True)
        return 12
    print(f"[test_plugins] Wrote {wav_path.name}", flush=True)

    if args.soundcard_sine:
        print("[test_plugins] Playing soundcard sine burst ...", flush=True)
        rc = play_soundcard_sine(seconds=args.seconds, freq=440.0, frames_per_buffer=args.frames)
        if rc != 0:
            return rc

    print("[test_plugins] OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
