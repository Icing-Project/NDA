# Prompt: Plan + Implement `soundcard`-based Cross‑OS Python Audio Plugins (NDA)

You are **GPT‑5.2 Extra High** acting as an agentic coding assistant inside the NDA repository.

Your mission: **rapidly deliver a real, working set of Python audio I/O plugins** based on the Python `soundcard` library, targeting **Windows + Linux** (macOS optional if feasible), with robust behavior under NDA’s real‑time pipeline.

This is not just ideation: produce a **careful, holistic plan** and then (unless blocked) **implement the plan** as a patch in this repo, with clear manual verification steps.

---

## Repository Context (Do Not Assume External Knowledge)

- Repo root contains:
  - Python plugins under `plugins_py/`
  - Python plugin API in `plugins_py/base_plugin.py`
  - C++ embedded Python bridge in `src/plugins/PythonPluginBridge.cpp` and plugin discovery in `src/plugins/PluginManager.cpp`
  - Processing pipeline in `src/core/ProcessingPipeline.cpp`
- NDA loads Python plugins by importing each `*.py` module and calling `create_plugin()`.
- The runtime pipeline calls:
  - `AudioSourcePlugin.read_audio(buffer)` once per frame
  - optional `AudioProcessorPlugin.process_audio(buffer)`
  - `AudioSinkPlugin.write_audio(buffer)` once per frame
  - backpressure is handled using `AudioSinkPlugin.get_available_space()`
- The pipeline uses **fixed frame sizes** (typically ~256–512 frames), and expects reasonable real‑time behavior.

### Existing Python plugin patterns to follow

- Look at existing source/sink plugins:
  - `plugins_py/sounddevice_microphone.py`
  - `plugins_py/sounddevice_speaker.py`
  - `plugins_py/pulseaudio_microphone.py`
  - `plugins_py/pulseaudio_speaker.py`
- Maintain the same interface style (methods, naming, state transitions, logging style).

---

## Goal

Create **new Python plugins** using the `soundcard` library that are:

1. **Cross‑OS**:
   - Windows: WASAPI via `soundcard`
   - Linux: PulseAudio via `soundcard` (works with PipeWire when PulseAudio compatibility is enabled)
2. **More reliable than `sounddevice`** under NDA’s pipeline pacing (fewer underruns/overflows, more stable timing).
3. **Fast to integrate**: minimal dependencies, clear install instructions, good failure messages.

---

## Required Deliverables

### 1) New plugins (at least)

- `plugins_py/soundcard_microphone.py` (Audio Source)
- `plugins_py/soundcard_speaker.py` (Audio Sink)

Each must:
- Implement `initialize`, `shutdown`, `start`, `stop`, `get_info`, `set_parameter`, `get_parameter`, and the required I/O methods (`read_audio` or `write_audio`).
- Provide meaningful `PluginInfo` strings (name, description, version).
- Use **numpy float32** consistently.

### 2) Dependency + docs updates

- Add dependency to `requirements.txt` (and `pyproject.toml` if that is used for Python plugin deps in this repo).
- Update `plugins_py/README.md` to list these plugins and document installation per OS.
- If there is a plugin test runner script (e.g. `plugins_py/test_plugins.py`), either update it or add a minimal sanity script that can:
  - list devices
  - capture a few seconds and/or play a short sine burst

### 3) Manual verification steps (must be practical)

Provide step‑by‑step instructions to verify:
- Plugin appears in NDA UI
- Can start a pipeline using the new source/sink
- Logs show stable operation (and any overruns/underruns counters)

---

## Key Engineering Constraints (Must Respect)

### A) NDA Pipeline behavior

Read `src/core/ProcessingPipeline.cpp` and plan around:
- The pipeline loops at real‑time cadence and calls `readAudio` / `writeAudio` per frame.
- `read_audio` should ideally be **non‑blocking** or “bounded blocking” (very short, consistent), otherwise NDA drifts.
- `write_audio` should ideally **not block**. Use queueing + a playback thread; expose backpressure via `get_available_space()`.

### B) Threading model + safety

- NDA will call plugin methods from its processing thread.
- For best stability, implement:
  - A capture thread (for microphone) that continuously records and fills a ring buffer.
  - A playback thread (for speaker) that drains a ring buffer and calls `soundcard` playback APIs.
- Use thread‑safe data structures (`queue.Queue`, `collections.deque` + locks) and keep memory bounded.

### C) Buffering + latency bounds

Design explicit buffering policy:
- Bounded latency: e.g. keep at most ~100–250ms of audio buffered.
- Underflow: return silence / “no data” with logging and counters.
- Overflow: drop oldest or newest deterministically (choose and justify), with counters and occasional logs.

### D) Device selection UX

Support selecting devices without editing code:
- `set_parameter("device", "...")` should accept a **substring match** on device name.
- Also accept `set_parameter("deviceIndex", "N")` as a fallback.
- Default should be “system default device” if not set.
- Expose `deviceName` in `get_parameter`.

### E) Sample rate + channels

Be explicit about how sample rate works with `soundcard`:
- Decide whether the plugin:
  1) forces 48kHz internally (preferred), or
  2) uses device default and reports it.
- Respect `set_sample_rate` and `set_channel_count` as much as the library allows.
- If `soundcard` cannot truly honor requested sample rate, document behavior and keep NDA’s resampler integration in mind.

### F) Failure behavior

If `soundcard` is missing or backend unavailable:
- `initialize()` must print a **clear actionable message** and return `False`.
- Avoid stack traces by default; only print tracebacks when helpful (or behind a verbose flag).

---

## “Act Fast” Requirements

Prioritize shipping something that works end‑to‑end:

- Minimal feature surface first (default devices, 48kHz mono/stereo, fixed buffer size).
- Then add device selection and advanced options.
- Avoid big refactors; keep changes localized to `plugins_py/` and dependency docs.

If a feature risks delaying delivery (e.g., perfect device enumeration UI), defer it with a clear TODO.

---

## Planning Output Requirements (What You Must Produce First)

Before coding, produce:

1. A short **design summary** (how capture/playback works, buffer strategy, how you handle sample rate/channels).
2. A **risk list** with mitigations (backend availability, latency spikes, device mismatch, pipewire‑pulse).
3. A step‑by‑step **implementation plan** with checkpoints and quick validation at each step.
4. A **manual test plan** for Windows + Linux.

Then implement the patch.

---

## Implementation Notes (Strong Recommendations)

- Implement a reusable ring buffer helper inside each plugin (or a small shared helper module under `plugins_py/` if appropriate).
- Keep logs concise but informative. Add counters:
  - capture underruns / overflows
  - playback underruns / queue full drops
  - device open failures
- Use `flush=True` in prints to ensure logs appear in embedded mode.

---

## Acceptance Criteria (Done Means Done)

- NDA starts and auto‑loads the plugins.
- Selecting `SoundCard Microphone` + `SoundCard Speaker` in a pipeline can run for ≥ 60 seconds without obvious drift/crackling under normal system load.
- On Linux, the plugins work when PulseAudio is available (or pipewire‑pulse).
- On Windows, the plugins can open default devices reliably.
- Error messages are actionable when the environment is missing dependencies/backends.

---

## Start Here (Concrete Pointers You Should Read in Repo)

Open and use these files as ground truth:
- `plugins_py/base_plugin.py`
- `plugins_py/sounddevice_microphone.py`
- `plugins_py/sounddevice_speaker.py`
- `src/core/ProcessingPipeline.cpp`
- `plugins_py/README.md`
- `requirements.txt`

---

## Output Format

1) Plan (with sections described above)  
2) Patch / file list of changes (what you edited/added)  
3) Manual repro steps for Windows + Linux  

Do not hand‑wave: when you suggest install steps, make them concrete (commands + package names), and note any OS caveats.

