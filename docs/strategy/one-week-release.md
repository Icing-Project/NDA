# One-Week Release Strategy: Stable Audio Bridge (Windows ⇄ AIOC)

**Status:** ACTIVE (release-triage mode)  
**Timebox:** 7 days  
**Owner:** Solo maintainer (AI-assisted), limited budget  

This document defines a short-term strategy to ship a usable build where the **core value works**:

- **TX:** Windows microphone → AIOC output
- **RX:** AIOC microphone → Windows output

It intentionally **does not remove** encryption or Python plugin work; it **pauses** it behind toggles/presets until the bridge is proven stable.

---

## 1) Definition of “Done” (Release Acceptance Criteria)

### Must-pass (hard gates)
- **No frequent dropouts**: continuous run of **20 minutes** with no repeating crackle/drop pattern.
- **No corruption**: audio is intelligible (no “robot”/bitcrush distortion), stable pitch, stable channel mapping.
- **Two bridges work** (TX and RX) with the target devices on Windows.
- **Failure is diagnosable**: logs/counters clearly indicate whether issues are capture underrun, render overrun, pacing drift, plugin failure, or device format mismatch.

### Allowed for the one-week release
- Higher latency than ideal (target <50ms, but stability first).
- Limited device matrix (documented “known-good” settings).
- Encryption and Python features present but **disabled by default**.

### Not allowed
- Shipping “sometimes works” audio with no telemetry explaining why it fails.
- Regressions in start/stop stability (no deadlocks, no hangs).

---

## 2) Scope Cuts (What We Stop Doing This Week)

This is how we buy certainty with limited time:

- **Freeze features**: no new UI features, no new plugin types, no new architecture refactors.
- **Pause Python plugins**:
  - Keep the Python bridge code and build option intact.
  - Default release build should run with Python support **OFF** unless/until the audio bridge is stable.
- **Pause encryption processors** in the default preset:
  - Keep encryptor/decryptor plugin infrastructure intact.
  - Default preset runs with **Processor = None** (or a vetted no-op passthrough).
- **Avoid resampling in the golden path**:
  - Prefer locking both sides to **48kHz** when possible.
  - If resampling is required for some devices, treat it as a controlled exception and measure its CPU/time cost.

---

## 3) Golden Path: “Bridge Mode” Preset

The release ships a single preset that we defend with testing and instrumentation.

### Bridge Mode (default)
- **TX pipeline:** Windows Mic (WASAPI) → *(no processor)* → AIOC Sink
- **RX pipeline:** AIOC Source → *(no processor)* → Windows Output (WASAPI)

### Operational constraints (week-release)
- Prefer **48kHz** end-to-end.
- Prefer a conservative **frame/buffer size** (stability over latency).
- If the pipeline supports negotiation, log the negotiated values at start.

### UX requirement
- “Bridge Mode” must be one click (or the simplest possible path) and must clearly show:
  - Current devices
  - Running state
  - Underrun/overrun counters
  - A visible “health” indicator (OK / Degraded / Failing)

---

## 4) Engineering Priorities (In Order)

### Priority 0: Make failures observable
Before changing algorithms, ensure we can answer:
- Is the audio thread missing deadlines?
- Is capture returning empty buffers?
- Is render rejecting writes due to padding/full buffer?
- Is there format mismatch (float/int16, channels)?
- Where is the time spent per frame (read/process/write/resample)?

Use the existing pipeline profiling/long-frame warnings where available and ensure the default logs are readable.

### Priority 1: Fix correctness issues in the hot path
Common “unusable audio” causes to eliminate first:
- Writing/reading the wrong sample format (float vs int16, `WAVEFORMATEXTENSIBLE` handling).
- Doing expensive or blocking work per frame (allocations, device queries, locks, conversions).
- Failing to treat “no data” / “silent” conditions correctly (returning garbage instead of silence).

### Priority 2: Stabilize real-time scheduling
The bridge must be robust under normal Windows scheduling jitter:
- If using `sleep_until` pacing, ensure short jitter does not create audible artifacts.
- Prefer device-driven pacing (event/callback timing) or buffering that smooths jitter.

### Priority 3: Only then, reintroduce features
After the bridge is stable, re-enable additional behaviors behind toggles:
- Resampling (measured and bounded)
- C++ processors
- Python processors (last)

---

## 5) Feature Gating Policy (Pause, Don’t Remove)

### Default behavior for the one-week release
- **Processor slot** present but defaults to **None**.
- **Python plugin support** present but defaults to **OFF** (build-time and/or runtime toggle).
- Any “experimental” path must be opt-in with a clear label.

### Rules for gating
- Do not delete code or interfaces required for the post-release roadmap.
- Prefer:
  - Compile-time flags (e.g., keep `NDA_ENABLE_PYTHON`, default OFF for release build)
  - Runtime toggles/env vars for diagnostics and pacing tweaks
  - UI presets that expose stable paths while keeping advanced paths accessible

---

## 6) Week Plan (Execution Checklist)

### Day 1: Repro + baseline metrics
- Lock the exact hardware, devices, and settings for the golden path.
- Capture baseline logs/counters for 5–10 minutes.
- Decide “known-good” buffer/frame defaults for stability.

### Days 2–4: Fix dropouts at the root cause
- Eliminate incorrect format handling and per-frame overhead on capture/render.
- Reduce blocking/locking in the hot path.
- Confirm counters move in the expected direction (underruns/overruns down).

### Day 5: Integrate + harden start/stop
- Ensure start/stop does not hang and can be repeated.
- Ensure device unplug/replug fails clearly (and ideally recovers).

### Day 6: Soak test + regression checks
- Run 20–60 minute soak tests on Bridge Mode.
- Try at least one “slower machine” scenario (background load) to validate jitter tolerance.

### Day 7: Packaging + release notes
- Package the build and document:
  - Supported golden-path configuration
  - How to enable experimental features (processors/Python)
  - How to collect diagnostics

---

## 7) Post-Release Re-Enablement (Immediately After the Week)

Once Bridge Mode is stable:

1. Reintroduce **C++ processors** (encryptor) with a strict real-time budget and clear failure policy.
2. Reintroduce **Python processors** behind a hard “experimental” toggle until performance/latency budgets are proven.
3. Expand device matrix support only after the bridge remains stable across changes.

The guiding principle remains: **protect the audio bridge first**; plugins must conform to real-time constraints, not the other way around.

