# NDA (Nade Desktop App)

## Testing the NDA

NDA is a **Qt desktop application** that runs two independent real-time audio pipelines (TX + RX). It sits between audio endpoints and routes audio through plugins (optionally including encryption/decryption processors).

Your job as a beta tester is to verify that NDA:
- launches reliably,
- loads plugins reliably,
- starts/stops TX and RX reliably,
- stays stable while running,
- produces usable logs when something fails.

## What is NDA exactly?

NDA (Nade Desktop App) is a **real-time audio processing bridge**:

- You choose a **Source** (input), an optional **Processor**, and a **Sink** (output).
- NDA moves audio buffers through that chain in real time.

NDA provides two independent pipelines:

- **TX (Transmit):** Source -> (Processor) -> Sink
- **RX (Receive):** Source -> (Processor) -> Sink

## How does it work?

At runtime, each pipeline runs its own processing loop and does the same core operation repeatedly:

1. Read audio frames from the configured Source plugin
2. Optionally transform them in the Processor plugin
3. Write the result to the Sink plugin

NDA is designed for long-running stability and “start/stop without drama”.

## What to test

Focus on these user-visible things:

- **Startup:** App opens, UI renders, no immediate crash.
- **Plugin loading:** Plugins appear in the Source/Processor/Sink dropdowns.
- **Start/Stop:** `Start TX`, `Start RX`, and `Start Both` work; Stop buttons always stop.
- **Independence:** TX can run while RX is stopped (and vice versa).
- **Stability:** Running for 5–10 minutes doesn’t degrade or crash.
- **Logging:** Errors are understandable (missing devices, missing deps, plugin load failures).

## Instructions

### Setup

- Run NDA from a terminal when possible so you can capture console logs.
- Make sure NDA can see plugins:
  - Python plugins are in `plugins_py/` (requires `numpy`; device plugins require additional audio deps).
  - C++ plugins are built into the plugin output directory (commonly under `build/plugins/`).

### Running tests

#### Test 1 — Launch + plugin discovery

Objective: verify NDA launches and populates dropdowns.

Steps:
1. Launch NDA.
2. Confirm the Source/Processor/Sink dropdowns are populated (not empty).

Expected outcome:
- No crash, and at least a few plugins are selectable.

Failure indicators:
- Empty dropdowns, plugin load errors, or immediate crash.

#### Test 2 — TX start/stop (safe, no hardware required)

Objective: validate the processing loop and lifecycle.

Steps:
1. Configure TX:
   - Source: `Sine Wave Generator` (if present)
   - Sink: `Null Sink (Console Monitor)` or `WAV File Recorder` (if present)
2. Click `Start TX`, wait ~10–30 seconds, then click `Stop TX`.

Expected outcome:
- TX starts, runs, and stops cleanly; no UI freeze.
- If `WAV File Recorder` is used, a `recording_*.wav` file is created.

Failure indicators:
- Start does nothing, Stop does nothing, or the app becomes unstable.

#### Test 3 — Dual pipeline behavior

Objective: verify TX and RX are truly independent and “Start Both” behaves.

Steps:
1. Configure both TX and RX (any working Source/Sink pair).
2. Click `Start Both`, wait ~10–30 seconds, then click `Stop Both`.
3. Repeat by starting/stopping TX and RX separately.

Expected outcome:
- “Start Both” starts both pipelines; “Stop Both” stops both pipelines.
- Stopping TX does not stop RX (and vice versa).

#### Test 4 — Real devices (careful)

Objective: validate real audio I/O without feedback or screaming audio.

Safety:
- Use headphones if possible.
- Start with low volume.

Steps:
1. Configure TX with a real microphone source and a non-speaker sink (or route to a virtual cable).
2. Start TX, observe stability for ~1 minute, stop TX.
3. Configure RX with a virtual/loopback input and speaker sink, start RX, stop RX.

Expected outcome:
- No crash; pipeline starts/stops; audio flows if routing is correct.

### Reporting

For each test, include:
- OS (Windows/Linux), audio backend (WASAPI/ALSA/PulseAudio), and whether Python plugins were enabled
- Which plugins you selected for TX and RX
- Pass/fail + the exact steps
- Console logs (copy/paste) and any generated `recording_*.wav` files
