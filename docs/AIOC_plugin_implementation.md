# AIOC Duplex Plugin Implementation Notes

Detailed walkthrough of the C++ AIOC source/sink plugins, the shared session helper, and how HID/CDC/PTT plus WASAPI audio are wired. This is the current implementation state (prototype quality) and still requires on-device validation and tuning.

## Components
- `plugins_src/AIOCPluginCommon.h/.cpp`
  - `AIOCSession` owns device discovery/handles (HID, CDC COM, WASAPI render/capture), state, ring buffers, and PTT/VOX config.
  - Thread-safety via mutex; keeps telemetry counters and last-status messages.
- `plugins_src/AIOCSourcePlugin.cpp`
  - Implements `AudioSourcePlugin`; pulls from `AIOCSession::readCapture`, applies volume/mute, supports loopback, exposes VCOS threshold params.
- `plugins_src/AIOCSinkPlugin.cpp`
  - Implements `AudioSinkPlugin`; applies volume/mute, drives PTT (HID/CDC manual or VOX auto), calls `AIOCSession::writePlayback`.

## Audio Path (WASAPI)
- COM init (`CoInitializeEx`) gated in `AIOCSession::ensureComInitialized`; teardown on disconnect.
- Device selection: uses endpoint IDs if provided via parameters (`device_id` in/out); otherwise defaults to console render/capture.
- Render:
  - `IAudioClient` initialized shared mode with requested sample rate/channels; falls back to mix format if the request fails.
  - `IAudioRenderClient::GetBuffer`/`ReleaseBuffer` to write interleaved PCM; float path preferred, 16-bit fallback.
  - Tracks overruns when requested frames exceed available space.
- Capture:
  - `IAudioClient` + `IAudioCaptureClient`; shared mode, requested SR/channels, fallback to mix format on failure.
  - Copies interleaved packet to `AudioBuffer`, converts 16-bit->float if needed, applies volume/mute, counts underruns on empty buffers.
- Loopback/testing: if enabled, render pushes into a ring buffer that capture consumes (no hardware required).

## PTT/COS Control
- Modes: `hid_manual`, `cdc_manual`, `vptt_auto` (VOX).
- HID (when `HAVE_HIDAPI`): opens by VID/PID (0x1209:0x7388). Manual PTT sends a 4-byte OUT report with bit0 reflecting asserted state (aligns with CM108-style GPIO). HID init is optional; if missing, HID control is disabled.
- CDC (Windows): opens named COM port (e.g., `COM5`), toggles DTR high / RTS low when PTT is asserted, clears both when released.
- VOX (sink): computes peak per buffer; compares to `vptt_threshold` (scaled to float); asserts PTT while above threshold and for `vptt_hang_ms` afterwards. Threshold/hang also map to firmware registers conceptually; here they’re used locally.
- COS/VCOS: thresholds stored; the session marks `vcosActive` when configured. Actual COS polling from firmware registers is not yet wired; telemetry flag is heuristic.

## Parameters (set/get)
- Common/session: `sampleRate`, `channels`, `bufferFrames`, `volume_in`, `volume_out`, `mute_in`, `mute_out`, `device_id` (per plugin), `loopback_test`.
- Sink-specific: `ptt_mode` (`hid_manual`|`cdc_manual`|`vptt_auto`), `ptt_state` (manual assert), `cdc_port`, `vptt_threshold`, `vptt_hang_ms`.
- Source-specific: `vcos_threshold`, `vcos_hang_ms`.
- Telemetry via `AIOCSession::getTelemetry` (connected/running/PTT, frames, underrun/overrun, lastMessage).

## Lifecycle
1. `initialize`: set SR/channels/buffer size, optional loopback; open HID (if available), CDC (if port provided), and WASAPI endpoints; set state Initialized.
2. `start`: start WASAPI clients (render/capture); set Running.
3. `writeAudio`/`readAudio`: apply gain/mute, VOX/ptt handling (sink), move audio via WASAPI or loopback queue; counts frames/underruns/overruns.
4. `stop`: stop audio clients; release PTT; preserve connections.
5. `shutdown`/`disconnect`: close audio, HID, CDC, COM uninit; clear queues; reset flags.

## Dependencies & Build
- Windows: link `ole32` and `avrt` (WASAPI/COM). Uses shared-mode WASAPI; exclusive not implemented.
- HID: requires hidapi and compile-time `HAVE_HIDAPI` define; otherwise HID control is skipped gracefully.
- CDC: Windows-only code path; other platforms currently no-op.

## Limitations / TODOs
- COS/INFO register polling via HID feature reports not yet implemented; telemetry uses thresholds only.
- Bearer decrypt callback not yet wired to `pushIncoming`; planned integration point for full duplex through the network path.
- No resampler/channel adapter beyond basic buffer resize; relies on matching SR/ch layout.
- Error handling/logging is minimal; UI-facing error propagation still needed.
- Linux/macOS audio paths are not implemented; WASAPI-only for now.
- HID OUT bit usage assumes CM108-compatible mapping; adjust if firmware expects different bits.

## Manual Test Ideas
- Loopback: enable `loopback_test=true`, set `sampleRate=48000`, `bufferFrames=512`; run sine source -> AIOC sink -> AIOC source -> null sink; check underrun/overrun counters.
- HID PTT: set `ptt_mode=hid_manual`, toggle `ptt_state` true/false, observe radio or HID trace.
- VOX: set `ptt_mode=vptt_auto`, tune `vptt_threshold`/`vptt_hang_ms`, play tone through sink, verify PTT hold.
- CDC: set `cdc_port=COMx`, `ptt_mode=cdc_manual`, toggle `ptt_state` and observe DTR/RTS on scope/loopback.

## Risk/Uncertainty Buffer
- WASAPI format negotiation may reject custom SR/ch; fallback path uses device mix—verify actual format in logs.
- HID report format may need alignment to firmware; confirm with `aioc-util`/firmware docs and adjust bits.
- VOX thresholds are provisional; expect tuning per radio levels.
- No underrun recovery beyond silence; consider buffering or latency adjustments if glitches observed.
