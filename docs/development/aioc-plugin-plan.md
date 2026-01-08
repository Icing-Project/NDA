# AIOC Duplex Plugin Implementation Plan

Plan to deliver an NDA plugin pair that treats AIOC as simultaneous audio source and sink, exposes rich control for PTT/VOX/device selection, and is ready for future full-duplex encryption/decryption. Includes risk/uncertainty notes for tuning.

## Objectives and Scope
- Deliver C++ plugins under `plugins_src/` (`AIOCSourcePlugin`, `AIOCSinkPlugin`) sharing a common `AIOCSession` helper for HID/CDC/UAC2.
- Provide optional Python prototype in `plugins_py/` with same control surface for rapid iteration.
- Support full-duplex path: NDA pipeline encrypts outbound before sink; decrypts inbound bearer packets before source read.
- Expose controls for PTT (manual HID/CDC, VPTT auto), COS/VCOS, volume/mute, device IDs, and telemetry to be consumed by future UI.
- Leave ~10-20% buffer in timelines and configuration values for debugging/tuning latency, thresholds, and HID retries.

## Architecture Overview
- Shared helper: `AIOCSession`
  - Responsibilities: enumerate AIOC by VID/PID (default 0x1209/0x7388), open HID feature + OUT endpoints, optional CDC ACM, and UAC2 mic/speaker streams.
  - Cache/register map snapshot (`AIOC_IOMUX*`, `VPTT_*`, `VCOS_*`, `INFO_*`), expose thread-safe read/write with retries/timeouts mirroring `aioc-util`.
  - Manage ring buffers/queues for audio capture/playback; provide callbacks for PTT state changes and COS events.
  - Reconnect handling: detect disconnect, attempt re-open, reapply cached config.
- Source plugin: `AIOCSourcePlugin` (implements `AudioSourcePlugin`)
  - Mono capture at 48 kHz (default), optional resample/upmix if pipeline demands stereo.
  - Pulls PCM from UAC2 mic, surfaces COS/VCOS state (from `INFO_AIOC0` and `CM108_IOMUX*`) and decrypts inbound bearer packets into a queue feeding `readAudio`.
- Sink plugin: `AIOCSinkPlugin` (implements `AudioSinkPlugin`)
  - Accepts NDA buffers, handles PTT gating (manual HID/CDC or VPTT), writes to UAC2 speaker, tracks buffer feedback (`INFO_AUDIO10-15`).
- Bearer integration
  - Register packet-received callback: decrypt via active encryptor, enqueue for source-side consumption.
  - Document that bearer must be configured to deliver raw PCM payloads matching pipeline sample rate/channels.

## Control Surface (setParameter/getParameter Keys)
- Device selection: `in_device_id`, `out_device_id` (Win: WASAPI/ASIO IDs; Linux: ALSA/Pulse names); `auto_select` fallback by VID/PID.
- Audio: `sample_rate` (default 48000), `channels` (default 1), `volume_in`, `volume_out`, `mute_in`, `mute_out`, `buffer_frames` (align with pipeline, default 512).
- PTT: `ptt_mode` (`hid_manual`, `cdc_manual`, `vptt_auto`), `ptt_state` (boolean toggle), `ptt_output` (PTT1/PTT2 via `AIOC_IOMUX0/1` bit masks), `ptt_threshold` (maps to `VPTT_LVLCTRL`), `ptt_hold_ms` (maps to `VPTT_TIMCTRL`), `cdc_mapping` (DTR/RTS bit masks), `ptt_timeout_ms` (safety auto-release).
- COS/VOX: `vcos_enable`, `vcos_threshold` (`VCOS_LVLCTRL`), `vcos_hang_ms` (`VCOS_TIMCTRL`), `cos_source` (button/IOMUX mapping).
- Telemetry/read-only: `info_ptt_state`, `info_vptt_state`, `info_vcos_state`, `info_buffer_stats` (from `INFO_AUDIO10-15`), `firmware_version`, `usb_strings`.
- Persistence: `persist_settings` flag to issue `STORE` after writes; otherwise volatile.

## Audio Data Path
- Sink (TX): on `writeAudio`, ensure PTT asserted per `ptt_mode`:
  - `hid_manual`: send HID OUT GPIO bits mapped by `AIOC_IOMUX*`.
  - `cdc_manual`: toggle DTR/RTS per `cdc_mapping`.
  - `vptt_auto`: program `VPTT_LVLCTRL/TIMCTRL` once, rely on firmware VOX; optionally mirror state from `INFO_AIOC0`.
  - Push frames to UAC2 speaker; monitor underruns/feedback (`INFO_AUDIO*`).
- Source (RX): `readAudio` blocks until buffer filled from UAC2 mic or decrypted bearer queue; convert to pipeline channel layout; update COS telemetry from `INFO_AIOC0`.
- Sample rate: default 48 kHz mono; expose parameter to switch if future firmware rates are needed; handle minor drift via feedback stats.

## HID/CDC/Register Handling
- Implement feature report helper:
  - Commands: `WRITESTROBE`, `DEFAULTS`, `STORE`, `RECALL`, `REBOOT` per firmware guide.
  - Retry/backoff strategy (3-5 attempts, 50-100 ms spacing); surface errors to plugin log/status.
- HID OUT GPIO for PTT: match IOMUX bits (PTT1/2) and verify via `INFO_AIOC0`.
- CDC path (optional): open COM port, assert DTR/RTS according to `cdc_mapping`; set `SERIAL_CTRL` bits to avoid UART/PTT collisions when necessary.
- Register batching: on start, read `AIOC_IOMUX*`, `VPTT_*`, `VCOS_*`, `INFO_*`; on telemetry poll, refresh `INFO_*` only to reduce USB traffic.

## Lifecycle and Threading
- `initialize`: locate device, open HID/CDC/UAC2, prime register cache, set plugin state to Initialized.
- `start`: start audio streams, spawn worker threads (capture loop, playback loop, telemetry poller), register bearer callback; set Running state.
- `stop`: stop threads/streams, release PTT if asserted, flush queues, set Initialized.
- `shutdown`: close handles, clear caches, set Unloaded.
- Thread safety: guard shared state (register cache, queues) with mutexes; avoid holding locks during HID I/O to prevent stalls.
- Reconnect: detect disconnection, try re-open every N seconds with exponential backoff; reapply cached parameters on success.

## Testing and Validation
- Unit-level (with mocks): verify `ptt_mode` toggles call expected HID feature/OUT payloads; thresholds map to correct register values; CDC mapping sets DTR/RTS as configured.
- Integration: pipeline loopback with UDP bearer + simple encryptor; send known PCM, ensure decrypted output matches input after passing through source/sink; measure latency and underruns.
- Manual device tests: with real AIOC, validate PTT assert/release (manual and VPTT), COS/VCOS reflection, buffer stats stable at target buffer level, and reconnect behavior.
- Performance checks: confirm no audio glitches at 48 kHz/512f; adjust buffer sizes if underruns seen (keep 10-20% margin).

## Documentation Deliverables
- Update `docs/ARCHITECTURE_REPORT.md` and `docs/NDA-SPECS.md` to reference AIOC duplex plugins and control surface.
- Add README section (or new doc) describing setup, parameters, and known limitations.
- Note any platform-specific drivers (hidapi/WinUSB/udev rules) and expected VID/PID defaults.

## Risks, Unknowns, and Buffer
- HID/CDC driver availability on Windows/Linux may require installer steps (WinUSB/libusb/permissions).
- UAC2 timing drift/feedback quirks may need tuning of buffer sizes or feedback handling; leave buffer headroom.
- Exact register values for thresholds/hang times may need empirical tuning per radio; plan UI sliders with safe defaults.
- Reconnect robustness depends on HID stack behavior; include logging and fallback to manual re-init.
- Keep schedule slack (10-20%) for debugging latency, PTT races, and HID retries once hardware is under test.
