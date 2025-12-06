# AIOC Duplex I/O & PTT Plugin Spec

> Builds on `docs/ARCHITECTURE_REPORT.md` and `docs/NDA-SPECS.md` plus the `aioc-firmware-software-guide.md` to define a complete duplex plugin that uses the AIOC radio interface for both encrypted transmit and receive, exposing PTT/VOX controls for users.

## 1. Objectives
1. Treat the AIOC device as an audio source (radio RX) and sink (radio TX) in NDA’s processing pipeline, while driving PTT via HID/CDC registers so the user can key the radio on demand.
2. Support the encryptor step bidirectionally: encrypt outbound audio before it reaches the AIOC sink and decrypt inbound packets that arrive via the bearer so transparent push/pull occurs between NDA and the radio.
3. Surface manual/automatic PTT gating and VOX thresholds through NDA settings so the radio can be keyed exactly when the UI indicates “talking.”
4. Keep the plugin lifecycle compatible with NDA’s plugin manager, maintaining logging, stats, and configuration endpoints so downstream automation (e.g., AI agents) can implement or extend it reliably.

## 2. Architecture Overview
### 2.1 NDA Pipeline Context
- `ProcessingPipeline`: `AudioSource → (Encryptor?) → Bearer → AudioSink`. The new plugin must provide both a source and sink pair (or a unified duplex plugin) so NDA can schedule streaming and encryption exactly as documented (`docs/ARCHITECTURE_REPORT.md:52-64`).
- The pipeline already knows about bearer/encryptor slots; the plugin should ensure decrypt logic is triggered when bearer packets arrive by registering with the bearer’s `PacketReceivedCallback` and routing decrypted audio back into NDA’s sink path.

### 2.2 AIOC Capabilities
- Host-visible USB interfaces:
  - USB Audio Class 2.0 mono speaker (AIOC→radio TX) and microphone (radio RX→host) with asynchronous feedback (`aioc-firmware-software-guide.md:68-105`).
  - HID feature reports/register map for `AIOC_IOMUX`, `VPTT/VCOS`, `INFO_*` telemetry, and PTT toggles (`˚md:24-139`).
  - CDC ACM port that can toggle DTR/RTS for PTT and provide UART bridging if needed (`...md:131-140`).
- Firmware timers generate virtual PTT (VPTT) and COS (VCOS) events, letting the hardware assert lines automatically when NDA supplies audio (`...md:81-103`).

## 3. Plugin Specification
### 3.1 Plugin role
- Provide:
  - `AIOCSourcePlugin`: implements `AudioSourcePlugin`, reads from the AIOC microphone endpoint, surfaces COS/VOX state via NDA telemetry, and runs the decrypt path.
  - `AIOCSinkPlugin`: implements `AudioSinkPlugin`, writes encryptor output to the AIOC speaker endpoint and asserts/deasserts PTT through HID registers or CDC line states.
  - Optional helper layer for shared configuration (register access, HID/CDC bridging).

### 3.2 Lifecycle Hooks
- `initialize()`: open HID/CDC handles (use `aioc-util` logic). Sample initial register states (`VPTT`, `VOX`, `INFO_AIOC0`, etc.) and report them via the plugin’s metadata.
- `start()`: start audio streaming threads or pump loops; register bearer receive callback to decrypt inbound packets and push to NDA’s sink by writing into a local `AudioBuffer`.
- `stop()`: stop USB audio streaming, release HID/CDC handles, and ensure PTT lines are released. Reset VOX timers if hardware manages them.
- `shutdown()`: clean up helper threads and release resources gracefully.

### 3.3 Audio Flow
- Transmit path:
  1. NDA writes frames into the plugin’s `writeAudio()` buffer.
  2. The plugin optionally triggers VOX timers (tight integration with `VPTT_LVLCTRL/TIMCTRL`), or enforces explicit user-requested PTT via HID/CDC writes before delivering frames through the USB speaker endpoint.
  3. The bearer receives packets and sends them over the radio after encryptor processing as in the pipeline.
- Receive path:
  1. The bearer’s `PacketReceivedCallback` decrypts incoming packets via the chosen encryptor (call `encryptor_->decrypt(...)` or similar) and then enqueues the decrypted PCM into a buffer accessible by `readAudio()`.
  2. `readAudio()` blocks until data is available from this queue or the USB microphone endpoint, ensuring NDA sees the radio input at pipeline sample rates (48 kHz mono to match the AIOC descriptor).
  3. The plugin also watches `INFO_AUDIO*` / `VCOS` registers to update COS states for the UI and can notify the pipeline when carrier is present.

### 3.4 PTT & VOX Controls
- Expose parameters in NDA’s UI (via `setParameter`/`getParameter`) for:
  - `ptt_mode`: {Manual HID, Manual CDC, VOX auto} controlling how PTT lines are asserted.
  - `ptt_hold_ms`: extra hang time after audio stops (maps to `VPTT_TIMCTRL`).
  - `ptt_threshold`: threshold for `VPTT_LVLCTRL`.
  - `voxon`: bool toggling VOX gating vs manual gating.
  - `cos_source`: map VCOS to UI carrier and align with `INFO_AIOC0`.
- Manual PTT:
  - On `setParameter("ptt_force", true)`, write the HID GPIO bits (as described in §5) to assert the desired AIOC PTT pin via feature report (write `AIOC_IOMUX0/1` appropriately).
  - Mirror state in `INFO_AIOC0` by reading back the register and exposing it via status queries.
- Automatic PTT (VOX):
  - When VOX is enabled, rely on `VPTT_*` registers; optionally, reconfigure them from NDA so the plugin can reduce hang time/threshold without direct HID writes.
  - Provide UI readouts for buffer fill/feedback registers (`INFO_AUDIO10`–`INFO_AUDIO15`) to monitor underruns.

### 3.5 Bearer Integration
- Use a bearer plugin (e.g., UDP or a custom radio bearer) to send/receive packets.
- The AIOC plugin should register a packet callback so decrypted frames flow directly into NDA’s sink, staying compatible with duplex expectations. Document how the callback is hooked inside the plugin (the payload is normalized to NDA’s `AudioBuffer` layout).
- Provide a factory helper so `AIOC` bearer binding can reuse the same metadata each time (e.g., `AIOCBearerPlugin` referencing `PacketReceivedCallback`).

## 4. Configuration & Control APIs
1. HID register read/write helpers (wrap `hidapi` or similar) with retries/timeouts matching `aioc-util`: send feature reports, handle `WRITESTROBE`/`STORE`, and ensure `INFO_` registers are refreshed on demand.
2. Optional CLI/callback that allows NDA to call `aioc-util --ptt1 VPTT` style commands, so future automation can be built on top of the plugin.
3. Parameter set/get for VOX/PTT and for toggling CDC DTR/RTS mapping if the plugin chooses to toggle CDC lines instead of HID (the plugin should record which interface is managing PTT).
4. Expose telemetry (latency, buffer levels, bytes sent/received) via NDA’s dashboard by mapping AIOC `INFO_` registers to plugin stats.

## 5. Testing & AI Implementation Guidance
- Provide unit tests stubs (C++/Python) that simulate HID register writes and carrier detect, verifying `ptt_mode` toggles work; describe how to mock `aioc-util`.
- Add integration test scenario:
  1. Start NDA pipeline with the AIOC plugin as both source and sink.
  2. Configure a loopback bearer (e.g., UDP) plus a simple encryptor plugin.
  3. Feed known PCM data and assert that decrypted output matches input after passing through the AIOC audio drivers and bearer.
- Suggest instrumentation: track packets sent/received per `INFO_AIOC0`, log HID register values on errors, and time the bearer callback path to ensure decrypt latency stays under target.
- Recommend building additional docs that show how an AI (e.g., `gpt-5.1-codex-high`) could implement the plugin by following the described parameters, callback wiring, and HID register map usage.

## 6. Dependencies & Implementation Notes
- The plugin will depend on `hidapi` or platform HID access, the NDA plugin interfaces (`AudioSourcePlugin`, `AudioSinkPlugin`, `BearerPlugin`, `EncryptorPlugin`), and the AIOC register definitions (`settings.h` fields).
- Highlight that the plugin must honor NDA’s thread-safety and buffer sizes (512 frames default) so the USB audio sinks/drivers do not underrun.
- If writing a Python prototype, reuse the `plugins_py` bridge to wrap HID communication while implementing the `AudioSourcePlugin` and `AudioSinkPlugin` interfaces for faster iteration; note that HID/CDC access may require platform-specific drivers.

## 7. Documentation Alignment
- Update `docs/ARCHITECTURE_REPORT.md` with references to the new plugin once implemented, and add a section describing how the bearer callback and dual encryptor/decryptor flow work.
- Append usage notes to `docs/NDA-SPECS.md` (pipeline flow) and maybe add a short entry in `docs/FINAL_DELIVERABLES.txt` once the feature ships.
