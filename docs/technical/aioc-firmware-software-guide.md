# AIOC Firmware & Host Integration Guide

This document explains how the AIOC firmware works internally and how a host application can communicate with the cable to stream audio, toggle PTT/COS lines, and reconfigure the hardware. It targets developers who know audio/DSP or desktop software but may not be familiar with embedded systems.

The source files referenced below live under `stm32/aioc-fw/Src`. The companion CLI (`aioc-util/aioc-util.py`) demonstrates how to exercise the HID register interface from a desktop application.

---

## 1. High-Level Architecture

- **MCU & toolchain:** STM32F302CB running bare metal firmware built with STM32Cube HAL and TinyUSB (`main.c`).
- **USB composite device:** One firmware image exposes USB Audio Class 2.0 (mono in/out with asynchronous feedback), a CDC ACM serial port, a CM108-compatible HID interface, and the DFU runtime interface (`usb_descriptors.c`).
- **Main loop:** `main.c` sets clocks, recalls persisted settings, enables the LED and IO subsystems, initializes USB + Fox Hunt mode, then services TinyUSB (`USB_Task`) while a watchdog ensures automatic recovery.
- **Interrupt-driven subsystems:**
  - *Audio path (`usb_audio.c`):* TIM3/ADC capture radio audio, TIM6/DAC drive TX audio, with TIM16/TIM17 timers implementing virtual PTT (VPTT) and virtual COS (VCOS).
  - *Serial bridge (`usb_serial.c`):* USART1 ISR ties the virtual CDC port to the radio’s K1 programming pins.
  - *HID/IO (`usb_hid.c`, `io.c`):* GPIO interrupts translate the radio’s COS/aux inputs into HID and CDC line-state events; HID output reports can assert PTTs.
  - *Configuration storage (`settings.c`):* a 256 × 32-bit register map persists in flash and is mirrored in RAM for quick access.

This structure lets desktop software treat AIOC as a regular USB audio + serial interface while still exposing low-level controls through HID feature reports.

---

## 2. Settings Register Map & Persistence

All runtime knobs live inside `settingsRegMap` (`settings.c`). Host software reads/writes those registers through the HID feature endpoint (see §5). The register block:

- Contains 256 32-bit entries (`SETTINGS_REGMAP_SIZE`).
- Lives in the `.eeprom` linker section so it survives power cycles.
- Uses a magic token (`'AIOC'` at address `0x00`) to validate flash data on boot.
- Accepts updates only for addresses `< 0xC0`; the remainder mirror live status (PTT/COS/Audio stats).

### 2.1 Commands exposed through HID feature reports

The HID feature report uses a single byte of command bits followed by the target address and 32-bit payload (`usb_hid.c`):

| Bit | Meaning                                  |
|-----|------------------------------------------|
|0x01 | `WRITESTROBE` – write 32-bit value to `address` |
|0x10 | `DEFAULTS` – reset RAM map to compiled defaults |
|0x20 | `REBOOT` – watchdog-reset into the DFU/system bootloader |
|0x40 | `RECALL` – reload from flash |
|0x80 | `STORE` – erase/program flash with current RAM map |

`aioc-util.py` (using `hidapi`) demonstrates the transactions: to read, it writes the register address with `Command.NONE`; to write it sends `WRITESTROBE`. Persisted settings stick only after issuing `STORE`.

### 2.2 Practical register overview

| Address | Register | Purpose |
|---------|----------|---------|
|`0x08`|`USBID`|Override USB VID/PID for the entire composite device. Defaults to 0x1209/0x7388 – change only if you know what you are doing (`settings.h`).|
|`0x24`|`AIOC_IOMUX0`|PTT1 source bit-field – OR together HID GPIO bits, CDC DTR/RTS combinations, and VPTT. Default ties CM108 GPIO3 + serial DTR && !RTS.|
|`0x25`|`AIOC_IOMUX1`|PTT2 sources. Default: CM108 GPIO4.|
|`0x44`–`0x47`|`CM108_IOMUX[0..3]`|Maps hardware inputs or virtual COS to HID button events (VolUp/VolDn/PlaybackMute/RecordMute). Defaults expose VCOS on VolDn so host software sees carrier detect even without a physical COS wire.|
|`0x60`|`SERIAL_CTRL`|Masks for “force PTT low while TXing serial” and “ignore radio data while PTT asserted” to avoid UART contention with shared TIP/RING pins.|
|`0x64`–`0x67`|`SERIAL_IOMUX[0..3]`|Maps COS sources to CDC line-status signals (DCD/DSR/RI/Break).|
|`0x72`|`AUDIO_RX`|Hardware version ≥1.2: select ADC front-end gain (1×..16×) to match radio audio levels.|
|`0x78`|`AUDIO_TX`|Enable TX boost (line-level vs mic-level) for firmware ≥1.4 + hardware 1.2 line driver.|
|`0x82`/`0x84`|`VPTT_LVLCTRL/TIMCTRL`|Amplitude threshold and hang timer (12.4 fixed-point ms) for automatic PTT based on outgoing audio energy.|
|`0x92`/`0x94`|`VCOS_LVLCTRL/TIMCTRL`|Threshold + hang timer for automatic COS reporting from incoming audio.|
|`0xA0`–`0xA5`|`FOXHUNT_CTRL/MSG[0..3]`|Enable the beacon mode: interval seconds, WPM, volume, and 16 ASCII bytes of Morse payload.|
|`0xC0+`|Info/debug|Read-only status bits: `INFO_AIOC0` shows live PTT/VPTT/VCOS states; `INFO_AUDIO*` capture mute flags, sample rates, buffer depths, and feedback stats, useful when tuning host audio pipelines.|

`aioc-util.py --dump` prints all registers, while other CLI switches set specific fields (PTT sources, audio gain, foxhunt message, etc.). Software can embed the same register IDs/enums from `settings.h` to manipulate the map directly via HID.

---

## 3. USB Composite Layout

The descriptors in `usb_descriptors.c` declare one configuration with seven interfaces:

1. **Audio Control / Streaming Out / Streaming In** – USB Audio Class 2.0, mono speaker (host→radio) and mono microphone (radio→host) endpoints plus asynchronous feedback (UAC2 async mode meets macOS/Windows quirks).
2. **CM108-compatible HID** – 4-button consumer-control report + 32-bit GPIO payload + HID feature transfer window for register access.
3. **CDC ACM** – Virtual COM port bridging to the radio’s programming UART, plus notification endpoint (not currently used for line-state packets).
4. **DFU Runtime** – Allows `dfu-util -d 1209:7388 ...` to drop the device into ST’s ROM bootloader without jumpering the hardware.

Because TinyUSB handles each class independently, you can use every interface in parallel (e.g., stream APRS audio while toggling PTT through HID and programming the radio over CDC).

---

## 4. Audio Signal Path (`usb_audio.c`)

### 4.1 Capture (Radio → Host, “Microphone” interface)

- **Hardware pipeline:** Radio audio enters OPAMP1 → ADC1/ADC2 driven by TIM3. Firmware chooses the ADC path and gain depending on `AUDIO_RX` (1× bypass vs. PGA for higher sensitivity).
- **Sample rates:** Only one channel is exposed, but the firmware advertises multiple fixed sample rates (48 kHz preferred, down to 8 kHz). For APRSdroid, it deliberately reports 22.05 kHz even though the crystal yields ~22.052 kHz – the stack compensates for the ppm error.
- **Stream lifecycle:** When the host activates the streaming interface (alternate setting 1), TinyUSB calls `tud_audio_set_itf_cb`. Firmware reconfigures timers, marks the state as `START`, and begins filling USB packets once the ADC ISR runs. Host applications should configure `ARATE 48000` (Direwolf example) to avoid format conversions.
- **Gain/Mute reporting:** UAC2 feature unit controls adjust per-channel dB values; firmware converts to linear multipliers and reflects them in `INFO_AUDIO3`. hosts see/ set mute & volume through standard mixer UIs.
- **Virtual COS:** Every ADC interrupt compares the absolute sample value against `VCOS_LVLCTRL`. When energy exceeds the threshold, TIM17 restarts and eventually sets COS true via HID buttons and CDC DCD/DSR depending on your IOMUX settings. The hang time uses `VCOS_TIMCTRL` (12.4 fixed-point milliseconds). This emulates hardware COS when your radio lacks such line.

### 4.2 Playback (Host → Radio, “Speaker” interface)

- **Hardware pipeline:** TIM6 triggers DAC1. GPIOA3 toggles an analog attenuator: mic-level (`TXBOOST` off) or boosted line-level (`TXBOOST` on) for hardware revision 1.2.
- **Asynchronous feedback:** `tud_audio_feedback_interval_isr` measures the timer cycles between SOF events (TIM2) and adjusts the feedback endpoint so PCs maintain the target buffer level (`SPEAKER_BUFFERLVL_TARGET`). This makes the device stable on macOS/Windows without manual drift-correction.
- **Virtual PTT:** TIM6 ISR monitors outgoing samples. If audio magnitude exceeds `VPTT_LVLCTRL`, TIM16 restarts. When the timer starts running, it asserts any PTT outputs mapped to `VPTT` in `AIOC_IOMUX[0/1]` and latches the state in `INFO_AIOC0`. When audio drops below the threshold for the configured timeout, TIM16 automatically deasserts those PTTs.
- **Buffer stats:** `INFO_AUDIO10`–`INFO_AUDIO15` expose min/max/avg buffer fill levels and feedback codes, which helps when diagnosing underruns in host software.

### 4.3 Audio-specific interrupts

- `ADC1_2_IRQHandler`: reads signed samples, applies microphone gain/mute, pushes audio into the TinyUSB TX FIFO, and kicks VCOS timers.
- `TIM6_DAC_IRQHandler`: consumes host samples, applies speaker gain/mute, writes DAC words, and drives VPTT timers.
- `TIM16_IRQHandler`/`TIM17_IRQHandler`: implement the on/off state machines for virtual PTT/COS.

These routines run at audio rate, so high-priority code should avoid blocking audio interrupts by keeping HID/CDC interactions non-blocking (TinyUSB handles this in interrupt context).

---

## 5. HID Interface & Register Transport (`usb_hid.c`)

The HID interface replicates a CM108 USB sound card so existing software (Direwolf, AllStarLink, etc.) can toggle PTT via HID consumer controls. Two payload types are used:

1. **Interrupt IN reports (4 bytes):** Bits [0:3] represent the four consumer buttons. Bytes 1–3 carry a snapshot of GPIO output states and are informational by default.
2. **Interrupt OUT reports (4 bytes):** Host writes GPIO drive states; firmware routes each bit through `ControlPTT()` which checks `AIOC_IOMUX` to see which PTTs should follow which CM108 GPIO.

Additionally, **Feature reports (6 bytes)** expose the register map:

```
byte0 = command bitfield (WRITESTROBE, DEFAULTS, STORE, etc.)
byte1 = register address (0x00-0xFF)
byte2..5 = 32-bit little-endian data
```

`aioc-util.py` encapsulates these transactions to make scripting easy, but any host language with HID access can perform the same exchange.

**PTT via HID (CM108):** Applications such as Direwolf can select “CM108 PTT” so they only need HID write access. Firmware allows bitwise OR of multiple sources (CM108 GPIO1..4, CDC DTR/RTS combinations, VPTT) so you can keep HID and CDC PTTs enabled simultaneously for redundancy.

**COS via HID:** `io.c` raises HID button events whenever the hardware IN1/IN2 lines change. The default IOMUX places IN2 onto VolUp and VCOS onto VolDn so host software sees direction-specific events. You can remap these bits to match your workflow (e.g., IN1 = VCOS, IN2 = external COS).

---

## 6. CDC ACM Serial Bridge (`usb_serial.c`)

- **Physical pins:** USART1 on PA9/PA10 connects to the Kenwood K1 tip/ring, sharing conductors with PTT lines. The firmware therefore cooperates with the PTT system to avoid collisions.
- **Baud configuration:** Host writes line coding through standard CDC control requests. Firmware reconfigures parity/stop/baud inside `tud_cdc_line_coding_cb`.
- **PTT guarding:** `SERIAL_CTRL` contains `TXFRCPTT` and `RXIGNPTT` bitmasks. Before sending host data to the radio (`tud_cdc_rx_cb`), the firmware checks whether any selected PTT is active; it can force those PTTs low while UART data is in flight. Conversely, incoming UART bytes can be ignored while PTT is asserted.
- **PTT via DTR/RTS:** `tud_cdc_line_state_cb` inspects DTR/RTS transitions and toggles PTT1/PTT2 for each mask bit set in `AIOC_IOMUX[0/1]`. Four modes exist for each output: DTR only, RTS only, DTR && !RTS, or !DTR && RTS.
- **Line state back-channel:** `USB_SerialSendLineState` can notify the host via the CDC notification endpoint (currently stubbed), but line statuses are still mirrored in `settingsRegMap` info registers and toggled through HID to keep compatibility with existing CM108 clients.

For host applications, the CDC port behaves like a conventional programming cable: open `/dev/ttyACMx`, toggle DTR/RTS for PTT if desired, and stream UART data at the required baud.

---

## 7. IO Layer, PTT Outputs, and COS Inputs (`io.c`, `io.h`, `cos.h`)

- **PTT outputs:** Two GPIOs on PA1 (PTT1) and PA0 (PTT2) drive the two sleeves on the K1 connector. `IO_PTTAssert/IO_PTTDeassert` also update LEDs and the `INFO_AIOC0` register so the host can confirm the state.
- **Input lines:** PB6 (IN1) and PB7 (IN2) are configured as edge-triggered interrupts. When they change, firmware consults the CM108 and CDC IOMUX bitfields to decide which HID buttons and CDC line signals to raise.
- **Virtual COS helper (`cos.h`):** Provides a consistent function to broadcast VCOS state across HID buttons and CDC line states simultaneously.

Because the AIOC PCB ties the K1 “PTT” conductor to both TX audio and UART TX, the flexible IOMUX + serial guard bits are crucial for preventing radios from seeing random serial pulses as PTT toggles.

---

## 8. Fox Hunt Beacon Mode (`fox_hunt.c`)

When `FOXHUNT_CTRL.Interval` is non-zero, the firmware periodically keys PTT1 and plays a 750 Hz sine wave keyed in Morse:

- `FoxHunt_Init()` reads the 16-byte message registers, converts them to Morse timings (`morse.c`), sets up TIM15 + DAC1 for standalone playback, and enables TIM15 interrupts.
- `FoxHunt_Tick()` runs once per second from the main loop to track the interval counter.
- TIM15 ISR steps through the precomputed timings, toggles the DAC output, scales the amplitude using the configured `Volume`, and deasserts PTT when the message completes.

Use `aioc-util` to set the message, WPM, volume, and repeat interval. A zero interval disables the mode entirely.

---

## 9. Working with `aioc-util`

The CLI under `aioc-util/aioc-util.py` is both a ready-to-use management tool and example code for host integrations:

- `--dump` shows every register plus USB strings/magic value.
- `--ptt1/--ptt2` accept `PTTSource` bitmask names and rewrite the IOMUX registers.
- `--vol-up/--vol-dn` remap COS/button sources.
- `--vptt-*` / `--vcos-*` tune thresholds and timers.
- `--audio-rx-gain` / `--audio-tx-boost` configure analog levels.
- `--foxhunt-*` manage beacon mode.
- `--set-ptt1-state on/off` toggles the raw HID GPIO bits (useful for testing without an audio modem).

Example session:

```bash
python aioc-util/aioc-util.py --dump                          # inspect current config
python aioc-util/aioc-util.py --ptt1 VPTT --ptt2 CM108GPIO1   # tie PTT1 to auto-PTT, PTT2 to HID GPIO1
python aioc-util/aioc-util.py --audio-rx-gain 4x --store      # persist higher RX gain
python aioc-util/aioc-util.py --vptt-lvlctrl 0x00000040 \
                              --vptt-timctrl 0x00000140      # lower VOX threshold, extend hang time
```

Borrowing the code: the script uses `hid.Device(vid, pid)` and wraps the feature report format, so you can port the same logic to another language if you need in-app configuration panels.

---

## 10. Integration Patterns

### 10.1 Audio modem (Direwolf, VaraFM, etc.)

1. **Audio interface:** Select the “AIOC Audio” device, pick 48 kHz mono (`ADEVICE plughw:<x>,0` on Linux, `ARATE 48000`). Direwolf tolerates the extra sample rates if you must use 22.05 kHz for Android compatibility.
2. **PTT:** Either configure `PTT CM108` so Direwolf writes to the HID GPIOs, or use `PTT <tty> DTR -RTS` to drive CDC line states. Use `aioc-util --ptt1 CM108GPIO1|SERIALDTR` if you want redundancy.
3. **COS:** The HID VolDn button reflects VCOS by default, so enable CM108 COS in Direwolf if you need carrier detect. To use a physical COS wire (hardware rev ≥1.2), run `aioc-util.py --enable-hwcos`.
4. **VOX fallback:** Map PTT1 to `VPTT` so outgoing audio automatically keys the radio when CM108/PTT is unavailable (e.g., apps without HID access).

### 10.2 Radio programming cable (CHIRP, OEM tools)

1. Open the CDC port (`/dev/ttyACMx` or COMx) at the radio’s programming baud (default 9600).
2. Ensure `SERIAL_CTRL.TXFRCPTT` includes the PTT that shares the UART pin (`aioc-util --auto-ptt1` or manual bit selection). This prevents PTT assertion from corrupting programming sessions.
3. Toggle `DTR/RTS` as required by the radio’s firmware. Since firmware 1.2.0 the default is “assert PTT when `DTR=1` and `RTS=0`” to match most vendor cables.

### 10.3 Custom desktop software

1. **Discovery:** Enumerate HID devices for VID/PID (default 0x1209/0x7388) or plan to update the IDs via register `USBID` if you have your own vendor ID.
2. **Configuration UI:** Use the HID feature protocol to read `AUDIO_RX`/`TX`, `VPTT_*`, etc., display them, and write back when the user saves settings. Follow up with `STORE` if you want persistence.
3. **PTT control:** If your app runs on Windows/macOS where direct HID writes are easiest, send 4-byte HID OUT reports to toggle GPIO bits. Otherwise, open the CDC port and drive DTR/RTS.
4. **Monitoring:** Subscribe to HID input reports to detect COS/button events, or poll `INFO_AIOC0` via feature reads to check PTT/VPTT/VCOS states.
5. **Fox Hunt:** Provide specialized UI: write ASCII text (max 16 chars) split across `FOXHUNT_MSG0`..`MSG3`, choose WPM/interval/volume, and tell the user to disconnect when running unattended.

### 10.4 Android / APRSdroid

- Use USB audio at 22.05 kHz (APRSdroid requirement).
- Since APRSdroid lacks PTT control, map PTT1 to `VPTT` so outgoing audio asserts VOX automatically, or let the radio’s built-in VOX handle it.
- `aioc-util`’s `--auto-ptt1` is a shortcut for this configuration when preparing the cable on a PC beforehand.

---

## 11. Reference: Common Bit Fields

### 11.1 PTT source bits (`AIOC_IOMUX0/1`)

| Bit mask | Source |
|----------|--------|
|`0x00000001`|`CM108GPIO1` (HID button)|
|`0x00000002`|`CM108GPIO2`|
|`0x00000004`|`CM108GPIO3`|
|`0x00000008`|`CM108GPIO4`|
|`0x00000100`|`SERIALDTR`|
|`0x00000200`|`SERIALRTS`|
|`0x00000400`|`SERIALDTRNRTS` (DTR=1 and RTS=0)|
|`0x00000800`|`SERIALNDTRRTS` (DTR=0 and RTS=1)|
|`0x00001000`|`VPTT` (virtual VOX)|

Combine any number of masks to create fallbacks.

### 11.2 CM108 button sources (`CM108_IOMUX*`)

Each button can follow:

| Bit mask | Meaning |
|----------|---------|
|`0x00010000`|`IN1` (physical input pin PB6)|
|`0x00020000`|`IN2` (physical input pin PB7)|
|`0x01000000`|`VCOS` (virtual carrier detect)|

Set to zero to detach a button from any source.

### 11.3 Serial line-state sources (`SERIAL_IOMUX*`)

Same bit definitions as above, but map to CDC line status pins:

| Register | USB bit |
|----------|---------|
|`SERIAL_IOMUX0`|DCD|
|`SERIAL_IOMUX1`|DSR|
|`SERIAL_IOMUX2`|RI|
|`SERIAL_IOMUX3`|BREAK (used by some digital mode software for squelch)|

---

## 12. Firmware Files of Interest

- `main.c` – System clock setup, USB reset logic, watchdog handling, FoxHunt tick loop.
- `settings.c/h` – Register definitions, flash storage, helper macros (`SETTINGS_GET`).
- `usb_audio.c/h` – Audio capture/playback ISR, VCOS/VPTT implementation, asynchronous feedback.
- `usb_serial.c/h` – CDC ↔ UART bridge with PTT/line-state handling.
- `usb_hid.c/h` – CM108 emulation plus register transport.
- `io.c/h` – GPIO configuration and EXTI ISR for COS inputs and PTT drives.
- `fox_hunt.c/h` & `morse.c/h` – Beacon generator.
- `aioc-util/aioc-util.py` – Example host-side control plane over HID.

Reading these modules provides a complete view of how the firmware treats each USB endpoint and how configuration registers map back to hardware behavior.

---

With this understanding you can build software that streams audio through the AIOC, asserts PTT either manually or via virtual VOX, reads COS, and reconfigures gains/PTT routing on the fly—all without touching embedded code. Use the HID register map for persistent settings, TinyUSB interfaces for streaming, and the provided CLI as a reference implementation.
