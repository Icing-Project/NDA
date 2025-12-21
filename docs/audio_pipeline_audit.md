# NDA Audio Pipeline Audit
- Purpose: document the current audio treatment path in NDA, spanning C++ core, plugin contracts, and Python bridges, and provide an in-depth analysis of the timing mismatches observed in recent runs.
- Scope: covers processing loop semantics in `src/core/ProcessingPipeline.cpp`, plugin behaviors in `plugins_src/` and `plugins_py/`, buffer and latency handling, and concrete remediation steps.
- Context: observations are based on the latest repository state under `C:\Users\Steph\Desktop\Icing\Dev\NDA` with Qt UI driving pipeline selection via `src/ui/PipelineView.cpp`.
- Audience: developers debugging throughput/latency issues and maintainers integrating new sources or sinks.
- Terminology: “frame” refers to one sample per channel; “buffer” refers to `frameCount` frames per channel; “audio seconds” = processed frames ÷ sample rate.

## Top-Level Architecture
- Entry point resides in `src/main.cpp`, which configures a Qt Fusion theme and instantiates `MainWindow` with tabs for dashboard and pipeline configuration.
- The audio pipeline is orchestrated by `nda::ProcessingPipeline` (header: `include/core/ProcessingPipeline.h`, implementation: `src/core/ProcessingPipeline.cpp`).
- Plugins implement the `BasePlugin` contract (`include/plugins/BasePlugin.h`) and are categorized by `PluginType` (`include/plugins/PluginTypes.h`) into source, sink, bearer, encryptor, and processor roles.
- Plugin lifecycle states are defined by `PluginState` (Unloaded, Loaded, Initialized, Running, Error) and are expected to be honored by each plugin implementation.
- The Qt UI (`src/ui/PipelineView.cpp`) lets users pick a single audio source, optional encryptor, optional bearer, and a single audio sink; pipeline start/stop is bound to GUI buttons.
- `PluginManager` (`include/plugins/PluginManager.h`, `src/plugins/PluginManager.cpp`) dynamically loads C++ plugins via shared libraries and Python plugins via `PythonPluginBridge` when `NDA_ENABLE_PYTHON` is enabled.
- Python plugins live under `plugins_py/` and are discovered by scanning `.py` files excluding loader/setup helpers; C++ plugins live under `plugins_src/` and are built into shared objects placed in plugin directories.
- The audio processing loop runs in a dedicated std::thread owned by `ProcessingPipeline`, with all plugin calls executed from that thread after initialization.
- The pipeline uses a single work buffer (`AudioBuffer` from `include/audio/AudioBuffer.h`) that is resized during initialization to the negotiated frame size and channel count.
- There is no explicit resampling layer; the pipeline assumes all components share the same sample rate and channel count negotiated at init time.
- Backpressure is not implemented; `ProcessingPipeline` calls source->readAudio then sink->writeAudio every loop without checking sink capacity beyond initialization buffer size alignment.
- Metrics exposed by the pipeline are limited to processed sample count, computed latency estimate, and a placeholder CPU load; no real-time scheduling or drift compensation exists.

## ProcessingPipeline Lifecycle
- `setAudioSource`, `setBearer`, `setEncryptor`, and `setAudioSink` simply assign shared_ptrs when the pipeline is not running.
- `initialize` logs plugin names and states, initializes Unloaded plugins, and rejects plugins not in Initialized state; it does not validate matching sample rates or channel counts between components.
- Frame size negotiation picks the minimum of the pipeline default (512), `audioSource->getBufferSize()`, and `audioSink->getBufferSize()`; bearers/encryptors are ignored in this negotiation.
- After negotiation, `frameCount_` is applied to source and sink via `setBufferSize`, but sources that do not override `setBufferSize` remain at their internal defaults.
- `workBuffer_` is resized to the source-reported channel count (defaults to 2 when source is absent) and the negotiated frame count; sample rate defaults to 48000 when source is absent.
- `start` sequentially calls `start()` on source, bearer, encryptor, and sink; failure anywhere aborts start without rollback of already-started plugins.
- When start succeeds, `isRunning_` is set true, `processedSamples_` is reset, and a joinable thread is created to run `processingThread`.
- `stop` sets `isRunning_` false, stops the audio source first to unblock reads, joins the processing thread, then stops bearer, encryptor, and sink.
- Stopping logs processed sample count and derived audio seconds using the source sample rate (or 48000 when source is absent).
- `shutdown` simply calls `shutdown()` on all components; it does not join threads (thread is already joined in stop) or clear shared_ptrs.
- Latency estimation (`getLatency`) sums source buffer time (workBuffer frames ÷ source sample rate), bearer latency, and sink buffer time; it ignores OS/hardware latency and Python bridge overhead.
- CPU load reporting is hardcoded to 5.0f; there is no measurement of thread runtime vs wall time or load per plugin.

## Processing Thread Behavior
- `processingThread` records `startTime` via `std::chrono::steady_clock::now` and enters a while loop while `isRunning_` is true.
- Each loop calls `processAudioFrame`, then increments a local `frameCount`.
- Progress logging occurs every 100 iterations (`frameCount % 100 == 0`), computing `elapsed` as integer seconds and `audioSeconds` as `processedSamples_ / sampleRate`.
- The loop contains no sleep except when `processAudioFrame` sleeps after failed reads; pacing relies entirely on plugin blocking characteristics.
- `isRunning_` is an atomic flag; there is no condition variable or sleep to yield CPU when running faster than real time.
- The thread is joinable and joined in `stop`; there is no detaching or thread reuse between runs.
- Because logging truncates elapsed to whole seconds, sub-second drift is not visible; large drift accumulates before being reported.
- The thread does not check return values from bearer send or sink writes when counting `processedSamples_`; it increments after every call regardless of success.
- No exception handling is present inside the processing loop; any thrown exception would terminate the thread without clearing `isRunning_`.
- There is no mechanism to detect or correct for sample-rate mismatch between source and sink; drift is unchecked.

## Per-Frame Audio Path
- Step 1: `audioSource_->readAudio(workBuffer_)` fills the buffer; on failure, consecutive failure counts trigger a 1 ms sleep and skip the rest of the loop.
- Step 2: If encryptor exists, the buffer is reinterpreted as bytes and encrypted in place with a zeroed nonce; output sizing assumes tag size 16 bytes.
- Step 3: If bearer exists and is connected, a `Packet` is serialized from the buffer (interleaved manually) and sent; packet timestamp = `processedSamples_`, sequence = `processedSamples_ / frameCount`.
- Step 4: `audioSink_->writeAudio(workBuffer_)` is called unconditionally; return value is ignored for sample counting.
- `processedSamples_` is incremented by `workBuffer_.getFrameCount()` even when bearer send or sink write failed.
- No gain staging, mixing, or resampling occurs in the core loop; plugins are expected to handle format and timing compatibility themselves.
- Work buffer data layout is channel-major (`std::vector<std::vector<float>>`), with contiguous frames per channel.
- There is no separate input/output buffer; the same buffer is used across pipeline stages, so plugins must avoid retaining references beyond the call.
- Encryption operates on raw float bytes, so downstream sinks receive encrypted float data if an encryptor is inserted without a decryptor, which is not currently modeled.

## Buffering and Data Layout
- `AudioBuffer` defaults to 2 channels, 512 frames; resize resets channel and frame counts and zero-fills channels.
- `AudioBuffer::mixWith` performs per-channel accumulation with optional gain; it is not used in the current pipeline.
- C++ plugins accessing `AudioBuffer` operate on channel-major arrays; Python plugins receive a NumPy array shaped (channels, frames) constructed by `PythonPluginBridge`.
- Channel count mismatches are not reconciled centrally; sources and sinks can set their own channel counts independently during initialization.
- `ProcessingPipeline` sets sink and source buffer sizes to the negotiated `frameCount_`, but only sinks that implement `setBufferSize` honor it; the C++ sine source ignores it.
- `getLatency` assumes sink and source buffer sizes reflect runtime queueing, which is not true for sinks with internal queues (e.g., sounddevice speaker uses an 8-buffer queue in Python).
- Sample format is 32-bit float throughout; there is no provision for integer PCM in the core buffer structure.
- `AudioBuffer` copies and clears are O(channels * frames); there is no reuse of preallocated interleaved buffers for I/O heavy sinks.

## Plugin Manager and Loading
- `PluginManager::loadPlugin` routes `.py` plugins to the Python bridge and other extensions to C++ loader, using `LoadLibraryA` on Windows.
- C++ plugin validation ensures API version matches `NDA_PLUGIN_API_VERSION` via `getInfo().apiVersion`.
- Python plugin validation relies on successful object creation and `initialize()` returning true; no API version check is enforced on the Python side.
- Loaded plugins are stored by name in a map; duplicate names overwrite previous entries without warning.
- `scanPluginDirectory` looks for `.dll` (Windows) or `.so` (Linux) and Python files (excluding setup/loader files); directories that cannot be read are silently ignored.
- Unloading a plugin calls `shutdown`, resets the shared_ptr, and unloads the library; Python plugins have no library handle to unload.
- There is no dependency tracking between plugins; unloading a plugin in use by the pipeline is not prevented.
- Plugin parameters are string-based; there is no schema or validation beyond per-plugin parsing.

## C++ Audio Sources
- `SineWaveSourcePlugin` (`plugins_src/SineWaveSourcePlugin.cpp`) defaults to 48 kHz, 2 channels, 512-frame blocks, and generates a 440 Hz sine at 0.5 amplitude.
- Sine generator start sets phase to 0 and immediately returns; `readAudio` is non-blocking and always succeeds when Running, producing data as fast as the pipeline calls it.
- `AIOCSourcePlugin` interfaces with custom USB audio endpoints via `AIOCSession`, defaulting to 48 kHz mono and 512-frame buffers; it enforces device IDs and exposes PTT/VOX telemetry.
- AIOC source calls `session_.readCapture` each loop; capture failure clears the buffer and returns false, which causes the pipeline to sleep 1 ms and log failures.
- `AlsaMicrophonePlugin` (`plugins_src/AlsaMicrophonePlugin.cpp`) uses ALSA capture (placeholder scaffolding) with default 48 kHz stereo and 512 frames, expected to block on capture in a real implementation.
- `LinuxMicrophonePlugin` is a placeholder stub with similar defaults, intended for Linux WDM/OSS style input.
- Source plugins do not expose per-call timestamps; processedSamples_ is the only timing reference in the pipeline.

## C++ Audio Sinks
- `WavFileSinkPlugin` writes 32-bit float PCM to disk, default 48 kHz stereo and 512-frame buffers, with filenames timestamped at start.
- On start, WAV sink writes a placeholder header and logs target filename and format; progress is logged whenever totalFrames_ crosses a one-second boundary.
- `writeAudio` interleaves channel-major buffer into a single float array and writes with `fwrite`; errors print "[WavFileSink] Write error!" and return false.
- WAV sink reports `getAvailableSpace` as a large constant (1,000,000 frames), so the pipeline never throttles based on disk I/O.
- `NullSinkPlugin` discards audio while reporting RMS levels every 0.1 seconds of processed audio; it is useful for monitoring but provides no pacing.
- `LinuxSpeakerPlugin` and `Alsa` sinks are placeholders representing device output paths; timing characteristics depend on future implementations.
- `AIOCSinkPlugin` pairs with AIOC hardware, writing playback via `session_.writePlayback`, mirroring the source side with the same buffer sizing defaults.

## Python Audio Sources
- Python plugins derive from `AudioSourcePlugin` in `plugins_py/base_plugin.py`, using NumPy-backed buffers with shape (channels, frames).
- `sine_wave_source.py` mirrors the C++ sine generator at 48 kHz stereo; `read_audio` vectorizes phase computation and is non-blocking.
- `sounddevice_microphone.py` captures via `sounddevice.InputStream` with default 48 kHz, 1 channel, 256-frame blocks, and an 8-buffer queue to absorb jitter.
- SoundDevice mic uses the stream callback to push blocks into a bounded queue; overflow triggers `_handle_overflow` and optional latency bump (switching to "high" latency).
- `read_audio` in sounddevice mic blocks up to 0.2 seconds waiting on queue data, then pads/truncates to the pipeline frame count; this blocking paces the pipeline toward real time.
- SoundDevice mic allows dynamic buffer sizing and latency mode ("low", "high", "default", "auto") when Unloaded or Initialized; changes while Running are ignored.
- `pulseaudio_microphone.py` captures via PyAudio in callback mode with a ring buffer sized for ~2 seconds; default 48 kHz stereo and 512-frame blocks.
- PulseAudio mic handles underruns by clearing buffers and logging every 100 underruns; it reshapes interleaved float32 data back to channel-major NumPy arrays.
- All Python sources rely on the pipeline setting buffer size to match their internal block size; mismatch triggers padding/truncation each read.

## Python Audio Sinks
- Python sinks derive from `AudioSinkPlugin` in `plugins_py/base_plugin.py` and expect channel-major NumPy arrays.
- `sounddevice_speaker.py` outputs via `sounddevice.OutputStream`, default 48 kHz, 1 channel, 256-frame blocks, with an 8-buffer queue for jitter tolerance.
- SoundDevice speaker auto-scales latency: three underflows trigger `_needs_latency_bump`, which restarts the stream with "high" latency and logs the change.
- The speaker sink's write path transposes buffer data to (frames, channels), enqueues it non-blockingly, and drops the oldest buffer on queue full, incrementing overflow count.
- `pulseaudio_speaker.py` outputs via PyAudio blocking writes, default 48 kHz stereo and 512-frame buffers; missing PyAudio raises an initialization error (as seen in logs).
- `wav_file_sink.py` (Python) mirrors the C++ WAV writer using NumPy interleaving and file I/O; format defaults are float32 with timestamped filenames.
- `null_sink.py` discards audio and tracks peak levels; useful for testing pipeline pacing without hardware.
- Python sinks expose `get_available_space`, typically as queue free slots times buffer size; the pipeline ignores this signal.

## Encryptor and Bearer Placeholders
- `EncryptorPlugin` is defined but no concrete C++ encryptor implementation exists in the repository; pipeline encryption step is effectively a no-op unless a plugin is added.
- `BearerPlugin` defines packet send/receive with latency, jitter, packet loss configuration; no concrete bearer implementation is present under `plugins_src/` or `plugins_py/`.
- Pipeline still reserves step 3 for bearer send; when absent, sample counters advance without network timing influence.

## Python Plugin Bridge Mechanics
- `PythonPluginBridge` (`src/plugins/PythonPluginBridge.cpp`) embeds the Python interpreter and initializes NumPy once per process.
- Plugin modules are imported by stripping ".py" from provided path and added to `sys.path` if not already present.
- Plugin instance is created via the `create_plugin` factory; absence or failure prints to stderr and aborts loading.
- Bridge state mirrors plugin state: Loaded after creation, Initialized after `initialize()` returns true, Running after `start()` returns true.
- Each `readAudio` and `writeAudio` call acquires the GIL via `PyGILState_Ensure`, constructs a new Python `AudioBuffer`, and copies data between C++ and NumPy arrays.
- `createPythonAudioBuffer` imports `base_plugin` on every call, allocates an `AudioBuffer`, accesses its NumPy `data` attribute, and copies channel-major data element by element.
- `copyFromPythonBuffer` reverses the copy, iterating over channels and frames; both directions are O(channels * frames) per call and allocate fresh Python objects.
- No caching of Python buffer objects or NumPy views is implemented, causing significant per-buffer allocation overhead.
- Buffer shape is fixed to (channels, frames); sinks expecting (frames, channels) transpose internally, adding more CPU overhead.
- Bridge logging is minimal; failures print Python exceptions but do not propagate error states to the pipeline beyond returning false.
- Sample rate, channels, and buffer size queries call Python methods on demand, acquiring and releasing the GIL each time.
- `getAvailableSpace` delegates to Python; pipeline does not consume this information.
- Encryption and bearer plugins written in Python would incur the same per-call marshalling overhead.

## UI and Control Path
- `MainWindow` creates a shared `ProcessingPipeline` and `PluginManager`, wiring both to `PipelineView` and `Dashboard`.
- PipelineView `autoLoad` scans `plugins_py` and `plugins/Release` or `plugins/Debug` relative to the application directory, loading all discovered plugins.
- UI enables Start only when both a source and sink are selected; bearer and encryptor are optional.
- Dashboard can only stop the pipeline; it mirrors pipeline state to UI labels.
- No UI fields exist for configuring buffer sizes, sample rates, latency modes, or device selections; these must be preconfigured via plugin parameters or defaults.

## Observed Logs (Run 1: Sine Wave -> WAV File)
- Log shows `[Pipeline] Audio source: Sine Wave Generator (state: 2)` meaning state Running, plugin name matches both C++ and Python sine generators.
- Sink reported as `WAV File Recorder (state: 2)` referencing the C++ WAV sink.
- Initialization reported 2 channels @ 48000 Hz with frame size 512, matching defaults.
- Progress logs: at elapsed 3 s processed 51,200 samples (1.0667 s audio); at 7 s processed 102,400 (2.1333 s audio); at 10 s processed 153,600 (3.2 s audio).
- Stop message indicates 171,008 samples processed (3.5627 s of audio) while wall time exceeded 10 seconds, showing pipeline throughput ~0.36x real-time.
- WAV sink progress reported "Recording: 1s", "2s", "3s" consistent with processed sample count, confirming processedSamples_ matches written frames.
- Timing mismatch: wall-clock elapsed was ~3x audio time, implying per-iteration overhead or blocking beyond the nominal 10.7 ms per 512-frame block required for real-time.

## Observed Logs (Run 2: Sine Wave -> SoundDevice Speaker)
- Source again reported as Sine Wave Generator; sink reported as SoundDevice Speaker (Python).
- Initialization set 2 channels @ 48000 Hz with frame size 256 (min of default 512 and sink's 256).
- SoundDevice speaker logged "Requesting higher latency to avoid underruns" and later "Latency bumped to high after underruns", indicating consumer callback drained queue faster than producer filled it initially.
- Progress logs (every 100 frames): elapsed 0 s at 25,600 samples (0.533 s audio), elapsed 6 s at 512,000 samples (10.6667 s audio), indicating processedSamples_ grew faster than reported elapsed.
- Elapsed values are integer seconds; 10.6 audio seconds vs 6 elapsed implies pipeline throughput ~1.77x real-time on average, but underrun logs show intermittent starvation.
- Queue size in sink is limited to 8 buffers (~42.6 ms at 256 frames), so even brief stalls in `readAudio` or Python bridge marshalling can cause underruns despite average overproduction.
- No confirmation exists that all 512,000 samples were actually played; overflow handling in sink drops oldest buffers silently while processedSamples_ continues to increment.

## Timing Mismatch Root Causes
- Lack of explicit pacing: `processingThread` runs as fast as source/sink calls permit; synthetic sources (sine) do not block, so loop can outrun real-time or stall depending on marshalling overhead.
- Python bridge overhead: each buffer allocates Python objects, imports modules, and copies channel-major arrays, adding milliseconds of overhead per 256–512 frame buffer.
- Blocking behaviors diverge: Python sounddevice mic blocks on queue get (real-time pacing), but sine source does not, leading to different timing characteristics per plugin pairing.
- Ignored backpressure: `audioSink->getAvailableSpace()` is never consulted, so sinks with small queues (sounddevice) cannot slow the producer; overflow leads to drops while counters still advance.
- Processed sample accounting ignores failures: `processedSamples_` increments even when sink write fails or drops data, overstating actual audio delivered.
- Sample rate negotiation is one-sided: pipeline assumes source rate; sink hardware may operate at a different rate, causing drift without resampling or rate matching.
- Buffer size mismatches: pipeline forces a single frame size, but sounddevice streams may internally adjust latency/blocksize when bumping to "high", causing mismatched pacing.
- Logging granularity: elapsed time is truncated to whole seconds, hiding sub-second jitter and making throughput comparisons coarse.
- Encryption/bearer steps are synchronous and would add further overhead if enabled, worsening real-time performance.
- WAV sink uses synchronous `fwrite` without buffering hints; on slow disks or antivirus interference, write time could exceed frame time budget.
- No priority or real-time thread scheduling is requested; Windows default thread priorities can be preempted by UI or other tasks, extending loop duration.

## Detailed Component Timing Implications
- C++ sine source cost per buffer is minimal (<0.1 ms) and non-blocking; pipeline loop timing is dominated by sink and bridge overhead when this source is used.
- Python sine source adds NumPy allocation and Python call overhead per buffer; rough estimates place this at 1–3 ms per 512-frame buffer depending on environment.
- WAV sink interleaving and fwrite typically fit within a 10 ms budget on SSDs but can exceed it on HDDs or under antivirus scanning; lack of double-buffering can amplify stalls.
- SoundDevice speaker enqueue is fast (<1 ms) but uses a small queue; underruns occur if producer stalls for >queue depth * block duration.
- SoundDevice mic blocks ~block duration per buffer (5.33 ms at 256 frames), naturally pacing the loop near real time when used as source.
- PulseAudio plugins (PyAudio) in blocking mode will pace to hardware; bridge overhead still applies per buffer.
- AIOC plugins depend on USB round-trip and WASAPI events; session queues capture underruns/overruns but the pipeline does not act on them.

## Data Integrity Considerations
- Encryptor step operates in place on float buffers, corrupting data for sinks unless decryptor symmetry is applied; currently unused but a risk if enabled inadvertently.
- Bearer serialization copies floats into a byte vector without endian or format negotiation; resumption on receiver side is undefined in this repository.
- Channel count changes mid-run are unsupported; source or sink `setChannels` calls after start will not adjust work buffer or pipeline assumptions.
- Python bridge copies assume contiguous channel-major layout; any deviation in Python plugin data shape will cause copy errors or silent misalignment.
- SoundDevice speaker mono fallback averages channels if channels mismatch; this can mask channel-specific content and alter levels.
- SoundDevice mic and speaker drop or pad samples to match frame size, introducing discontinuities when sizes drift.
- WAV sink rewrites header on stop; abrupt termination without stop call leaves an invalid WAV header.

## Proposed Remediations (Pacing and Timing)
- Add explicit pacing in `processingThread`: compute target next-frame timestamp based on sample rate and frameCount_, sleep when ahead, warn when behind.
- Use sink backpressure: query `audioSink_->getAvailableSpace()` and throttle or drop frames when sink queues are near capacity.
- Track actual wall-clock throughput: record per-iteration duration and compute moving average of samples per second, logging deviations from target 48 kHz.
- Promote processing thread to higher priority (e.g., `THREAD_PRIORITY_TIME_CRITICAL` on Windows) when real-time mode is requested.
- Introduce configurable `maxLatencyMs` and enforce by limiting queue depth or adjusting buffer sizes dynamically.

## Proposed Remediations (Buffering and Flow Control)
- Implement a small ring buffer between source and sink to absorb jitter and provide true backpressure signals to the producer.
- Allow independent source and sink frame sizes with an internal adaptor that batches/splits buffers while preserving timestamps.
- Expose buffer size and latency controls in the UI to avoid mismatched defaults and allow quick tuning per hardware.
- Respect sink `getAvailableSpace()` before writing; if insufficient, either wait or drop with explicit counters surfaced to the UI.
- In SoundDevice speaker, allow configurable queue depth and provide a blocking enqueue path so the pipeline thread naturally paces to playback.
- Provide a dry-run mode where the pipeline measures achievable throughput per plugin combination without real audio output, reporting headroom.

## Proposed Remediations (Python Bridge Performance)
- Cache `base_plugin.AudioBuffer` objects and reuse NumPy arrays to avoid per-call allocation and module import in `PythonPluginBridge`.
- Use `PyMemoryView_FromMemory` or NumPy frombuffer to share C++ buffer memory with Python, eliminating copy loops for sinks.
- Batch GIL acquisition by grouping operations (e.g., reuse held GIL for consecutive property queries) or minimize cross-language calls per frame.
- Add optional Cython or pybind11 adapters for high-throughput sinks/sources to reduce overhead compared to the generic bridge.
- Measure per-buffer bridge latency with timestamps before/after Python calls and surface metrics to the UI.

## Proposed Remediations (Accounting and Telemetry)
- Increment `processedSamples_` only when sink write succeeds; maintain separate counters for generated, dropped, and delivered frames.
- Extend logs to include sink return status, queue fill levels (where available), and underflow/overflow counts per sink.
- Record elapsed as fractional seconds to improve visibility into drift; include both wall-clock and audio-seconds deltas per log.
- Surface AIOC telemetry (framesCaptured/framesPlayed/underruns/overruns) to the dashboard for hardware-based pacing feedback.
- Add simple JSON telemetry output for automated analysis during soak tests.

## Proposed Remediations (Sample Rate and Format)
- Negotiate sample rate and channels across source and sink; if mismatched, insert a resampler (e.g., speexdsp or libsamplerate) and channel mapper.
- Validate buffer size compatibility after any latency bump (e.g., sounddevice switching to "high") and adjust workBuffer_ accordingly.
- Allow sinks to request specific formats; currently all paths assume float32, which may not match hardware-native formats without conversion.
- Add warning when sample rate reported by sink/device differs from source; refuse to start unless resampling is enabled.

## Proposed Remediations (File I/O)
- Move WAV writing to a dedicated worker thread with a lock-free queue to decouple disk stalls from the processing loop.
- Use buffered I/O (`std::ofstream` with streambuf tuning) and explicit flush intervals instead of per-buffer fwrite.
- Preallocate header and data size estimates to minimize fseek calls at stop time on slow storage.
- Add an option to skip disk writes during throughput tests to isolate CPU vs I/O issues.

## Validation and Test Plan
- Reproduce Run 1 with sine -> WAV, measure wall time vs audio time using high-resolution timers and confirm current ~0.36x throughput baseline.
- Reproduce Run 2 with sine -> SoundDevice speaker, collect sink underrun/overflow counts via `get_parameter("underruns")` and `("overflows")` to quantify drop/underflow rates.
- Add a synthetic pacing source that sleeps for frame duration to act as a real-time baseline; compare processedSamples_ to wall time.
- Run soak test for 60 seconds with sounddevice mic -> null sink to verify capture pacing and absence of drift or underruns.
- Profile Python bridge with cProfile around read/write calls to attribute per-buffer cost.
- Capture task manager CPU usage during runs to confirm whether CPU saturation or blocking dominates.
- Validate WAV files produced match expected duration (frames / 48000) and check header integrity with an external tool.
- When resampling is introduced, verify latency and frequency response with test tones and spectrogram analysis.

## Risks and Open Questions
- Without backpressure, pipeline can silently drop or duplicate audio depending on plugin behavior, leading to non-deterministic logs.
- Python environment availability (sounddevice, pyaudio, numpy) directly affects sink/source initialization; missing dependencies currently surface only via console logs.
- Encryptor and bearer paths are untested; enabling them could further disturb timing due to synchronous processing and buffer copying.
- UI lacks controls for device selection and latency, so users rely on defaults that may not match their hardware capabilities.
- AIOC plugins rely on device-specific IDs and WASAPI/CDC handles; behavior under failure or hotplug conditions is not covered in this audit.
- Thread priority changes may impact UI responsiveness; careful balancing is required when applying real-time priorities.

## Immediate Action Items
- Implement pacing in `ProcessingPipeline::processingThread` and integrate sink backpressure to align processedSamples_ with wall time.
- Add detailed logging (elapsed ms, success/failure per stage, sink queue depth) every N frames to observe drift in real runs.
- Cache Python buffer objects in `PythonPluginBridge` to cut per-buffer allocation overhead and remeasure throughput with sine -> WAV.
- Expose sink/source buffer size and latency mode controls in the UI to allow quick tuning away from mismatch-induced underruns.
- Add counters for dropped/failed sink writes and expose them via the dashboard for real-time visibility.
- Create a developer doc outlining recommended buffer sizes per plugin pairing (e.g., 256 frames for sounddevice, 512 for WAV).
- Verify PyAudio installation or switch default speaker to sounddevice sink to avoid initialization errors on systems without PyAudio.

## Summary of Timing Problem
- Observed mismatch: wall-clock elapsed diverges from processed audio seconds in both directions (slower than real-time with WAV sink; faster yet underrun-prone with sounddevice sink).
- Root causes: absence of pacing/backpressure, Python bridge marshalling overhead, small sink queues, and unvalidated sample rate/buffer negotiations.
- Remediation: enforce pacing, honor sink availability, optimize Python bridge, and surface telemetry to detect and correct drift early.

## Quantitative Throughput Reference
- At 48,000 Hz, each sample period is 20.833 microseconds; 512-frame buffers represent 10.6667 ms of audio.
- With 512-frame buffers, the pipeline must complete ~93.75 iterations per second to be real-time; any per-iteration cost above ~10 ms causes underruns.
- At 48,000 Hz with 256-frame buffers, each buffer spans 5.3333 ms; ~187.5 iterations per second are needed for real-time playback.
- Logging every 100 iterations at 512 frames corresponds to ~1.0667 s of audio; at 256 frames, ~0.5333 s of audio between logs.
- Run 1 processed 153,600 samples over 10 s, equating to 15,360 samples/s throughput (32% of required 48,000 samples/s).
- Run 1 final count 171,008 samples over ~10+ s equals ~16,000 samples/s, confirming sub-real-time behavior by ~3x.
- Run 2 processed 512,000 samples in 6 s, equating to ~85,333 samples/s, exceeding the 48,000 target by ~78% on average.
- SoundDevice speaker queue (8 buffers * 256 frames) holds 2,048 frames, equal to 42.67 ms of audio headroom at 48 kHz.
- A 42.67 ms queue tolerates at most a single missed 5.33 ms production cycle; two consecutive misses induce underflow.
- WAV sink writes 512 frames * 2 channels * 4 bytes = 4,096 bytes per buffer; at real-time throughput, disk writes are ~384 KB/s.
- Python bridge copies 512 frames * 2 channels = 1,024 floats per call; at 10 ms budget, copy must stay below ~2 GB/s effective to avoid overruns.
- If Python bridge overhead is 3 ms per call at 512 frames, throughput caps near 170 buffers/s, equal to ~87,000 samples/s, matching the overproduction observed in Run 2 when no blocking occurs.
- If Python bridge overhead is 15 ms per call at 512 frames, throughput caps near 33 buffers/s, equal to ~16,896 samples/s, matching the underproduction observed in Run 1.
- These derived ceilings align with measured data, suggesting per-call overhead dominates pacing when synthetic sources are used.

## Detailed Log Timeline (Run 1: WAV File)
- T0: Pipeline initialization logs 2 channels @ 48 kHz, frame 512, implying `frameCount_` remained default.
- T0+: WAV sink prints recording path and format, indicating start succeeded and file handle opened.
- T≈3s (log): 51,200 samples processed; expected real-time audio would have processed ~144,000 samples by 3 s, highlighting shortfall.
- T≈7s (log): 102,400 samples processed; expected real-time ~336,000 samples, widening deficit.
- T≈10s (log): 153,600 samples processed; expected real-time ~480,000 samples, confirming ~3.1x slower rate.
- T≈10s+: WAV sink progress logs "Recording: 1s", "2s", "3s" near sample counts, showing internal counters align with processedSamples_.
- T_stop: Pipeline logs 171,008 samples (3.56 s audio) before exit; wall time exceeds triple the audio duration.
- No bearer/encryptor logs, ruling out encryption/network overhead as contributors.
- No read failure logs appear, implying `readAudio` returned true but still incurred large per-call cost (likely Python marshalling).
- No sink write errors printed, suggesting disk writes succeeded but loop still lagged.

## Detailed Log Timeline (Run 2: SoundDevice Speaker)
- T0: Initialization sets frame size 256, channels 2, matching sink default and negotiated minimum.
- T0+: SoundDevice speaker logs start with block 256 and device info string truncated ("Realtek HD Audio 2nd output").
- T0+ shortly: Underflow notice triggers auto latency bump, indicating initial queue fill was insufficient.
- Progress log 1: elapsed 0 s, processed 25,600 samples (0.533 s audio), indicating near-instant production of the first 100 buffers.
- Progress log 2: elapsed 0 s, processed 51,200 samples (1.066 s audio), confirming pipeline outpaced the one-second mark before elapsed crossed 1 s.
- Progress log 6: elapsed 2 s, processed 153,600 samples (3.2 s audio), still ahead of wall time despite earlier underrun.
- Progress log 12: elapsed 4 s, processed 307,200 samples (6.4 s audio), maintaining ~1.6x real-time throughput.
- Progress log 20: elapsed 6 s, processed 512,000 samples (10.6667 s audio), cumulative throughput ~85k samples/s.
- Underflow handling switched latency to "high", but buffer size stayed at 256; pipeline continued producing without pacing, risking queue oscillation between underflow and overflow.
- No sink overflow logs appear by default, so potential drops are not visible without querying parameters.
- Audio heard (if any) likely contained gaps corresponding to initial underflows; later overproduction could cause pitch-up artifacts if sink consumed faster than wall time, but sounddevice stream pins playback to hardware clock, so actual audible duration would be ~10.7 s while pipeline thought 6 s elapsed.

## SoundDevice Speaker Deep Dive
- File: `plugins_py/sounddevice_speaker.py`; default sample_rate=48000, channels=1, buffer_size=256, max_queue_buffers=8.
- Initialization queries device info via `sd.query_devices`; failure sets state ERROR.
- Start resets queues and counters, seeds queue with two silence buffers to avoid immediate underflow, and opens an `sd.OutputStream`.
- Latency mode is "auto" by default; stream opens with `_effective_latency()` returning "low" unless upscaled.
- If stream open fails for requested channels, fallback to mono and reopen is attempted.
- Audio callback pulls from `audio_queue` non-blockingly; on Empty, fills zeros and records underflow.
- `_handle_underflow` increments count and requests latency bump after three underflows when auto_scale is enabled.
- `_maybe_bump_latency` closes the stream, resets queue, reopens with latency="high", restarts stream, and marks `_latency_upscaled`.
- `_maybe_bump_block_size` is stubbed (disabled) to avoid mid-run frame-size mismatch with pipeline.
- `write_audio` transposes channel-major data to (frames, channels), casts to float32, and enqueues with `put_nowait`.
- On queue full, sink drops the oldest buffer (via get_nowait), enqueues new data, increments overflow count, and returns false.
- `get_available_space` computes free slots * buffer_size; pipeline ignores this metric, so overflow handling remains local to the sink.
- Parameters allow adjusting autoScale, bufferSize, latencyMode, device, sampleRate, channels; only honored when Unloaded/Initialized.
- State transitions: initialize -> start -> running; stop stops and closes stream, sets state back to Initialized.

## SoundDevice Microphone Deep Dive
- File: `plugins_py/sounddevice_microphone.py`; default sample_rate=48000, channels=1, buffer_size=256, max_queue_buffers=8.
- Initialization queries input device via `sd.query_devices`; missing sounddevice triggers ERROR.
- Start resets queues and counters, opens `sd.InputStream` with callback `_audio_callback` and chosen latency.
- Callback pushes copies of input blocks into the bounded queue; overflow leads to dropping the oldest buffer and requesting a latency bump.
- `_maybe_bump_latency` stops and reopens the stream with latency="high" upon repeated overflows, similar to speaker behavior.
- Read path blocks up to 0.2 s on queue.get, pacing the pipeline to input availability when used as source.
- If obtained block shape differs from pipeline frame_count, data is padded or truncated to fit before transpose into channel-major layout.
- Channel mismatch handling: if channels=1 but pipeline expects more, buffer.data mean is repeated per channel within sink or source? (mic sets channels via set_channels before start).
- Parameters include bufferSize, autoScale, latencyMode, device, sampleRate, channels; only effective before start.
- Underflows are not logged directly; queue.get timeout returns false and clears buffer, incrementing no explicit counter (would be useful to add).
- stop closes the stream and reinitializes state, logging "Stopped" on success.

## WAV File Sink Deep Dive
- File: `plugins_src/WavFileSinkPlugin.cpp`; defaults: sampleRate 48000, channels 2, bufferSize 512.
- Start builds timestamped filename `recording_YYYYMMDD_HHMMSS.wav` using std::localtime and std::snprintf.
- WAV header uses RIFF chunk with audioFormat=3 (float32), blockAlign=channels*4, byteRate=sampleRate*channels*4, data chunk size patched on stop.
- Recording progress prints when `totalFrames_ % sampleRate_ < frameCount`, i.e., on crossing whole seconds boundaries.
- Write path interleaves channel-major buffer into contiguous float array; fwrite writes the full block; mismatch logs "[WavFileSink] Write error!".
- getAvailableSpace returns 1,000,000 regardless of disk state, preventing pipeline backpressure.
- setBufferSize, setSampleRate, setChannels only act when Unloaded or Initialized; changing after start is ignored.
- Stop seeks to file start, rewrites header with correct sizes, flushes, and closes; abrupt termination leaves stale header.
- Duration logged on stop equals totalFrames_/sampleRate, matching processedSamples_ counter increments.

## Python Bridge Copy Path (Expanded)
- Each call to `createPythonAudioBuffer` re-imports `base_plugin`, fetches `AudioBuffer` class, and constructs a new instance with channel/frame parameters.
- NumPy array pointer is obtained via PyArray_DATA; data copy iterates over channels outer, frames inner, assigning floats individually.
- For sinks, a fresh Python buffer is created even though sink only reads; returning false still discards allocated Python objects.
- For sources, data is copied back via `copyFromPythonBuffer`, iterating channels then frames; no vectorized memcpy is used despite contiguous layout.
- The bridge never reuses Python objects across calls, so GC pressure is proportional to buffer rate (e.g., ~180 allocations/s at 256 frames).
- GIL acquisition happens per call, not batched across multiple plugin interactions inside one frame (e.g., source read + sink write each reacquire the GIL).
- Module path manipulation occurs on each load but not on each call; however, repeated imports inside copy functions still add overhead.
- Error handling prints Python tracebacks but does not set plugin state to Error, so pipeline continues attempting calls after repeated failures.
- Buffer shape assumptions (channels, frames) may clash with plugins expecting interleaved arrays; current audio plugins transpose internally as needed.

## Failure Modes and Edge Cases
- Starting the pipeline without both source and sink returns false; UI already disables start, but programmatic calls could still attempt it.
- If source start succeeds but sink start fails, source remains running until stop is called; no rollback is performed in `start`.
- If `readAudio` returns false repeatedly, pipeline sleeps 1 ms per loop but continues logging; no automatic stop after prolonged failure.
- If sink write returns false, processedSamples_ still increments, masking dropped audio; UI sees misleading progress.
- Python bridge exceptions leave plugin state unchanged; repeated failures could spam logs without recovery.
- SoundDevice speaker overflow drops oldest buffers, causing audible skips while pipeline counters continue increasing.
- SoundDevice mic overflow requests latency bump but does not signal pipeline; processedSamples_ may undercount when reads fail.
- WAV sink crash or disk full scenarios are not handled; fwrite failure only logs once per failure occurrence.
- AIOC device disconnect mid-run would cause session reads/writes to fail; pipeline would continue sleeping without explicit stop.
- Encryption step could overrun buffer if encryptor expands data beyond buffer size; current placeholder adds 16 bytes without capacity check.

## Instrumentation and Code Change Backlog
- Add high-resolution timing around `processAudioFrame` to log per-iteration duration and identify slow stages.
- Track counts for read failures, write failures, bearer send failures, encryptor failures, and expose via getters for UI.
- Add warning when `processedSamples_ / elapsedSeconds` deviates from source sampleRate by more than 5%.
- Introduce a debug flag to log sink queue depth and underrun/overflow counts each second for sounddevice and pulseaudio sinks.
- Add a `pacingMode` enum (None, SleepToRate, SinkBackpressure, Hybrid) to configure processing loop behavior.
- Implement a monotonic clock-aligned sleep using `std::chrono::steady_clock` to target next buffer deadline.
- Provide a configuration hook to set thread priority and MMCSS task for Windows audio threads.
- Cache Python AudioBuffer objects and NumPy array descriptors in `PythonPluginBridge` to remove repeated allocation.
- Replace element-wise copies with `memcpy` where channel-major layout matches contiguous blocks to reduce copy time.
- Add a compile-time toggle to bypass Python bridge copying for plugins that can accept pointers (via PyCapsule) to shared buffers.
- Extend PluginManager to prevent unloading plugins in use and to warn on duplicate names overwriting existing entries.
- Enhance UI to show plugin parameters (buffer size, latency mode, device) and allow editing before start.
- Integrate a small metrics pane showing processedSamples_, throughput (samples/s), sink underruns/overflows, and queue depth.
- Add an optional diagnostics log file capturing timestamps, iteration durations, and plugin return codes for offline analysis.

## Manual Test Recipes
- Test A: Sine (C++) -> WAV (C++), frame 512, run 15 s; verify wall time vs audio time and WAV duration equals ~15 s.
- Test B: Sine (Python) -> WAV (C++), frame 512; compare throughput to Test A to quantify Python bridge overhead on source side.
- Test C: SoundDevice mic -> Null sink, frame 256; monitor underrun/overflow counts and ensure throughput ≈ 48,000 samples/s for 60 s.
- Test D: Sine (C++) -> SoundDevice speaker, frame 256; collect underrun/overflow counts and confirm queue remains mostly full after latency bump.
- Test E: SoundDevice mic -> SoundDevice speaker, frame 256; validate end-to-end latency subjectively and via recorded timestamps in logs.
- Test F: AIOC source -> Null sink; check telemetry for underruns/overruns and PTT state transitions under load.
- Test G: WAV sink file integrity; open recording in an external editor to verify header correctness and duration.
- Test H: Negative path; simulate sink write failure (e.g., read-only directory) and confirm pipeline handles gracefully with accurate counters.
- Test I: Mixed sample rates; modify source sampleRate to 44100, sink to 48000, observe behavior and confirm resampling guardrails once implemented.
- Test J: Python bridge stress; reduce buffer size to 128 and run sine -> WAV to measure per-buffer overhead impact.

## Deployment and Environment Considerations
- Ensure `sounddevice` and its backend (PortAudio) are installed when using sounddevice plugins; missing module produces ERROR state during initialize.
- PyAudio is optional but required for PulseAudio plugins; logs already suggest installing via `pip install pyaudio`.
- NumPy is required for all Python plugins; Python bridge initialization prints failure if NumPy import fails.
- On Windows, `NDA_ENABLE_PYTHON` must be set during CMake configure to build Python bridge; otherwise Python plugins cannot be loaded.
- Qt runtime and OpenSSL paths must remain configured per repository guidelines for UI and deploy scripts; unrelated but critical for full app operation.
- Antivirus or disk encryption can increase fwrite latency for WAV sink; consider excluding output directory during testing.
- Sounddevice device indices may differ across machines; plugin accepts `device` parameter but UI currently lacks a selector, so defaults apply.
- AIOC plugins expect specific device GUIDs; running on machines without the hardware will cause connection failures.
- High-DPI or UI theme changes do not impact audio but can affect CPU scheduling if GPU drivers are heavy; keep display driver overhead minimal during audio tests.

## Next-Step Sequencing (Suggested Order)
- Implement processing loop pacing with optional backpressure awareness to immediately address timing drift.
- Add logging and counters for sink underruns/overflows and processedSamples_ accuracy to gain visibility.
- Optimize Python bridge allocations and measure throughput again on the same scenarios to quantify gains.
- Surface buffer size and latency controls in the UI for quick tuning during manual validation.
- Add ring buffer/backpressure handling for sounddevice sinks to prevent queue starvation/drops.
- Introduce sample-rate validation and optional resampling to prevent unbounded drift when components disagree.
- Move WAV sink I/O to an async writer to isolate disk latency from the processing thread.
- After these changes, rerun Tests A–E and compare wall time vs audio time to confirm mismatch resolution.

## Diagnostics Checklist
- Verify plugin states after start: source, sink should report Running (state code 2) before entering processing loop.
- Capture wall-clock timestamps at pipeline start/stop to compare against processedSamples_/sampleRate.
- Query sounddevice sink parameters for underruns/overflows via `get_parameter("underruns")` and `("overflows")` after runs.
- For WAV runs, compute duration = frames / sampleRate from header and compare to log-derived processedSamples_.
- Enable additional verbose logging around Python bridge to catch any repeated exceptions not currently surfaced.
- Monitor CPU utilization of the processing thread; values near 100% indicate loop pacing issues vs hardware blocking.
- Inspect disk throughput during WAV recording to rule out I/O stalls; use Resource Monitor on Windows during tests.
- Confirm buffer size negotiation results via logs; mismatches between source and sink expectations should be noted before start.
- If bearer/encryptor are added, log per-packet send duration and encryption latency per buffer to detect bottlenecks early.
- Check that thread join occurs on stop to avoid dangling processing threads influencing subsequent tests.
- Validate that UI reflects pipeline stopped state and buttons toggle appropriately after failures.
- Ensure Python dependencies (sounddevice, pyaudio, numpy) import without warnings before starting audio runs.
- Verify that `processingThread` remains alive only once; multiple concurrent threads could distort timing metrics.
- Collect stack traces on any unhandled exception in the processing loop to attribute crash causes.
- Record device names and IDs used for sounddevice to reproduce results across systems.

## Plugin Parameter Matrix (Defaults and Tuning Levers)
- Sine Wave Generator (C++): sampleRate=48000, channels=2, bufferSize=512, parameter `frequency`.
- Sine Wave Generator (Python): sampleRate=48000, channels=2, bufferSize inherits from pipeline, parameter `frequency`.
- SoundDevice Microphone: sampleRate=48000, channels=1, bufferSize=256, parameters `bufferSize`, `autoScale`, `latencyMode`, `device`, `sampleRate`, `channels`.
- SoundDevice Speaker: sampleRate=48000, channels=1 (auto fallback), bufferSize=256, parameters `autoScale`, `bufferSize`, `latencyMode`, `device`, `sampleRate`, `channels`.
- PulseAudio Microphone: sampleRate=48000, channels=2, bufferSize=512, minimal parameters (sampleRate, channels).
- PulseAudio Speaker: sampleRate=48000, channels=2, bufferSize=512, minimal parameters.
- WAV File Sink (C++): sampleRate=48000, channels=2, bufferSize=512, parameter `filename` (used to override timestamp).
- Null Sink (C++): sampleRate=48000, channels=2, bufferSize=512, parameter `showMetrics`.
- AIOC Source: sampleRate=48000, channels=1, bufferFrames=512, parameters include `volume_in`, `mute_in`, `device_id`, `loopback_test`, `vcos_threshold`, `vcos_hang_ms`.
- AIOC Sink: sampleRate=48000, channels=1, bufferFrames=512, parameters include `volume_out`, `mute_out`, `device_id`, `loopback_test`.
- PluginManager: no tunable parameters via UI; plugin directories discovered via scan or auto-load paths.
- ProcessingPipeline: no external parameters exposed; frameCount_ negotiation depends solely on plugin bufferSize getters.
- Future UI work should expose buffer sizes and latency modes to allow tuning of the above parameters without code changes.

## Future Enhancements
- Add automated benchmark mode that runs synthetic pipelines (sine -> null, sine -> wav, sine -> speaker) and reports throughput/latency automatically.
- Build a small CLI harness around ProcessingPipeline to allow headless testing without Qt UI, simplifying continuous integration.
- Implement a resampling adaptor with quality presets (fast, medium, high) to reconcile mismatched sample rates.
- Add channel mapping utilities (stereo to mono, mono to stereo, channel swap) selectable per sink.
- Support metadata embedding in WAV recordings (e.g., plugin names, buffer sizes) for later forensic analysis.
- Introduce a structured logging format (JSON lines) to make ingestion into observability tools straightforward.
- Provide a watchdog that stops the pipeline if processedSamples_ stalls for N seconds to avoid runaway threads.
- Allow per-plugin profiling hooks so individual read/write durations can be measured without modifying plugin code.
- Consider using a lock-free ring buffer between C++ pipeline and Python sinks to reduce GIL contention.
- Explore moving the sine generator into the core as an optional built-in source to avoid bridge overhead when testing sinks.
- Add a synthetic sink that simply counts samples and reports throughput, isolating source performance independently of I/O.

## Validation Tools and Scripts (Proposed)
- Add a CMake option to build a `nda_bench` executable that instantiates ProcessingPipeline with configurable plugins and durations.
- Provide a Python script in `scripts/` to parse log output and compute throughput, underruns, and drift automatically.
- Include a small WAV duration checker in `scripts/` to confirm recorded files match expected lengths.
- Create a sounddevice dependency checker script to list available devices and their default sample rates for debugging initialization failures.
- Add a profiler configuration (e.g., Windows Performance Recorder preset) focused on the processing thread for timing analysis.
- Supply a PowerShell script to run the suite of manual tests (A–J) with timestamps and save logs to a results directory.
- Add a simple visualization notebook in `docs/` or `examples/` to plot processedSamples_ over time from logged data.
- Include a unit-style test that mocks sink backpressure to validate new pacing logic without hardware.

## Potential Regression Risks After Fixes
- Adding sleeps for pacing could underutilize CPU but must not introduce jitter beyond sink queue capacity.
- Backpressure logic might block pipeline thread if sink reports zero space; require timeouts to avoid deadlock.
- Thread priority elevation could starve UI or other threads; needs careful configuration and opt-in behavior.
- Shared buffer reuse in Python bridge must avoid lifetime bugs; incorrect reference management could crash the interpreter.
- Asynchronous WAV writing introduces queueing; insufficient queue depth could still drop audio if disk stalls are prolonged.
- Resampling could add latency and CPU load; quality settings should be tunable to balance performance vs fidelity.
- Expanded logging could impact performance; consider rate-limiting or sampling logs in high-throughput scenarios.

## Action Items Specific to Timing Mismatch
- Instrument elapsed vs audioSeconds per log entry with millisecond resolution to quantify drift precisely.
- Add a per-buffer timestamp to the work buffer to trace when data was generated vs consumed by sinks.
- Implement optional busy-wait spin only when behind schedule; otherwise sleep to align with target cadence.
- Respect sink backpressure: if `writeAudio` returns false, pause until sink reports available space rather than incrementing counters.
- Add detection for queue overflows/underflows in sounddevice sink and propagate warnings to the pipeline log.
- Create a configuration that forces sine generator to sleep for frame duration, providing a real-time baseline for throughput comparisons.
- Measure Python bridge per-call time using a scoped timer and log the distribution to identify heavy-tailed delays.
- Add UI indicator when processedSamples_ deviates from wall-clock derived samples by more than 5%, prompting user intervention.
- Run WAV recording with and without bridge caching to validate performance gains after optimization.
- Re-evaluate default frame size: consider 256 for all Python sinks/sources to reduce per-buffer overhead while maintaining latency bounds.
- Document recommended pairings (e.g., sine C++ -> sounddevice speaker for sink testing; sounddevice mic -> null sink for source testing) to standardize validation runs.
- Ensure stop() waits for sink queues to drain or explicitly flushes/clears to avoid misleading processedSamples_ counts at shutdown.
- After implementing fixes, rerun the exact scenarios from the provided logs and verify processed audio seconds match wall clock within ±5%.
