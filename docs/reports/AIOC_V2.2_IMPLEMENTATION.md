# AIOC Plugin v2.2 Implementation Report

**Date:** 2026-01-11
**Version:** v2.2
**Status:** ✅ Complete

## Executive Summary

This document details the complete implementation of AIOC (Amateur Radio Audio Interface) plugin updates to v2.2 architecture and full PTT (Push-to-Talk) integration. The implementation includes:

- Migration from polling to event-driven architecture with ring buffers
- Full keyboard PTT support (T and Space keys)
- Dynamic WASAPI device selection with AIOC auto-detection
- PTT mode selection (HID Manual, CDC Manual, VPTT Auto)
- Critical bug fixes for PTT routing

---

## Table of Contents

1. [Architecture Changes](#architecture-changes)
2. [PTT Integration](#ptt-integration)
3. [Device Selection UI](#device-selection-ui)
4. [Bug Fixes](#bug-fixes)
5. [Files Modified](#files-modified)
6. [Testing Guidelines](#testing-guidelines)

---

## Architecture Changes

### 1. Ring Buffer Implementation

**Objective:** Migrate AIOC plugins from blocking WASAPI calls to non-blocking ring buffer architecture.

#### Changes to `plugins_src/AIOCPluginCommon.h`

Added ring buffer members and thread infrastructure:

```cpp
// v2.2: Ring buffers for async operation (200ms capacity at 48kHz = 9600 frames)
RingBuffer captureRingBuffer_;    // AIOC mic → host
RingBuffer playbackRingBuffer_;   // Host → AIOC speaker

// v2.2: Background threads (decouple WASAPI from plugin)
std::thread captureThread_;
std::thread playbackThread_;
std::atomic<bool> captureThreadRunning_;
std::atomic<bool> playbackThreadRunning_;

// v2.2: Event-driven callbacks
std::function<void()> dataReadyCallback_;
std::function<void()> spaceAvailableCallback_;
int dataReadyThreshold_ = 512;
int spaceAvailableThreshold_ = 512;
```

Added method declarations:

```cpp
void captureThreadFunc();
void playbackThreadFunc();
void setDataReadyCallback(std::function<void()> callback);
void setSpaceAvailableCallback(std::function<void()> callback);
int getCaptureRingBufferAvailable() const;
int getPlaybackRingBufferAvailable() const;
```

#### Changes to `plugins_src/AIOCPluginCommon.cpp`

**Background Capture Thread (217 lines):**
- Polls WASAPI capture client for available packets
- Converts interleaved audio to planar format
- Writes to `captureRingBuffer_` (lock-free, non-blocking)
- Triggers `dataReadyCallback_` when threshold met
- Runs at `THREAD_PRIORITY_TIME_CRITICAL`

**Background Playback Thread (220 lines):**
- Reads from `playbackRingBuffer_` when WASAPI has space
- Converts planar to interleaved format
- Writes to WASAPI render client
- Triggers `spaceAvailableCallback_` when space available
- Runs at `THREAD_PRIORITY_TIME_CRITICAL`

**Modified `start()` method:**
```cpp
bool AIOCSession::start() {
    // ... WASAPI initialization ...

    // v2.2: Initialize ring buffers (200ms capacity at 48kHz = 9600 frames)
    int ringBufferCapacity = (sampleRate_ * 200) / 1000;
    captureRingBuffer_.initialize(channels_, ringBufferCapacity);
    playbackRingBuffer_.initialize(channels_, ringBufferCapacity);

    // Start background threads
    captureThreadRunning_.store(true, std::memory_order_release);
    playbackThreadRunning_.store(true, std::memory_order_release);
    captureThread_ = std::thread([this]() { this->captureThreadFunc(); });
    playbackThread_ = std::thread([this]() { this->playbackThreadFunc(); });

    return true;
}
```

**Modified `stop()` method:**
```cpp
void AIOCSession::stop() {
    // Signal threads to stop
    captureThreadRunning_.store(false, std::memory_order_release);
    playbackThreadRunning_.store(false, std::memory_order_release);

    // CRITICAL: Unlock mutex before joining threads to prevent deadlock
    if (captureThread_.joinable()) captureThread_.join();
    if (playbackThread_.joinable()) playbackThread_.join();

    // ... WASAPI cleanup ...
}
```

**Modified `readCapture()` method:**
- Now reads from `captureRingBuffer_` instead of WASAPI directly
- Non-blocking operation (returns false if insufficient data)
- Preserves volume/mute processing

**Modified `writePlayback()` method:**
- Now writes to `playbackRingBuffer_` instead of WASAPI directly
- Non-blocking operation (returns false on overrun)
- Preserves PTT handling via `handlePtt()`

### 2. Plugin v2.2 Interface

#### Changes to `plugins_src/AIOCSourcePlugin.cpp`

Added v2.2 async interface methods:

```cpp
// v2.2: Event-driven async mode support
bool supportsAsyncMode() const override {
    return true;  // We use ring buffer + background thread
}

void setDataReadyCallback(DataReadyCallback callback) override {
    dataReadyCallback_ = callback;
    session_.setDataReadyCallback(callback);  // Propagate to AIOCSession
}

int getDataReadyThreshold() const override {
    return dataReadyThreshold_;  // Default: 512 frames
}
```

Added private members:
```cpp
DataReadyCallback dataReadyCallback_;
int dataReadyThreshold_;
```

#### Changes to `plugins_src/AIOCSinkPlugin.cpp`

Added v2.2 async interface methods:

```cpp
bool supportsAsyncMode() const override {
    return true;  // Ring buffer + background thread
}

bool isNonBlocking() const override {
    return true;  // writeAudio() writes to ring buffer, never blocks WASAPI
}

void setSpaceAvailableCallback(SpaceAvailableCallback callback) override {
    spaceAvailableCallback_ = callback;
    session_.setSpaceAvailableCallback(callback);  // Propagate to AIOCSession
}

int getAvailableSpace() const override {
    if (state_ != PluginState::Running) return 0;
    // v2.2: Return ring buffer space (not bufferFrames_)
    return session_.getPlaybackRingBufferAvailable();
}
```

Added private members:
```cpp
SpaceAvailableCallback spaceAvailableCallback_;
int spaceAvailableThreshold_;
```

---

## PTT Integration

### 1. Keyboard Input Handling

#### Changes to `include/ui/UnifiedPipelineView.h`

Added protected method declarations:
```cpp
protected:
    // Keyboard input for PTT (T and Space keys)
    void keyPressEvent(QKeyEvent* event) override;
    void keyReleaseEvent(QKeyEvent* event) override;
    void focusOutEvent(QFocusEvent* event) override;
```

Added private helper methods:
```cpp
private:
    // PTT helpers
    bool isAIOCSink(std::shared_ptr<AudioSinkPlugin> sink) const;
    void updatePTTButtonState();
```

#### Changes to `src/ui/UnifiedPipelineView.cpp`

**Constructor addition:**
```cpp
UnifiedPipelineView::UnifiedPipelineView(QWidget *parent)
    : QWidget(parent), pttActive_(false), bridgeModeActive_(false)
{
    // ... existing code ...

    setFocusPolicy(Qt::StrongFocus);  // Enable keyboard input

    // ... rest of constructor ...
}
```

**Keyboard event handlers:**
```cpp
void UnifiedPipelineView::keyPressEvent(QKeyEvent* event)
{
    // CRITICAL: Prevent OS key repeat
    if (event->isAutoRepeat()) {
        event->accept();
        return;
    }

    // PTT keys: T and Space
    if ((event->key() == Qt::Key_T || event->key() == Qt::Key_Space)
        && pttButton_->isEnabled()) {
        onPTTPressed();
        event->accept();
        return;
    }

    event->ignore();
}

void UnifiedPipelineView::keyReleaseEvent(QKeyEvent* event)
{
    if (event->isAutoRepeat()) {
        event->accept();
        return;
    }

    if ((event->key() == Qt::Key_T || event->key() == Qt::Key_Space)
        && pttActive_) {
        onPTTReleased();
        event->accept();
        return;
    }

    event->ignore();
}

void UnifiedPipelineView::focusOutEvent(QFocusEvent* event)
{
    // CRITICAL: Force PTT release if focus lost while PTT held
    // (User won't receive keyReleaseEvent if they Alt-Tab while holding PTT)
    if (pttActive_) {
        onPTTReleased();
    }
    QWidget::focusOutEvent(event);
}
```

Added includes:
```cpp
#include <QKeyEvent>
#include <QFocusEvent>
#include <algorithm>  // For std::transform
```

### 2. PTT Routing Bug Fix (CRITICAL)

**Problem:** PTT handlers were targeting `txSource_` instead of `txSink_`. AIOC Sink owns PTT control, not Source.

**Original (INCORRECT) code:**
```cpp
void UnifiedPipelineView::onPTTPressed()
{
    // ... UI updates ...

    if (txSource_) {
        std::string supportsPTT = txSource_->getParameter("supports_ptt");
        if (supportsPTT == "true") {
            txSource_->setParameter("ptt_active", "true");  // WRONG!
        }
    }
}
```

**Fixed code:**
```cpp
void UnifiedPipelineView::onPTTPressed()
{
    pttActive_ = true;
    pttButton_->setProperty("active", true);
    pttButton_->style()->unpolish(pttButton_);
    pttButton_->style()->polish(pttButton_);

    // CORRECTED: Target TX Sink (AIOC Sink owns PTT control)
    if (txSink_ && isAIOCSink(txSink_)) {
        txSink_->setParameter("ptt_state", "true");
    }
}

void UnifiedPipelineView::onPTTReleased()
{
    pttActive_ = false;
    pttButton_->setProperty("active", false);
    pttButton_->style()->unpolish(pttButton_);
    pttButton_->style()->polish(pttButton_);

    // CORRECTED: Target TX Sink (AIOC Sink owns PTT control)
    if (txSink_ && isAIOCSink(txSink_)) {
        txSink_->setParameter("ptt_state", "false");
    }
}
```

**Parameter name corrected:**
- Was: `"ptt_active"` (wrong)
- Now: `"ptt_state"` (matches AIOCSinkPlugin.cpp line 110)

### 3. Conditional PTT Button Enabling

**Helper method for AIOC detection:**
```cpp
bool UnifiedPipelineView::isAIOCSink(std::shared_ptr<AudioSinkPlugin> sink) const
{
    if (!sink) return false;
    auto info = sink->getInfo();
    std::string name = info.name;
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);
    return name.find("aioc") != std::string::npos;
}
```

**PTT button state management:**
```cpp
void UnifiedPipelineView::updatePTTButtonState()
{
    bool isAIOC = isAIOCSink(txSink_);

    pttButton_->setEnabled(isAIOC);
    pttButton_->setToolTip(isAIOC ?
        "Push-to-Talk (Hold T or Space)" :
        "PTT only available with AIOC Sink");

    // Force release if disabled while active
    if (!isAIOC && pttActive_) {
        onPTTReleased();
    }
}
```

**Integration with status updates:**
```cpp
void UnifiedPipelineView::updateTXStatus()
{
    // ... existing status update code ...

    // Update PTT button state (enabled only for AIOC sink)
    updatePTTButtonState();
}
```

### 4. PTT Safety Features

**Force PTT release when stopping TX pipeline:**
```cpp
void UnifiedPipelineView::onStopTXClicked()
{
    if (!txPipeline_) return;

    txPipeline_->stop();

    // Force PTT release when stopping TX pipeline
    if (pttActive_) {
        onPTTReleased();
    }

    stopTXButton_->setEnabled(false);
    updateTXStatus();
    emit txPipelineStopped();
}
```

**CSS for disabled PTT button state:**
```cpp
#pttButton:disabled {
    background-color: #334155;
    color: #64748b;
    opacity: 0.5;
}
```

---

## Device Selection UI

### 1. WASAPI Device Enumeration

#### Changes to `plugins_src/AIOCPluginCommon.h`

Added device info struct:
```cpp
// v2.2: WASAPI device enumeration support
struct WASAPIDeviceInfo
{
    std::string id;           // GUID string for setParameter("device_id")
    std::string friendlyName; // Display name for UI
};

// Enumerate available WASAPI audio devices
// direction: 0 = eCapture (microphones), 1 = eRender (speakers)
std::vector<WASAPIDeviceInfo> enumerateWASAPIDevices(int direction);
```

#### Changes to `plugins_src/AIOCPluginCommon.cpp`

Implemented device enumeration function (82 lines):

```cpp
std::vector<WASAPIDeviceInfo> enumerateWASAPIDevices(int direction)
{
    std::vector<WASAPIDeviceInfo> devices;

#ifdef _WIN32
    // direction: 0 = eCapture (microphones), 1 = eRender (speakers)
    EDataFlow dataFlow = static_cast<EDataFlow>((direction == 0) ? eCapture : eRender);

    // Initialize COM
    HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    bool comInitialized = SUCCEEDED(hr);

    // Create device enumerator
    IMMDeviceEnumerator* enumerator = nullptr;
    hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr,
                          CLSCTX_ALL, __uuidof(IMMDeviceEnumerator),
                          (void**)&enumerator);
    if (FAILED(hr)) {
        if (comInitialized) CoUninitialize();
        return devices;
    }

    // Enumerate active audio endpoints
    IMMDeviceCollection* collection = nullptr;
    hr = enumerator->EnumAudioEndpoints(dataFlow, DEVICE_STATE_ACTIVE, &collection);
    if (FAILED(hr)) {
        enumerator->Release();
        if (comInitialized) CoUninitialize();
        return devices;
    }

    UINT count = 0;
    collection->GetCount(&count);

    for (UINT i = 0; i < count; ++i) {
        IMMDevice* device = nullptr;
        if (FAILED(collection->Item(i, &device))) continue;

        // Get device ID (GUID string)
        LPWSTR pwszID = nullptr;
        device->GetId(&pwszID);
        if (pwszID) {
            std::wstring wid(pwszID);
            std::string id(wid.begin(), wid.end());
            CoTaskMemFree(pwszID);

            // Get friendly name from property store
            IPropertyStore* props = nullptr;
            std::string friendlyName = "Unknown Device";
            if (SUCCEEDED(device->OpenPropertyStore(STGM_READ, &props))) {
                PROPVARIANT varName;
                PropVariantInit(&varName);
                if (SUCCEEDED(props->GetValue(PKEY_Device_FriendlyName, &varName))) {
                    if (varName.pwszVal) {
                        std::wstring wname(varName.pwszVal);
                        friendlyName = std::string(wname.begin(), wname.end());
                    }
                    PropVariantClear(&varName);
                }
                props->Release();
            }

            WASAPIDeviceInfo info;
            info.id = id;
            info.friendlyName = friendlyName;
            devices.push_back(info);
        }

        device->Release();
    }

    collection->Release();
    enumerator->Release();
    if (comInitialized) CoUninitialize();
#endif

    return devices;
}
```

### 2. PluginSidebar AIOC UI

#### Changes to `include/ui/PluginSidebar.h`

Added method declarations:
```cpp
// v2.2: AIOC-specific UI helpers
void addAIOCDeviceSelector(const QString& label, const QString& key, int direction);
void addPTTModeSelector(const QString& label, const QString& key);
```

#### Changes to `src/ui/PluginSidebar.cpp`

**Forward declarations:**
```cpp
// v2.2: Forward declarations for WASAPI device enumeration
namespace nda {
    struct WASAPIDeviceInfo {
        std::string id;
        std::string friendlyName;
    };
    std::vector<WASAPIDeviceInfo> enumerateWASAPIDevices(int direction);
}
```

**AIOC device selector implementation:**
```cpp
void PluginSidebar::addAIOCDeviceSelector(const QString& label, const QString& key, int direction)
{
    QVBoxLayout *contentLayout = qobject_cast<QVBoxLayout*>(contentWidget_->layout());
    if (!contentLayout) return;

    QLabel *labelWidget = new QLabel(label, this);
    labelWidget->setObjectName("paramLabel");
    contentLayout->addWidget(labelWidget);

    QComboBox *combo = new QComboBox(this);
    combo->setObjectName("paramCombo");
    combo->setMinimumHeight(35);

    // Enumerate WASAPI devices
    // direction: 0 = eCapture (microphones), 1 = eRender (speakers)
    auto devices = enumerateWASAPIDevices(direction);

    int aiocIndex = -1;
    for (size_t i = 0; i < devices.size(); ++i) {
        combo->addItem(QString::fromStdString(devices[i].friendlyName),
                       QString::fromStdString(devices[i].id));

        // Pre-select first device containing "aioc" (case-insensitive)
        if (aiocIndex == -1) {
            std::string name = devices[i].friendlyName;
            std::transform(name.begin(), name.end(), name.begin(), ::tolower);
            if (name.find("aioc") != std::string::npos) {
                aiocIndex = static_cast<int>(i);
            }
        }
    }

    if (aiocIndex >= 0) {
        combo->setCurrentIndex(aiocIndex);
    }

    contentLayout->addWidget(combo);
    parameterWidgets_[key.toStdString()] = combo;
}
```

**PTT mode selector implementation:**
```cpp
void PluginSidebar::addPTTModeSelector(const QString& label, const QString& key)
{
    QVBoxLayout *contentLayout = qobject_cast<QVBoxLayout*>(contentWidget_->layout());
    if (!contentLayout) return;

    QLabel *labelWidget = new QLabel(label, this);
    labelWidget->setObjectName("paramLabel");
    contentLayout->addWidget(labelWidget);

    QComboBox *combo = new QComboBox(this);
    combo->setObjectName("paramCombo");
    combo->setMinimumHeight(35);

    // PTT mode options with userData for parameter value
    combo->addItem("HID Manual (Recommended)", "hid_manual");
    combo->addItem("CDC Manual (COM Port)", "cdc_manual");
    combo->addItem("VPTT Auto (Voice Activated)", "vptt_auto");

    // Default to HID Manual (index 0)
    combo->setCurrentIndex(0);

    contentLayout->addWidget(combo);
    parameterWidgets_[key.toStdString()] = combo;
}
```

**AIOC-specific UI generation in `showPluginConfig()`:**
```cpp
void PluginSidebar::showPluginConfig(std::shared_ptr<BasePlugin> plugin)
{
    // ... existing code ...

    // v2.2: AIOC Sink specific UI
    if (info.name.find("AIOC") != std::string::npos && info.type == PluginType::AudioSink) {
        addAIOCDeviceSelector("AIOC Output Device:", "device_id", 1);  // 1 = eRender (speakers)
        addPTTModeSelector("PTT Mode:", "ptt_mode");
        addSlider("VPTT Threshold:", "vptt_threshold", 0, 32768, 64);
        addSlider("VPTT Hang (ms):", "vptt_hang_ms", 0, 2000, 200);
        addTextInput("CDC Port (auto-detect):", "cdc_port");
    }

    // v2.2: AIOC Source specific UI
    if (info.name.find("AIOC") != std::string::npos && info.type == PluginType::AudioSource) {
        addAIOCDeviceSelector("AIOC Input Device:", "device_id", 0);  // 0 = eCapture (microphones)
        addSlider("VCOS Threshold:", "vcos_threshold", 0, 32768, 32);
        addSlider("VCOS Hang (ms):", "vcos_hang_ms", 0, 2000, 200);
    }

    // ... rest of method ...
}
```

**QComboBox userData support in `onApplyClicked()`:**
```cpp
void PluginSidebar::onApplyClicked()
{
    if (!currentPlugin_) return;

    for (const auto& [key, widget] : parameterWidgets_) {
        QString value;

        if (auto* combo = qobject_cast<QComboBox*>(widget)) {
            // v2.2: Prefer userData (device GUID, PTT mode string) over display text
            value = combo->currentData().toString();
            if (value.isEmpty()) {
                value = combo->currentText();
            }
        } else if (auto* slider = qobject_cast<QSlider*>(widget)) {
            value = QString::number(slider->value());
        } else if (auto* lineEdit = qobject_cast<QLineEdit*>(widget)) {
            value = lineEdit->text();
        } else if (auto* checkbox = qobject_cast<QCheckBox*>(widget)) {
            value = checkbox->isChecked() ? "true" : "false";
        }

        currentPlugin_->setParameter(key, value.toStdString());
        emit parameterChanged(key, value.toStdString());
    }
}
```

---

## Bug Fixes

### 1. Hardcoded Device GUIDs Removed

**Problem:** AIOC plugins had hardcoded device GUIDs from developer's system, causing "Audio device open failed" errors on other systems.

#### Changes to `plugins_src/AIOCSourcePlugin.cpp`

**Before:**
```cpp
AIOCSourcePlugin()
{
    session_.setDeviceIds(
        "{0D94B72A-8A15-4C85-B8F7-5AC442A88BFB}", // Hardcoded GUID
        session_.deviceOutId());
    // ...
}
```

**After:**
```cpp
AIOCSourcePlugin()
    : state_(PluginState::Unloaded),
      sampleRate_(48000),
      channels_(1),
      bufferFrames_(512),
      loopbackTest_(false),
      dataReadyThreshold_(512)
{
    // v2.2: Use default WASAPI capture device initially
    // User can select specific AIOC device via PluginSidebar UI
    session_.setSampleRate(sampleRate_);
    session_.setChannels(channels_);
    session_.setBufferFrames(bufferFrames_);
}
```

#### Changes to `plugins_src/AIOCSinkPlugin.cpp`

**Before:**
```cpp
AIOCSinkPlugin()
{
    session_.setDeviceIds(
        session_.deviceInId(),
        "{DF6E2579-254F-44A3-AFD9-301BDD499759}"); // Hardcoded GUID
    session_.setCdcPort("COM8"); // Hardcoded COM port
    // ...
}
```

**After:**
```cpp
AIOCSinkPlugin()
    : state_(PluginState::Unloaded),
      sampleRate_(48000),
      channels_(1),
      bufferFrames_(512),
      pttArmed_(false),
      pttMode_(AIOCPttMode::HidManual),
      loopbackTest_(false),
      spaceAvailableThreshold_(512)
{
    // v2.2: Use default WASAPI render device initially
    // User can select specific AIOC device via PluginSidebar UI
    // COM port will be set by user or auto-detected
    session_.setSampleRate(sampleRate_);
    session_.setChannels(channels_);
    session_.setBufferFrames(bufferFrames_);
    session_.setPttMode(pttMode_);
}
```

**Result:** AIOC plugins now use default WASAPI device on startup, allowing immediate operation. Users can select specific AIOC device via UI.

### 2. Compilation Error Fixes

**Issue:** Duplicate struct definition and type conversion errors.

**Fixes:**
- Removed duplicate `WASAPIDeviceInfo` struct definition (was defined twice in header)
- Added `static_cast<EDataFlow>` for proper enum type conversion
- Changed brace initialization to explicit object creation for MSVC compatibility
- Used forward declarations in PluginSidebar.cpp to avoid circular dependencies

---

## Files Modified

### Plugin Core Files
| File | Lines Changed | Description |
|------|---------------|-------------|
| `plugins_src/AIOCPluginCommon.h` | +30 | Ring buffer members, thread declarations, device struct |
| `plugins_src/AIOCPluginCommon.cpp` | +520 | Background threads, ring buffer logic, device enumeration |
| `plugins_src/AIOCSourcePlugin.cpp` | +15, -3 | v2.2 async interface, removed hardcoded GUID |
| `plugins_src/AIOCSinkPlugin.cpp` | +20, -4 | v2.2 non-blocking interface, removed hardcoded GUID |

### UI Files
| File | Lines Changed | Description |
|------|---------------|-------------|
| `include/ui/UnifiedPipelineView.h` | +8 | Keyboard event overrides, PTT helper declarations |
| `src/ui/UnifiedPipelineView.cpp` | +120 | Keyboard handlers, PTT routing fix, button state management |
| `include/ui/PluginSidebar.h` | +3 | AIOC UI method declarations |
| `src/ui/PluginSidebar.cpp` | +110 | Device/PTT selectors, AIOC UI blocks, userData support |

### Total Impact
- **8 files modified**
- **~800+ lines added**
- **~10 lines removed**
- **0 files deleted**

---

## Testing Guidelines

### Unit Testing

#### 1. Keyboard PTT Testing
**Objective:** Verify keyboard input triggers PTT correctly.

**Steps:**
1. Configure TX pipeline: Any Source → AIOC Sink
2. Start TX pipeline
3. Hold **T key** → Verify PTT button turns green, AIOC PTT LED lights
4. Release **T key** → Verify PTT button turns gray, LED off
5. Repeat with **Space bar**
6. Hold T + **Alt-Tab** away → Verify PTT releases (focus loss handler)

**Acceptance:**
- Both T and Space keys trigger PTT
- PTT releases on focus loss
- No key repeat issues (should not flicker)

#### 2. AIOC Device Detection
**Objective:** Verify PTT button conditionally enables.

**Steps:**
1. Configure TX pipeline: Any Source → **AIOC Sink** → Verify PTT button enabled (not gray)
2. Switch to: Any Source → **Null Sink** → Verify PTT button disabled (gray)
3. While PTT held, switch to Null Sink → Verify PTT releases

**Acceptance:**
- PTT button enabled only for AIOC Sink
- Tooltip changes based on state
- PTT forced release when switching away from AIOC

#### 3. Device Selection UI
**Objective:** Verify WASAPI device enumeration and AIOC pre-selection.

**Steps:**
1. Select AIOC Sink in TX pipeline
2. Click AIOC Sink to open PluginSidebar
3. Verify "AIOC Output Device" dropdown populated with WASAPI devices
4. Verify first device containing "AIOC" or "aioc" is pre-selected
5. Change device → Click Apply → Verify parameter sent to plugin

**Acceptance:**
- All WASAPI render devices listed
- AIOC device auto-selected if present
- Device GUID (not friendly name) sent to plugin

#### 4. PTT Mode Selection
**Objective:** Verify PTT mode dropdown functionality.

**Steps:**
1. Open AIOC Sink in PluginSidebar
2. Verify "PTT Mode" dropdown shows:
   - HID Manual (Recommended) [default selected]
   - CDC Manual (COM Port)
   - VPTT Auto (Voice Activated)
3. Select each mode → Click Apply
4. Verify mode string sent to plugin (`hid_manual`, `cdc_manual`, `vptt_auto`)

**Acceptance:**
- All three modes available
- HID Manual selected by default
- Mode string (not display text) sent to plugin

### Integration Testing

#### Golden Path: Bridge Mode Test
**Objective:** End-to-end test of AIOC v2.2 with PTT.

**Steps:**
1. Configure TX: Windows Mic → AIOC Sink (HID Manual PTT)
2. Configure RX: AIOC Source → Windows Speaker
3. Open AIOC Sink in sidebar → Select AIOC device → Apply
4. Open AIOC Source in sidebar → Select AIOC device → Apply
5. Click "Start Both"
6. Press **T key** → Speak into Windows mic
7. Release **T** → Verify AIOC PTT LED off
8. Verify audio from AIOC source plays through Windows speaker
9. Run for **20 minutes** (soak test)

**Acceptance:**
- No audio dropouts
- Underruns < 10, overruns < 10 (check telemetry)
- PTT latency < 50ms (keyboard to LED)
- Audio remains intelligible under load

### Ring Buffer Stability Test

**Objective:** Verify 200ms ring buffers prevent underruns/overruns.

**Steps:**
1. Start TX: Windows Mic → AIOC Sink
2. Start RX: AIOC Source → Windows Speaker
3. Run for 20 minutes continuous
4. Simulate load: Open browser, play video, run CPU stress
5. Monitor telemetry: `txUnderruns`, `txOverruns`, `rxUnderruns`, `rxOverruns`

**Acceptance:**
- Total underruns + overruns < 20 over 20 minutes
- No audible clicks or pops
- Latency remains stable (check `txLatency`, `rxLatency`)

### Edge Case Testing

#### 1. Rapid PTT Toggling
**Steps:**
1. Configure TX with AIOC Sink
2. Rapidly press/release T key 100 times
3. Verify no stuck PTT state
4. Verify no memory leaks (check Task Manager)

#### 2. Pipeline Stop While PTT Held
**Steps:**
1. Start TX pipeline with AIOC Sink
2. Hold **T key** (PTT active)
3. Click "Stop TX"
4. Verify PTT released (LED off)

#### 3. Device Unplug During Operation
**Steps:**
1. Start TX/RX pipelines with AIOC device
2. Physically disconnect AIOC USB
3. Verify pipeline stops gracefully (no crash)
4. Reconnect AIOC → Restart pipelines → Verify recovery

---

## Known Limitations

### Deferred to Future Releases

1. **Multiple AIOC Devices**
   - Current: Selects first device containing "AIOC"
   - Future: Support multiple AIOC devices by index

2. **COM Port Auto-Detection**
   - Current: Manual COM port entry required
   - Future: Auto-detect AIOC CDC via VID/PID (0x1209:0x7388)

3. **Linux/macOS Support**
   - Current: Windows-only (WASAPI)
   - Future: Linux ALSA/PulseAudio, macOS Core Audio

4. **PTT Latency Metrics**
   - Current: Telemetry exists but not exposed in UI
   - Future: Real-time PTT latency display

### Accepted Limitations

1. **Sample Rate Mismatch**
   - WASAPI shared mode handles resampling transparently
   - May introduce slight latency increase

2. **HID Device Access**
   - Requires Windows HID drivers
   - Some systems may need manual driver installation

---

## References

### Related Documentation
- [v2.2 Architecture Overview](./V2.1_DOCUMENTATION_INDEX.md)
- [WASAPI Ring Buffer Implementation](./WASAPI_RING_BUFFER_IMPLEMENTATION_PLAN.md)
- [Windows Plugin Development](../development/plugins.md)
- [One-Week Release Strategy](../strategy/one-week-release.md)

### Code Patterns Referenced
- `plugins_src/WindowsMicrophoneSourcePlugin.cpp` - Ring buffer capture pattern
- `plugins_src/WindowsSpeakerSinkPlugin.cpp` - Ring buffer playback pattern
- `include/audio/RingBuffer.h` - Lock-free SPSC ring buffer API

### External Standards
- [USB Audio Class 2.0 Specification](https://www.usb.org/document-library/audio-device-class-specification-20)
- [WASAPI Documentation (Microsoft)](https://docs.microsoft.com/en-us/windows/win32/coreaudio/wasapi)
- [Qt Event Handling](https://doc.qt.io/qt-6/eventsandfilters.html)

---

## Changelog

### v2.2 (2026-01-11)
- ✅ Migrated AIOC plugins to ring buffer architecture
- ✅ Implemented keyboard PTT (T and Space keys)
- ✅ Fixed critical PTT routing bug (txSource → txSink)
- ✅ Added WASAPI device enumeration and selection UI
- ✅ Added PTT mode selector (HID/CDC/VPTT)
- ✅ Removed hardcoded device GUIDs
- ✅ Implemented conditional PTT button enabling
- ✅ Added PTT safety features (focus loss, pipeline stop)

---

**End of Report**
