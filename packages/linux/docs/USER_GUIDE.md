# NADE User Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Audio Plugins](#audio-plugins)
3. [Recording Audio](#recording-audio)
4. [Playback](#playback)
5. [Encryption](#encryption)
6. [Troubleshooting](#troubleshooting)

## Getting Started

### First Launch

**Windows:**
```cmd
NADE.bat
```

**Linux/Mac:**
```bash
./NADE.sh
```

### Main Window

The application has 4 tabs:
- **Pipeline Configuration** - Set up your audio processing chain
- **Dashboard** - Monitor live performance metrics
- **Plugins** - Manage installed plugins (removed in favor of auto-load)
- **Settings** - Configure application preferences

## Audio Plugins

### Available Plugins

#### Audio Sources (Input)
- **SoundDevice Microphone** - System microphone input (recommended)
- **PulseAudio Microphone** - Linux-specific microphone (legacy)
- **Sine Wave Generator** - Test tone generator

#### Audio Sinks (Output)
- **WAV File Recorder** - Record to WAV file
- **SoundDevice Speaker** - Play through system speakers (recommended)
- **PulseAudio Speaker** - Linux-specific speaker output (legacy)
- **Null Sink** - Debug/monitoring (console output)

### Loading Plugins

1. Click **"Auto-Load Python Plugins"** button
2. All plugins from `plugins/` folder are loaded automatically
3. They appear in the dropdown menus

Or manually:
1. Click **"Load Plugins from Directory"**
2. Select the `plugins/` folder
3. Plugins are scanned and loaded

## Recording Audio

### Step-by-Step

1. **Load Plugins** (if not already loaded)
   - Click "Auto-Load Python Plugins"

2. **Configure Pipeline**
   - Audio Source: Select "SoundDevice Microphone"
   - Audio Sink: Select "WAV File Recorder"
   - (Encryptor and Network Transport are optional)

3. **Start Recording**
   - Click "Start Pipeline" button
   - Status shows "Pipeline Running"
   - Switch to Dashboard tab to see metrics

4. **Stop Recording**
   - Click "Stop Pipeline" button
   - WAV file is saved automatically
   - Filename: `recording_YYYYMMDD_HHMMSS.wav`

### Recording Settings

Default settings:
- **Sample Rate**: 48000 Hz
- **Channels**: 2 (stereo)
- **Format**: 32-bit float PCM
- **Location**: Current directory

### Checking Recording

The console shows:
```
[WavFileSink] Recording to: recording_20251016_213232.wav
[WavFileSink] Format: 48000 Hz, 2 channels, 32-bit float
[WavFileSink] Recording: 1s
[WavFileSink] Recording: 2s
...
[WavFileSink] Saved 240000 frames (5.00 seconds)
```

## Playback

### Live Monitoring

1. **Audio Source**: SoundDevice Microphone
2. **Audio Sink**: SoundDevice Speaker
3. Click "Start Pipeline"
4. You'll hear your microphone through speakers (may cause feedback!)

### Playing Test Tone

1. **Audio Source**: Sine Wave Generator
2. **Audio Sink**: SoundDevice Speaker
3. Click "Start Pipeline"
4. You'll hear a 440 Hz test tone

## Encryption

### Basic Encryption

1. Go to **Encryption** tab
2. Enable "Use Encryption"
3. Select algorithm: AES-256-GCM (recommended)
4. Enter encryption key or generate one
5. Configure pipeline with Encryptor plugin
6. Start pipeline

**Note**: Encrypted audio is stored in encrypted format in WAV file.

### Network Streaming (Advanced)

1. Configure encryption (both sender and receiver)
2. Add **Network Transport** plugin to pipeline
3. Set IP address and port
4. Sender: Microphone → Encryptor → Network
5. Receiver: Network → Decryptor → Speaker

## Troubleshooting

### No Audio Captured

**Check:**
- Is microphone connected?
- Is correct microphone selected in system settings?
- Try "Auto-Load Python Plugins" again
- Check console for error messages

**Fix:**
```bash
# Verify sounddevice works
python -c "import sounddevice as sd; print(sd.query_devices())"

# Reinstall if needed
pip install --upgrade sounddevice
```

### Poor Audio Quality

**Symptoms**: Crackling, dropouts, distortion

**Solutions**:
1. Increase buffer size (in settings)
2. Close other audio applications
3. Use SoundDevice plugins (not PulseAudio)
4. Check CPU usage in Dashboard tab

### Recording Too Short

**Symptoms**: Recording stops early or is shorter than expected

**Check:**
- Console for "Audio read failed" messages
- Dashboard for processed samples count
- System audio settings

**Fix:**
- Use SoundDevice Microphone (most reliable)
- Check `[Pipeline] Stopped after processing X samples` message
- Ensure Python dependencies are up to date

### Application Crashes

**On Start:**
- Missing DLLs (Windows): Run `windeployqt.exe`
- Missing Python: Install Python 3.7+
- Missing Qt: Install Qt6

**During Recording:**
- Buffer overflow: Increase buffer size
- Plugin error: Check console for Python tracebacks
- Memory: Close other applications

**On Stop:**
- Should now stop cleanly with new version
- If crash persists, check console for error

## Performance Optimization

### For Low Latency
- Reduce buffer size (Settings tab)
- Use SoundDevice plugins
- Disable encryption if not needed
- Close unnecessary applications

### For Stability
- Increase buffer size to 1024 or 2048
- Use default sample rate (48000 Hz)
- Monitor CPU usage in Dashboard
- Ensure adequate disk space for recordings

## Keyboard Shortcuts

- **Ctrl+Q**: Quit application
- **F1**: Help (opens this guide)
- **Tab switching**: Navigate between tabs

## File Formats

### WAV Files
- **Format**: RIFF WAVE
- **Encoding**: IEEE Float (32-bit)
- **Channels**: 2 (stereo)
- **Sample Rate**: 48000 Hz
- **Compatible with**: Audacity, VLC, Windows Media Player, etc.

### Plugin Files
- **Format**: Python (.py)
- **Location**: `plugins/` directory
- **Base class**: Defined in `base_plugin.py`

## Advanced Usage

### Custom Plugins

Create your own audio plugin:

1. Copy `sine_wave_source.py` as template
2. Modify `create_plugin()` function
3. Implement required methods (initialize, start, stop, etc.)
4. Save in `plugins/` folder
5. Restart NADE and auto-load plugins

### Batch Processing

Record multiple files:
1. Start recording
2. Stop and wait for save
3. Start again (new filename generated)
4. Repeat as needed

### Monitoring Metrics

Dashboard shows:
- **Latency**: End-to-end processing delay
- **CPU Load**: Processing overhead
- **Processed Samples**: Total audio processed
- **Buffer Status**: Queue health

## Support

- **Documentation**: See `docs/` folder
- **Build Issues**: See `BUILD_WINDOWS.md`
- **Bug Reports**: Check console output for errors
- **Updates**: Check repository for latest version

## Tips & Tricks

1. **Always use SoundDevice plugins** - Most reliable
2. **Test with Sine Wave first** - Verify pipeline works
3. **Check VERSION.txt** - Know which version you're running
4. **Monitor Dashboard** - Watch for issues in real-time
5. **Keep plugins updated** - Run `pip install -r requirements.txt`

---

**NADE v1.0.0** - Plugin-Based Audio Encryption System
