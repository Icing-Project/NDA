# NADE - Windows Installation & Usage Guide

## 🚀 Quick Start

### Prerequisites

1. **Windows 10/11** (64-bit)
2. **Python 3.7+** - [Download from python.org](https://www.python.org/downloads/)
3. **Visual C++ Redistributable** - Usually included with Windows

### Installation Steps

1. **Install Python Dependencies**
   ```cmd
   pip install -r requirements.txt
   ```

2. **Run NADE**
   ```cmd
   cd bin
   NADE.exe
   ```

## 📦 What's Included

```
readytoship/
├── bin/
│   ├── NADE.exe              # Main application
│   ├── Qt6*.dll               # Qt libraries
│   ├── python3*.dll           # Python runtime
│   └── *.dll                  # Other dependencies
├── plugins/
│   ├── base_plugin.py         # Base plugin classes
│   ├── sounddevice_microphone.py
│   ├── sounddevice_speaker.py
│   ├── wav_file_sink.py
│   ├── sine_wave_source.py
│   └── null_sink.py
├── lib/
│   └── (Qt plugins, platform files)
├── docs/
│   ├── WINDOWS_README.md      # This file
│   └── USER_GUIDE.md          # User guide
└── requirements.txt           # Python dependencies
```

## 🎵 Using Audio Plugins

### 1. Load Plugins
- Click **"Auto-Load Python Plugins"** button
- Or manually select plugin directory

### 2. Configure Pipeline
Choose plugins from dropdowns:
- **Audio Source**: SoundDevice Microphone, Sine Wave Generator
- **Audio Sink**: WAV File Recorder, SoundDevice Speaker

### 3. Start Recording/Playback
- Click **"Start Pipeline"**
- Monitor on Dashboard tab
- Click **"Stop Pipeline"** when done

## 🔧 Troubleshooting

### Audio Not Working
- **Check Python Dependencies**:
  ```cmd
  pip list | findstr sounddevice
  pip list | findstr numpy
  ```
- **Install/Reinstall**:
  ```cmd
  pip install --upgrade sounddevice numpy
  ```

### Application Won't Start
- **Missing DLLs**: Install Visual C++ Redistributable
- **Qt Issues**: Check that all Qt DLLs are in bin/ folder
- **Python Issues**: Verify Python 3.7+ is installed

### Plugin Errors
- Check `plugins/` folder contains all .py files
- Ensure `base_plugin.py` is present
- Run: `python -c "import sounddevice; print('OK')"`

## 📊 Audio Settings

- **Sample Rate**: 48000 Hz (default)
- **Channels**: 2 (stereo)
- **Buffer Size**: 512 samples
- **Format**: 32-bit float

## 🔐 Encryption (Optional)

NADE supports AES-256-GCM encryption for audio streams.
Configure in the Encryption tab before starting pipeline.

## 📝 File Locations

- **Recordings**: Saved in current directory as `recording_YYYYMMDD_HHMMSS.wav`
- **Logs**: Console output (redirect with `NADE.exe > log.txt`)

## 🛠️ Building from Source

See `BUILD_WINDOWS.md` for compilation instructions.

## 📄 License

See LICENSE file in the repository.

## 🐛 Support

Report issues at: https://github.com/your-repo/NADE/issues
