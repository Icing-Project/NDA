# NADE Desktop Application

Professional Audio Encryption Bridge System for Windows

## Overview

NADE Desktop is a Windows-native audio processing system designed to provide real-time encryption for various audio communication channels. Built specifically for Windows 10/11, it leverages Windows audio APIs for optimal performance while maintaining a flexible plugin architecture for different audio sources and encryption methods.

## Features

- **Windows Native Performance**: Optimized for Windows audio subsystem
- **Universal Audio Support**: WASAPI, WDM, ASIO support for professional audio
- **Plugin Architecture**: Hot-swappable modules for different audio sources
- **Low Latency**: <10ms with ASIO, <20ms with WASAPI
- **Modern UI**: Windows 11 styled interface built with Electron
- **Real-time Encryption**: AES-256-GCM encryption with minimal latency
- **Multi-device Support**: USB radio interfaces, Bluetooth devices, and more

## System Requirements

### Minimum
- Windows 10 20H2 or later
- Intel i3 / AMD Ryzen 3 or equivalent
- 4GB RAM
- Onboard audio device

### Recommended
- Windows 11
- Intel i5 / AMD Ryzen 5 or equivalent
- 8GB RAM
- USB Audio Interface

### Professional
- Windows 11 Pro
- Intel i7 / AMD Ryzen 7 or equivalent
- 16GB RAM
- ASIO-compatible audio interface

## Installation

### Prerequisites

1. Install Node.js (v16 or later)
2. Install Windows Build Tools (Admin PowerShell):
   ```powershell
   npm install --global windows-build-tools
   ```

### Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/nade-team/nade-desktop.git
   cd nade-desktop
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Build native modules:
   ```bash
   npm run build:native
   ```

4. Run in development mode:
   ```bash
   npm run dev
   ```

## Building

### Build for Windows

```bash
# Build installer
npm run build:win

# Build portable version
npm run dist
```

The built application will be in the `dist` folder.

## Architecture

### Core Components

- **Electron Main Process**: Application lifecycle and system integration
- **Windows Audio Engine**: Native audio processing with WASAPI/ASIO
- **Plugin System**: Modular input/output/processor plugins
- **Encryption Core**: Real-time audio encryption/decryption

### Plugin Types

1. **Input Plugins**: Audio capture from various sources
   - WASAPI Input
   - ASIO Input
   - Bluetooth Input
   - Radio USB Interface

2. **Processor Plugins**: Audio processing and encryption
   - NADE Encryption Processor
   - Audio Effects
   - Noise Reduction

3. **Output Plugins**: Audio playback to devices
   - WASAPI Output
   - ASIO Output
   - Network Output

## Usage

### Basic Operation

1. Launch NADE Desktop
2. Select input and output audio devices
3. Configure encryption settings
4. Click "Start Stream" to begin encrypted audio transmission

### Advanced Features

- **Plugin Management**: Add/remove plugins via the Plugin Manager
- **Audio Settings**: Configure buffer size, sample rate, and API
- **Encryption Key Exchange**: Support for ECDH, RSA, and X25519

## Development

### Project Structure

```
nade-desktop-windows/
├── src/
│   ├── main/              # Electron main process
│   ├── renderer/          # UI (HTML/CSS/JS)
│   ├── native/           # Native Windows audio modules
│   └── plugins/          # Plugin modules
├── resources/            # Application resources
├── build/               # Build configuration
└── dist/               # Built application
```

### Creating Plugins

Plugins should extend the appropriate base class and implement required methods:

```javascript
class MyInputPlugin extends EventEmitter {
    constructor() {
        super();
        this.name = 'My Input Plugin';
        this.description = 'Custom audio input';
    }

    async initialize() {
        // Plugin initialization
    }

    async cleanup() {
        // Cleanup resources
    }
}
```

## Troubleshooting

### Common Issues

1. **"Windows cannot access audio device"**
   - Run as Administrator
   - Check Windows privacy settings for microphone access

2. **High latency**
   - Use ASIO drivers for lowest latency
   - Reduce buffer size in audio settings

3. **Audio crackling**
   - Increase buffer size
   - Check CPU usage and close unnecessary applications

## License

MIT License - See LICENSE file for details

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

For issues and feature requests, please use the GitHub issue tracker.