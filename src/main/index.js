const { app, BrowserWindow, ipcMain, Menu } = require('electron');
const path = require('path');
const { WindowsAudioEngine } = require('./windowsAudioEngine');
const { PluginManager } = require('./pluginManager');

class NADEWindowsApp {
  constructor() {
    this.mainWindow = null;
    this.audioEngine = null;
    this.pluginManager = null;
    this.isDev = process.argv.includes('--dev');
  }

  async initialize() {
    // Initialize audio engine
    this.audioEngine = new WindowsAudioEngine({
      api: 'WASAPI',
      latency: 'low',
      exclusive: true
    });

    // Initialize plugin manager
    this.pluginManager = new PluginManager();
    await this.pluginManager.loadPlugins();

    // Create main window
    this.createMainWindow();

    // Set up IPC handlers
    this.setupIPCHandlers();

    // Initialize audio engine
    try {
      await this.audioEngine.initialize();
      console.log('Audio engine initialized successfully');
    } catch (error) {
      console.error('Failed to initialize audio engine:', error);
    }
  }

  createMainWindow() {
    this.mainWindow = new BrowserWindow({
      width: 1200,
      height: 800,
      minWidth: 800,
      minHeight: 600,
      webPreferences: {
        nodeIntegration: false,
        contextIsolation: true,
        preload: path.join(__dirname, '../preload.js')
      },
      // Windows 11 style
      titleBarStyle: process.platform === 'win32' ? 'hidden' : 'default',
      titleBarOverlay: process.platform === 'win32' ? {
        color: '#2f3241',
        symbolColor: '#74b1be',
        height: 40
      } : false,
      icon: path.join(__dirname, '../../resources/icon.ico')
    });

    // Load the renderer
    if (this.isDev) {
      this.mainWindow.loadFile(path.join(__dirname, '../renderer/index.html'));
      this.mainWindow.webContents.openDevTools();
    } else {
      this.mainWindow.loadFile(path.join(__dirname, '../renderer/index.html'));
    }

    // Handle window closed
    this.mainWindow.on('closed', () => {
      this.mainWindow = null;
    });

    // Create application menu
    this.createApplicationMenu();
  }

  createApplicationMenu() {
    const template = [
      {
        label: 'File',
        submenu: [
          {
            label: 'Settings',
            accelerator: 'CmdOrCtrl+,',
            click: () => {
              this.mainWindow.webContents.send('open-settings');
            }
          },
          { type: 'separator' },
          {
            label: 'Exit',
            accelerator: process.platform === 'darwin' ? 'Cmd+Q' : 'Ctrl+Q',
            click: () => {
              app.quit();
            }
          }
        ]
      },
      {
        label: 'Audio',
        submenu: [
          {
            label: 'Select Input Device',
            click: () => {
              this.mainWindow.webContents.send('select-audio-device', 'input');
            }
          },
          {
            label: 'Select Output Device',
            click: () => {
              this.mainWindow.webContents.send('select-audio-device', 'output');
            }
          },
          { type: 'separator' },
          {
            label: 'Audio Settings',
            click: () => {
              this.mainWindow.webContents.send('open-audio-settings');
            }
          }
        ]
      },
      {
        label: 'Plugins',
        submenu: [
          {
            label: 'Manage Plugins',
            click: () => {
              this.mainWindow.webContents.send('open-plugin-manager');
            }
          },
          {
            label: 'Reload Plugins',
            click: async () => {
              await this.pluginManager.reloadPlugins();
              this.mainWindow.webContents.send('plugins-reloaded');
            }
          }
        ]
      },
      {
        label: 'View',
        submenu: [
          { role: 'reload' },
          { role: 'forceReload' },
          { role: 'toggleDevTools' },
          { type: 'separator' },
          { role: 'resetZoom' },
          { role: 'zoomIn' },
          { role: 'zoomOut' },
          { type: 'separator' },
          { role: 'togglefullscreen' }
        ]
      },
      {
        label: 'Help',
        submenu: [
          {
            label: 'Documentation',
            click: () => {
              require('electron').shell.openExternal('https://nade.app/docs');
            }
          },
          {
            label: 'About',
            click: () => {
              this.mainWindow.webContents.send('show-about');
            }
          }
        ]
      }
    ];

    const menu = Menu.buildFromTemplate(template);
    Menu.setApplicationMenu(menu);
  }

  setupIPCHandlers() {
    // Audio device enumeration
    ipcMain.handle('enumerate-audio-devices', async () => {
      return await this.audioEngine.enumerateDevices();
    });

    // Audio device selection
    ipcMain.handle('select-audio-device', async (event, deviceId, type) => {
      return await this.audioEngine.selectDevice(deviceId, type);
    });

    // Start/stop audio stream
    ipcMain.handle('toggle-audio-stream', async (event, shouldStart) => {
      if (shouldStart) {
        return await this.audioEngine.startStream();
      } else {
        return await this.audioEngine.stopStream();
      }
    });

    // Plugin management
    ipcMain.handle('get-available-plugins', async () => {
      return this.pluginManager.getAvailablePlugins();
    });

    // Plugin activation
    ipcMain.handle('activate-plugin', async (event, pluginId) => {
      return await this.pluginManager.activatePlugin(pluginId);
    });

    // Plugin deactivation
    ipcMain.handle('deactivate-plugin', async (event, pluginId) => {
      return await this.pluginManager.deactivatePlugin(pluginId);
    });

    // Get audio statistics
    ipcMain.handle('get-audio-stats', async () => {
      return this.audioEngine.getStatistics();
    });

    // Update audio settings
    ipcMain.handle('update-audio-settings', async (event, settings) => {
      return await this.audioEngine.updateSettings(settings);
    });

    // Encryption settings
    ipcMain.handle('update-encryption-settings', async (event, settings) => {
      return await this.audioEngine.updateEncryptionSettings(settings);
    });
  }

  async cleanup() {
    if (this.audioEngine) {
      await this.audioEngine.cleanup();
    }
    if (this.pluginManager) {
      await this.pluginManager.cleanup();
    }
  }
}

// Application lifecycle
let nadeApp = null;

app.whenReady().then(async () => {
  // Set application name
  app.setName('NADE Desktop');

  // Create and initialize app
  nadeApp = new NADEWindowsApp();
  await nadeApp.initialize();

  // Handle macOS activation
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      nadeApp.createMainWindow();
    }
  });
});

// Handle all windows closed
app.on('window-all-closed', async () => {
  if (process.platform !== 'darwin') {
    if (nadeApp) {
      await nadeApp.cleanup();
    }
    app.quit();
  }
});

// Handle app quit
app.on('before-quit', async (event) => {
  event.preventDefault();
  if (nadeApp) {
    await nadeApp.cleanup();
  }
  app.exit();
});

// Handle unhandled promise rejections
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});