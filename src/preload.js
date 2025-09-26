const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('nadeAPI', {
  // Audio device management
  enumerateAudioDevices: () => ipcRenderer.invoke('enumerate-audio-devices'),
  selectAudioDevice: (deviceId, type) => ipcRenderer.invoke('select-audio-device', deviceId, type),

  // Audio stream control
  toggleAudioStream: (shouldStart) => ipcRenderer.invoke('toggle-audio-stream', shouldStart),
  getAudioStats: () => ipcRenderer.invoke('get-audio-stats'),

  // Settings
  updateAudioSettings: (settings) => ipcRenderer.invoke('update-audio-settings', settings),
  updateEncryptionSettings: (settings) => ipcRenderer.invoke('update-encryption-settings', settings),

  // Plugin management
  getAvailablePlugins: () => ipcRenderer.invoke('get-available-plugins'),
  activatePlugin: (pluginId) => ipcRenderer.invoke('activate-plugin', pluginId),
  deactivatePlugin: (pluginId) => ipcRenderer.invoke('deactivate-plugin', pluginId),

  // Event listeners
  on: (channel, callback) => {
    const validChannels = [
      'open-settings',
      'open-audio-settings',
      'open-plugin-manager',
      'select-audio-device',
      'show-about',
      'plugins-reloaded',
      'audio-level-update',
      'encryption-status-update',
      'plugin-status-update',
      'error'
    ];

    if (validChannels.includes(channel)) {
      ipcRenderer.on(channel, (event, ...args) => callback(...args));
    }
  },

  // Remove listener
  removeListener: (channel, callback) => {
    ipcRenderer.removeListener(channel, callback);
  },

  // One-time listeners
  once: (channel, callback) => {
    const validChannels = [
      'audio-device-selected',
      'settings-updated',
      'plugin-activated',
      'plugin-deactivated'
    ];

    if (validChannels.includes(channel)) {
      ipcRenderer.once(channel, (event, ...args) => callback(...args));
    }
  },

  // System information
  platform: process.platform,
  version: process.versions.electron
});