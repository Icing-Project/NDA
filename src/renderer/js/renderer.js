// Renderer process main script
const { nadeAPI } = window;

// Application state
const appState = {
    currentView: 'dashboard',
    audioDevices: {
        inputs: [],
        outputs: [],
        selectedInput: null,
        selectedOutput: null
    },
    isStreaming: false,
    plugins: [],
    stats: {
        latency: 0,
        cpuUsage: 0,
        bufferUnderruns: 0
    }
};

// Initialize the application
document.addEventListener('DOMContentLoaded', async () => {
    initializeTitleBar();
    initializeNavigation();
    initializeEventListeners();
    await loadInitialData();
    showView('dashboard');
});

// Title bar controls for Windows
function initializeTitleBar() {
    if (nadeAPI.platform === 'win32') {
        document.getElementById('minimizeBtn').addEventListener('click', () => {
            window.electronAPI?.minimize();
        });

        document.getElementById('maximizeBtn').addEventListener('click', () => {
            window.electronAPI?.maximize();
        });

        document.getElementById('closeBtn').addEventListener('click', () => {
            window.electronAPI?.close();
        });
    } else {
        document.getElementById('titleBar').style.display = 'none';
    }
}

// Navigation handling
function initializeNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const view = item.getAttribute('data-view');
            showView(view);

            // Update active state
            navItems.forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');
        });
    });
}

// Event listeners
function initializeEventListeners() {
    // Audio level updates
    nadeAPI.on('audio-level-update', (data) => {
        updateAudioLevels(data);
    });

    // Encryption status updates
    nadeAPI.on('encryption-status-update', (status) => {
        document.getElementById('encryptionStatus').textContent = status.enabled ? 'Active' : 'Disabled';
    });

    // Plugin status updates
    nadeAPI.on('plugin-status-update', (data) => {
        updatePluginStatus(data);
    });

    // Error handling
    nadeAPI.on('error', (error) => {
        showNotification('error', error.message);
    });

    // Settings updates
    nadeAPI.on('open-settings', () => {
        showView('settings');
    });

    // Audio device selection
    nadeAPI.on('select-audio-device', (type) => {
        showAudioDeviceSelector(type);
    });
}

// Load initial data
async function loadInitialData() {
    try {
        // Get audio devices
        const devices = await nadeAPI.enumerateAudioDevices();
        appState.audioDevices = devices;

        // Get available plugins
        const plugins = await nadeAPI.getAvailablePlugins();
        appState.plugins = plugins;

        // Get current stats
        updateStats();
    } catch (error) {
        console.error('Failed to load initial data:', error);
        showNotification('error', 'Failed to initialize application');
    }
}

// View management
function showView(viewName) {
    appState.currentView = viewName;
    const contentView = document.getElementById('contentView');

    switch (viewName) {
        case 'dashboard':
            contentView.innerHTML = renderDashboard();
            break;
        case 'audio-devices':
            contentView.innerHTML = renderAudioDevices();
            break;
        case 'encryption':
            contentView.innerHTML = renderEncryption();
            break;
        case 'plugins':
            contentView.innerHTML = renderPlugins();
            break;
        case 'settings':
            contentView.innerHTML = renderSettings();
            break;
    }
}

// Dashboard view
function renderDashboard() {
    return `
        <div class="dashboard">
            <h1>Dashboard</h1>

            <div class="card">
                <h2 class="card-header">Audio Stream Control</h2>
                <div class="stream-controls">
                    <button id="toggleStreamBtn" class="button button-primary">
                        ${appState.isStreaming ? 'Stop Stream' : 'Start Stream'}
                    </button>
                    <div class="stream-status">
                        Status: <span class="${appState.isStreaming ? 'text-success' : 'text-secondary'}">
                            ${appState.isStreaming ? 'Active' : 'Idle'}
                        </span>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2 class="card-header">Audio Levels</h2>
                <div class="audio-levels">
                    <div class="level-meter">
                        <label>Input Level</label>
                        <div class="audio-meter">
                            <div class="audio-meter-bar" id="inputLevelBar" style="width: 0%"></div>
                        </div>
                    </div>
                    <div class="level-meter">
                        <label>Output Level</label>
                        <div class="audio-meter">
                            <div class="audio-meter-bar" id="outputLevelBar" style="width: 0%"></div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2 class="card-header">System Performance</h2>
                <div class="performance-stats">
                    <div class="stat-item">
                        <span class="stat-label">Latency:</span>
                        <span class="stat-value">${appState.stats.latency} ms</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">CPU Usage:</span>
                        <span class="stat-value">${appState.stats.cpuUsage}%</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Buffer Underruns:</span>
                        <span class="stat-value">${appState.stats.bufferUnderruns}</span>
                    </div>
                </div>
            </div>
        </div>
    `;
}

// Audio devices view
function renderAudioDevices() {
    return `
        <div class="audio-devices">
            <h1>Audio Devices</h1>

            <div class="card">
                <h2 class="card-header">Input Devices</h2>
                <div class="device-list">
                    ${appState.audioDevices.inputs.map(device => `
                        <div class="device-item ${device.id === appState.audioDevices.selectedInput ? 'selected' : ''}">
                            <div class="device-info">
                                <div class="device-name">${device.name}</div>
                                <div class="device-type">${device.type}</div>
                            </div>
                            <button class="button button-secondary" onclick="selectAudioDevice('${device.id}', 'input')">
                                Select
                            </button>
                        </div>
                    `).join('')}
                </div>
            </div>

            <div class="card">
                <h2 class="card-header">Output Devices</h2>
                <div class="device-list">
                    ${appState.audioDevices.outputs.map(device => `
                        <div class="device-item ${device.id === appState.audioDevices.selectedOutput ? 'selected' : ''}">
                            <div class="device-info">
                                <div class="device-name">${device.name}</div>
                                <div class="device-type">${device.type}</div>
                            </div>
                            <button class="button button-secondary" onclick="selectAudioDevice('${device.id}', 'output')">
                                Select
                            </button>
                        </div>
                    `).join('')}
                </div>
            </div>
        </div>
    `;
}

// Encryption view
function renderEncryption() {
    return `
        <div class="encryption">
            <h1>Encryption Settings</h1>

            <div class="card">
                <h2 class="card-header">Encryption Configuration</h2>
                <form id="encryptionForm">
                    <div class="form-group">
                        <label class="form-label">Encryption Algorithm</label>
                        <select class="form-select" id="encryptionAlgorithm">
                            <option value="aes-256-gcm">AES-256-GCM</option>
                            <option value="chacha20-poly1305">ChaCha20-Poly1305</option>
                            <option value="aes-256-ctr">AES-256-CTR</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label class="form-label">Key Exchange Method</label>
                        <select class="form-select" id="keyExchange">
                            <option value="ecdh">ECDH (Elliptic Curve)</option>
                            <option value="rsa">RSA-2048</option>
                            <option value="x25519">X25519</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label class="form-label">Encryption Key</label>
                        <input type="password" class="form-input" id="encryptionKey" placeholder="Enter encryption key">
                    </div>

                    <button type="submit" class="button button-primary">Apply Settings</button>
                </form>
            </div>
        </div>
    `;
}

// Plugins view
function renderPlugins() {
    return `
        <div class="plugins">
            <h1>Plugin Manager</h1>

            <div class="card">
                <h2 class="card-header">Available Plugins</h2>
                <div class="plugin-list">
                    ${appState.plugins.map(plugin => `
                        <div class="plugin-item">
                            <div class="plugin-info">
                                <div class="plugin-name">${plugin.name}</div>
                                <div class="plugin-description">${plugin.description}</div>
                                <div class="plugin-version">Version: ${plugin.version}</div>
                            </div>
                            <div class="plugin-actions">
                                <button class="button ${plugin.active ? 'button-secondary' : 'button-primary'}"
                                        onclick="togglePlugin('${plugin.id}', ${plugin.active})">
                                    ${plugin.active ? 'Deactivate' : 'Activate'}
                                </button>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        </div>
    `;
}

// Settings view
function renderSettings() {
    return `
        <div class="settings">
            <h1>Settings</h1>

            <div class="card">
                <h2 class="card-header">Audio Settings</h2>
                <form id="audioSettingsForm">
                    <div class="form-group">
                        <label class="form-label">Buffer Size</label>
                        <select class="form-select" id="bufferSize">
                            <option value="64">64 samples (lowest latency)</option>
                            <option value="128">128 samples</option>
                            <option value="256">256 samples</option>
                            <option value="512">512 samples</option>
                            <option value="1024">1024 samples (highest stability)</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label class="form-label">Sample Rate</label>
                        <select class="form-select" id="sampleRate">
                            <option value="44100">44.1 kHz</option>
                            <option value="48000">48 kHz</option>
                            <option value="96000">96 kHz</option>
                            <option value="192000">192 kHz</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label class="form-label">Audio API</label>
                        <select class="form-select" id="audioApi">
                            <option value="WASAPI">WASAPI</option>
                            <option value="ASIO">ASIO</option>
                            <option value="WDM-KS">WDM-KS</option>
                        </select>
                    </div>

                    <button type="submit" class="button button-primary">Save Settings</button>
                </form>
            </div>

            <div class="card">
                <h2 class="card-header">Application Settings</h2>
                <form id="appSettingsForm">
                    <div class="form-group">
                        <label class="form-label">
                            <input type="checkbox" id="startMinimized"> Start minimized
                        </label>
                    </div>

                    <div class="form-group">
                        <label class="form-label">
                            <input type="checkbox" id="runAtStartup"> Run at Windows startup
                        </label>
                    </div>

                    <div class="form-group">
                        <label class="form-label">
                            <input type="checkbox" id="enableNotifications"> Enable notifications
                        </label>
                    </div>

                    <button type="submit" class="button button-primary">Save Settings</button>
                </form>
            </div>
        </div>
    `;
}

// Helper functions
function updateAudioLevels(data) {
    if (data.input !== undefined) {
        const inputBar = document.getElementById('inputLevelBar');
        if (inputBar) {
            inputBar.style.width = `${Math.min(100, data.input * 100)}%`;
        }
    }

    if (data.output !== undefined) {
        const outputBar = document.getElementById('outputLevelBar');
        if (outputBar) {
            outputBar.style.width = `${Math.min(100, data.output * 100)}%`;
        }
    }
}

async function selectAudioDevice(deviceId, type) {
    try {
        await nadeAPI.selectAudioDevice(deviceId, type);
        if (type === 'input') {
            appState.audioDevices.selectedInput = deviceId;
        } else {
            appState.audioDevices.selectedOutput = deviceId;
        }
        showNotification('success', `${type} device selected successfully`);
        showView('audio-devices'); // Refresh view
    } catch (error) {
        showNotification('error', `Failed to select ${type} device: ${error.message}`);
    }
}

async function togglePlugin(pluginId, isActive) {
    try {
        if (isActive) {
            await nadeAPI.deactivatePlugin(pluginId);
        } else {
            await nadeAPI.activatePlugin(pluginId);
        }

        // Update local state
        const plugin = appState.plugins.find(p => p.id === pluginId);
        if (plugin) {
            plugin.active = !isActive;
        }

        showView('plugins'); // Refresh view
        showNotification('success', `Plugin ${isActive ? 'deactivated' : 'activated'} successfully`);
    } catch (error) {
        showNotification('error', `Failed to toggle plugin: ${error.message}`);
    }
}

function showNotification(type, message) {
    // TODO: Implement notification system
    console.log(`[${type}] ${message}`);
}

async function updateStats() {
    try {
        const stats = await nadeAPI.getAudioStats();
        appState.stats = stats;

        // Update status bar
        document.getElementById('latencyValue').textContent = `${stats.latency} ms`;
        document.getElementById('cpuUsage').textContent = `${stats.cpuUsage}%`;
    } catch (error) {
        console.error('Failed to update stats:', error);
    }

    // Update every second
    setTimeout(updateStats, 1000);
}

// Make functions available globally for inline handlers
window.selectAudioDevice = selectAudioDevice;
window.togglePlugin = togglePlugin;