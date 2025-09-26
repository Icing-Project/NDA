const EventEmitter = require('events');

class WASAPIOutput extends EventEmitter {
    constructor() {
        super();
        this.id = 'wasapi-output';
        this.name = 'WASAPI Output';
        this.description = 'Windows Audio Session API output plugin for low-latency audio playback';
        this.version = '1.0.0';

        this.isInitialized = false;
        this.isPlaying = false;
        this.deviceId = null;
        this.playbackOptions = {
            sampleRate: 48000,
            channels: 2,
            bitDepth: 16,
            exclusive: false,
            bufferSize: 256
        };

        this.statistics = {
            packetsPlayed: 0,
            bufferUnderruns: 0,
            latency: 0
        };
    }

    async initialize() {
        console.log(`Initializing ${this.name}...`);

        // In production, this would initialize the WASAPI playback device
        // For now, we'll simulate initialization
        this.isInitialized = true;

        this.emit('status', {
            message: 'WASAPI output initialized',
            initialized: true
        });

        return true;
    }

    async selectDevice(deviceId) {
        if (!this.isInitialized) {
            throw new Error('Plugin not initialized');
        }

        this.deviceId = deviceId;
        console.log(`Selected WASAPI output device: ${deviceId}`);

        this.emit('status', {
            message: `Output device selected: ${deviceId}`,
            deviceId: deviceId
        });

        return true;
    }

    async startPlayback(options = {}) {
        if (!this.isInitialized) {
            throw new Error('Plugin not initialized');
        }

        if (!this.deviceId) {
            throw new Error('No device selected');
        }

        if (this.isPlaying) {
            return { success: true, message: 'Already playing' };
        }

        // Merge options
        this.playbackOptions = { ...this.playbackOptions, ...options };

        console.log(`Starting WASAPI playback with options:`, this.playbackOptions);

        // Calculate latency
        this.statistics.latency = (this.playbackOptions.bufferSize / this.playbackOptions.sampleRate) * 1000;

        this.isPlaying = true;

        this.emit('status', {
            message: 'Playback started',
            playing: true,
            options: this.playbackOptions,
            latency: this.statistics.latency
        });

        return { success: true, message: 'Playback started' };
    }

    async stopPlayback() {
        if (!this.isPlaying) {
            return { success: true, message: 'Not playing' };
        }

        this.isPlaying = false;

        this.emit('status', {
            message: 'Playback stopped',
            playing: false
        });

        return { success: true, message: 'Playback stopped' };
    }

    async playAudio(audioData) {
        if (!this.isPlaying) {
            throw new Error('Playback not started');
        }

        try {
            // In production, this would send audio to the WASAPI device
            // For now, we'll simulate playback
            await this.simulatePlayback(audioData);

            this.statistics.packetsPlayed++;

            // Simulate occasional buffer underruns for testing
            if (Math.random() < 0.001) {
                this.statistics.bufferUnderruns++;
                this.emit('warning', {
                    message: 'Buffer underrun detected',
                    underruns: this.statistics.bufferUnderruns
                });
            }

            return { success: true, packetsPlayed: this.statistics.packetsPlayed };

        } catch (error) {
            this.emit('error', {
                message: 'Playback error',
                error: error.message
            });
            throw error;
        }
    }

    async simulatePlayback(audioData) {
        // Simulate processing time based on buffer size
        const processingTime = (this.playbackOptions.bufferSize / this.playbackOptions.sampleRate) * 1000;

        return new Promise((resolve) => {
            setTimeout(() => {
                // Emit playback event
                this.emit('data', {
                    timestamp: audioData.timestamp,
                    bufferSize: audioData.buffer.length,
                    latency: this.statistics.latency,
                    packetsPlayed: this.statistics.packetsPlayed
                });
                resolve();
            }, processingTime / 2); // Simulate half the buffer time
        });
    }

    async updateSettings(settings) {
        const validSettings = ['sampleRate', 'channels', 'bitDepth', 'exclusive', 'bufferSize'];
        const wasPlaying = this.isPlaying;

        if (wasPlaying) {
            await this.stopPlayback();
        }

        for (const [key, value] of Object.entries(settings)) {
            if (validSettings.includes(key)) {
                this.playbackOptions[key] = value;
            }
        }

        // Recalculate latency
        this.statistics.latency = (this.playbackOptions.bufferSize / this.playbackOptions.sampleRate) * 1000;

        if (wasPlaying) {
            await this.startPlayback();
        }

        this.emit('status', {
            message: 'Settings updated',
            settings: this.playbackOptions,
            latency: this.statistics.latency
        });

        return this.playbackOptions;
    }

    getStatistics() {
        return {
            ...this.statistics,
            isPlaying: this.isPlaying,
            deviceId: this.deviceId
        };
    }

    async cleanup() {
        console.log(`Cleaning up ${this.name}...`);

        if (this.isPlaying) {
            await this.stopPlayback();
        }

        this.removeAllListeners();
        this.isInitialized = false;

        return true;
    }
}

module.exports = WASAPIOutput;