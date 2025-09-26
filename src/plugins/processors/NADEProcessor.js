const EventEmitter = require('events');
const crypto = require('crypto');

class NADEProcessor extends EventEmitter {
    constructor() {
        super();
        this.id = 'nade-processor';
        this.name = 'NADE Encryption Processor';
        this.description = 'Core audio encryption/decryption processor using NADE protocol';
        this.version = '1.0.0';

        this.isInitialized = false;
        this.encryptionEnabled = false;
        this.encryptionSettings = {
            algorithm: 'aes-256-gcm',
            keyExchange: 'ecdh',
            key: null,
            iv: null
        };

        this.statistics = {
            packetsEncrypted: 0,
            packetsDecrypted: 0,
            processingTime: 0,
            errors: 0
        };
    }

    async initialize() {
        console.log(`Initializing ${this.name}...`);

        // Initialize encryption components
        this.initializeEncryption();

        this.isInitialized = true;

        this.emit('status', {
            message: 'NADE processor initialized',
            initialized: true
        });

        return true;
    }

    initializeEncryption() {
        // Generate default encryption key and IV if not provided
        if (!this.encryptionSettings.key) {
            this.encryptionSettings.key = crypto.randomBytes(32); // 256-bit key
        }

        if (!this.encryptionSettings.iv) {
            this.encryptionSettings.iv = crypto.randomBytes(16); // 128-bit IV
        }

        console.log('Encryption initialized with algorithm:', this.encryptionSettings.algorithm);
    }

    async processAudio(audioData, mode = 'encrypt') {
        if (!this.isInitialized) {
            throw new Error('Processor not initialized');
        }

        if (!this.encryptionEnabled) {
            // Pass through without processing
            return audioData;
        }

        const startTime = Date.now();

        try {
            let processedData;

            if (mode === 'encrypt') {
                processedData = await this.encryptAudio(audioData);
                this.statistics.packetsEncrypted++;
            } else {
                processedData = await this.decryptAudio(audioData);
                this.statistics.packetsDecrypted++;
            }

            const processingTime = Date.now() - startTime;
            this.statistics.processingTime = processingTime;

            this.emit('data', {
                mode: mode,
                inputSize: audioData.buffer.length,
                outputSize: processedData.buffer.length,
                processingTime: processingTime,
                timestamp: audioData.timestamp
            });

            return processedData;

        } catch (error) {
            this.statistics.errors++;
            this.emit('error', {
                message: `Failed to ${mode} audio`,
                error: error.message
            });
            throw error;
        }
    }

    async encryptAudio(audioData) {
        const cipher = crypto.createCipheriv(
            this.encryptionSettings.algorithm,
            this.encryptionSettings.key,
            this.encryptionSettings.iv
        );

        const encryptedBuffer = Buffer.concat([
            cipher.update(audioData.buffer),
            cipher.final()
        ]);

        // For GCM mode, append auth tag
        let authTag = null;
        if (this.encryptionSettings.algorithm.includes('gcm')) {
            authTag = cipher.getAuthTag();
        }

        return {
            ...audioData,
            buffer: encryptedBuffer,
            encrypted: true,
            authTag: authTag
        };
    }

    async decryptAudio(audioData) {
        const decipher = crypto.createDecipheriv(
            this.encryptionSettings.algorithm,
            this.encryptionSettings.key,
            this.encryptionSettings.iv
        );

        // For GCM mode, set auth tag
        if (this.encryptionSettings.algorithm.includes('gcm') && audioData.authTag) {
            decipher.setAuthTag(audioData.authTag);
        }

        const decryptedBuffer = Buffer.concat([
            decipher.update(audioData.buffer),
            decipher.final()
        ]);

        return {
            ...audioData,
            buffer: decryptedBuffer,
            encrypted: false,
            authTag: null
        };
    }

    async setEncryptionKey(key, format = 'hex') {
        if (!key) {
            throw new Error('Encryption key is required');
        }

        try {
            if (format === 'hex') {
                this.encryptionSettings.key = Buffer.from(key, 'hex');
            } else if (format === 'base64') {
                this.encryptionSettings.key = Buffer.from(key, 'base64');
            } else if (format === 'utf8') {
                // Derive key from passphrase using PBKDF2
                const salt = crypto.randomBytes(16);
                this.encryptionSettings.key = crypto.pbkdf2Sync(key, salt, 100000, 32, 'sha256');
            } else {
                this.encryptionSettings.key = key;
            }

            // Generate new IV for the new key
            this.encryptionSettings.iv = crypto.randomBytes(16);

            this.emit('status', {
                message: 'Encryption key updated',
                keyLength: this.encryptionSettings.key.length * 8
            });

            return true;

        } catch (error) {
            this.emit('error', {
                message: 'Failed to set encryption key',
                error: error.message
            });
            throw error;
        }
    }

    enableEncryption(enable = true) {
        this.encryptionEnabled = enable;

        this.emit('status', {
            message: `Encryption ${enable ? 'enabled' : 'disabled'}`,
            encryptionEnabled: enable
        });

        return enable;
    }

    async updateSettings(settings) {
        const validSettings = ['algorithm', 'keyExchange'];

        for (const [key, value] of Object.entries(settings)) {
            if (validSettings.includes(key)) {
                this.encryptionSettings[key] = value;
            }
        }

        // Reinitialize encryption with new settings
        this.initializeEncryption();

        this.emit('status', {
            message: 'Settings updated',
            settings: this.encryptionSettings
        });

        return this.encryptionSettings;
    }

    getStatistics() {
        return {
            ...this.statistics,
            encryptionEnabled: this.encryptionEnabled,
            algorithm: this.encryptionSettings.algorithm
        };
    }

    async cleanup() {
        console.log(`Cleaning up ${this.name}...`);

        // Clear sensitive data
        if (this.encryptionSettings.key) {
            this.encryptionSettings.key.fill(0);
        }
        if (this.encryptionSettings.iv) {
            this.encryptionSettings.iv.fill(0);
        }

        this.removeAllListeners();
        this.isInitialized = false;

        return true;
    }
}

module.exports = NADEProcessor;