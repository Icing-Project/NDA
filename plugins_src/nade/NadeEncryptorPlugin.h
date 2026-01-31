/**
 * @file NadeEncryptorPlugin.h
 * @brief NDA Processor Plugin for TX (transmit) path.
 *
 * This plugin reads pre-buffered FSK audio from NadeExternalIO and outputs
 * it to the NDA TX pipeline. The audio thread never blocks on Python/GIL.
 *
 * Responsibilities:
 *   - Receive "text as audio" input and decode it using TextEncoding.h
 *   - Queue decoded text to NadeExternalIO for Python processing
 *   - Read pre-buffered FSK audio from NadeExternalIO
 *   - Output FSK to NDA TX pipeline
 *
 * Threading:
 *   - processAudio() called from audio thread (non-blocking)
 *   - Reads from lock-free pre-buffer (filled by Python worker thread)
 *
 * @version 3.0
 * @date 2026-01-30
 */

#pragma once

#include "plugins/AudioProcessorPlugin.h"
#include "NadeExternalIO.h"
#include <memory>
#include <vector>

namespace nade {

/**
 * @brief Processor plugin for TX path - generates FSK audio from text messages.
 */
class NadeEncryptorPlugin : public nda::AudioProcessorPlugin {
public:
    NadeEncryptorPlugin();
    ~NadeEncryptorPlugin() override;

    // =========================================================================
    // Plugin Metadata (BasePlugin)
    // =========================================================================

    nda::PluginInfo getInfo() const override;
    nda::PluginType getType() const override { return nda::PluginType::Processor; }
    nda::PluginState getState() const override { return state_; }

    // =========================================================================
    // Lifecycle (BasePlugin)
    // =========================================================================

    bool initialize() override;
    void shutdown() override;
    bool start() override;
    void stop() override;

    // =========================================================================
    // Configuration (BasePlugin)
    // =========================================================================

    void setParameter(const std::string& key, const std::string& value) override;
    std::string getParameter(const std::string& key) const override;

    // =========================================================================
    // Audio Processing (AudioProcessorPlugin)
    // =========================================================================

    bool processAudio(AudioBuffer& buffer) override;

    int getSampleRate() const override { return sampleRate_; }
    int getChannelCount() const override { return channelCount_; }
    void setSampleRate(int rate) override { sampleRate_ = rate; }
    void setChannelCount(int channels) override { channelCount_ = channels; }

    double getProcessingLatency() const override { return 0.0; }

    // =========================================================================
    // Key Delivery (Called by NDA CryptoManager)
    // =========================================================================

    /**
     * @brief Called by NDA when crypto keys are ready.
     *
     * Creates or retrieves NadeExternalIO singleton.
     *
     * @param localPrivateKey 32-byte X25519 private key
     * @param localPublicKey 32-byte X25519 public key
     * @param remotePublicKey 32-byte peer's X25519 public key
     */
    void onCryptoKeysReady(
        const std::vector<uint8_t>& localPrivateKey,
        const std::vector<uint8_t>& localPublicKey,
        const std::vector<uint8_t>& remotePublicKey
    );

private:
    std::shared_ptr<NadeExternalIO> nade_;
    nda::PluginState state_ = nda::PluginState::Unloaded;
    bool keysReceived_ = false;

    int sampleRate_ = 48000;
    int channelCount_ = 2;

    // Diagnostic tracking
    size_t messagesProcessed_ = 0;      // Total messages successfully queued
    size_t messagesDecoded_ = 0;        // Total text decode attempts
    size_t decodeFailures_ = 0;         // Failed text decodes
    size_t queueFailures_ = 0;          // Failed sendTextMessage() calls
    std::string lastError_;             // Last error message
    std::string lastMessage_;           // Last message queued (for debugging)
};

}  // namespace nade
