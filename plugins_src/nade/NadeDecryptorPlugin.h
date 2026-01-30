/**
 * @file NadeDecryptorPlugin.h
 * @brief NDA Processor Plugin for RX (receive) path.
 *
 * This plugin receives FSK audio from the NDA RX pipeline, writes it to
 * NadeExternalIO's post-buffer for demodulation, and outputs decoded
 * messages as "text as audio".
 *
 * Responsibilities:
 *   - Receive FSK audio from NDA RX pipeline
 *   - Write to post-buffer (for Python worker thread)
 *   - Retrieve decoded messages from Nade
 *   - Encode messages as "audio" for sink plugin
 *
 * Threading:
 *   - processAudio() called from audio thread (non-blocking)
 *   - Writes to lock-free post-buffer (read by Python worker thread)
 *
 * @version 3.0
 * @date 2026-01-30
 */

#pragma once

#include "plugins/AudioProcessorPlugin.h"
#include "NadeExternalIO.h"
#include <memory>
#include <vector>
#include <string>
#include <queue>

namespace nade {

/**
 * @brief Processor plugin for RX path - demodulates FSK audio to text messages.
 */
class NadeDecryptorPlugin : public nda::AudioProcessorPlugin {
public:
    NadeDecryptorPlugin();
    ~NadeDecryptorPlugin() override;

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

    // Message output queue (encoded as audio)
    std::queue<std::vector<float>> pendingMessageAudio_;
    std::vector<float> currentMessageAudio_;
    size_t audioPosition_ = 0;
};

}  // namespace nade
