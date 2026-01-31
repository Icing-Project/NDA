/**
 * @file NadeEncryptorPlugin.cpp
 * @brief Implementation of NadeEncryptorPlugin.
 *
 * @version 3.0
 * @date 2026-01-30
 */

#include "NadeEncryptorPlugin.h"
#include "TextEncoding.h"
#include "crypto/CryptoManager.h"
#include <iostream>

namespace nade {

NadeEncryptorPlugin::NadeEncryptorPlugin() = default;

NadeEncryptorPlugin::~NadeEncryptorPlugin() {
    shutdown();
}

// =============================================================================
// Plugin Metadata
// =============================================================================

nda::PluginInfo NadeEncryptorPlugin::getInfo() const {
    nda::PluginInfo info;
    info.name = "Nade Encryptor";
    info.version = "3.0.0";
    info.author = "Icing Project";
    info.description = "Text -> Noise XK -> FSK modulation (TX path)";
    info.type = nda::PluginType::Processor;
    info.apiVersion = 1;
    return info;
}

// =============================================================================
// Lifecycle
// =============================================================================

bool NadeEncryptorPlugin::initialize() {
    state_ = nda::PluginState::Initialized;

    // Get keys from CryptoManager
    nda::CryptoManager& cm = nda::CryptoManager::instance();

    if (cm.hasX25519KeyPair() && cm.hasX25519PeerPublicKey()) {
        auto privKey = cm.exportX25519PrivateKeyBytes();
        auto pubKey = cm.exportX25519PublicKeyBytes();
        auto peerKey = cm.exportX25519PeerPublicKeyBytes();

        nade_ = NadeExternalIO::createInstance(privKey, pubKey, peerKey, sampleRate_, true);
    }

    return true;
}

void NadeEncryptorPlugin::shutdown() {
    stop();

    // Don't destroy the singleton - decryptor may still need it
    nade_.reset();

    state_ = nda::PluginState::Unloaded;
}

bool NadeEncryptorPlugin::start() {
    // Try to get existing singleton (shared between encryptor and decryptor)
    if (!nade_) {
        nade_ = NadeExternalIO::getInstance();
    }

    if (!nade_) {
        std::cout << "[NadeEncryptor] Starting without keys - import X25519 keys via Crypto menu" << std::endl;
    }

    state_ = nda::PluginState::Running;
    return true;
}

void NadeEncryptorPlugin::stop() {
    state_ = nda::PluginState::Initialized;
}

// =============================================================================
// Configuration
// =============================================================================

void NadeEncryptorPlugin::setParameter(const std::string& key, const std::string& value) {
    if (key == "force_handshake" && nade_) {
        bool isInitiator = (value == "initiator");
        nade_->forceHandshake(isInitiator);
        return;
    }
    if (key == "restart_discovery" && nade_) {
        nade_->restartDiscovery();
        return;
    }
    // Configuration happens via onCryptoKeysReady(), not setParameter()
    (void)key;
    (void)value;
}

std::string NadeEncryptorPlugin::getParameter(const std::string& key) const {
    if (key == "initialized") {
        return nade_ ? "true" : "false";
    }
    if (key == "mode") {
        return nade_ ? nade_->getMode() : "none";
    }
    if (key == "transmitting") {
        return (nade_ && nade_->isTransmitting()) ? "true" : "false";
    }
    if (key == "handshake_phase") {
        return nade_ ? std::to_string(nade_->getHandshakePhase()) : "0";
    }
    // Diagnostic parameters (v3.0.1)
    if (key == "messages_processed") {
        return std::to_string(messagesProcessed_);
    }
    if (key == "messages_decoded") {
        return std::to_string(messagesDecoded_);
    }
    if (key == "decode_failures") {
        return std::to_string(decodeFailures_);
    }
    if (key == "queue_failures") {
        return std::to_string(queueFailures_);
    }
    if (key == "last_error") {
        return lastError_;
    }
    if (key == "last_message") {
        return lastMessage_;
    }
    return "";
}

// =============================================================================
// Audio Processing
// =============================================================================

bool NadeEncryptorPlugin::processAudio(AudioBuffer& buffer) {
    if (!nade_) {
        // Not initialized - output silence
        for (int ch = 0; ch < buffer.getChannelCount(); ++ch) {
            float* data = buffer.getChannelData(ch);
            if (data) {
                std::fill_n(data, buffer.getFrameCount(), 0.0f);
            }
        }
        return true;
    }

    int frameCount = buffer.getFrameCount();

    // =========================================================================
    // TEXT INPUT: Decode text from input audio (from NadeTextSourcePlugin)
    // =========================================================================

    // Check if input buffer contains encoded text message
    // Note: Buffer size is fixed at 512 samples, max encoded message is 258 samples
    // Therefore, complete messages always fit in a single buffer

    float* inputData = buffer.getChannelData(0);

    if (inputData && frameCount >= 2) {
        // Attempt to decode text from input buffer
        auto maybeText = audioBufferToText(inputData, frameCount);

        if (maybeText.has_value()) {
            // Successfully decoded text from input
            std::string text = maybeText.value();
            messagesDecoded_++;

            // Queue message for Python processing (via NadeExternalIO)
            NadeResult result = nade_->sendTextMessage(text);

            if (result.success) {
                // Message successfully queued
                messagesProcessed_++;
                lastMessage_ = text;
                lastError_ = "";

                std::cout << "[NadeEncryptor] Message queued for transmission ("
                          << text.length() << " chars): \""
                          << text << "\"" << std::endl;
            } else {
                // Failed to queue message
                queueFailures_++;
                lastError_ = result.error_message;

                std::cerr << "[NadeEncryptor] ERROR: Failed to queue message: "
                          << result.error_message << std::endl;
                std::cerr << "[NadeEncryptor]        Message was: \""
                          << text << "\" (" << text.length() << " chars)" << std::endl;
            }
        } else {
            // No text detected in input buffer - this is normal
            // (Input may be silence or real audio)
            // No logging needed for this case to avoid spam
        }
    }

    // =========================================================================
    // FSK OUTPUT: Read from pre-buffer and output
    // =========================================================================

    // Calculate buffer duration
    float durationMs = (static_cast<float>(frameCount) / sampleRate_) * 1000.0f;

    // Get FSK audio from pre-buffer (non-blocking)
    // Pre-buffer is filled by Python worker thread
    std::vector<float> fskAudio = nade_->getTxAudio(durationMs);

    // Write FSK to all output channels
    for (int ch = 0; ch < buffer.getChannelCount(); ++ch) {
        float* channelData = buffer.getChannelData(ch);
        if (channelData) {
            int toCopy = std::min(frameCount, static_cast<int>(fskAudio.size()));
            std::copy_n(fskAudio.data(), toCopy, channelData);

            // Zero remaining samples (if any)
            if (toCopy < frameCount) {
                std::fill_n(channelData + toCopy, frameCount - toCopy, 0.0f);
            }
        }
    }

    return true;
}

// =============================================================================
// Key Delivery
// =============================================================================

void NadeEncryptorPlugin::onCryptoKeysReady(
    const std::vector<uint8_t>& localPrivateKey,
    const std::vector<uint8_t>& localPublicKey,
    const std::vector<uint8_t>& remotePublicKey)
{
    if (keysReceived_) {
        return;  // Already initialized
    }

    // Create or get singleton
    nade_ = NadeExternalIO::createInstance(
        localPrivateKey,
        localPublicKey,
        remotePublicKey,
        sampleRate_,
        true  // Encryptor is initiator
    );

    keysReceived_ = true;
}

}  // namespace nade

// Plugin factory using NDA macro
NDA_DECLARE_PLUGIN(nade::NadeEncryptorPlugin)
