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
    // Try to get existing singleton (may have been created by text source plugin)
    if (!nade_) {
        nade_ = NadeExternalIO::getInstance();
    }

    if (!nade_) {
        std::cout << "[NadeEncryptor] Starting without keys (configure in Text Source window)" << std::endl;
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

    // Calculate buffer duration
    float durationMs = (static_cast<float>(buffer.getFrameCount()) / sampleRate_) * 1000.0f;

    // Get FSK audio from pre-buffer (non-blocking)
    std::vector<float> fskAudio = nade_->getTxAudio(durationMs);

    // Write to all channels
    int frameCount = buffer.getFrameCount();
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
