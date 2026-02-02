/**
 * @file NadeDecryptorPlugin.cpp
 * @brief Implementation of NadeDecryptorPlugin.
 *
 * @version 3.0
 * @date 2026-01-30
 */

#include "NadeDecryptorPlugin.h"
#include "TextEncoding.h"
#include "crypto/CryptoManager.h"
#include <iostream>

namespace nade {

NadeDecryptorPlugin::NadeDecryptorPlugin() = default;

NadeDecryptorPlugin::~NadeDecryptorPlugin() {
    shutdown();
}

// =============================================================================
// Plugin Metadata
// =============================================================================

nda::PluginInfo NadeDecryptorPlugin::getInfo() const {
    nda::PluginInfo info;
    info.name = "Nade Decryptor";
    info.version = "3.0.0";
    info.author = "Icing Project";
    info.description = "FSK demod -> Noise XK -> Text (RX path)";
    info.type = nda::PluginType::Processor;
    info.apiVersion = 1;
    return info;
}

// =============================================================================
// Lifecycle
// =============================================================================

bool NadeDecryptorPlugin::initialize() {
    state_ = nda::PluginState::Initialized;

    // Get keys from CryptoManager
    nda::CryptoManager& cm = nda::CryptoManager::instance();

    if (cm.hasX25519KeyPair() && cm.hasX25519PeerPublicKey()) {
        auto privKey = cm.exportX25519PrivateKeyBytes();
        auto pubKey = cm.exportX25519PublicKeyBytes();
        auto peerKey = cm.exportX25519PeerPublicKeyBytes();

        nade_ = NadeExternalIO::createInstance(privKey, pubKey, peerKey, sampleRate_, false);
    }

    return true;
}

void NadeDecryptorPlugin::shutdown() {
    stop();

    // Decryptor is typically last to shutdown - destroy singleton
    if (nade_) {
        NadeExternalIO::destroyInstance();
        nade_.reset();
    }

    state_ = nda::PluginState::Unloaded;
}

bool NadeDecryptorPlugin::start() {
    // Try to get existing singleton (shared between encryptor and decryptor)
    if (!nade_) {
        nade_ = NadeExternalIO::getInstance();
    }

    // If still no instance, check if keys are now available (may have been imported after initialize())
    if (!nade_) {
        nda::CryptoManager& cm = nda::CryptoManager::instance();

        if (cm.hasX25519KeyPair() && cm.hasX25519PeerPublicKey()) {
            std::cout << "[NadeDecryptor] Keys detected - creating NadeExternalIO instance" << std::endl;

            auto privKey = cm.exportX25519PrivateKeyBytes();
            auto pubKey = cm.exportX25519PublicKeyBytes();
            auto peerKey = cm.exportX25519PeerPublicKeyBytes();

            nade_ = NadeExternalIO::createInstance(privKey, pubKey, peerKey, sampleRate_, false);
            keysReceived_ = true;
        } else {
            std::cout << "[NadeDecryptor] Starting without keys - import X25519 keys via Crypto menu" << std::endl;
        }
    }

    state_ = nda::PluginState::Running;
    return true;
}

void NadeDecryptorPlugin::stop() {
    state_ = nda::PluginState::Initialized;
}

// =============================================================================
// Configuration
// =============================================================================

void NadeDecryptorPlugin::setParameter(const std::string& key, const std::string& value) {
    // Configuration happens via onCryptoKeysReady(), not setParameter()
    (void)key;
    (void)value;
}

std::string NadeDecryptorPlugin::getParameter(const std::string& key) const {
    if (key == "initialized") {
        return nade_ ? "true" : "false";
    }
    if (key == "mode") {
        return nade_ ? nade_->getMode() : "none";
    }
    if (key == "rx_signal_quality") {
        if (nade_) {
            char buf[32];
            snprintf(buf, sizeof(buf), "%.3f", nade_->getRxSignalQuality());
            return buf;
        }
        return "0.0";
    }
    if (key == "rx_synchronized") {
        return (nade_ && nade_->isRxSynchronized()) ? "true" : "false";
    }
    return "";
}

// =============================================================================
// Audio Processing
// =============================================================================

bool NadeDecryptorPlugin::processAudio(AudioBuffer& buffer) {
    if (!nade_) {
        // Not initialized - pass through silence
        for (int ch = 0; ch < buffer.getChannelCount(); ++ch) {
            float* data = buffer.getChannelData(ch);
            if (data) {
                std::fill_n(data, buffer.getFrameCount(), 0.0f);
            }
        }
        return true;
    }

    int frameCount = buffer.getFrameCount();

    // Extract mono audio from input (mix down if stereo)
    std::vector<float> monoInput(frameCount, 0.0f);
    if (buffer.getChannelCount() > 1) {
        // Average channels
        for (int i = 0; i < frameCount; ++i) {
            float sum = 0.0f;
            for (int ch = 0; ch < buffer.getChannelCount(); ++ch) {
                const float* channelData = buffer.getChannelData(ch);
                if (channelData) {
                    sum += channelData[i];
                }
            }
            monoInput[i] = sum / buffer.getChannelCount();
        }
    } else {
        const float* channelData = buffer.getChannelData(0);
        if (channelData) {
            std::copy_n(channelData, frameCount, monoInput.data());
        }
    }

    // Calculate buffer duration
    float durationMs = (static_cast<float>(frameCount) / sampleRate_) * 1000.0f;

    // Feed FSK audio to post-buffer (non-blocking)
    nade_->processRxAudio(monoInput.data(), monoInput.size(), durationMs);

    // Retrieve any decoded messages
    std::vector<std::string> messages = nade_->getReceivedMessages();
    for (const auto& msg : messages) {
        // Encode message as "audio" and queue for output
        std::vector<float> msgAudio = textToAudioBuffer(msg);
        pendingMessageAudio_.push(std::move(msgAudio));
    }

    // Output pending message audio (or silence)
    std::vector<float> output(frameCount, 0.0f);
    size_t outputPos = 0;

    while (outputPos < static_cast<size_t>(frameCount)) {
        // If current message is exhausted, get next one
        if (audioPosition_ >= currentMessageAudio_.size()) {
            if (pendingMessageAudio_.empty()) {
                break;  // No more messages
            }
            currentMessageAudio_ = std::move(pendingMessageAudio_.front());
            pendingMessageAudio_.pop();
            audioPosition_ = 0;
        }

        // Copy audio
        size_t samplesRemaining = frameCount - outputPos;
        size_t samplesAvailable = currentMessageAudio_.size() - audioPosition_;
        size_t toCopy = std::min(samplesRemaining, samplesAvailable);

        std::copy_n(currentMessageAudio_.data() + audioPosition_, toCopy, output.data() + outputPos);

        outputPos += toCopy;
        audioPosition_ += toCopy;
    }

    // Write to all output channels
    for (int ch = 0; ch < buffer.getChannelCount(); ++ch) {
        float* channelData = buffer.getChannelData(ch);
        if (channelData) {
            std::copy_n(output.data(), frameCount, channelData);
        }
    }

    return true;
}

// =============================================================================
// Key Delivery
// =============================================================================

void NadeDecryptorPlugin::onCryptoKeysReady(
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
        false  // Decryptor is responder
    );

    keysReceived_ = true;
}

}  // namespace nade

// Plugin factory using NDA macro
NDA_DECLARE_PLUGIN(nade::NadeDecryptorPlugin)
