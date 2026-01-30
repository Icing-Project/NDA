/**
 * @file NadeTextSourcePlugin.cpp
 * @brief Implementation of NadeTextSourcePlugin with Qt GUI.
 *
 * @version 3.0
 * @date 2026-01-30
 */

#include "NadeTextSourcePlugin.h"
#include "TextEncoding.h"
#include "crypto/CryptoManager.h"

#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QTextEdit>
#include <QPushButton>
#include <QLabel>
#include <QTimer>
#include <QMessageBox>

namespace nade {

NadeTextSourcePlugin::NadeTextSourcePlugin() = default;

NadeTextSourcePlugin::~NadeTextSourcePlugin() {
    shutdown();
}

// =============================================================================
// Plugin Metadata
// =============================================================================

nda::PluginInfo NadeTextSourcePlugin::getInfo() const {
    nda::PluginInfo info;
    info.name = "Nade Text Source";
    info.version = "3.0.0";
    info.author = "Icing Project";
    info.description = "Text input GUI for encrypted messaging (TX path)";
    info.type = nda::PluginType::AudioSource;
    info.apiVersion = 1;
    return info;
}

// =============================================================================
// Lifecycle
// =============================================================================

bool NadeTextSourcePlugin::initialize() {
    state_ = nda::PluginState::Initialized;

    // Get keys from CryptoManager
    nda::CryptoManager& cm = nda::CryptoManager::instance();

    if (cm.hasX25519KeyPair() && cm.hasX25519PeerPublicKey()) {
        auto privKey = cm.exportX25519PrivateKeyBytes();
        auto pubKey = cm.exportX25519PublicKeyBytes();
        auto peerKey = cm.exportX25519PeerPublicKeyBytes();

        // Create NadeExternalIO instance
        nade_ = NadeExternalIO::createInstance(privKey, pubKey, peerKey, sampleRate_, true);
    }

    return true;
}

void NadeTextSourcePlugin::shutdown() {
    stop();

    if (txCheckTimer_) {
        txCheckTimer_->stop();
        txCheckTimer_ = nullptr;
    }

    // Clean up GUI (all child widgets are owned by guiWidget_)
    if (guiWidget_) {
        guiWidget_->close();
        delete guiWidget_;
        guiWidget_ = nullptr;
        inputText_ = nullptr;
        sendButton_ = nullptr;
        statusLabel_ = nullptr;
    }

    nade_.reset();
    state_ = nda::PluginState::Unloaded;
}

bool NadeTextSourcePlugin::start() {
    state_ = nda::PluginState::Running;
    // GUI created and managed by MainWindow's QDockWidget system
    return true;
}

void NadeTextSourcePlugin::stop() {
    if (txCheckTimer_) {
        txCheckTimer_->stop();
    }
    if (guiWidget_) {
        guiWidget_->hide();
    }
    state_ = nda::PluginState::Initialized;
}

// =============================================================================
// Configuration
// =============================================================================

void NadeTextSourcePlugin::setParameter(const std::string& key, const std::string& value) {
    (void)key;
    (void)value;
}

std::string NadeTextSourcePlugin::getParameter(const std::string& key) const {
    (void)key;
    return "";
}

// =============================================================================
// Audio Source Interface
// =============================================================================

void NadeTextSourcePlugin::setAudioCallback(nda::AudioSourceCallback callback) {
    audioCallback_ = callback;
}

bool NadeTextSourcePlugin::readAudio(AudioBuffer& buffer) {
    int frameCount = buffer.getFrameCount();

    // If no pending audio, output silence
    if (pendingAudio_.empty() || audioPosition_ >= pendingAudio_.size()) {
        for (int ch = 0; ch < buffer.getChannelCount(); ++ch) {
            float* data = buffer.getChannelData(ch);
            if (data) {
                std::fill_n(data, frameCount, 0.0f);
            }
        }
        return true;
    }

    // Output pending audio
    size_t samplesAvailable = pendingAudio_.size() - audioPosition_;
    size_t toCopy = std::min(static_cast<size_t>(frameCount), samplesAvailable);

    for (int ch = 0; ch < buffer.getChannelCount(); ++ch) {
        float* data = buffer.getChannelData(ch);
        if (data) {
            std::copy_n(pendingAudio_.data() + audioPosition_, toCopy, data);

            // Zero remaining samples
            if (toCopy < static_cast<size_t>(frameCount)) {
                std::fill_n(data + toCopy, frameCount - toCopy, 0.0f);
            }
        }
    }

    audioPosition_ += toCopy;

    // Clear pending audio if fully consumed
    if (audioPosition_ >= pendingAudio_.size()) {
        pendingAudio_.clear();
        audioPosition_ = 0;
    }

    return true;
}

// =============================================================================
// GUI
// =============================================================================

QWidget* NadeTextSourcePlugin::createDockableGui() {
    guiWidget_ = new QWidget();
    QVBoxLayout* mainLayout = new QVBoxLayout(guiWidget_);

    // Title label
    QLabel* titleLabel = new QLabel("Nade Text Messaging (TX)");
    titleLabel->setStyleSheet("font-weight: bold; font-size: 14px;");
    mainLayout->addWidget(titleLabel);

    // Status label
    statusLabel_ = new QLabel("Status: Waiting for keys...");
    mainLayout->addWidget(statusLabel_);

    // Message input
    QLabel* inputLabel = new QLabel("Message to Send (max 256 chars):");
    mainLayout->addWidget(inputLabel);

    inputText_ = new QTextEdit();
    inputText_->setMaximumHeight(80);
    inputText_->setPlaceholderText("Type your message here...");
    inputText_->setEnabled(false);
    mainLayout->addWidget(inputText_);

    // Send button
    sendButton_ = new QPushButton("Send");
    sendButton_->setEnabled(false);
    mainLayout->addWidget(sendButton_);

    // Connect send button
    QObject::connect(sendButton_, &QPushButton::clicked, [this]() {
        onSendClicked();
    });

    // Timer to check TX status and key availability
    txCheckTimer_ = new QTimer(guiWidget_);
    QObject::connect(txCheckTimer_, &QTimer::timeout, [this]() {
        checkTxStatus();
    });
    txCheckTimer_->start(100);

    guiWidget_->setLayout(mainLayout);
    return guiWidget_;
}

void NadeTextSourcePlugin::onSendClicked() {
    if (!nade_) {
        if (statusLabel_) {
            statusLabel_->setText("Status: Error - Not initialized");
        }
        return;
    }

    if (!inputText_) {
        return;
    }

    QString text = inputText_->toPlainText().trimmed();
    if (text.isEmpty()) {
        return;
    }

    std::string stdText = text.toStdString();

    // Validate length
    if (stdText.length() > MAX_MESSAGE_LENGTH) {
        QMessageBox::warning(guiWidget_, "Message Too Long",
            QString("Message exceeds %1 characters. Please shorten it.")
                .arg(MAX_MESSAGE_LENGTH));
        return;
    }

    // Send to Nade
    auto result = nade_->sendTextMessage(stdText);
    if (result.success) {
        // Disable send button during transmission
        sendButton_->setEnabled(false);
        sendButton_->setText("Transmitting...");
        statusLabel_->setText("Status: Transmitting...");

        // Clear input for next message
        inputText_->clear();

        // Encode text as audio for pipeline trigger
        pendingAudio_ = textToAudioBuffer(stdText);
        audioPosition_ = 0;

    } else {
        QMessageBox::warning(guiWidget_, "Send Failed",
            QString::fromStdString(result.error_message));
    }
}

void NadeTextSourcePlugin::checkTxStatus() {
    if (!nade_) {
        // Check if keys are now available
        nda::CryptoManager& cm = nda::CryptoManager::instance();
        if (cm.hasX25519KeyPair() && cm.hasX25519PeerPublicKey()) {
            auto privKey = cm.exportX25519PrivateKeyBytes();
            auto pubKey = cm.exportX25519PublicKeyBytes();
            auto peerKey = cm.exportX25519PeerPublicKeyBytes();

            nade_ = NadeExternalIO::createInstance(privKey, pubKey, peerKey, sampleRate_, true);

            if (nade_) {
                if (inputText_) inputText_->setEnabled(true);
                if (sendButton_) sendButton_->setEnabled(true);
                if (statusLabel_) statusLabel_->setText("Status: Connected - Ready to send");
            }
        }
        return;
    }

    if (!nade_->isTransmitting()) {
        // Re-enable send button after transmission completes
        if (sendButton_ && !sendButton_->isEnabled()) {
            sendButton_->setEnabled(true);
            sendButton_->setText("Send");
            statusLabel_->setText("Status: Connected - Ready to send");
        }
    }
}

}  // namespace nade

// Plugin factory using NDA macro
NDA_DECLARE_PLUGIN(nade::NadeTextSourcePlugin)
