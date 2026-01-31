/**
 * @file NadeTextSourcePlugin.cpp
 * @brief Implementation of NadeTextSourcePlugin with Qt GUI.
 *
 * @version 3.0
 * @date 2026-01-30
 */

#include "NadeTextSourcePlugin.h"
#include "TextEncoding.h"

#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QTextEdit>
#include <QPushButton>
#include <QLabel>
#include <QTimer>
#include <QMessageBox>
#include <iostream>

namespace nade {

// Initialize static GUI widgets and audio buffer (shared across all instances)
QWidget* NadeTextSourcePlugin::sharedGuiWidget_ = nullptr;
QTextEdit* NadeTextSourcePlugin::sharedInputText_ = nullptr;
QPushButton* NadeTextSourcePlugin::sharedSendButton_ = nullptr;
QLabel* NadeTextSourcePlugin::sharedStatusLabel_ = nullptr;
QTimer* NadeTextSourcePlugin::sharedTxCheckTimer_ = nullptr;
std::atomic<nda::PluginState> NadeTextSourcePlugin::sharedState_(nda::PluginState::Unloaded);
std::vector<float> NadeTextSourcePlugin::sharedPendingAudio_;
std::atomic<size_t> NadeTextSourcePlugin::sharedAudioPosition_(0);

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

    // Note: This plugin uses simple text-as-audio encoding (TextEncoding.h)
    // It does NOT require NadeExternalIO or crypto keys
    // Text flows through the pipeline as encoded audio samples

    return true;
}

void NadeTextSourcePlugin::shutdown() {
    stop();

    // Note: Don't clean up static GUI widgets here - they're shared across all instances
    // Only clean them up when the last instance is destroyed

    state_ = nda::PluginState::Unloaded;
    sharedState_ = nda::PluginState::Unloaded;
}

bool NadeTextSourcePlugin::start() {
    state_ = nda::PluginState::Running;
    sharedState_ = nda::PluginState::Running;  // Update shared state


    // Enable text input when pipeline is running (use shared widgets)
    if (sharedInputText_) {
        sharedInputText_->setEnabled(true);
        std::cout << "[NadeTextSourcePlugin@" << this << "] Text input enabled (shared)" << std::endl;
    } else {
        std::cout << "[NadeTextSourcePlugin@" << this << "] WARNING: sharedInputText_ is null!" << std::endl;
    }

    if (sharedSendButton_) {
        sharedSendButton_->setEnabled(true);
        std::cout << "[NadeTextSourcePlugin@" << this << "] Send button enabled (shared)" << std::endl;
    } else {
        std::cout << "[NadeTextSourcePlugin@" << this << "] WARNING: sharedSendButton_ is null!" << std::endl;
    }

    if (sharedStatusLabel_) {
        sharedStatusLabel_->setText("Status: Ready");
    } else {
        std::cout << "[NadeTextSourcePlugin@" << this << "] WARNING: sharedStatusLabel_ is null!" << std::endl;
    }

    return true;
}

void NadeTextSourcePlugin::stop() {
    std::cout << "[NadeTextSourcePlugin@" << this << "] stop() called" << std::endl;

    state_ = nda::PluginState::Initialized;
    sharedState_ = nda::PluginState::Initialized;  // Update shared state

    // Disable text input when pipeline is stopped (use shared widgets)
    if (sharedInputText_) {
        sharedInputText_->setEnabled(false);
    }
    if (sharedSendButton_) {
        sharedSendButton_->setEnabled(false);
    }
    if (sharedStatusLabel_) {
        sharedStatusLabel_->setText("Status: Pipeline stopped");
    }

    // Clear any pending audio (use shared buffer)
    sharedPendingAudio_.clear();
    sharedAudioPosition_ = 0;
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
    size_t currentPos = sharedAudioPosition_.load();
    if (sharedPendingAudio_.empty() || currentPos >= sharedPendingAudio_.size()) {
        for (int ch = 0; ch < buffer.getChannelCount(); ++ch) {
            float* data = buffer.getChannelData(ch);
            if (data) {
                std::fill_n(data, frameCount, 0.0f);
            }
        }
        return true;
    }

    // Output pending audio (use shared buffer)
    size_t samplesAvailable = sharedPendingAudio_.size() - currentPos;
    size_t toCopy = std::min(static_cast<size_t>(frameCount), samplesAvailable);

    for (int ch = 0; ch < buffer.getChannelCount(); ++ch) {
        float* data = buffer.getChannelData(ch);
        if (data) {
            std::copy_n(sharedPendingAudio_.data() + currentPos, toCopy, data);

            // Zero remaining samples
            if (toCopy < static_cast<size_t>(frameCount)) {
                std::fill_n(data + toCopy, frameCount - toCopy, 0.0f);
            }
        }
    }

    // Update position atomically
    size_t newPos = currentPos + toCopy;
    sharedAudioPosition_ = newPos;

    // Clear pending audio if fully consumed
    if (newPos >= sharedPendingAudio_.size()) {
        sharedPendingAudio_.clear();
        sharedAudioPosition_ = 0;
    }

    return true;
}

// =============================================================================
// GUI
// =============================================================================

QWidget* NadeTextSourcePlugin::createDockableGui() {

    // Only create GUI once (singleton pattern)
    if (sharedGuiWidget_) {
        return sharedGuiWidget_;
    }

    sharedGuiWidget_ = new QWidget();
    QVBoxLayout* mainLayout = new QVBoxLayout(sharedGuiWidget_);
    mainLayout->setContentsMargins(10, 10, 10, 10);
    mainLayout->setSpacing(8);

    // Title label
    QLabel* titleLabel = new QLabel("Text Input");
    titleLabel->setStyleSheet("font-weight: bold; font-size: 14px;");
    mainLayout->addWidget(titleLabel);

    // Status label
    sharedStatusLabel_ = new QLabel("Status: Pipeline stopped");
    sharedStatusLabel_->setStyleSheet("font-size: 11px; color: #888;");
    mainLayout->addWidget(sharedStatusLabel_);

    // Message input
    QLabel* inputLabel = new QLabel("Message (max 256 chars):");
    inputLabel->setStyleSheet("font-size: 11px; margin-top: 5px;");
    mainLayout->addWidget(inputLabel);

    sharedInputText_ = new QTextEdit();
    sharedInputText_->setMaximumHeight(80);
    sharedInputText_->setPlaceholderText("Type your message here...");
    mainLayout->addWidget(sharedInputText_, 1);

    // Send button
    sharedSendButton_ = new QPushButton("Send");
    sharedSendButton_->setMaximumHeight(30);
    mainLayout->addWidget(sharedSendButton_);

    // Add stretch to push everything to the top
    mainLayout->addStretch();

    // Connect send button (capture 'this' is safe because we're storing a global instance reference)
    QObject::connect(sharedSendButton_, &QPushButton::clicked, [this]() {
        onSendClicked();
    });

    // Timer to check TX status
    sharedTxCheckTimer_ = new QTimer(sharedGuiWidget_);
    QObject::connect(sharedTxCheckTimer_, &QTimer::timeout, [this]() {
        checkTxStatus();
    });
    sharedTxCheckTimer_->start(100);

    // Set initial state based on current shared state
    bool shouldEnable = (sharedState_ == nda::PluginState::Running);
    sharedInputText_->setEnabled(shouldEnable);
    sharedSendButton_->setEnabled(shouldEnable);

    sharedGuiWidget_->setLayout(mainLayout);
    return sharedGuiWidget_;
}

void NadeTextSourcePlugin::onSendClicked() {
    // Check if pipeline is running (use shared state)
    if (sharedState_ != nda::PluginState::Running) {
        if (sharedStatusLabel_) {
            sharedStatusLabel_->setText("Status: Pipeline not running");
        }
        return;
    }

    if (!sharedInputText_) {
        return;
    }

    QString text = sharedInputText_->toPlainText().trimmed();
    if (text.isEmpty()) {
        return;
    }

    std::string stdText = text.toStdString();

    // Validate length
    if (stdText.length() > MAX_MESSAGE_LENGTH) {
        QMessageBox::warning(sharedGuiWidget_, "Message Too Long",
            QString("Message exceeds %1 characters. Please shorten it.")
                .arg(MAX_MESSAGE_LENGTH));
        return;
    }

    // Encode text as audio for pipeline (use shared buffer)
    sharedPendingAudio_ = textToAudioBuffer(stdText);
    sharedAudioPosition_ = 0;

    std::cout << "[NadeTextSourcePlugin] Sending message: \"" << stdText
              << "\" (" << sharedPendingAudio_.size() << " samples)" << std::endl;

    // Update UI
    if (sharedSendButton_) {
        sharedSendButton_->setEnabled(false);
        sharedSendButton_->setText("Transmitting...");
    }
    if (sharedStatusLabel_) {
        sharedStatusLabel_->setText("Status: Transmitting...");
    }

    // Clear input for next message
    sharedInputText_->clear();
}

void NadeTextSourcePlugin::checkTxStatus() {
    // Ensure widgets are in correct state based on shared plugin state
    if (sharedInputText_ && sharedSendButton_) {
        bool shouldBeEnabled = (sharedState_ == nda::PluginState::Running);

        // Check if transmission is complete (all audio has been sent) - use shared buffer
        size_t pos = sharedAudioPosition_.load();
        bool isTransmitting = !sharedPendingAudio_.empty() && (pos < sharedPendingAudio_.size());

        if (shouldBeEnabled) {
            // Pipeline is running
            if (!isTransmitting) {
                // Not transmitting - enable send button
                if (!sharedSendButton_->isEnabled()) {
                    sharedSendButton_->setEnabled(true);
                    sharedSendButton_->setText("Send");
                    if (sharedStatusLabel_) {
                        sharedStatusLabel_->setText("Status: Ready");
                    }
                }
            }
            // Enable text input if pipeline is running
            if (!sharedInputText_->isEnabled()) {
                sharedInputText_->setEnabled(true);
            }
        } else {
            // Pipeline is not running - disable everything
            if (sharedInputText_->isEnabled()) {
                sharedInputText_->setEnabled(false);
            }
            if (sharedSendButton_->isEnabled()) {
                sharedSendButton_->setEnabled(false);
                if (sharedStatusLabel_) {
                    sharedStatusLabel_->setText("Status: Pipeline stopped");
                }
            }
        }
    }
}

}  // namespace nade

// Plugin factory using NDA macro
NDA_DECLARE_PLUGIN(nade::NadeTextSourcePlugin)
