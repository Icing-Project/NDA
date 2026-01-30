/**
 * @file NadeTextSinkPlugin.cpp
 * @brief Implementation of NadeTextSinkPlugin with Qt GUI.
 *
 * @version 3.0
 * @date 2026-01-30
 */

#include "NadeTextSinkPlugin.h"
#include "TextEncoding.h"

#include <QWidget>
#include <QVBoxLayout>
#include <QTextEdit>
#include <QLabel>
#include <QDateTime>

namespace nade {

NadeTextSinkPlugin::NadeTextSinkPlugin() = default;

NadeTextSinkPlugin::~NadeTextSinkPlugin() {
    shutdown();
}

// =============================================================================
// Plugin Metadata
// =============================================================================

nda::PluginInfo NadeTextSinkPlugin::getInfo() const {
    nda::PluginInfo info;
    info.name = "Nade Text Sink";
    info.version = "3.0.0";
    info.author = "Icing Project";
    info.description = "Text output display for received messages (RX path)";
    info.type = nda::PluginType::AudioSink;
    info.apiVersion = 1;
    return info;
}

// =============================================================================
// Lifecycle
// =============================================================================

bool NadeTextSinkPlugin::initialize() {
    state_ = nda::PluginState::Initialized;
    return true;
}

void NadeTextSinkPlugin::shutdown() {
    stop();

    // Clean up GUI
    if (guiWidget_) {
        guiWidget_->close();
        delete guiWidget_;
        guiWidget_ = nullptr;
        outputText_ = nullptr;
        statusLabel_ = nullptr;
    }

    state_ = nda::PluginState::Unloaded;
}

bool NadeTextSinkPlugin::start() {
    state_ = nda::PluginState::Running;
    // GUI created and managed by MainWindow's QDockWidget system
    return true;
}

void NadeTextSinkPlugin::stop() {
    if (guiWidget_) {
        guiWidget_->hide();
    }
    state_ = nda::PluginState::Initialized;
}

// =============================================================================
// Configuration
// =============================================================================

void NadeTextSinkPlugin::setParameter(const std::string& key, const std::string& value) {
    (void)key;
    (void)value;
}

std::string NadeTextSinkPlugin::getParameter(const std::string& key) const {
    if (key == "message_count") {
        return std::to_string(messageCount_);
    }
    return "";
}

// =============================================================================
// Audio Sink Interface
// =============================================================================

bool NadeTextSinkPlugin::writeAudio(const AudioBuffer& buffer) {
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

    // Accumulate audio
    size_t oldSize = accumulatedAudio_.size();
    accumulatedAudio_.resize(oldSize + monoInput.size());
    std::copy(monoInput.begin(), monoInput.end(), accumulatedAudio_.begin() + oldSize);

    // Try to decode text from accumulated audio
    tryDecodeBuffer();

    return true;
}

// =============================================================================
// Text Decoding
// =============================================================================

void NadeTextSinkPlugin::tryDecodeBuffer() {
    // Need at least magic + length
    if (accumulatedAudio_.size() < 2) {
        return;
    }

    // Check for text-encoded audio
    if (!isTextEncodedAudio(accumulatedAudio_)) {
        // Not text data - clear buffer (might be silence or noise)
        // Only clear if we have a reasonable amount of non-text data
        if (accumulatedAudio_.size() > 1024) {
            accumulatedAudio_.clear();
        }
        return;
    }

    // Try to decode
    auto text = audioBufferToText(accumulatedAudio_);
    if (text.has_value()) {
        // Success! Display message
        displayMessage(text.value());

        // Calculate consumed samples: magic + length + text bytes
        size_t textLength = static_cast<size_t>(accumulatedAudio_[1]);
        size_t consumedSamples = textLength + 2;

        // Remove consumed samples from buffer
        if (consumedSamples >= accumulatedAudio_.size()) {
            accumulatedAudio_.clear();
        } else {
            accumulatedAudio_.erase(
                accumulatedAudio_.begin(),
                accumulatedAudio_.begin() + consumedSamples
            );
        }

        // Try to decode more messages
        tryDecodeBuffer();
    }
    // If decode fails, we might need more data - keep accumulating
}

// =============================================================================
// GUI
// =============================================================================

QWidget* NadeTextSinkPlugin::createDockableGui() {
    guiWidget_ = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(guiWidget_);

    // Title label
    QLabel* titleLabel = new QLabel("Nade Text Messaging (RX)");
    titleLabel->setStyleSheet("font-weight: bold; font-size: 14px;");
    layout->addWidget(titleLabel);

    // Status label
    statusLabel_ = new QLabel("Received Messages: 0");
    layout->addWidget(statusLabel_);

    // Text output (read-only)
    outputText_ = new QTextEdit();
    outputText_->setReadOnly(true);
    outputText_->setPlaceholderText("Waiting for messages...");
    layout->addWidget(outputText_);

    guiWidget_->setLayout(layout);
    return guiWidget_;
}

void NadeTextSinkPlugin::displayMessage(const std::string& text) {
    messageCount_++;

    if (outputText_) {
        // Get timestamp
        QString timestamp = QDateTime::currentDateTime().toString("hh:mm:ss");

        // Format message
        QString formattedMsg = QString("[%1] %2")
            .arg(timestamp)
            .arg(QString::fromStdString(text));

        // Append to output
        outputText_->append(formattedMsg);

        // Scroll to bottom
        QTextCursor cursor = outputText_->textCursor();
        cursor.movePosition(QTextCursor::End);
        outputText_->setTextCursor(cursor);
    }

    if (statusLabel_) {
        statusLabel_->setText(QString("Received Messages: %1").arg(messageCount_));
    }
}

}  // namespace nade

// Plugin factory using NDA macro
NDA_DECLARE_PLUGIN(nade::NadeTextSinkPlugin)
