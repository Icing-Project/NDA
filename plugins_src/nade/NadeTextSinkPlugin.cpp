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
#include <iostream>

namespace nade {

// Initialize static GUI widgets and buffers (shared across all instances)
QWidget* NadeTextSinkPlugin::sharedGuiWidget_ = nullptr;
QTextEdit* NadeTextSinkPlugin::sharedOutputText_ = nullptr;
QLabel* NadeTextSinkPlugin::sharedStatusLabel_ = nullptr;
std::vector<float> NadeTextSinkPlugin::sharedAccumulatedAudio_;
size_t NadeTextSinkPlugin::sharedMessageCount_ = 0;

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

    // Note: Don't clean up static GUI widgets here - they're shared across all instances

    state_ = nda::PluginState::Unloaded;
}

bool NadeTextSinkPlugin::start() {
    std::cout << "[NadeTextSinkPlugin@" << this << "] start() called" << std::endl;
    state_ = nda::PluginState::Running;

    // Clear accumulated audio from previous runs
    sharedAccumulatedAudio_.clear();

    return true;
}

void NadeTextSinkPlugin::stop() {
    std::cout << "[NadeTextSinkPlugin@" << this << "] stop() called" << std::endl;
    state_ = nda::PluginState::Initialized;

    // Clear accumulated audio
    sharedAccumulatedAudio_.clear();
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
        return std::to_string(sharedMessageCount_);
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

    // Debug: Check if we received text-encoded audio
    if (monoInput.size() >= 2) {
        if (isTextEncodedAudio(monoInput)) {
            std::cout << "[NadeTextSinkPlugin@" << this << "] Received text-encoded audio: "
                      << monoInput.size() << " samples" << std::endl;
        }
    }

    // Accumulate audio (use shared buffer)
    size_t oldSize = sharedAccumulatedAudio_.size();
    sharedAccumulatedAudio_.resize(oldSize + monoInput.size());
    std::copy(monoInput.begin(), monoInput.end(), sharedAccumulatedAudio_.begin() + oldSize);

    // Try to decode text from accumulated audio
    tryDecodeBuffer();

    return true;
}

// =============================================================================
// Text Decoding
// =============================================================================

void NadeTextSinkPlugin::tryDecodeBuffer() {
    // Need at least magic + length (use shared buffer)
    if (sharedAccumulatedAudio_.size() < 2) {
        return;
    }

    // Scan for text magic number throughout the buffer (not just at start)
    // Text messages might arrive at any position due to silence padding
    size_t scanStart = 0;
    bool foundText = false;

    while (scanStart + 2 <= sharedAccumulatedAudio_.size()) {
        // Check if magic number exists at this position
        if (std::abs(sharedAccumulatedAudio_[scanStart] - MAGIC_NUMBER) <= MAGIC_TOLERANCE) {
            // Found magic number! Try to decode from this position
            std::vector<float> textChunk(
                sharedAccumulatedAudio_.begin() + scanStart,
                sharedAccumulatedAudio_.end()
            );

            auto text = audioBufferToText(textChunk);
            if (text.has_value()) {
                // Success! Display message
                std::cout << "[NadeTextSinkPlugin@" << this << "] Found text at offset "
                          << scanStart << ", decoded: \"" << text.value() << "\"" << std::endl;
                displayMessage(text.value());

                // Calculate consumed samples: magic + length + text bytes
                size_t textLength = static_cast<size_t>(sharedAccumulatedAudio_[scanStart + 1]);
                size_t consumedSamples = scanStart + textLength + 2;

                // Remove everything up to and including this message
                if (consumedSamples >= sharedAccumulatedAudio_.size()) {
                    sharedAccumulatedAudio_.clear();
                } else {
                    sharedAccumulatedAudio_.erase(
                        sharedAccumulatedAudio_.begin(),
                        sharedAccumulatedAudio_.begin() + consumedSamples
                    );
                }

                foundText = true;
                // Try to decode more messages (recursively)
                tryDecodeBuffer();
                return;
            }
        }

        // Move to next sample
        scanStart++;
    }

    // No text found - clear old silence samples if buffer is getting large
    if (!foundText && sharedAccumulatedAudio_.size() > 1024) {
        std::cout << "[NadeTextSinkPlugin@" << this << "] No text found in "
                  << sharedAccumulatedAudio_.size() << " samples, clearing" << std::endl;
        sharedAccumulatedAudio_.clear();
    }
}

// =============================================================================
// GUI
// =============================================================================

QWidget* NadeTextSinkPlugin::createDockableGui() {
    std::cout << "[NadeTextSinkPlugin@" << this << "] createDockableGui() called" << std::endl;

    // Only create GUI once (singleton pattern)
    if (sharedGuiWidget_) {
        std::cout << "[NadeTextSinkPlugin@" << this << "] GUI already exists, returning existing widget" << std::endl;
        return sharedGuiWidget_;
    }

    sharedGuiWidget_ = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(sharedGuiWidget_);
    layout->setContentsMargins(10, 10, 10, 10);
    layout->setSpacing(8);

    // Title label
    QLabel* titleLabel = new QLabel("Text Output");
    titleLabel->setStyleSheet("font-weight: bold; font-size: 14px;");
    layout->addWidget(titleLabel);

    // Status label
    sharedStatusLabel_ = new QLabel("Received: 0");
    sharedStatusLabel_->setStyleSheet("font-size: 11px; color: #888;");
    layout->addWidget(sharedStatusLabel_);

    // Text output (read-only)
    sharedOutputText_ = new QTextEdit();
    sharedOutputText_->setReadOnly(true);
    sharedOutputText_->setPlaceholderText("Waiting for messages...");
    layout->addWidget(sharedOutputText_, 1);

    sharedGuiWidget_->setLayout(layout);

    std::cout << "[NadeTextSinkPlugin@" << this << "] GUI created (shared)" << std::endl;
    std::cout << "[NadeTextSinkPlugin@" << this << "] sharedOutputText_=" << sharedOutputText_ << std::endl;

    return sharedGuiWidget_;
}

void NadeTextSinkPlugin::displayMessage(const std::string& text) {
    sharedMessageCount_++;

    std::cout << "[NadeTextSinkPlugin@" << this << "] displayMessage: \"" << text
              << "\" (total: " << sharedMessageCount_ << ")" << std::endl;

    if (sharedOutputText_) {
        // Get timestamp
        QString timestamp = QDateTime::currentDateTime().toString("hh:mm:ss");

        // Format message
        QString formattedMsg = QString("[%1] %2")
            .arg(timestamp)
            .arg(QString::fromStdString(text));

        // Append to output
        sharedOutputText_->append(formattedMsg);

        // Scroll to bottom
        QTextCursor cursor = sharedOutputText_->textCursor();
        cursor.movePosition(QTextCursor::End);
        sharedOutputText_->setTextCursor(cursor);

        std::cout << "[NadeTextSinkPlugin@" << this << "] Message displayed in GUI" << std::endl;
    } else {
        std::cout << "[NadeTextSinkPlugin@" << this << "] WARNING: sharedOutputText_ is null!" << std::endl;
    }

    if (sharedStatusLabel_) {
        sharedStatusLabel_->setText(QString("Received: %1").arg(sharedMessageCount_));
    }
}

}  // namespace nade

// Plugin factory using NDA macro
NDA_DECLARE_PLUGIN(nade::NadeTextSinkPlugin)
