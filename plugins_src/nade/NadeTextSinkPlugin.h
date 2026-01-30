/**
 * @file NadeTextSinkPlugin.h
 * @brief NDA Sink Plugin for text output (RX path).
 *
 * This plugin receives "text as audio" samples from the decryptor plugin,
 * decodes them back to text, and displays them in a Qt GUI.
 *
 * Responsibilities:
 *   - Receive "audio" (encoded text) from decryptor plugin
 *   - Decode audio to text using audioBufferToText()
 *   - Display messages in Qt GUI with timestamps
 *   - Provide scrollable message history
 *
 * @version 3.0
 * @date 2026-01-30
 */

#pragma once

#include "plugins/AudioSinkPlugin.h"
#include <vector>
#include <memory>

// Forward declarations for Qt types
class QWidget;
class QTextEdit;
class QLabel;

namespace nade {

/**
 * @brief Sink plugin for text output (RX path) with Qt GUI.
 */
class NadeTextSinkPlugin : public nda::AudioSinkPlugin {
public:
    NadeTextSinkPlugin();
    ~NadeTextSinkPlugin() override;

    // =========================================================================
    // Plugin Metadata (BasePlugin)
    // =========================================================================

    nda::PluginInfo getInfo() const override;
    nda::PluginType getType() const override { return nda::PluginType::AudioSink; }
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
    // Audio Sink Interface
    // =========================================================================

    bool writeAudio(const AudioBuffer& buffer) override;

    int getSampleRate() const override { return sampleRate_; }
    int getChannels() const override { return channelCount_; }
    void setSampleRate(int sampleRate) override { sampleRate_ = sampleRate; }
    void setChannels(int channels) override { channelCount_ = channels; }

    int getBufferSize() const override { return 512; }
    void setBufferSize(int samples) override { (void)samples; }
    int getAvailableSpace() const override { return 512; }

    // =========================================================================
    // GUI (Qt)
    // =========================================================================

    /**
     * @brief Create Qt GUI widget for message display.
     * @return QWidget to be docked in NDA main window
     */
    QWidget* createDockableGui() override;

private:
    void displayMessage(const std::string& text);
    void tryDecodeBuffer();

    nda::PluginState state_ = nda::PluginState::Unloaded;

    int sampleRate_ = 48000;
    int channelCount_ = 2;

    // GUI widgets (owned by Qt parent)
    QWidget* guiWidget_ = nullptr;
    QTextEdit* outputText_ = nullptr;
    QLabel* statusLabel_ = nullptr;

    // Audio accumulation buffer for decoding
    std::vector<float> accumulatedAudio_;
    size_t messageCount_ = 0;
};

}  // namespace nade
