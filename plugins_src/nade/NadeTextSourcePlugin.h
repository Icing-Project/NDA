/**
 * @file NadeTextSourcePlugin.h
 * @brief NDA Source Plugin for text input (TX path).
 *
 * This plugin provides a Qt GUI for user text input and encodes the text
 * as "fake audio" samples that flow through NDA's TX pipeline.
 *
 * Responsibilities:
 *   - Provide Qt GUI with QTextEdit for text input
 *   - Encode user text as "audio" samples
 *   - Send text to NadeExternalIO for FSK transmission
 *   - Disable send button during transmission
 *
 * @version 3.0
 * @date 2026-01-30
 */

#pragma once

#include "plugins/AudioSourcePlugin.h"
#include <memory>
#include <vector>
#include <atomic>

// Forward declarations for Qt types (avoid Qt header in plugin header)
class QWidget;
class QTextEdit;
class QPushButton;
class QTimer;
class QLabel;
class QLineEdit;
class QGroupBox;

namespace nade {

/**
 * @brief Source plugin for text input (TX path) with Qt GUI.
 */
class NadeTextSourcePlugin : public nda::AudioSourcePlugin {
public:
    NadeTextSourcePlugin();
    ~NadeTextSourcePlugin() override;

    // =========================================================================
    // Plugin Metadata (BasePlugin)
    // =========================================================================

    nda::PluginInfo getInfo() const override;
    nda::PluginType getType() const override { return nda::PluginType::AudioSource; }
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
    // Audio Source Interface
    // =========================================================================

    void setAudioCallback(nda::AudioSourceCallback callback) override;
    bool readAudio(AudioBuffer& buffer) override;

    int getSampleRate() const override { return sampleRate_; }
    int getChannels() const override { return channelCount_; }
    void setSampleRate(int sampleRate) override { sampleRate_ = sampleRate; }
    void setChannels(int channels) override { channelCount_ = channels; }

    int getBufferSize() const override { return 512; }

    // =========================================================================
    // GUI (Qt)
    // =========================================================================

    /**
     * @brief Create Qt GUI widget for text input.
     * @return QWidget to be docked in NDA main window
     */
    QWidget* createDockableGui() override;

private:
    // GUI event handlers
    void onSendClicked();
    void checkTxStatus();

    nda::PluginState state_ = nda::PluginState::Unloaded;
    nda::AudioSourceCallback audioCallback_;

    int sampleRate_ = 48000;
    int channelCount_ = 2;

    // GUI widgets (shared across all instances - singleton pattern)
    // This allows the pipeline instance to control the GUI created by the dock instance
    static QWidget* sharedGuiWidget_;
    static QTextEdit* sharedInputText_;
    static QPushButton* sharedSendButton_;
    static QLabel* sharedStatusLabel_;
    static QTimer* sharedTxCheckTimer_;
    static std::atomic<nda::PluginState> sharedState_;

    // Pending audio output (shared across all instances)
    // The GUI instance writes to this, the pipeline instance reads from it
    static std::vector<float> sharedPendingAudio_;
    static std::atomic<size_t> sharedAudioPosition_;
};

}  // namespace nade
