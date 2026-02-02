#ifndef UNIFIEDPIPELINEVIEW_H
#define UNIFIEDPIPELINEVIEW_H

#include <QWidget>
#include <QComboBox>
#include <QPushButton>
#include <QLabel>
#include <QProgressBar>
#include <QTimer>
#include <QShortcut>
#include <QKeyEvent>
#include <QFocusEvent>
#include "core/ProcessingPipeline.h"
#include "plugins/PluginManager.h"
#include "ui/PluginSidebar.h"
#include <memory>

namespace nda {

/**
 * Unified pipeline view - combines pipeline config and dashboard
 * 
 * Layout:
 * - TX pipeline row (Source → Processor → Sink) with metrics
 * - RX pipeline row (Source → Processor → Sink) with metrics
 * - Global controls (Start Both, Stop Both)
 * - Collapsible plugin sidebar for parameters
 */
class UnifiedPipelineView : public QWidget
{
    Q_OBJECT

public:
    explicit UnifiedPipelineView(QWidget *parent = nullptr);
    ~UnifiedPipelineView();

    void setPluginManager(std::shared_ptr<PluginManager> manager);
    void setTXPipeline(std::shared_ptr<ProcessingPipeline> pipeline);
    void setRXPipeline(std::shared_ptr<ProcessingPipeline> pipeline);

    void refreshPluginLists();

    // Cancel any pending plugin selection changes (called before Apply)
    void cancelPendingPluginChange();

    // Get currently selected plugins (for key application)
    std::shared_ptr<AudioProcessorPlugin> getTXProcessor() const { return txProcessor_; }
    std::shared_ptr<AudioProcessorPlugin> getRXProcessor() const { return rxProcessor_; }

signals:
    void txPipelineStarted();
    void txPipelineStopped();
    void rxPipelineStarted();
    void rxPipelineStopped();
    void pluginSelected(const QString& pluginName, bool selected);

private slots:
    // TX pipeline
    void onTXSourceChanged(int index);
    void onTXProcessorChanged(int index);
    void onTXSinkChanged(int index);
    void onStartTXClicked();
    void onStopTXClicked();
    void onPTTPressed();
    void onPTTReleased();
    
    // RX pipeline
    void onRXSourceChanged(int index);
    void onRXProcessorChanged(int index);
    void onRXSinkChanged(int index);
    void onStartRXClicked();
    void onStopRXClicked();
    
    // Global controls
    void onStartBothClicked();
    void onStopBothClicked();

    // Bridge Mode (v2.1)
    void onBridgeModeClicked();

    // Metrics update
    void updateMetrics();
    // v2.2: Removed dead onPluginFocused slot

    // v2.2: Plugin configuration button handlers (cogwheel buttons)
    void onTXSourceConfigClicked();
    void onTXProcessorConfigClicked();
    void onTXSinkConfigClicked();
    void onRXSourceConfigClicked();
    void onRXProcessorConfigClicked();
    void onRXSinkConfigClicked();

protected:
    // Keyboard input for PTT (T and Space keys)
    void keyPressEvent(QKeyEvent* event) override;
    void keyReleaseEvent(QKeyEvent* event) override;
    void focusOutEvent(QFocusEvent* event) override;

private:
    void setupUI();
    void createTXPipelineRow(QVBoxLayout* layout);
    void createRXPipelineRow(QVBoxLayout* layout);
    void createControlBar(QVBoxLayout* layout);
    void updateTXStatus();
    void updateRXStatus();
    void applyModernStyles();

    // PTT helpers
    bool isAIOCSink(std::shared_ptr<AudioSinkPlugin> sink) const;
    void updatePTTButtonState();
    
    QWidget* createPipelineCard(const QString& title, bool isTX);
    QProgressBar* createAudioMeter();

    // v2.2: Cogwheel button helpers
    QPushButton* createConfigButton();
    void updateConfigButtonStates();

    // Handshake status helpers
    void updateHandshakeLabel(QLabel* label, int phase);
    void onManualOverride();
    void onRestartDiscovery();
    void forceHandshake(bool isInitiator);

    // Plugin manager
    std::shared_ptr<PluginManager> pluginManager_;
    
    // Pipelines
    std::shared_ptr<ProcessingPipeline> txPipeline_;
    std::shared_ptr<ProcessingPipeline> rxPipeline_;
    
    // TX pipeline UI components
    QComboBox *txSourceCombo_;
    QComboBox *txProcessorCombo_;
    QComboBox *txSinkCombo_;
    QLabel *txStatusLabel_;
    QLabel *txLatencyLabel_;
    QLabel *txCPULabel_;
    QPushButton *pttButton_;
    QPushButton *startTXButton_;
    QPushButton *stopTXButton_;
    QProgressBar *txInputMeterL_;
    QProgressBar *txInputMeterR_;

    // TX diagnostics (v2.1)
    QLabel *txHealthLabel_;
    QLabel *txHandshakeLabel_;  // Handshake status (discovery/handshaking/established)
    QLabel *txDriftLabel_;
    QLabel *txReadFailsLabel_;
    QLabel *txWriteFailsLabel_;

    // RX pipeline UI components
    QComboBox *rxSourceCombo_;
    QComboBox *rxProcessorCombo_;
    QComboBox *rxSinkCombo_;
    QLabel *rxStatusLabel_;
    QLabel *rxLatencyLabel_;
    QLabel *rxCPULabel_;
    QPushButton *startRXButton_;
    QPushButton *stopRXButton_;
    QProgressBar *rxOutputMeterL_;
    QProgressBar *rxOutputMeterR_;

    // RX diagnostics (v2.1)
    QLabel *rxHealthLabel_;
    QLabel *rxHandshakeLabel_;  // Handshake status (discovery/handshaking/established)
    QLabel *rxDriftLabel_;
    QLabel *rxReadFailsLabel_;
    QLabel *rxWriteFailsLabel_;
    
    // Global controls
    QPushButton *bridgeModeButton_;  // v2.1: One-click Bridge Mode setup
    QPushButton *startBothButton_;
    QPushButton *stopBothButton_;
    // v2.2: Removed dead settingsButton_

    // v2.2: Cogwheel settings buttons (explicit sidebar trigger)
    QPushButton *txSourceConfigBtn_;
    QPushButton *txProcessorConfigBtn_;
    QPushButton *txSinkConfigBtn_;
    QPushButton *rxSourceConfigBtn_;
    QPushButton *rxProcessorConfigBtn_;
    QPushButton *rxSinkConfigBtn_;

    // Bridge Mode state (v2.1)
    bool bridgeModeActive_;
    
    // Plugin sidebar
    PluginSidebar *pluginSidebar_;
    
    // Selected plugins
    std::shared_ptr<AudioSourcePlugin> txSource_;
    std::shared_ptr<AudioProcessorPlugin> txProcessor_;
    std::shared_ptr<AudioSinkPlugin> txSink_;
    std::shared_ptr<AudioSourcePlugin> rxSource_;
    std::shared_ptr<AudioProcessorPlugin> rxProcessor_;
    std::shared_ptr<AudioSinkPlugin> rxSink_;
    
    // Metrics timer (60 FPS)
    QTimer *metricsTimer_;

    // PTT state
    bool pttActive_;

    // Debounce timer for plugin selection (v2.2 stability fix)
    QTimer *pluginSelectionDebounceTimer_;
    enum class PendingPluginChange {
        None,
        TXSource,
        TXProcessor,
        TXSink,
        RXSource,
        RXProcessor,
        RXSink
    };
    PendingPluginChange pendingChange_;
    int pendingIndex_;

    // Batch update mode - bypasses debouncing for programmatic plugin changes
    bool batchUpdateMode_;

    // Process debounced plugin changes
    void processPendingPluginChange();
    void processTXSourceChange(int index);
    void processTXProcessorChange(int index);
    void processTXSinkChange(int index);
    void processRXSourceChange(int index);
    void processRXProcessorChange(int index);
    void processRXSinkChange(int index);
};

} // namespace nda

#endif // UNIFIEDPIPELINEVIEW_H

