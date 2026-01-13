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

signals:
    void txPipelineStarted();
    void txPipelineStopped();
    void rxPipelineStarted();
    void rxPipelineStopped();

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
    QLabel *rxDriftLabel_;
    QLabel *rxReadFailsLabel_;
    QLabel *rxWriteFailsLabel_;
    
    // Global controls
    QPushButton *bridgeModeButton_;  // v2.1: One-click Bridge Mode setup
    QPushButton *startBothButton_;
    QPushButton *stopBothButton_;
    // v2.2: Removed dead settingsButton_

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
};

} // namespace nda

#endif // UNIFIEDPIPELINEVIEW_H

