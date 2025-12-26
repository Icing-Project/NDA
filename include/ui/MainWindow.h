#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QMenuBar>
#include <QStatusBar>
#include "core/ProcessingPipeline.h"
#include "plugins/PluginManager.h"
#include <memory>

namespace nda {
    class UnifiedPipelineView;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    
    void autoLoadPlugins();  // v2.0: Auto-load plugins on startup

private slots:
    void onTXPipelineStarted();
    void onTXPipelineStopped();
    void onRXPipelineStarted();
    void onRXPipelineStopped();
    void onStatusUpdate(const QString &message);

private:
    void setupUI();
    void createMenus();
    void createStatusBar();

    // v2.0: Single unified view (no tabs)
    nda::UnifiedPipelineView *unifiedView_;

    // Core components
    std::shared_ptr<nda::PluginManager> pluginManager_;
    
    // v2.0: Dual pipeline architecture (TX + RX)
    std::shared_ptr<nda::ProcessingPipeline> txPipeline_;  // Transmit pipeline
    std::shared_ptr<nda::ProcessingPipeline> rxPipeline_;  // Receive pipeline
};

#endif // MAINWINDOW_H
