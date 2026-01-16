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

    // v2.1: Soak test support
    void startBothPipelines();
    void stopBothPipelines();
    void printSoakTestReport();

private slots:
    void onTXPipelineStarted();
    void onTXPipelineStopped();
    void onRXPipelineStarted();
    void onRXPipelineStopped();
    // v2.2: Removed dead onStatusUpdate slot

    // Crypto menu slots
    void onGenerateAESKey();
    void onGenerateX25519KeyPair();
    void onImportKeys();
    void onExportKeys();
    void onDeriveSharedKey();
    void onClearKeys();

private:
    void setupUI();
    void createMenus();
    void createStatusBar();

    // Crypto helper methods
    bool applyKeyToSelectedPlugin(const std::string& paramName, const std::string& hexValue);

    // v2.0: Single unified view (no tabs)
    nda::UnifiedPipelineView *unifiedView_;

    // Core components
    std::shared_ptr<nda::PluginManager> pluginManager_;
    
    // v2.0: Dual pipeline architecture (TX + RX)
    std::shared_ptr<nda::ProcessingPipeline> txPipeline_;  // Transmit pipeline
    std::shared_ptr<nda::ProcessingPipeline> rxPipeline_;  // Receive pipeline
};

#endif // MAINWINDOW_H
