#include "ui/MainWindow.h"
#include "ui/UnifiedPipelineView.h"
#include <QAction>
#include <QCoreApplication>
#include <QDebug>
#include <QDir>
#include <QMenu>
#include <QMenuBar>
#include <QStatusBar>
#include <iostream>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    setWindowTitle("NDA v2.0 - Real-Time Audio Encryption Bridge");
    setMinimumSize(1200, 800);

    // Create core components
    pluginManager_ = std::make_shared<nda::PluginManager>();

    // v2.0: Dual pipeline architecture (TX + RX)
    txPipeline_ = std::make_shared<nda::ProcessingPipeline>();
    rxPipeline_ = std::make_shared<nda::ProcessingPipeline>();

    setupUI();
    createMenus();
    createStatusBar();
}

MainWindow::~MainWindow()
{
}

void MainWindow::setupUI()
{
    // v2.0: No tabs - single unified view
    unifiedView_ = new nda::UnifiedPipelineView(this);
    unifiedView_->setPluginManager(pluginManager_);
    unifiedView_->setTXPipeline(txPipeline_);
    unifiedView_->setRXPipeline(rxPipeline_);

    setCentralWidget(unifiedView_);

    // Connect signals
    connect(unifiedView_, &nda::UnifiedPipelineView::txPipelineStarted,
            this, &MainWindow::onTXPipelineStarted);
    connect(unifiedView_, &nda::UnifiedPipelineView::txPipelineStopped,
            this, &MainWindow::onTXPipelineStopped);
    connect(unifiedView_, &nda::UnifiedPipelineView::rxPipelineStarted,
            this, &MainWindow::onRXPipelineStarted);
    connect(unifiedView_, &nda::UnifiedPipelineView::rxPipelineStopped,
            this, &MainWindow::onRXPipelineStopped);
}

void MainWindow::createMenus()
{
    QMenu *fileMenu = menuBar()->addMenu("&File");

    QAction *exitAction = new QAction("E&xit", this);
    exitAction->setShortcut(QKeySequence::Quit);
    connect(exitAction, &QAction::triggered, this, &QMainWindow::close);
    fileMenu->addAction(exitAction);

    QMenu *helpMenu = menuBar()->addMenu("&Help");
    QAction *aboutAction = new QAction("&About", this);
    helpMenu->addAction(aboutAction);
}

void MainWindow::createStatusBar()
{
    statusBar()->showMessage("Ready");
}

void MainWindow::onTXPipelineStarted()
{
    statusBar()->showMessage("TX Pipeline Running");
}

void MainWindow::onTXPipelineStopped()
{
    statusBar()->showMessage("TX Pipeline Stopped");
}

void MainWindow::onRXPipelineStarted()
{
    statusBar()->showMessage("RX Pipeline Running");
}

void MainWindow::onRXPipelineStopped()
{
    statusBar()->showMessage("RX Pipeline Stopped");
}

void MainWindow::onStatusUpdate(const QString &message)
{
    statusBar()->showMessage(message);
}

void MainWindow::autoLoadPlugins()
{
    if (!pluginManager_) return;

#ifndef NDA_ENABLE_PYTHON
    qDebug() << "Python plugin support is disabled in this build; .py plugins will not be loaded";
#endif

    // Auto-discover plugins from standard directories
    const QString appDir = QCoreApplication::applicationDirPath();

    // Python plugins
    QStringList pythonCandidates = {
        appDir + "/plugins_py",
        appDir + "/../plugins_py",
        appDir + "/../../plugins_py"
    };

    // C++ plugins
    QStringList cppCandidates = {
        appDir + "/plugins",
        appDir + "/plugins/Release",
        appDir + "/plugins/Debug",
        appDir + "/../build/plugins",
        appDir + "/../../build/plugins"
    };

    int loadedCount = 0;

#ifdef NDA_ENABLE_PYTHON
    // Load Python plugins
    for (const auto& dir : pythonCandidates) {
        if (QDir(dir).exists()) {
            auto paths = pluginManager_->scanPluginDirectory(dir.toStdString());
            for (const auto& path : paths) {
                if (pluginManager_->loadPlugin(path)) {
                    loadedCount++;
                }
            }
        }
    }
#endif

    // Load C++ plugins
    for (const auto& dir : cppCandidates) {
        if (QDir(dir).exists()) {
            auto paths = pluginManager_->scanPluginDirectory(dir.toStdString());
            for (const auto& path : paths) {
                if (pluginManager_->loadPlugin(path)) {
                    loadedCount++;
                }
            }
        }
    }

    qDebug() << "Auto-loaded" << loadedCount << "plugins on startup";

    // Notify unified view to refresh plugin lists
    if (unifiedView_) {
        unifiedView_->refreshPluginLists();
    }
}

// v2.1: Soak test support methods

void MainWindow::startBothPipelines()
{
    if (unifiedView_) {
        // Trigger the Start Both button click programmatically
        QMetaObject::invokeMethod(unifiedView_, "onStartBothClicked");
    }
}

void MainWindow::stopBothPipelines()
{
    if (unifiedView_) {
        // Trigger the Stop Both button click programmatically
        QMetaObject::invokeMethod(unifiedView_, "onStopBothClicked");
    }
}

void MainWindow::printSoakTestReport()
{
    std::cout << "\n========================================\n";
    std::cout << "       SOAK TEST REPORT\n";
    std::cout << "========================================\n\n";

    if (txPipeline_) {
        std::cout << "TX Pipeline:\n";
        std::cout << "  Status: " << (txPipeline_->isRunning() ? "Running" : "Stopped") << "\n";
        std::cout << "  Processed: " << txPipeline_->getProcessedSamples() << " samples\n";
        std::cout << "  Dropped: " << txPipeline_->getDroppedSamples() << " samples\n";
        std::cout << "  Read Failures: " << txPipeline_->getReadFailures() << "\n";
        std::cout << "  Write Failures: " << txPipeline_->getWriteFailures() << "\n";
        std::cout << "  Current Drift: " << txPipeline_->getCurrentDriftMs() << " ms\n";
        std::cout << "  Max Drift: " << txPipeline_->getMaxDriftMs() << " ms\n";

        auto txHealth = txPipeline_->getHealthStatus();
        std::cout << "  Health: ";
        switch (txHealth) {
            case nda::ProcessingPipeline::HealthStatus::OK:
                std::cout << "OK (0)\n";
                break;
            case nda::ProcessingPipeline::HealthStatus::Degraded:
                std::cout << "Degraded (1)\n";
                break;
            case nda::ProcessingPipeline::HealthStatus::Failing:
                std::cout << "Failing (2)\n";
                break;
        }
        std::cout << "\n";
    }

    if (rxPipeline_) {
        std::cout << "RX Pipeline:\n";
        std::cout << "  Status: " << (rxPipeline_->isRunning() ? "Running" : "Stopped") << "\n";
        std::cout << "  Processed: " << rxPipeline_->getProcessedSamples() << " samples\n";
        std::cout << "  Dropped: " << rxPipeline_->getDroppedSamples() << " samples\n";
        std::cout << "  Read Failures: " << rxPipeline_->getReadFailures() << "\n";
        std::cout << "  Write Failures: " << rxPipeline_->getWriteFailures() << "\n";
        std::cout << "  Current Drift: " << rxPipeline_->getCurrentDriftMs() << " ms\n";
        std::cout << "  Max Drift: " << rxPipeline_->getMaxDriftMs() << " ms\n";

        auto rxHealth = rxPipeline_->getHealthStatus();
        std::cout << "  Health: ";
        switch (rxHealth) {
            case nda::ProcessingPipeline::HealthStatus::OK:
                std::cout << "OK (0)\n";
                break;
            case nda::ProcessingPipeline::HealthStatus::Degraded:
                std::cout << "Degraded (1)\n";
                break;
            case nda::ProcessingPipeline::HealthStatus::Failing:
                std::cout << "Failing (2)\n";
                break;
        }
        std::cout << "\n";
    }

    std::cout << "========================================\n";
    std::cout << "       END OF REPORT\n";
    std::cout << "========================================\n\n";
}
