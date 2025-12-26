#include "ui/MainWindow.h"
#include "ui/UnifiedPipelineView.h"
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QStatusBar>
#include <QCoreApplication>
#include <QDir>
#include <QDebug>

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
