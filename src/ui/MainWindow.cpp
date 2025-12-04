#include "ui/MainWindow.h"
#include "ui/Dashboard.h"
#include "ui/PipelineView.h"
#include "ui/SettingsView.h"
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QStatusBar>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    setWindowTitle("NDA - Plugin-Based Audio Encryption System");
    setMinimumSize(1200, 800);

    // Create core components
    pluginManager_ = std::make_shared<nda::PluginManager>();
    pipeline_ = std::make_shared<nda::ProcessingPipeline>();

    setupUI();
    createMenus();
    createStatusBar();
}

MainWindow::~MainWindow()
{
}

void MainWindow::setupUI()
{
    tabWidget = new QTabWidget(this);
    setCentralWidget(tabWidget);

    // Create views
    pipelineView = new PipelineView(this);
    pipelineView->setPluginManager(pluginManager_);
    pipelineView->setPipeline(pipeline_);

    dashboard = new Dashboard(this);
    dashboard->setPipeline(pipeline_);

    settingsView = new SettingsView(this);

    // Add tabs
    tabWidget->addTab(pipelineView, "Pipeline Configuration");
    tabWidget->addTab(dashboard, "Dashboard");
    tabWidget->addTab(settingsView, "Settings");

    // Connect signals
    connect(pipelineView, &PipelineView::pipelineStarted, this, &MainWindow::onPipelineStarted);
    connect(pipelineView, &PipelineView::pipelineStopped, this, &MainWindow::onPipelineStopped);
    connect(dashboard, &Dashboard::streamStarted, this, &MainWindow::onPipelineStarted);
    connect(dashboard, &Dashboard::streamStopped, this, &MainWindow::onPipelineStopped);
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

void MainWindow::onPipelineStarted()
{
    statusBar()->showMessage("Pipeline Running - Processing Audio");
    // Update Dashboard UI state
    dashboard->updatePipelineState();
    // Auto-switch to Dashboard tab to show live metrics
    tabWidget->setCurrentWidget(dashboard);
}

void MainWindow::onPipelineStopped()
{
    statusBar()->showMessage("Pipeline Stopped");
    // Update Dashboard UI state
    dashboard->updatePipelineState();
}

void MainWindow::onStatusUpdate(const QString &message)
{
    statusBar()->showMessage(message);
}
