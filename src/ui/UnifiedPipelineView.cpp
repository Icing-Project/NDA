#include "ui/UnifiedPipelineView.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QMessageBox>
#include <QFrame>
#include <QSplitter>

namespace nda {

UnifiedPipelineView::UnifiedPipelineView(QWidget *parent)
    : QWidget(parent)
    , pttActive_(false)
{
    setupUI();
    applyModernStyles();
    
    // Setup metrics timer (60 FPS)
    metricsTimer_ = new QTimer(this);
    connect(metricsTimer_, &QTimer::timeout, this, &UnifiedPipelineView::updateMetrics);
    metricsTimer_->start(16); // ~60 FPS
}

UnifiedPipelineView::~UnifiedPipelineView()
{
}

void UnifiedPipelineView::setPluginManager(std::shared_ptr<PluginManager> manager)
{
    pluginManager_ = manager;
    refreshPluginLists();
}

void UnifiedPipelineView::setTXPipeline(std::shared_ptr<ProcessingPipeline> pipeline)
{
    txPipeline_ = pipeline;
}

void UnifiedPipelineView::setRXPipeline(std::shared_ptr<ProcessingPipeline> pipeline)
{
    rxPipeline_ = pipeline;
}

void UnifiedPipelineView::setupUI()
{
    QHBoxLayout *mainLayout = new QHBoxLayout(this);
    mainLayout->setSpacing(0);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    
    // Main content area
    QWidget *contentArea = new QWidget(this);
    QVBoxLayout *contentLayout = new QVBoxLayout(contentArea);
    contentLayout->setSpacing(10);
    contentLayout->setContentsMargins(20, 20, 20, 20);
    
    // Create TX and RX pipeline rows
    createTXPipelineRow(contentLayout);
    createRXPipelineRow(contentLayout);
    
    // Create control bar
    createControlBar(contentLayout);
    
    contentLayout->addStretch();
    
    // Plugin sidebar
    pluginSidebar_ = new PluginSidebar(this);
    pluginSidebar_->hide();
    
    // Add to splitter for resizable sidebar
    QSplitter *splitter = new QSplitter(Qt::Horizontal, this);
    splitter->addWidget(contentArea);
    splitter->addWidget(pluginSidebar_);
    splitter->setStretchFactor(0, 1);
    splitter->setStretchFactor(1, 0);
    
    mainLayout->addWidget(splitter);
}

void UnifiedPipelineView::createTXPipelineRow(QVBoxLayout* layout)
{
    QWidget *txCard = createPipelineCard("TRANSMIT (TX)", true);
    QVBoxLayout *cardLayout = new QVBoxLayout(txCard);
    cardLayout->setSpacing(10);
    
    // Add title label
    QLabel *titleLabel = new QLabel("TRANSMIT (TX)", this);
    titleLabel->setObjectName("pipelineTitle");
    cardLayout->addWidget(titleLabel);
    
    // Pipeline dropdowns row
    QHBoxLayout *pipelineRow = new QHBoxLayout();
    pipelineRow->setSpacing(10);
    
    // Source dropdown
    txSourceCombo_ = new QComboBox(this);
    txSourceCombo_->setObjectName("pipelineCombo");
    txSourceCombo_->addItem("(None)");
    txSourceCombo_->setMinimumHeight(35);
    connect(txSourceCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &UnifiedPipelineView::onTXSourceChanged);
    pipelineRow->addWidget(txSourceCombo_);
    
    // Arrow label
    QLabel *arrow1 = new QLabel("→", this);
    arrow1->setStyleSheet("color: #94a3b8; font-size: 18px;");
    pipelineRow->addWidget(arrow1);
    
    // Processor dropdown
    txProcessorCombo_ = new QComboBox(this);
    txProcessorCombo_->setObjectName("pipelineCombo");
    txProcessorCombo_->addItem("(None - Passthrough)");
    txProcessorCombo_->setMinimumHeight(35);
    connect(txProcessorCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &UnifiedPipelineView::onTXProcessorChanged);
    pipelineRow->addWidget(txProcessorCombo_);
    
    // Arrow label
    QLabel *arrow2 = new QLabel("→", this);
    arrow2->setStyleSheet("color: #94a3b8; font-size: 18px;");
    pipelineRow->addWidget(arrow2);
    
    // Sink dropdown
    txSinkCombo_ = new QComboBox(this);
    txSinkCombo_->setObjectName("pipelineCombo");
    txSinkCombo_->addItem("(None)");
    txSinkCombo_->setMinimumHeight(35);
    connect(txSinkCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &UnifiedPipelineView::onTXSinkChanged);
    pipelineRow->addWidget(txSinkCombo_);
    
    cardLayout->addLayout(pipelineRow);
    
    // Status and controls row
    QHBoxLayout *statusRow = new QHBoxLayout();
    statusRow->setSpacing(10);
    
    txStatusLabel_ = new QLabel("⚙️ Configure source and sink", this);
    txStatusLabel_->setObjectName("statusLabel");
    txStatusLabel_->setProperty("state", "stopped");
    statusRow->addWidget(txStatusLabel_);
    
    statusRow->addWidget(new QLabel("│", this));
    
    txLatencyLabel_ = new QLabel("--", this);
    txLatencyLabel_->setObjectName("metricLabel");
    statusRow->addWidget(new QLabel("Latency:", this));
    statusRow->addWidget(txLatencyLabel_);
    
    statusRow->addWidget(new QLabel("│", this));
    
    txCPULabel_ = new QLabel("--", this);
    txCPULabel_->setObjectName("metricLabel");
    statusRow->addWidget(new QLabel("CPU:", this));
    statusRow->addWidget(txCPULabel_);
    
    statusRow->addStretch();
    
    // PTT button
    pttButton_ = new QPushButton("🎤 PTT", this);
    pttButton_->setObjectName("pttButton");
    pttButton_->setMinimumHeight(35);
    pttButton_->setProperty("active", false);
    connect(pttButton_, &QPushButton::pressed, this, &UnifiedPipelineView::onPTTPressed);
    connect(pttButton_, &QPushButton::released, this, &UnifiedPipelineView::onPTTReleased);
    statusRow->addWidget(pttButton_);
    
    // Start TX button
    startTXButton_ = new QPushButton("▶ Start TX", this);
    startTXButton_->setObjectName("startButton");
    startTXButton_->setMinimumHeight(35);
    startTXButton_->setEnabled(false);
    connect(startTXButton_, &QPushButton::clicked, this, &UnifiedPipelineView::onStartTXClicked);
    statusRow->addWidget(startTXButton_);
    
    // Stop TX button
    stopTXButton_ = new QPushButton("■ Stop TX", this);
    stopTXButton_->setObjectName("stopButton");
    stopTXButton_->setMinimumHeight(35);
    stopTXButton_->setEnabled(false);
    connect(stopTXButton_, &QPushButton::clicked, this, &UnifiedPipelineView::onStopTXClicked);
    statusRow->addWidget(stopTXButton_);
    
    cardLayout->addLayout(statusRow);
    
    // Audio meters row
    QHBoxLayout *metersRow = new QHBoxLayout();
    metersRow->setSpacing(10);
    
    QLabel *inputLabel = new QLabel("Input:", this);
    inputLabel->setObjectName("metricLabel");
    metersRow->addWidget(inputLabel);
    
    txInputMeterL_ = createAudioMeter();
    metersRow->addWidget(txInputMeterL_, 1);
    
    txInputMeterR_ = createAudioMeter();
    metersRow->addWidget(txInputMeterR_, 1);
    
    cardLayout->addLayout(metersRow);
    
    layout->addWidget(txCard);
}

void UnifiedPipelineView::createRXPipelineRow(QVBoxLayout* layout)
{
    QWidget *rxCard = createPipelineCard("RECEIVE (RX)", false);
    QVBoxLayout *cardLayout = new QVBoxLayout(rxCard);
    cardLayout->setSpacing(10);
    
    // FIX: Add title label first (since createPipelineCard no longer adds it)
    QLabel *titleLabel = new QLabel("RECEIVE (RX)", this);
    titleLabel->setObjectName("pipelineTitle");
    cardLayout->addWidget(titleLabel);
    
    // Pipeline dropdowns row
    QHBoxLayout *pipelineRow = new QHBoxLayout();
    pipelineRow->setSpacing(10);
    
    // Source dropdown
    rxSourceCombo_ = new QComboBox(this);
    rxSourceCombo_->setObjectName("pipelineCombo");
    rxSourceCombo_->addItem("(None)");
    rxSourceCombo_->setMinimumHeight(35);
    connect(rxSourceCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &UnifiedPipelineView::onRXSourceChanged);
    pipelineRow->addWidget(rxSourceCombo_);
    
    // Arrow label
    QLabel *arrow1 = new QLabel("→", this);
    arrow1->setStyleSheet("color: #94a3b8; font-size: 18px;");
    pipelineRow->addWidget(arrow1);
    
    // Processor dropdown
    rxProcessorCombo_ = new QComboBox(this);
    rxProcessorCombo_->setObjectName("pipelineCombo");
    rxProcessorCombo_->addItem("(None - Passthrough)");
    rxProcessorCombo_->setMinimumHeight(35);
    connect(rxProcessorCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &UnifiedPipelineView::onRXProcessorChanged);
    pipelineRow->addWidget(rxProcessorCombo_);
    
    // Arrow label
    QLabel *arrow2 = new QLabel("→", this);
    arrow2->setStyleSheet("color: #94a3b8; font-size: 18px;");
    pipelineRow->addWidget(arrow2);
    
    // Sink dropdown
    rxSinkCombo_ = new QComboBox(this);
    rxSinkCombo_->setObjectName("pipelineCombo");
    rxSinkCombo_->addItem("(None)");
    rxSinkCombo_->setMinimumHeight(35);
    connect(rxSinkCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &UnifiedPipelineView::onRXSinkChanged);
    pipelineRow->addWidget(rxSinkCombo_);
    
    cardLayout->addLayout(pipelineRow);
    
    // Status and controls row
    QHBoxLayout *statusRow = new QHBoxLayout();
    statusRow->setSpacing(10);
    
    rxStatusLabel_ = new QLabel("⚙️ Configure source and sink", this);
    rxStatusLabel_->setObjectName("statusLabel");
    rxStatusLabel_->setProperty("state", "stopped");
    statusRow->addWidget(rxStatusLabel_);
    
    statusRow->addWidget(new QLabel("│", this));
    
    rxLatencyLabel_ = new QLabel("--", this);
    rxLatencyLabel_->setObjectName("metricLabel");
    statusRow->addWidget(new QLabel("Latency:", this));
    statusRow->addWidget(rxLatencyLabel_);
    
    statusRow->addWidget(new QLabel("│", this));
    
    rxCPULabel_ = new QLabel("--", this);
    rxCPULabel_->setObjectName("metricLabel");
    statusRow->addWidget(new QLabel("CPU:", this));
    statusRow->addWidget(rxCPULabel_);
    
    statusRow->addStretch();
    
    // Start RX button
    startRXButton_ = new QPushButton("▶ Start RX", this);
    startRXButton_->setObjectName("startButton");
    startRXButton_->setMinimumHeight(35);
    startRXButton_->setEnabled(false);
    connect(startRXButton_, &QPushButton::clicked, this, &UnifiedPipelineView::onStartRXClicked);
    statusRow->addWidget(startRXButton_);
    
    // Stop RX button
    stopRXButton_ = new QPushButton("■ Stop RX", this);
    stopRXButton_->setObjectName("stopButton");
    stopRXButton_->setMinimumHeight(35);
    stopRXButton_->setEnabled(false);
    connect(stopRXButton_, &QPushButton::clicked, this, &UnifiedPipelineView::onStopRXClicked);
    statusRow->addWidget(stopRXButton_);
    
    cardLayout->addLayout(statusRow);
    
    // Audio meters row
    QHBoxLayout *metersRow = new QHBoxLayout();
    metersRow->setSpacing(10);
    
    QLabel *outputLabel = new QLabel("Output:", this);
    outputLabel->setObjectName("metricLabel");
    metersRow->addWidget(outputLabel);
    
    rxOutputMeterL_ = createAudioMeter();
    metersRow->addWidget(rxOutputMeterL_, 1);
    
    rxOutputMeterR_ = createAudioMeter();
    metersRow->addWidget(rxOutputMeterR_, 1);
    
    cardLayout->addLayout(metersRow);
    
    layout->addWidget(rxCard);
}

void UnifiedPipelineView::createControlBar(QVBoxLayout* layout)
{
    QWidget *controlCard = createPipelineCard("", false);
    QHBoxLayout *controlLayout = new QHBoxLayout(controlCard);
    controlLayout->setSpacing(15);
    
    // Start Both button
    startBothButton_ = new QPushButton("▶ Start Both", this);
    startBothButton_->setObjectName("startBothButton");
    startBothButton_->setMinimumHeight(50);
    startBothButton_->setEnabled(false);
    connect(startBothButton_, &QPushButton::clicked, this, &UnifiedPipelineView::onStartBothClicked);
    controlLayout->addWidget(startBothButton_);
    
    // Stop Both button
    stopBothButton_ = new QPushButton("■ Stop Both", this);
    stopBothButton_->setObjectName("stopButton");
    stopBothButton_->setMinimumHeight(50);
    stopBothButton_->setEnabled(false);
    connect(stopBothButton_, &QPushButton::clicked, this, &UnifiedPipelineView::onStopBothClicked);
    controlLayout->addWidget(stopBothButton_);
    
    controlLayout->addStretch();
    
    // Settings button
    settingsButton_ = new QPushButton("📁 Settings", this);
    settingsButton_->setObjectName("secondaryButton");
    settingsButton_->setMinimumHeight(50);
    settingsButton_->setMinimumWidth(120);
    controlLayout->addWidget(settingsButton_);
    
    layout->addWidget(controlCard);
}

QWidget* UnifiedPipelineView::createPipelineCard(const QString& title, bool isTX)
{
    QWidget *card = new QWidget(this);
    card->setObjectName("pipelineCard");
    
    // Do NOT create layout here - let caller create it
    // This prevents the layout from being replaced when caller sets their own layout
    return card;
}

QProgressBar* UnifiedPipelineView::createAudioMeter()
{
    QProgressBar *meter = new QProgressBar(this);
    meter->setRange(0, 100);
    meter->setValue(0);
    meter->setTextVisible(false);
    meter->setMinimumHeight(10);
    meter->setMaximumHeight(10);
    return meter;
}

void UnifiedPipelineView::refreshPluginLists()
{
    if (!pluginManager_) return;
    
    // Clear existing items
    txSourceCombo_->clear();
    txProcessorCombo_->clear();
    txSinkCombo_->clear();
    rxSourceCombo_->clear();
    rxProcessorCombo_->clear();
    rxSinkCombo_->clear();
    
    txSourceCombo_->addItem("(None)");
    txProcessorCombo_->addItem("(None - Passthrough)");
    txSinkCombo_->addItem("(None)");
    rxSourceCombo_->addItem("(None)");
    rxProcessorCombo_->addItem("(None - Passthrough)");
    rxSinkCombo_->addItem("(None)");
    
    // Populate from plugin manager
    auto sources = pluginManager_->getPluginsByType(PluginType::AudioSource);
    for (const auto& plugin : sources) {
        QString name = QString::fromStdString(plugin.info.name);
        txSourceCombo_->addItem(name);
        rxSourceCombo_->addItem(name);
    }
    
    auto processors = pluginManager_->getPluginsByType(PluginType::Processor);
    for (const auto& plugin : processors) {
        QString name = QString::fromStdString(plugin.info.name);
        txProcessorCombo_->addItem(name);
        rxProcessorCombo_->addItem(name);
    }
    
    auto sinks = pluginManager_->getPluginsByType(PluginType::AudioSink);
    for (const auto& plugin : sinks) {
        QString name = QString::fromStdString(plugin.info.name);
        txSinkCombo_->addItem(name);
        rxSinkCombo_->addItem(name);
    }
}

void UnifiedPipelineView::onTXSourceChanged(int index)
{
    if (index <= 0 || !pluginManager_) {
        txSource_ = nullptr;
        updateTXStatus();
        return;
    }
    
    std::string pluginName = txSourceCombo_->currentText().toStdString();
    txSource_ = pluginManager_->getAudioSourcePlugin(pluginName);
    
    if (txSource_) {
        pluginSidebar_->showPluginConfig(txSource_);
    }
    
    updateTXStatus();
}

void UnifiedPipelineView::onTXProcessorChanged(int index)
{
    if (index <= 0 || !pluginManager_) {
        txProcessor_ = nullptr;
        updateTXStatus();
        return;
    }
    
    std::string pluginName = txProcessorCombo_->currentText().toStdString();
    txProcessor_ = pluginManager_->getAudioProcessorPlugin(pluginName);
    
    if (txProcessor_) {
        pluginSidebar_->showPluginConfig(txProcessor_);
    }
    
    updateTXStatus();
}

void UnifiedPipelineView::onTXSinkChanged(int index)
{
    if (index <= 0 || !pluginManager_) {
        txSink_ = nullptr;
        updateTXStatus();
        return;
    }
    
    std::string pluginName = txSinkCombo_->currentText().toStdString();
    txSink_ = pluginManager_->getAudioSinkPlugin(pluginName);
    
    if (txSink_) {
        pluginSidebar_->showPluginConfig(txSink_);
    }
    
    updateTXStatus();
}

void UnifiedPipelineView::onRXSourceChanged(int index)
{
    if (index <= 0 || !pluginManager_) {
        rxSource_ = nullptr;
        updateRXStatus();
        return;
    }
    
    std::string pluginName = rxSourceCombo_->currentText().toStdString();
    rxSource_ = pluginManager_->getAudioSourcePlugin(pluginName);
    
    if (rxSource_) {
        pluginSidebar_->showPluginConfig(rxSource_);
    }
    
    updateRXStatus();
}

void UnifiedPipelineView::onRXProcessorChanged(int index)
{
    if (index <= 0 || !pluginManager_) {
        rxProcessor_ = nullptr;
        updateRXStatus();
        return;
    }
    
    std::string pluginName = rxProcessorCombo_->currentText().toStdString();
    rxProcessor_ = pluginManager_->getAudioProcessorPlugin(pluginName);
    
    if (rxProcessor_) {
        pluginSidebar_->showPluginConfig(rxProcessor_);
    }
    
    updateRXStatus();
}

void UnifiedPipelineView::onRXSinkChanged(int index)
{
    if (index <= 0 || !pluginManager_) {
        rxSink_ = nullptr;
        updateRXStatus();
        return;
    }
    
    std::string pluginName = rxSinkCombo_->currentText().toStdString();
    rxSink_ = pluginManager_->getAudioSinkPlugin(pluginName);
    
    if (rxSink_) {
        pluginSidebar_->showPluginConfig(rxSink_);
    }
    
    updateRXStatus();
}

void UnifiedPipelineView::updateTXStatus()
{
    bool canStart = txSource_ && txSink_;
    bool isRunning = txPipeline_ && txPipeline_->isRunning();
    
    if (isRunning) {
        txStatusLabel_->setText("🟢 Running");
        txStatusLabel_->setProperty("state", "running");
    } else if (canStart) {
        txStatusLabel_->setText("✓ Ready to start");
        txStatusLabel_->setProperty("state", "ready");
    } else {
        txStatusLabel_->setText("⚙️ Configure source and sink");
        txStatusLabel_->setProperty("state", "stopped");
    }
    
    txStatusLabel_->style()->unpolish(txStatusLabel_);
    txStatusLabel_->style()->polish(txStatusLabel_);
    
    // Update individual start button
    startTXButton_->setEnabled(canStart && !isRunning);
    
    // Update Start Both button
    bool bothReady = (txSource_ && txSink_) && (rxSource_ && rxSink_);
    bool bothRunning = (txPipeline_ && txPipeline_->isRunning()) && 
                       (rxPipeline_ && rxPipeline_->isRunning());
    
    startBothButton_->setEnabled(bothReady && !bothRunning);
    stopBothButton_->setEnabled(bothRunning);
}

void UnifiedPipelineView::updateRXStatus()
{
    bool canStart = rxSource_ && rxSink_;
    bool isRunning = rxPipeline_ && rxPipeline_->isRunning();
    
    if (isRunning) {
        rxStatusLabel_->setText("🟢 Running");
        rxStatusLabel_->setProperty("state", "running");
    } else if (canStart) {
        rxStatusLabel_->setText("✓ Ready to start");
        rxStatusLabel_->setProperty("state", "ready");
    } else {
        rxStatusLabel_->setText("⚙️ Configure source and sink");
        rxStatusLabel_->setProperty("state", "stopped");
    }
    
    rxStatusLabel_->style()->unpolish(rxStatusLabel_);
    rxStatusLabel_->style()->polish(rxStatusLabel_);
    
    // Update individual start button
    startRXButton_->setEnabled(canStart && !isRunning);
    
    // Update Start Both button
    bool bothReady = (txSource_ && txSink_) && (rxSource_ && rxSink_);
    bool bothRunning = (txPipeline_ && txPipeline_->isRunning()) && 
                       (rxPipeline_ && rxPipeline_->isRunning());
    
    startBothButton_->setEnabled(bothReady && !bothRunning);
    stopBothButton_->setEnabled(bothRunning);
}

void UnifiedPipelineView::onStartTXClicked()
{
    if (!txPipeline_ || !txSource_ || !txSink_) return;
    
    txPipeline_->setSource(txSource_);
    if (txProcessor_) txPipeline_->setProcessor(txProcessor_);
    txPipeline_->setSink(txSink_);
    
    if (!txPipeline_->initialize()) {
        QMessageBox::critical(this, "TX Pipeline Error", "Failed to initialize TX pipeline");
        return;
    }
    
    if (!txPipeline_->start()) {
        QMessageBox::critical(this, "TX Pipeline Error", "Failed to start TX pipeline");
        return;
    }
    
    startTXButton_->setEnabled(false);
    stopTXButton_->setEnabled(true);
    updateTXStatus();
    emit txPipelineStarted();
}

void UnifiedPipelineView::onStopTXClicked()
{
    if (!txPipeline_) return;
    
    txPipeline_->stop();
    stopTXButton_->setEnabled(false);
    updateTXStatus();  // This will properly set startTXButton state
    emit txPipelineStopped();
}

void UnifiedPipelineView::onStartRXClicked()
{
    if (!rxPipeline_ || !rxSource_ || !rxSink_) return;
    
    rxPipeline_->setSource(rxSource_);
    if (rxProcessor_) rxPipeline_->setProcessor(rxProcessor_);
    rxPipeline_->setSink(rxSink_);
    
    if (!rxPipeline_->initialize()) {
        QMessageBox::critical(this, "RX Pipeline Error", "Failed to initialize RX pipeline");
        return;
    }
    
    if (!rxPipeline_->start()) {
        QMessageBox::critical(this, "RX Pipeline Error", "Failed to start RX pipeline");
        return;
    }
    
    startRXButton_->setEnabled(false);
    stopRXButton_->setEnabled(true);
    updateRXStatus();
    emit rxPipelineStarted();
}

void UnifiedPipelineView::onStopRXClicked()
{
    if (!rxPipeline_) return;
    
    rxPipeline_->stop();
    stopRXButton_->setEnabled(false);
    updateRXStatus();  // This will properly set startRXButton state
    emit rxPipelineStopped();
}

void UnifiedPipelineView::onStartBothClicked()
{
    // Start TX pipeline
    if (txPipeline_ && txSource_ && txSink_) {
        txPipeline_->setSource(txSource_);
        if (txProcessor_) txPipeline_->setProcessor(txProcessor_);
        txPipeline_->setSink(txSink_);
        
        if (!txPipeline_->initialize()) {
            QMessageBox::critical(this, "Pipeline Error", "Failed to initialize TX pipeline");
            return;
        }
        
        if (!txPipeline_->start()) {
            QMessageBox::critical(this, "Pipeline Error", "Failed to start TX pipeline");
            return;
        }
        
        startTXButton_->setEnabled(false);
        stopTXButton_->setEnabled(true);
        emit txPipelineStarted();
    }
    
    // Start RX pipeline
    if (rxPipeline_ && rxSource_ && rxSink_) {
        rxPipeline_->setSource(rxSource_);
        if (rxProcessor_) rxPipeline_->setProcessor(rxProcessor_);
        rxPipeline_->setSink(rxSink_);
        
        if (!rxPipeline_->initialize()) {
            QMessageBox::critical(this, "Pipeline Error", "Failed to initialize RX pipeline");
            // Stop TX if RX fails
            if (txPipeline_) {
                txPipeline_->stop();
                startTXButton_->setEnabled(true);
                stopTXButton_->setEnabled(false);
            }
            return;
        }
        
        if (!rxPipeline_->start()) {
            QMessageBox::critical(this, "Pipeline Error", "Failed to start RX pipeline");
            // Stop TX if RX fails
            if (txPipeline_) {
                txPipeline_->stop();
                startTXButton_->setEnabled(true);
                stopTXButton_->setEnabled(false);
            }
            return;
        }
        
        startRXButton_->setEnabled(false);
        stopRXButton_->setEnabled(true);
        emit rxPipelineStarted();
    }
    
    updateTXStatus();
    updateRXStatus();
}

void UnifiedPipelineView::onStopBothClicked()
{
    if (txPipeline_) {
        txPipeline_->stop();
        stopTXButton_->setEnabled(false);
        emit txPipelineStopped();
    }
    
    if (rxPipeline_) {
        rxPipeline_->stop();
        stopRXButton_->setEnabled(false);
        emit rxPipelineStopped();
    }
    
    updateTXStatus();  // This will properly set startTXButton state
    updateRXStatus();  // This will properly set startRXButton state
}

void UnifiedPipelineView::onPTTPressed()
{
    pttActive_ = true;
    pttButton_->setProperty("active", true);
    pttButton_->style()->unpolish(pttButton_);
    pttButton_->style()->polish(pttButton_);
    
    // If TX source plugin supports PTT, trigger it
    if (txSource_) {
        std::string supportsPTT = txSource_->getParameter("supports_ptt");
        if (supportsPTT == "true") {
            txSource_->setParameter("ptt_active", "true");
        } else {
            // Fallback: Unmute audio at pipeline level
            if (txPipeline_ && txPipeline_->isRunning()) {
                // Pipeline mute/unmute would be implemented here
            }
        }
    }
}

void UnifiedPipelineView::onPTTReleased()
{
    pttActive_ = false;
    pttButton_->setProperty("active", false);
    pttButton_->style()->unpolish(pttButton_);
    pttButton_->style()->polish(pttButton_);
    
    // If TX source plugin supports PTT, release it
    if (txSource_) {
        std::string supportsPTT = txSource_->getParameter("supports_ptt");
        if (supportsPTT == "true") {
            txSource_->setParameter("ptt_active", "false");
        } else {
            // Fallback: Mute audio at pipeline level
            if (txPipeline_ && txPipeline_->isRunning()) {
                // Pipeline mute/unmute would be implemented here
            }
        }
    }
}

void UnifiedPipelineView::updateMetrics()
{
    // TX pipeline metrics
    if (txPipeline_ && txPipeline_->isRunning()) {
        double txLatency = txPipeline_->getLatency();
        float txCPU = txPipeline_->getCPULoad();
        
        txLatencyLabel_->setText(QString::number(txLatency, 'f', 1) + " ms");
        txCPULabel_->setText(QString::number(static_cast<int>(txCPU)) + "%");
        
        float txPeakL = 0.0f;
        float txPeakR = 0.0f;
        txPipeline_->getPeakLevels(txPeakL, txPeakR);
        txInputMeterL_->setValue(static_cast<int>(txPeakL * 100.0f));
        txInputMeterR_->setValue(static_cast<int>(txPeakR * 100.0f));
    } else {
        txLatencyLabel_->setText("--");
        txCPULabel_->setText("--");
        txInputMeterL_->setValue(0);
        txInputMeterR_->setValue(0);
    }
    
    // RX pipeline metrics
    if (rxPipeline_ && rxPipeline_->isRunning()) {
        double rxLatency = rxPipeline_->getLatency();
        float rxCPU = rxPipeline_->getCPULoad();
        
        rxLatencyLabel_->setText(QString::number(rxLatency, 'f', 1) + " ms");
        rxCPULabel_->setText(QString::number(static_cast<int>(rxCPU)) + "%");
        
        float rxPeakL = 0.0f;
        float rxPeakR = 0.0f;
        rxPipeline_->getPeakLevels(rxPeakL, rxPeakR);
        rxOutputMeterL_->setValue(static_cast<int>(rxPeakL * 100.0f));
        rxOutputMeterR_->setValue(static_cast<int>(rxPeakR * 100.0f));
    } else {
        rxLatencyLabel_->setText("--");
        rxCPULabel_->setText("--");
        rxOutputMeterL_->setValue(0);
        rxOutputMeterR_->setValue(0);
    }
}

void UnifiedPipelineView::onPluginFocused(const std::string& pluginName, PluginType type)
{
    // Handle plugin focus for showing configuration sidebar
    // This would be called when user clicks on a dropdown
}

void UnifiedPipelineView::applyModernStyles()
{
    setStyleSheet(R"(
        QWidget {
            background-color: #1a1a2e;
            color: #eaeaea;
            font-family: 'Segoe UI', Arial, sans-serif;
        }

        #pipelineCard {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                      stop:0 rgba(40, 40, 55, 0.95),
                                      stop:1 rgba(30, 30, 45, 0.95));
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 15px;
            margin: 5px;
        }

        #pipelineTitle {
            font-size: 16px;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: 0.5px;
            margin-bottom: 10px;
        }

        #pipelineCombo {
            background-color: #2a2a3e;
            border: 1px solid #3a3a4e;
            border-radius: 6px;
            padding: 8px;
            color: #eaeaea;
            font-size: 13px;
            min-height: 30px;
            max-width: 250px;
        }

        #pipelineCombo:hover {
            border: 1px solid #3b82f6;
            background-color: #34344e;
        }

        #pipelineCombo::drop-down {
            border: none;
            width: 25px;
        }

        #pipelineCombo::down-arrow {
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 6px solid #94a3b8;
            margin-right: 8px;
        }

        #statusLabel {
            font-size: 13px;
            font-weight: 600;
            color: #94a3b8;
            padding: 4px 8px;
            border-radius: 4px;
        }

        #statusLabel[state="running"] {
            color: #4ade80;
            background-color: rgba(74, 222, 128, 0.1);
        }

        #statusLabel[state="ready"] {
            color: #4ade80;
            background-color: rgba(74, 222, 128, 0.05);
        }

        #statusLabel[state="stopped"] {
            color: #94a3b8;
            background-color: rgba(148, 163, 184, 0.1);
        }

        #metricLabel {
            font-size: 12px;
            color: #cbd5e1;
            padding: 3px;
        }

        #startButton {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                      stop:0 #10b981,
                                      stop:1 #059669);
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 600;
            padding: 8px 15px;
            min-width: 90px;
        }

        #startButton:hover:enabled {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                      stop:0 #059669,
                                      stop:1 #047857);
        }

        #startButton:disabled {
            background-color: #334155;
            color: #64748b;
        }

        #pttButton {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                      stop:0 #475569,
                                      stop:1 #334155);
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 600;
            padding: 8px 15px;
            min-width: 80px;
        }

        #pttButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                      stop:0 #64748b,
                                      stop:1 #475569);
        }

        #pttButton[active="true"] {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                      stop:0 #10b981,
                                      stop:1 #059669);
        }

        #stopButton {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                      stop:0 #dc2626,
                                      stop:1 #b91c1c);
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 600;
            padding: 8px 15px;
            min-width: 90px;
        }

        #stopButton:hover:enabled {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                      stop:0 #b91c1c,
                                      stop:1 #991b1b);
        }

        #stopButton:disabled {
            background-color: #334155;
            color: #64748b;
        }

        #startBothButton {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                      stop:0 #10b981,
                                      stop:1 #059669);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 700;
            padding: 12px 25px;
        }

        #startBothButton:hover:enabled {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                      stop:0 #059669,
                                      stop:1 #047857);
        }

        #startBothButton:disabled {
            background-color: #334155;
            color: #64748b;
        }

        #secondaryButton {
            background-color: #475569;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            padding: 12px;
        }

        #secondaryButton:hover {
            background-color: #64748b;
        }

        QProgressBar {
            border: none;
            border-radius: 4px;
            background-color: rgba(255, 255, 255, 0.04);
            height: 10px;
            text-align: center;
        }

        QProgressBar::chunk {
            background-color: #4ade80;
            border-radius: 4px;
        }
    )");
}

} // namespace nda
