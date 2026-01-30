#include "ui/UnifiedPipelineView.h"
#include <QFormLayout>
#include <QFrame>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QMessageBox>
#include <QSignalBlocker>
#include <QSplitter>
#include <QVBoxLayout>
#include <algorithm>
#include <iostream>

namespace nda {

UnifiedPipelineView::UnifiedPipelineView(QWidget *parent)
    : QWidget(parent)
    , pttActive_(false)
    , bridgeModeActive_(false)  // v2.1: Bridge Mode initially off
    , pendingChange_(PendingPluginChange::None)  // v2.2: Initialize debounce state
    , pendingIndex_(-1)
    , batchUpdateMode_(false)  // v2.2: Batch mode off by default
{
    setupUI();
    applyModernStyles();

    // Enable keyboard input for PTT (T and Space keys)
    setFocusPolicy(Qt::StrongFocus);

    // Setup metrics timer (60 FPS)
    metricsTimer_ = new QTimer(this);
    connect(metricsTimer_, &QTimer::timeout, this, &UnifiedPipelineView::updateMetrics);
    metricsTimer_->start(16); // ~60 FPS

    // Setup debounce timer for plugin selection (50ms delay) - v2.2 stability fix
    pluginSelectionDebounceTimer_ = new QTimer(this);
    pluginSelectionDebounceTimer_->setSingleShot(true);
    pluginSelectionDebounceTimer_->setInterval(50);
    connect(pluginSelectionDebounceTimer_, &QTimer::timeout,
            this, &UnifiedPipelineView::processPendingPluginChange);
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

    // Connect sidebar's aboutToApply signal to cancel pending plugin changes
    connect(pluginSidebar_, &PluginSidebar::aboutToApplyParameters,
            this, &UnifiedPipelineView::cancelPendingPluginChange);

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

    // Config button
    txSourceConfigBtn_ = createConfigButton();
    connect(txSourceConfigBtn_, &QPushButton::clicked,
            this, &UnifiedPipelineView::onTXSourceConfigClicked);
    pipelineRow->addWidget(txSourceConfigBtn_);

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

    // Config button
    txProcessorConfigBtn_ = createConfigButton();
    connect(txProcessorConfigBtn_, &QPushButton::clicked,
            this, &UnifiedPipelineView::onTXProcessorConfigClicked);
    pipelineRow->addWidget(txProcessorConfigBtn_);

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

    // Config button
    txSinkConfigBtn_ = createConfigButton();
    connect(txSinkConfigBtn_, &QPushButton::clicked,
            this, &UnifiedPipelineView::onTXSinkConfigClicked);
    pipelineRow->addWidget(txSinkConfigBtn_);

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

    // v2.1: Diagnostics section
    QGroupBox *txDiagGroup = new QGroupBox("Diagnostics", this);
    txDiagGroup->setObjectName("diagnosticsGroup");
    QFormLayout *txDiagLayout = new QFormLayout(txDiagGroup);
    txDiagLayout->setSpacing(5);

    txHealthLabel_ = new QLabel("⚙️ Not running", this);
    txHealthLabel_->setObjectName("healthLabel");
    txHealthLabel_->setStyleSheet("font-weight: bold;");
    txDiagLayout->addRow("Health:", txHealthLabel_);

    txDriftLabel_ = new QLabel("—", this);
    txDriftLabel_->setObjectName("metricLabel");
    txDiagLayout->addRow("Drift:", txDriftLabel_);

    txReadFailsLabel_ = new QLabel("—", this);
    txReadFailsLabel_->setObjectName("metricLabel");
    txDiagLayout->addRow("Read Fails:", txReadFailsLabel_);

    txWriteFailsLabel_ = new QLabel("—", this);
    txWriteFailsLabel_->setObjectName("metricLabel");
    txDiagLayout->addRow("Write Fails:", txWriteFailsLabel_);

    cardLayout->addWidget(txDiagGroup);

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

    // Config button
    rxSourceConfigBtn_ = createConfigButton();
    connect(rxSourceConfigBtn_, &QPushButton::clicked,
            this, &UnifiedPipelineView::onRXSourceConfigClicked);
    pipelineRow->addWidget(rxSourceConfigBtn_);

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

    // Config button
    rxProcessorConfigBtn_ = createConfigButton();
    connect(rxProcessorConfigBtn_, &QPushButton::clicked,
            this, &UnifiedPipelineView::onRXProcessorConfigClicked);
    pipelineRow->addWidget(rxProcessorConfigBtn_);

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

    // Config button
    rxSinkConfigBtn_ = createConfigButton();
    connect(rxSinkConfigBtn_, &QPushButton::clicked,
            this, &UnifiedPipelineView::onRXSinkConfigClicked);
    pipelineRow->addWidget(rxSinkConfigBtn_);

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

    // v2.1: Diagnostics section
    QGroupBox *rxDiagGroup = new QGroupBox("Diagnostics", this);
    rxDiagGroup->setObjectName("diagnosticsGroup");
    QFormLayout *rxDiagLayout = new QFormLayout(rxDiagGroup);
    rxDiagLayout->setSpacing(5);

    rxHealthLabel_ = new QLabel("⚙️ Not running", this);
    rxHealthLabel_->setObjectName("healthLabel");
    rxHealthLabel_->setStyleSheet("font-weight: bold;");
    rxDiagLayout->addRow("Health:", rxHealthLabel_);

    rxDriftLabel_ = new QLabel("—", this);
    rxDriftLabel_->setObjectName("metricLabel");
    rxDiagLayout->addRow("Drift:", rxDriftLabel_);

    rxReadFailsLabel_ = new QLabel("—", this);
    rxReadFailsLabel_->setObjectName("metricLabel");
    rxDiagLayout->addRow("Read Fails:", rxReadFailsLabel_);

    rxWriteFailsLabel_ = new QLabel("—", this);
    rxWriteFailsLabel_->setObjectName("metricLabel");
    rxDiagLayout->addRow("Write Fails:", rxWriteFailsLabel_);

    cardLayout->addWidget(rxDiagGroup);

    layout->addWidget(rxCard);
}

void UnifiedPipelineView::createControlBar(QVBoxLayout* layout)
{
    QWidget *controlCard = createPipelineCard("", false);
    QHBoxLayout *controlLayout = new QHBoxLayout(controlCard);
    controlLayout->setSpacing(15);

    // v2.2: Radio Mode - one-click AIOC radio setup
    bridgeModeButton_ = new QPushButton("📻 Radio Mode", this);
    bridgeModeButton_->setObjectName("bridgeModeButton");
    bridgeModeButton_->setMinimumHeight(50);
    bridgeModeButton_->setMinimumWidth(160);
    bridgeModeButton_->setStyleSheet("font-weight: bold; background-color: #4CAF50; color: white;");
    bridgeModeButton_->setToolTip(
        "One-click AIOC radio setup:\n"
        "• TX: Windows Mic → AIOC PTT Sink\n"
        "• RX: AIOC Source → Windows Speaker\n"
        "• Processors disabled (passthrough)"
    );
    connect(bridgeModeButton_, &QPushButton::clicked, this, &UnifiedPipelineView::onBridgeModeClicked);
    controlLayout->addWidget(bridgeModeButton_);

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
    // v2.2: Removed dead settingsButton_ (was never connected)

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

// v2.2: Create cogwheel settings button
QPushButton* UnifiedPipelineView::createConfigButton()
{
    QPushButton *btn = new QPushButton("⚙", this);
    btn->setObjectName("configButton");
    btn->setMinimumHeight(35);
    btn->setMinimumWidth(35);
    btn->setMaximumWidth(35);
    btn->setToolTip("Configure plugin settings");
    btn->setEnabled(false);  // Disabled by default (no plugin selected)
    return btn;
}

void UnifiedPipelineView::cancelPendingPluginChange()
{
    // Stop debounce timer and clear pending change
    pluginSelectionDebounceTimer_->stop();
    pendingChange_ = PendingPluginChange::None;
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

    // v2.2: Reset button states when plugins reloaded
    updateConfigButtonStates();
}

// v2.2: Debounced signal handlers - check batch mode to bypass debouncing
void UnifiedPipelineView::onTXSourceChanged(int index)
{
    if (batchUpdateMode_) {
        // Batch mode: immediate processing, no debouncing
        processTXSourceChange(index);
    } else {
        // Normal mode: debounced processing
        pendingChange_ = PendingPluginChange::TXSource;
        pendingIndex_ = index;
        pluginSelectionDebounceTimer_->start();
    }
}

void UnifiedPipelineView::onTXProcessorChanged(int index)
{
    if (batchUpdateMode_) {
        processTXProcessorChange(index);
    } else {
        pendingChange_ = PendingPluginChange::TXProcessor;
        pendingIndex_ = index;
        pluginSelectionDebounceTimer_->start();
    }
}

void UnifiedPipelineView::onTXSinkChanged(int index)
{
    if (batchUpdateMode_) {
        processTXSinkChange(index);
    } else {
        pendingChange_ = PendingPluginChange::TXSink;
        pendingIndex_ = index;
        pluginSelectionDebounceTimer_->start();
    }
}

void UnifiedPipelineView::onRXSourceChanged(int index)
{
    if (batchUpdateMode_) {
        processRXSourceChange(index);
    } else {
        pendingChange_ = PendingPluginChange::RXSource;
        pendingIndex_ = index;
        pluginSelectionDebounceTimer_->start();
    }
}

void UnifiedPipelineView::onRXProcessorChanged(int index)
{
    if (batchUpdateMode_) {
        processRXProcessorChange(index);
    } else {
        pendingChange_ = PendingPluginChange::RXProcessor;
        pendingIndex_ = index;
        pluginSelectionDebounceTimer_->start();
    }
}

void UnifiedPipelineView::onRXSinkChanged(int index)
{
    if (batchUpdateMode_) {
        processRXSinkChange(index);
    } else {
        pendingChange_ = PendingPluginChange::RXSink;
        pendingIndex_ = index;
        pluginSelectionDebounceTimer_->start();
    }
}

// v2.2: Process debounced plugin change
void UnifiedPipelineView::processPendingPluginChange()
{
    int index = pendingIndex_;

    switch (pendingChange_) {
        case PendingPluginChange::TXSource:
            processTXSourceChange(index);
            break;
        case PendingPluginChange::TXProcessor:
            processTXProcessorChange(index);
            break;
        case PendingPluginChange::TXSink:
            processTXSinkChange(index);
            break;
        case PendingPluginChange::RXSource:
            processRXSourceChange(index);
            break;
        case PendingPluginChange::RXProcessor:
            processRXProcessorChange(index);
            break;
        case PendingPluginChange::RXSink:
            processRXSinkChange(index);
            break;
        default:
            break;
    }

    pendingChange_ = PendingPluginChange::None;
}

// v2.2: Actual plugin change handlers (original logic)
void UnifiedPipelineView::processTXSourceChange(int index)
{
    // Hide previous text plugin if any
    if (txSource_) {
        auto prevInfo = txSource_->getInfo();
        if (prevInfo.name.find("Text") != std::string::npos) {
            emit pluginSelected(QString::fromStdString(prevInfo.name), false);
        }
    }

    if (index <= 0 || !pluginManager_) {
        txSource_ = nullptr;
        updateTXStatus();
        updateConfigButtonStates();
        return;
    }

    std::string pluginName = txSourceCombo_->currentText().toStdString();
    txSource_ = pluginManager_->getAudioSourcePlugin(pluginName);

    // v2.2: Removed auto-open sidebar - now requires explicit cogwheel button click
    // if (txSource_) {
    //     pluginSidebar_->showPluginConfig(txSource_);
    // }

    // Show text plugin dock if selected
    if (txSource_) {
        auto info = txSource_->getInfo();
        std::cout << "[UnifiedPipelineView] TX Source selected: " << info.name
                  << " (ptr=" << txSource_.get() << ")" << std::endl;
        if (info.name.find("Text") != std::string::npos) {
            emit pluginSelected(QString::fromStdString(info.name), true);
        }
    }

    updateTXStatus();
    updateConfigButtonStates();
}

void UnifiedPipelineView::processTXProcessorChange(int index)
{
    if (index <= 0 || !pluginManager_) {
        txProcessor_ = nullptr;
        updateTXStatus();
        updateConfigButtonStates();
        return;
    }

    std::string pluginName = txProcessorCombo_->currentText().toStdString();
    txProcessor_ = pluginManager_->getAudioProcessorPlugin(pluginName);

    // v2.2: Removed auto-open sidebar - now requires explicit cogwheel button click
    // if (txProcessor_) {
    //     pluginSidebar_->showPluginConfig(txProcessor_);
    // }

    updateTXStatus();
    updateConfigButtonStates();
}

void UnifiedPipelineView::processTXSinkChange(int index)
{
    // Hide previous text plugin if any
    if (txSink_) {
        auto prevInfo = txSink_->getInfo();
        if (prevInfo.name.find("Text") != std::string::npos) {
            emit pluginSelected(QString::fromStdString(prevInfo.name), false);
        }
    }

    if (index <= 0 || !pluginManager_) {
        txSink_ = nullptr;
        updateTXStatus();
        updateConfigButtonStates();
        return;
    }

    std::string pluginName = txSinkCombo_->currentText().toStdString();
    txSink_ = pluginManager_->getAudioSinkPlugin(pluginName);

    // v2.2: Removed auto-open sidebar - now requires explicit cogwheel button click
    // if (txSink_) {
    //     pluginSidebar_->showPluginConfig(txSink_);
    // }

    // Show text plugin dock if selected
    if (txSink_) {
        auto info = txSink_->getInfo();
        if (info.name.find("Text") != std::string::npos) {
            emit pluginSelected(QString::fromStdString(info.name), true);
        }
    }

    updateTXStatus();
    updateConfigButtonStates();
}

void UnifiedPipelineView::processRXSourceChange(int index)
{
    // Hide previous text plugin if any
    if (rxSource_) {
        auto prevInfo = rxSource_->getInfo();
        if (prevInfo.name.find("Text") != std::string::npos) {
            emit pluginSelected(QString::fromStdString(prevInfo.name), false);
        }
    }

    if (index <= 0 || !pluginManager_) {
        rxSource_ = nullptr;
        updateRXStatus();
        updateConfigButtonStates();
        return;
    }

    std::string pluginName = rxSourceCombo_->currentText().toStdString();
    rxSource_ = pluginManager_->getAudioSourcePlugin(pluginName);

    // v2.2: Removed auto-open sidebar - now requires explicit cogwheel button click
    // if (rxSource_) {
    //     pluginSidebar_->showPluginConfig(rxSource_);
    // }

    // Show text plugin dock if selected
    if (rxSource_) {
        auto info = rxSource_->getInfo();
        if (info.name.find("Text") != std::string::npos) {
            emit pluginSelected(QString::fromStdString(info.name), true);
        }
    }

    updateRXStatus();
    updateConfigButtonStates();
}

void UnifiedPipelineView::processRXProcessorChange(int index)
{
    if (index <= 0 || !pluginManager_) {
        rxProcessor_ = nullptr;
        updateRXStatus();
        updateConfigButtonStates();
        return;
    }

    std::string pluginName = rxProcessorCombo_->currentText().toStdString();
    rxProcessor_ = pluginManager_->getAudioProcessorPlugin(pluginName);

    // v2.2: Removed auto-open sidebar - now requires explicit cogwheel button click
    // if (rxProcessor_) {
    //     pluginSidebar_->showPluginConfig(rxProcessor_);
    // }

    updateRXStatus();
    updateConfigButtonStates();
}

void UnifiedPipelineView::processRXSinkChange(int index)
{
    // Hide previous text plugin if any
    if (rxSink_) {
        auto prevInfo = rxSink_->getInfo();
        if (prevInfo.name.find("Text") != std::string::npos) {
            emit pluginSelected(QString::fromStdString(prevInfo.name), false);
        }
    }

    if (index <= 0 || !pluginManager_) {
        rxSink_ = nullptr;
        updateRXStatus();
        updateConfigButtonStates();
        return;
    }

    std::string pluginName = rxSinkCombo_->currentText().toStdString();
    rxSink_ = pluginManager_->getAudioSinkPlugin(pluginName);

    // v2.2: Removed auto-open sidebar - now requires explicit cogwheel button click
    // if (rxSink_) {
    //     pluginSidebar_->showPluginConfig(rxSink_);
    // }

    // Show text plugin dock if selected
    if (rxSink_) {
        auto info = rxSink_->getInfo();
        if (info.name.find("Text") != std::string::npos) {
            emit pluginSelected(QString::fromStdString(info.name), true);
        }
    }

    updateRXStatus();
    updateConfigButtonStates();
}

// v2.2: Cogwheel button handlers - ONLY way to open plugin settings
void UnifiedPipelineView::onTXSourceConfigClicked()
{
    if (txSource_) {
        pluginSidebar_->showPluginConfig(txSource_);
    }
}

void UnifiedPipelineView::onTXProcessorConfigClicked()
{
    if (txProcessor_) {
        pluginSidebar_->showPluginConfig(txProcessor_);
    }
}

void UnifiedPipelineView::onTXSinkConfigClicked()
{
    if (txSink_) {
        pluginSidebar_->showPluginConfig(txSink_);
    }
}

void UnifiedPipelineView::onRXSourceConfigClicked()
{
    if (rxSource_) {
        pluginSidebar_->showPluginConfig(rxSource_);
    }
}

void UnifiedPipelineView::onRXProcessorConfigClicked()
{
    if (rxProcessor_) {
        pluginSidebar_->showPluginConfig(rxProcessor_);
    }
}

void UnifiedPipelineView::onRXSinkConfigClicked()
{
    if (rxSink_) {
        pluginSidebar_->showPluginConfig(rxSink_);
    }
}

// v2.2: Update config button states based on plugin selection
void UnifiedPipelineView::updateConfigButtonStates()
{
    // Enable button only if valid plugin selected (not nullptr)
    txSourceConfigBtn_->setEnabled(txSource_ != nullptr);
    txProcessorConfigBtn_->setEnabled(txProcessor_ != nullptr);
    txSinkConfigBtn_->setEnabled(txSink_ != nullptr);
    rxSourceConfigBtn_->setEnabled(rxSource_ != nullptr);
    rxProcessorConfigBtn_->setEnabled(rxProcessor_ != nullptr);
    rxSinkConfigBtn_->setEnabled(rxSink_ != nullptr);
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

    // Update PTT button state (enabled only for AIOC sink)
    updatePTTButtonState();
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

    std::cout << "[UnifiedPipelineView] Starting TX pipeline..." << std::endl;
    std::cout << "[UnifiedPipelineView] TX Source: " << txSource_->getInfo().name
              << " (ptr=" << txSource_.get() << ")" << std::endl;
    if (txProcessor_) {
        std::cout << "[UnifiedPipelineView] TX Processor: " << txProcessor_->getInfo().name
                  << " (ptr=" << txProcessor_.get() << ")" << std::endl;
    }
    std::cout << "[UnifiedPipelineView] TX Sink: " << txSink_->getInfo().name
              << " (ptr=" << txSink_.get() << ")" << std::endl;

    txPipeline_->setSource(txSource_);
    if (txProcessor_) txPipeline_->setProcessor(txProcessor_);
    txPipeline_->setSink(txSink_);

    if (!txPipeline_->initialize()) {
        QMessageBox::critical(this, "TX Pipeline Error", "Failed to initialize TX pipeline");
        return;
    }

    std::cout << "[UnifiedPipelineView] Calling txPipeline_->start()..." << std::endl;
    if (!txPipeline_->start()) {
        QMessageBox::critical(this, "TX Pipeline Error", "Failed to start TX pipeline");
        return;
    }
    std::cout << "[UnifiedPipelineView] TX pipeline started successfully" << std::endl;

    startTXButton_->setEnabled(false);
    stopTXButton_->setEnabled(true);
    updateTXStatus();
    emit txPipelineStarted();
}

void UnifiedPipelineView::onStopTXClicked()
{
    if (!txPipeline_) return;

    txPipeline_->stop();

    // Force PTT release when stopping TX pipeline
    if (pttActive_) {
        onPTTReleased();
    }

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

void UnifiedPipelineView::onBridgeModeClicked()
{
    // v2.2: Radio Mode - one-click AIOC radio setup
    // Auto-configures: Windows Mic → AIOC Sink (TX), AIOC Source → Windows Speaker (RX)

    // Step 1: Stop running pipelines
    if (txPipeline_ && txPipeline_->isRunning()) {
        txPipeline_->stop();
        std::cout << "[UI] Stopped TX pipeline for Radio Mode" << std::endl;
    }
    if (rxPipeline_ && rxPipeline_->isRunning()) {
        rxPipeline_->stop();
        std::cout << "[UI] Stopped RX pipeline for Radio Mode" << std::endl;
    }

    // Step 2: Find AIOC radio devices
    int txSourceIdx = txSourceCombo_->findText("Windows Microphone (WASAPI)");
    int txSinkIdx = txSinkCombo_->findText("AIOC Sink");
    int rxSourceIdx = rxSourceCombo_->findText("AIOC Source");
    int rxSinkIdx = rxSinkCombo_->findText("Windows Speaker (WASAPI)");

    // Check if all required plugins are loaded
    QStringList missing;
    if (txSourceIdx < 0) missing << "Windows Microphone (WASAPI)";
    if (txSinkIdx < 0) missing << "AIOC Sink";
    if (rxSourceIdx < 0) missing << "AIOC Source";
    if (rxSinkIdx < 0) missing << "Windows Speaker (WASAPI)";

    if (!missing.isEmpty()) {
        QMessageBox::warning(this, "Radio Mode",
            "Cannot configure Radio Mode - missing plugins:\n\n• " +
            missing.join("\n• ") +
            "\n\nEnsure all plugins are loaded and AIOC device is connected.");
        return;
    }

    // Step 3: Block signals to avoid triggering handlers during batch update
    QSignalBlocker txSourceBlock(txSourceCombo_);
    QSignalBlocker txProcBlock(txProcessorCombo_);
    QSignalBlocker txSinkBlock(txSinkCombo_);
    QSignalBlocker rxSourceBlock(rxSourceCombo_);
    QSignalBlocker rxProcBlock(rxProcessorCombo_);
    QSignalBlocker rxSinkBlock(rxSinkCombo_);

    // Step 4: Apply golden path selections
    txSourceCombo_->setCurrentIndex(txSourceIdx);
    txProcessorCombo_->setCurrentIndex(0);  // None
    txSinkCombo_->setCurrentIndex(txSinkIdx);

    rxSourceCombo_->setCurrentIndex(rxSourceIdx);
    rxProcessorCombo_->setCurrentIndex(0);  // None
    rxSinkCombo_->setCurrentIndex(rxSinkIdx);

    // Step 5: Disable processor combos (passthrough only in Bridge Mode)
    txProcessorCombo_->setEnabled(false);
    rxProcessorCombo_->setEnabled(false);

    // Step 6: Enable batch update mode to bypass debouncing
    batchUpdateMode_ = true;

    // Step 7: Manually trigger selection handlers - now executes immediately
    onTXSourceChanged(txSourceIdx);
    onTXSinkChanged(txSinkIdx);
    onRXSourceChanged(rxSourceIdx);
    onRXSinkChanged(rxSinkIdx);

    // Step 8: Disable batch update mode
    batchUpdateMode_ = false;

    // Step 9: Apply timing preset to pipelines
    if (txPipeline_) txPipeline_->enableBridgeMode();
    if (rxPipeline_) rxPipeline_->enableBridgeMode();

    bridgeModeActive_ = true;

    // Step 10: Show confirmation
    QMessageBox::information(this, "Radio Mode",
        "Radio Mode configured!\n\n"
        "TX: Windows Mic → AIOC Radio\n"
        "RX: AIOC Radio → Windows Speaker\n\n"
        "Click 'Start Both' to begin.");

    std::cout << "[UI] Radio Mode: AIOC setup configured" << std::endl;
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

    // CORRECTED: Target TX Sink (AIOC Sink owns PTT control)
    if (txSink_ && isAIOCSink(txSink_)) {
        txSink_->setParameter("ptt_state", "true");
    }
}

void UnifiedPipelineView::onPTTReleased()
{
    pttActive_ = false;
    pttButton_->setProperty("active", false);
    pttButton_->style()->unpolish(pttButton_);
    pttButton_->style()->polish(pttButton_);

    // CORRECTED: Target TX Sink (AIOC Sink owns PTT control)
    if (txSink_ && isAIOCSink(txSink_)) {
        txSink_->setParameter("ptt_state", "false");
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

        // v2.1: Update diagnostics
        const double drift = txPipeline_->getCurrentDriftMs();
        const double maxDrift = txPipeline_->getMaxDriftMs();
        txDriftLabel_->setText(QString("%1 ms (max: %2 ms)")
            .arg(drift, 0, 'f', 1)
            .arg(maxDrift, 0, 'f', 1));

        txReadFailsLabel_->setText(QString::number(txPipeline_->getReadFailures()));
        txWriteFailsLabel_->setText(QString::number(txPipeline_->getWriteFailures()));

        // Update health indicator
        auto health = txPipeline_->getHealthStatus();
        switch (health) {
            case ProcessingPipeline::HealthStatus::OK:
                txHealthLabel_->setText("🟢 OK");
                txHealthLabel_->setStyleSheet("color: green; font-weight: bold;");
                break;
            case ProcessingPipeline::HealthStatus::Degraded:
                txHealthLabel_->setText("🟡 Degraded");
                txHealthLabel_->setStyleSheet("color: orange; font-weight: bold;");
                break;
            case ProcessingPipeline::HealthStatus::Failing:
                txHealthLabel_->setText("🔴 Failing");
                txHealthLabel_->setStyleSheet("color: red; font-weight: bold;");
                break;
        }
    } else {
        txLatencyLabel_->setText("--");
        txCPULabel_->setText("--");
        txInputMeterL_->setValue(0);
        txInputMeterR_->setValue(0);

        // v2.1: Reset diagnostics
        txHealthLabel_->setText("⚙️ Not running");
        txHealthLabel_->setStyleSheet("font-weight: bold;");
        txDriftLabel_->setText("—");
        txReadFailsLabel_->setText("—");
        txWriteFailsLabel_->setText("—");
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

        // v2.1: Update diagnostics
        const double drift = rxPipeline_->getCurrentDriftMs();
        const double maxDrift = rxPipeline_->getMaxDriftMs();
        rxDriftLabel_->setText(QString("%1 ms (max: %2 ms)")
            .arg(drift, 0, 'f', 1)
            .arg(maxDrift, 0, 'f', 1));

        rxReadFailsLabel_->setText(QString::number(rxPipeline_->getReadFailures()));
        rxWriteFailsLabel_->setText(QString::number(rxPipeline_->getWriteFailures()));

        // Update health indicator
        auto health = rxPipeline_->getHealthStatus();
        switch (health) {
            case ProcessingPipeline::HealthStatus::OK:
                rxHealthLabel_->setText("🟢 OK");
                rxHealthLabel_->setStyleSheet("color: green; font-weight: bold;");
                break;
            case ProcessingPipeline::HealthStatus::Degraded:
                rxHealthLabel_->setText("🟡 Degraded");
                rxHealthLabel_->setStyleSheet("color: orange; font-weight: bold;");
                break;
            case ProcessingPipeline::HealthStatus::Failing:
                rxHealthLabel_->setText("🔴 Failing");
                rxHealthLabel_->setStyleSheet("color: red; font-weight: bold;");
                break;
        }
    } else {
        rxLatencyLabel_->setText("--");
        rxCPULabel_->setText("--");
        rxOutputMeterL_->setValue(0);
        rxOutputMeterR_->setValue(0);

        // v2.1: Reset diagnostics
        rxHealthLabel_->setText("⚙️ Not running");
        rxHealthLabel_->setStyleSheet("font-weight: bold;");
        rxDriftLabel_->setText("—");
        rxReadFailsLabel_->setText("—");
        rxWriteFailsLabel_->setText("—");
    }
}

// v2.2: Removed dead onPluginFocused() method

// Keyboard event handlers for PTT (T and Space keys)
void UnifiedPipelineView::keyPressEvent(QKeyEvent* event)
{
    // CRITICAL: Prevent OS key repeat from triggering multiple PTT press/release cycles
    if (event->isAutoRepeat()) {
        event->accept();
        return;
    }

    // PTT keys: T and Space
    if ((event->key() == Qt::Key_T || event->key() == Qt::Key_Space)
        && pttButton_->isEnabled()) {
        onPTTPressed();
        event->accept();
        return;
    }

    event->ignore();  // Let parent handle unprocessed keys
}

void UnifiedPipelineView::keyReleaseEvent(QKeyEvent* event)
{
    if (event->isAutoRepeat()) {
        event->accept();
        return;
    }

    if ((event->key() == Qt::Key_T || event->key() == Qt::Key_Space)
        && pttActive_) {
        onPTTReleased();
        event->accept();
        return;
    }

    event->ignore();
}

void UnifiedPipelineView::focusOutEvent(QFocusEvent* event)
{
    // CRITICAL: Force PTT release if focus lost while PTT held
    // (User won't receive keyReleaseEvent if they Alt-Tab while holding PTT)
    if (pttActive_) {
        onPTTReleased();
    }
    QWidget::focusOutEvent(event);
}

// Helper: Detect if sink is AIOC-related
bool UnifiedPipelineView::isAIOCSink(std::shared_ptr<AudioSinkPlugin> sink) const
{
    if (!sink) return false;

    auto info = sink->getInfo();
    std::string name = info.name;

    // Case-insensitive check for "AIOC"
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);
    return name.find("aioc") != std::string::npos;
}

// Update PTT button state based on selected TX sink
void UnifiedPipelineView::updatePTTButtonState()
{
    bool isAIOC = isAIOCSink(txSink_);

    pttButton_->setEnabled(isAIOC);
    pttButton_->setToolTip(isAIOC ?
        "Push-to-Talk (Hold T or Space)" :
        "PTT only available with AIOC Sink");

    // Force release if disabled while active
    if (!isAIOC && pttActive_) {
        onPTTReleased();
    }
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

        #pttButton:disabled {
            background-color: #334155;
            color: #64748b;
            opacity: 0.5;
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

        /* v2.2: Cogwheel settings buttons */
        #configButton {
            background-color: #475569;
            color: #cbd5e1;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            font-weight: 600;
            padding: 0px;
            min-width: 35px;
            max-width: 35px;
            min-height: 35px;
        }

        #configButton:hover:enabled {
            background-color: #64748b;
        }

        #configButton:disabled {
            background-color: #334155;
            color: #64748b;
            opacity: 0.5;
        }
    )");
}

} // namespace nda
