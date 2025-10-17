#include "ui/PipelineView.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QFileDialog>
#include <QMessageBox>
#include <QStyle>
#include <QScrollArea>
#include <QDir>
#include <QDirIterator>
#include <QFileInfo>

PipelineView::PipelineView(QWidget *parent)
    : QWidget(parent)
{
    setupUI();
}

PipelineView::~PipelineView()
{
}

void PipelineView::setupUI()
{
    // Create scroll area
    QScrollArea *scrollArea = new QScrollArea(this);
    scrollArea->setWidgetResizable(true);
    scrollArea->setFrameShape(QFrame::NoFrame);
    scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    // Content widget
    QWidget *contentWidget = new QWidget();
    scrollArea->setWidget(contentWidget);

    QVBoxLayout *mainLayout = new QVBoxLayout(contentWidget);
    mainLayout->setSpacing(20);
    mainLayout->setContentsMargins(30, 30, 30, 30);

    // Set this widget's layout to contain the scroll area
    QVBoxLayout *thisLayout = new QVBoxLayout(this);
    thisLayout->setContentsMargins(0, 0, 0, 0);
    thisLayout->addWidget(scrollArea);

    // === HEADER ===
    QLabel *titleLabel = new QLabel("Configure Audio Pipeline", this);
    titleLabel->setObjectName("pageTitle");
    mainLayout->addWidget(titleLabel);

    QLabel *subtitleLabel = new QLabel("Build your audio processing chain by selecting plugins for each stage", this);
    subtitleLabel->setObjectName("pageSubtitle");
    mainLayout->addWidget(subtitleLabel);

    mainLayout->addSpacing(10);

    // === PIPELINE STEPS ===
    // Step 1: Audio Source
    QWidget *sourceCard = createStepCard();
    QVBoxLayout *sourceLayout = new QVBoxLayout(sourceCard);
    QLabel *sourceLabel = new QLabel("1. Audio Source", this);
    sourceLabel->setObjectName("stepLabel");
    audioSourceCombo = new QComboBox(this);
    audioSourceCombo->addItem("(None)");
    audioSourceCombo->setObjectName("pipelineCombo");
    audioSourceCombo->setMinimumHeight(40);
    connect(audioSourceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &PipelineView::onAudioSourceChanged);
    sourceLayout->addWidget(sourceLabel);
    sourceLayout->addWidget(audioSourceCombo);
    mainLayout->addWidget(sourceCard);

    // Step 2: Encryptor (Optional)
    QWidget *encryptorCard = createStepCard();
    QVBoxLayout *encryptorLayout = new QVBoxLayout(encryptorCard);
    QLabel *encryptorLabel = new QLabel("2. Encryptor (Optional)", this);
    encryptorLabel->setObjectName("stepLabel");
    encryptorCombo = new QComboBox(this);
    encryptorCombo->addItem("(None)");
    encryptorCombo->setObjectName("pipelineCombo");
    encryptorCombo->setMinimumHeight(40);
    connect(encryptorCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &PipelineView::onEncryptorChanged);
    encryptorLayout->addWidget(encryptorLabel);
    encryptorLayout->addWidget(encryptorCombo);
    mainLayout->addWidget(encryptorCard);

    // Step 3: Bearer (Optional)
    QWidget *bearerCard = createStepCard();
    QVBoxLayout *bearerLayout = new QVBoxLayout(bearerCard);
    QLabel *bearerLabel = new QLabel("3. Network Transport (Optional)", this);
    bearerLabel->setObjectName("stepLabel");
    bearerCombo = new QComboBox(this);
    bearerCombo->addItem("(None)");
    bearerCombo->setObjectName("pipelineCombo");
    bearerCombo->setMinimumHeight(40);
    connect(bearerCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &PipelineView::onBearerChanged);
    bearerLayout->addWidget(bearerLabel);
    bearerLayout->addWidget(bearerCombo);
    mainLayout->addWidget(bearerCard);

    // Step 4: Audio Sink
    QWidget *sinkCard = createStepCard();
    QVBoxLayout *sinkLayout = new QVBoxLayout(sinkCard);
    QLabel *sinkLabel = new QLabel("4. Audio Sink", this);
    sinkLabel->setObjectName("stepLabel");
    audioSinkCombo = new QComboBox(this);
    audioSinkCombo->addItem("(None)");
    audioSinkCombo->setObjectName("pipelineCombo");
    audioSinkCombo->setMinimumHeight(40);
    connect(audioSinkCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &PipelineView::onAudioSinkChanged);
    sinkLayout->addWidget(sinkLabel);
    sinkLayout->addWidget(audioSinkCombo);
    mainLayout->addWidget(sinkCard);

    // === STATUS CARD ===
    QWidget *statusCard = createStepCard();
    QVBoxLayout *statusLayout = new QVBoxLayout(statusCard);
    pipelineStatusLabel = new QLabel("⚙️ Configure source and sink to begin", this);
    pipelineStatusLabel->setObjectName("pipelineStatus");
    pipelineStatusLabel->setProperty("state", "notready");
    statusLayout->addWidget(pipelineStatusLabel);
    mainLayout->addWidget(statusCard);

    // === CONTROL BUTTONS ===
    QWidget *controlCard = createStepCard();
    QVBoxLayout *controlLayout = new QVBoxLayout(controlCard);

    // Plugin loading buttons in horizontal layout
    QHBoxLayout *pluginButtonsLayout = new QHBoxLayout();
    pluginButtonsLayout->setSpacing(15);

    loadPluginsButton = new QPushButton("📁  Load Plugins from Directory", this);
    loadPluginsButton->setObjectName("secondaryButton");
    loadPluginsButton->setMinimumHeight(50);
    loadPluginsButton->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    connect(loadPluginsButton, &QPushButton::clicked, this, &PipelineView::onLoadPluginsClicked);
    pluginButtonsLayout->addWidget(loadPluginsButton);

    autoLoadPythonButton = new QPushButton("🐍  Auto-Load Python Plugins", this);
    autoLoadPythonButton->setObjectName("secondaryButton");
    autoLoadPythonButton->setMinimumHeight(50);
    autoLoadPythonButton->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    connect(autoLoadPythonButton, &QPushButton::clicked, this, &PipelineView::onAutoLoadPythonPlugins);
    pluginButtonsLayout->addWidget(autoLoadPythonButton);

    controlLayout->addLayout(pluginButtonsLayout);
    controlLayout->addSpacing(10);

    QHBoxLayout *pipelineButtonsLayout = new QHBoxLayout();
    pipelineButtonsLayout->setSpacing(15);

    startPipelineButton = new QPushButton("▶  Start Pipeline", this);
    startPipelineButton->setObjectName("startButton");
    startPipelineButton->setMinimumHeight(55);
    startPipelineButton->setMinimumWidth(150);
    startPipelineButton->setEnabled(false);
    startPipelineButton->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    connect(startPipelineButton, &QPushButton::clicked, this, &PipelineView::onStartPipelineClicked);
    pipelineButtonsLayout->addWidget(startPipelineButton);

    stopPipelineButton = new QPushButton("■  Stop Pipeline", this);
    stopPipelineButton->setObjectName("stopButton");
    stopPipelineButton->setMinimumHeight(55);
    stopPipelineButton->setMinimumWidth(150);
    stopPipelineButton->setEnabled(false);
    stopPipelineButton->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    connect(stopPipelineButton, &QPushButton::clicked, this, &PipelineView::onStopPipelineClicked);
    pipelineButtonsLayout->addWidget(stopPipelineButton);

    controlLayout->addLayout(pipelineButtonsLayout);
    mainLayout->addWidget(controlCard);

    mainLayout->addStretch();

    applyModernStyles();
}

QWidget* PipelineView::createStepCard()
{
    QWidget *card = new QWidget(this);
    card->setObjectName("stepCard");
    return card;
}

void PipelineView::applyModernStyles()
{
    setStyleSheet(R"(
        #pageTitle {
            font-size: 28px;
            font-weight: bold;
            color: #ffffff;
            margin-bottom: 5px;
        }

        #pageSubtitle {
            font-size: 14px;
            color: #94a3b8;
            margin-bottom: 10px;
        }

        #stepCard {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                      stop:0 rgba(45, 45, 60, 0.9),
                                      stop:1 rgba(35, 35, 50, 0.9));
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 20px;
        }

        #stepLabel {
            font-size: 15px;
            font-weight: 600;
            color: #cbd5e1;
            margin-bottom: 8px;
        }

        #pipelineCombo {
            background-color: #1e293b;
            border: 1px solid #475569;
            border-radius: 8px;
            padding: 10px;
            color: #f1f5f9;
            font-size: 14px;
        }

        #pipelineCombo:hover {
            border: 1px solid #3b82f6;
            background-color: #334155;
        }

        #pipelineCombo:focus {
            border: 1px solid #3b82f6;
        }

        #pipelineCombo::drop-down {
            border: none;
            width: 30px;
        }

        #pipelineCombo::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 7px solid #94a3b8;
            margin-right: 10px;
        }

        #pipelineStatus {
            font-size: 16px;
            font-weight: 600;
            padding: 8px;
            border-radius: 6px;
        }

        #pipelineStatus[state="notready"] {
            color: #fb923c;
            background-color: rgba(251, 146, 60, 0.1);
        }

        #pipelineStatus[state="ready"] {
            color: #4ade80;
            background-color: rgba(74, 222, 128, 0.1);
        }

        #pipelineStatus[state="running"] {
            color: #3b82f6;
            background-color: rgba(59, 130, 246, 0.1);
        }

        #pipelineStatus[state="stopped"] {
            color: #ef4444;
            background-color: rgba(239, 68, 68, 0.1);
        }

        #secondaryButton {
            background-color: #475569;
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 15px;
            font-weight: 600;
            padding: 10px;
        }

        #secondaryButton:hover {
            background-color: #64748b;
        }

        #secondaryButton:pressed {
            background-color: #334155;
        }

        #startButton {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                      stop:0 #10b981,
                                      stop:1 #059669);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
        }

        #startButton:hover:enabled {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                      stop:0 #059669,
                                      stop:1 #047857);
        }

        #startButton:pressed {
            background: #065f46;
        }

        #startButton:disabled {
            background-color: #334155;
            color: #64748b;
        }

        #stopButton {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                      stop:0 #dc2626,
                                      stop:1 #b91c1c);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
        }

        #stopButton:hover:enabled {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                      stop:0 #b91c1c,
                                      stop:1 #991b1b);
        }

        #stopButton:pressed {
            background: #7f1d1d;
        }

        #stopButton:disabled {
            background-color: #334155;
            color: #64748b;
        }
    )");
}

void PipelineView::setPluginManager(std::shared_ptr<NADE::PluginManager> manager)
{
    pluginManager_ = manager;
    refreshPluginLists();
}

void PipelineView::setPipeline(std::shared_ptr<NADE::ProcessingPipeline> pipeline)
{
    pipeline_ = pipeline;
}

void PipelineView::onLoadPluginsClicked()
{
    QString directory = QFileDialog::getExistingDirectory(this,
        "Select Plugin Directory", "", QFileDialog::ShowDirsOnly);

    if (!directory.isEmpty() && pluginManager_) {
        auto pluginPaths = pluginManager_->scanPluginDirectory(directory.toStdString());

        int loadedCount = 0;
        for (const auto& path : pluginPaths) {
            if (pluginManager_->loadPlugin(path)) {
                loadedCount++;
            }
        }

        refreshPluginLists();

        QMessageBox::information(this, "Plugins Loaded",
            QString("Loaded %1 plugin(s) from directory").arg(loadedCount));
    }
}

void PipelineView::onAutoLoadPythonPlugins()
{
    if (!pluginManager_) {
        QMessageBox::warning(this, "Plugin Manager Not Available",
            "Plugin manager is not initialized");
        return;
    }

    // Define the plugins_py directory path
    QString pluginsDir = QDir::currentPath() + "/plugins_py";

    QDir dir(pluginsDir);
    if (!dir.exists()) {
        // Try parent directory
        pluginsDir = QDir::currentPath() + "/../plugins_py";
        dir.setPath(pluginsDir);

        if (!dir.exists()) {
            QMessageBox::warning(this, "Directory Not Found",
                "Could not find plugins_py directory at:\n" +
                QDir::currentPath() + "/plugins_py\nor\n" +
                QDir::currentPath() + "/../plugins_py");
            return;
        }
    }

    // Use plugin manager to load plugins from the directory
    auto pluginPaths = pluginManager_->scanPluginDirectory(pluginsDir.toStdString());

    int loadedCount = 0;
    int skippedCount = 0;
    QString loadedPlugins;

    for (const auto& path : pluginPaths) {
        // Extract filename from path
        QString fullPath = QString::fromStdString(path);
        QString fileName = QFileInfo(fullPath).fileName();

        // Skip base files and test files
        if (fileName == "base_plugin.py" || fileName == "__init__.py" ||
            fileName == "plugin_loader.py" || fileName == "test_plugins.py" ||
            fileName.startsWith("setup_")) {
            skippedCount++;
            continue;
        }

        if (pluginManager_->loadPlugin(path)) {
            QString displayName = fileName;
            displayName.replace(".py", "");
            displayName.replace("_", " ");

            // Capitalize first letter of each word
            QStringList words = displayName.split(" ");
            for (int i = 0; i < words.size(); ++i) {
                if (!words[i].isEmpty()) {
                    words[i][0] = words[i][0].toUpper();
                }
            }
            displayName = words.join(" ");

            loadedPlugins += "  • " + displayName + "\n";
            loadedCount++;
        }
    }

    // Refresh the plugin lists in the UI
    refreshPluginLists();

    // Show summary
    QString message;
    if (loadedCount > 0) {
        message = QString("Successfully loaded %1 Python plugin(s):\n\n%2")
                      .arg(loadedCount)
                      .arg(loadedPlugins);
    } else {
        message = "No new Python plugins found in:\n" + pluginsDir;
    }

    if (skippedCount > 0) {
        message += QString("\n\nSkipped %1 system/base file(s)").arg(skippedCount);
    }

    QMessageBox::information(this, "Auto-Load Complete", message);
}

void PipelineView::refreshPluginLists()
{
    if (!pluginManager_) return;

    // Clear existing items (keep "None")
    audioSourceCombo->clear();
    bearerCombo->clear();
    encryptorCombo->clear();
    audioSinkCombo->clear();

    audioSourceCombo->addItem("(None)");
    bearerCombo->addItem("(None)");
    encryptorCombo->addItem("(None)");
    audioSinkCombo->addItem("(None)");

    // Populate from plugin manager
    auto sources = pluginManager_->getPluginsByType(NADE::PluginType::AudioSource);
    for (const auto& plugin : sources) {
        audioSourceCombo->addItem(QString::fromStdString(plugin.info.name));
    }

    auto bearers = pluginManager_->getPluginsByType(NADE::PluginType::Bearer);
    for (const auto& plugin : bearers) {
        bearerCombo->addItem(QString::fromStdString(plugin.info.name));
    }

    auto encryptors = pluginManager_->getPluginsByType(NADE::PluginType::Encryptor);
    for (const auto& plugin : encryptors) {
        encryptorCombo->addItem(QString::fromStdString(plugin.info.name));
    }

    auto sinks = pluginManager_->getPluginsByType(NADE::PluginType::AudioSink);
    for (const auto& plugin : sinks) {
        audioSinkCombo->addItem(QString::fromStdString(plugin.info.name));
    }
}

void PipelineView::onAudioSourceChanged(int index)
{
    if (index <= 0 || !pluginManager_) {
        selectedSource_ = nullptr;
        return;
    }

    std::string pluginName = audioSourceCombo->currentText().toStdString();
    selectedSource_ = pluginManager_->getAudioSourcePlugin(pluginName);
    updatePipelineStatus();
}

void PipelineView::onBearerChanged(int index)
{
    if (index <= 0 || !pluginManager_) {
        selectedBearer_ = nullptr;
        return;
    }

    std::string pluginName = bearerCombo->currentText().toStdString();
    selectedBearer_ = pluginManager_->getBearerPlugin(pluginName);
    updatePipelineStatus();
}

void PipelineView::onEncryptorChanged(int index)
{
    if (index <= 0 || !pluginManager_) {
        selectedEncryptor_ = nullptr;
        return;
    }

    std::string pluginName = encryptorCombo->currentText().toStdString();
    selectedEncryptor_ = pluginManager_->getEncryptorPlugin(pluginName);
    updatePipelineStatus();
}

void PipelineView::onAudioSinkChanged(int index)
{
    if (index <= 0 || !pluginManager_) {
        selectedSink_ = nullptr;
        return;
    }

    std::string pluginName = audioSinkCombo->currentText().toStdString();
    selectedSink_ = pluginManager_->getAudioSinkPlugin(pluginName);
    updatePipelineStatus();
}

void PipelineView::updatePipelineStatus()
{
    bool canStart = selectedSource_ && selectedSink_;

    if (canStart) {
        pipelineStatusLabel->setText("✓ Ready to start");
        pipelineStatusLabel->setProperty("state", "ready");
        startPipelineButton->setEnabled(true);
    } else {
        pipelineStatusLabel->setText("⚙️ Configure source and sink to begin");
        pipelineStatusLabel->setProperty("state", "notready");
        startPipelineButton->setEnabled(false);
    }

    // Refresh styles
    pipelineStatusLabel->style()->unpolish(pipelineStatusLabel);
    pipelineStatusLabel->style()->polish(pipelineStatusLabel);
}

void PipelineView::onStartPipelineClicked()
{
    if (!pipeline_ || !selectedSource_ || !selectedSink_) {
        QMessageBox::warning(this, "Cannot Start",
            "Please configure audio source and sink first");
        return;
    }

    // Configure pipeline
    pipeline_->setAudioSource(selectedSource_);
    if (selectedBearer_) pipeline_->setBearer(selectedBearer_);
    if (selectedEncryptor_) pipeline_->setEncryptor(selectedEncryptor_);
    pipeline_->setAudioSink(selectedSink_);

    // Initialize and start
    if (!pipeline_->initialize()) {
        QMessageBox::critical(this, "Initialization Failed",
            "Failed to initialize pipeline");
        return;
    }

    if (!pipeline_->start()) {
        QMessageBox::critical(this, "Start Failed",
            "Failed to start pipeline");
        return;
    }

    pipelineStatusLabel->setText("🟢 Pipeline Running");
    pipelineStatusLabel->setProperty("state", "running");
    pipelineStatusLabel->style()->unpolish(pipelineStatusLabel);
    pipelineStatusLabel->style()->polish(pipelineStatusLabel);

    startPipelineButton->setEnabled(false);
    stopPipelineButton->setEnabled(true);

    emit pipelineStarted();
}

void PipelineView::onStopPipelineClicked()
{
    if (!pipeline_) return;

    pipeline_->stop();

    pipelineStatusLabel->setText("⚫ Pipeline Stopped");
    pipelineStatusLabel->setProperty("state", "stopped");
    pipelineStatusLabel->style()->unpolish(pipelineStatusLabel);
    pipelineStatusLabel->style()->polish(pipelineStatusLabel);

    startPipelineButton->setEnabled(true);
    stopPipelineButton->setEnabled(false);

    emit pipelineStopped();
}
