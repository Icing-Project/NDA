#include "ui/Dashboard.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QTimer>
#include <QFrame>
#include <QGridLayout>
#include <QStyle>
#include <QScrollArea>
#include <random>

Dashboard::Dashboard(QWidget *parent)
    : QWidget(parent), isStreaming(false)
{
    setupUI();
    applyModernStyles();
}

Dashboard::~Dashboard()
{
}

void Dashboard::setPipeline(std::shared_ptr<nda::ProcessingPipeline> pipeline)
{
    pipeline_ = pipeline;
}

void Dashboard::updatePipelineState()
{
    if (!pipeline_) return;

    bool running = pipeline_->isRunning();

    if (running) {
        startStopButton->setText("■  Stop Pipeline");
        startStopButton->setEnabled(true);
        startStopButton->setProperty("running", true);
        statusLabel->setText("🟢 Running");
        statusLabel->setProperty("status", "running");
        isStreaming = true;
    } else {
        startStopButton->setText("■  Stop Pipeline");
        startStopButton->setEnabled(false);
        startStopButton->setProperty("running", false);
        statusLabel->setText("⚫ Stopped");
        statusLabel->setProperty("status", "stopped");
        isStreaming = false;
    }

    // Refresh styles after property change
    startStopButton->style()->unpolish(startStopButton);
    startStopButton->style()->polish(startStopButton);
    statusLabel->style()->unpolish(statusLabel);
    statusLabel->style()->polish(statusLabel);
}

void Dashboard::setupUI()
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

    // === HEADER SECTION ===
    QWidget *headerCard = createCard();
    QVBoxLayout *headerLayout = new QVBoxLayout(headerCard);

    QLabel *titleLabel = new QLabel("Audio Pipeline Monitor", this);
    titleLabel->setObjectName("dashboardTitle");
    headerLayout->addWidget(titleLabel);

    statusLabel = new QLabel("⚫ Stopped", this);
    statusLabel->setObjectName("statusLabel");
    statusLabel->setProperty("status", "stopped");
    headerLayout->addWidget(statusLabel);

    mainLayout->addWidget(headerCard);

    // === CONTROL SECTION ===
    QWidget *controlCard = createCard();
    QVBoxLayout *controlLayout = new QVBoxLayout(controlCard);

    startStopButton = new QPushButton("▶  Start Pipeline", this);
    startStopButton->setObjectName("primaryButton");
    startStopButton->setMinimumHeight(60);
    connect(startStopButton, &QPushButton::clicked, this, &Dashboard::onStartStopClicked);
    controlLayout->addWidget(startStopButton);

    mainLayout->addWidget(controlCard);

    // === METRICS CARDS ROW ===
    QGridLayout *metricsGrid = new QGridLayout();
    metricsGrid->setSpacing(20);

    // Latency Card
    QWidget *latencyCard = createMetricCard();
    QVBoxLayout *latencyLayout = new QVBoxLayout(latencyCard);
    QLabel *latencyTitle = new QLabel("LATENCY", this);
    latencyTitle->setObjectName("metricTitle");
    latencyLabel = new QLabel("--", this);
    latencyLabel->setObjectName("metricValue");
    QLabel *latencyUnit = new QLabel("milliseconds", this);
    latencyUnit->setObjectName("metricUnit");
    latencyLayout->addWidget(latencyTitle);
    latencyLayout->addWidget(latencyLabel, 0, Qt::AlignCenter);
    latencyLayout->addWidget(latencyUnit);
    latencyCard->setMinimumWidth(150);
    metricsGrid->addWidget(latencyCard, 0, 0);

    // CPU Card
    QWidget *cpuCard = createMetricCard();
    QVBoxLayout *cpuLayout = new QVBoxLayout(cpuCard);
    QLabel *cpuTitle = new QLabel("CPU USAGE", this);
    cpuTitle->setObjectName("metricTitle");
    cpuLabel = new QLabel("--", this);
    cpuLabel->setObjectName("metricValue");
    QLabel *cpuUnit = new QLabel("percent", this);
    cpuUnit->setObjectName("metricUnit");
    cpuLayout->addWidget(cpuTitle);
    cpuLayout->addWidget(cpuLabel, 0, Qt::AlignCenter);
    cpuLayout->addWidget(cpuUnit);
    cpuCard->setMinimumWidth(150);
    metricsGrid->addWidget(cpuCard, 0, 1);

    // Memory Card
    QWidget *memoryCard = createMetricCard();
    QVBoxLayout *memoryLayout = new QVBoxLayout(memoryCard);
    QLabel *memoryTitle = new QLabel("MEMORY", this);
    memoryTitle->setObjectName("metricTitle");
    memoryLabel = new QLabel("--", this);
    memoryLabel->setObjectName("metricValue");
    QLabel *memoryUnit = new QLabel("megabytes", this);
    memoryUnit->setObjectName("metricUnit");
    memoryLayout->addWidget(memoryTitle);
    memoryLayout->addWidget(memoryLabel, 0, Qt::AlignCenter);
    memoryLayout->addWidget(memoryUnit);
    memoryCard->setMinimumWidth(150);
    metricsGrid->addWidget(memoryCard, 0, 2);

    // Make grid responsive - set column stretch
    metricsGrid->setColumnStretch(0, 1);
    metricsGrid->setColumnStretch(1, 1);
    metricsGrid->setColumnStretch(2, 1);

    mainLayout->addLayout(metricsGrid);

    // === AUDIO LEVELS CARD ===
    QWidget *audioCard = createCard();
    QVBoxLayout *audioLayout = new QVBoxLayout(audioCard);

    QLabel *audioTitle = new QLabel("Audio Levels", this);
    audioTitle->setObjectName("sectionTitle");
    audioLayout->addWidget(audioTitle);

    audioLayout->addSpacing(10);

    // Input meters
    QLabel *inputLabel = new QLabel("INPUT", this);
    inputLabel->setObjectName("meterLabel");
    audioLayout->addWidget(inputLabel);

    inputMeterL = createModernProgressBar();
    inputMeterL->setObjectName("audioMeterL");
    audioLayout->addWidget(inputMeterL);

    inputMeterR = createModernProgressBar();
    inputMeterR->setObjectName("audioMeterR");
    audioLayout->addWidget(inputMeterR);

    audioLayout->addSpacing(15);

    // Output meters
    QLabel *outputLabel = new QLabel("OUTPUT", this);
    outputLabel->setObjectName("meterLabel");
    audioLayout->addWidget(outputLabel);

    outputMeterL = createModernProgressBar();
    outputMeterL->setObjectName("audioMeterL");
    audioLayout->addWidget(outputMeterL);

    outputMeterR = createModernProgressBar();
    outputMeterR->setObjectName("audioMeterR");
    audioLayout->addWidget(outputMeterR);

    mainLayout->addWidget(audioCard);

    mainLayout->addStretch();

    // Timer for updating meters (60 FPS for smooth animation)
    QTimer *meterTimer = new QTimer(this);
    connect(meterTimer, &QTimer::timeout, this, &Dashboard::updateAudioMeters);
    meterTimer->start(16); // ~60 FPS
}

QWidget* Dashboard::createCard()
{
    QWidget *card = new QWidget(this);
    card->setObjectName("card");
    return card;
}

QWidget* Dashboard::createMetricCard()
{
    QWidget *card = new QWidget(this);
    card->setObjectName("metricCard");
    return card;
}

QProgressBar* Dashboard::createModernProgressBar()
{
    QProgressBar *bar = new QProgressBar(this);
    bar->setRange(0, 100);
    bar->setTextVisible(false);
    bar->setMinimumHeight(12);
    bar->setMaximumHeight(12);
    return bar;
}

void Dashboard::onStartStopClicked()
{
    if (!pipeline_) return;

    // Dashboard can only stop the pipeline (start happens in PipelineView)
    if (pipeline_->isRunning()) {
        pipeline_->stop();

        startStopButton->setText("■  Stop Pipeline");
        startStopButton->setEnabled(false);
        startStopButton->setProperty("running", false);
        statusLabel->setText("⚫ Stopped");
        statusLabel->setProperty("status", "stopped");
        emit streamStopped();

        // Refresh styles after property change
        startStopButton->style()->unpolish(startStopButton);
        startStopButton->style()->polish(startStopButton);
        statusLabel->style()->unpolish(statusLabel);
        statusLabel->style()->polish(statusLabel);
    }
}

void Dashboard::updateAudioMeters()
{
    if (pipeline_ && pipeline_->isRunning()) {
        // Get real metrics from pipeline
        double latency = pipeline_->getLatency();
        float cpuLoad = pipeline_->getCPULoad();
        uint64_t samples = pipeline_->getProcessedSamples();

        // Update metric labels with real data
        latencyLabel->setText(QString::number(latency, 'f', 1));
        cpuLabel->setText(QString::number(static_cast<int>(cpuLoad)));

        // Estimate memory usage based on processed samples (rough calculation)
        // Assuming 32-bit float stereo at 48kHz = 384KB/s
        uint64_t memoryMB = (samples * sizeof(float) * 2) / (1024 * 1024);
        memoryLabel->setText(QString::number(memoryMB));

        // Audio meters - simulate for now (TODO: add real peak metering to pipeline)
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_int_distribution<> inputDis(35, 75);
        static std::uniform_int_distribution<> outputDis(30, 70);

        int inputL = inputDis(gen);
        int inputR = inputDis(gen);
        int outputL = outputDis(gen);
        int outputR = outputDis(gen);

        inputMeterL->setValue(inputL);
        inputMeterR->setValue(inputR);
        outputMeterL->setValue(outputL);
        outputMeterR->setValue(outputR);

        // Update meter colors based on levels
        updateMeterColor(inputMeterL, inputL);
        updateMeterColor(inputMeterR, inputR);
        updateMeterColor(outputMeterL, outputL);
        updateMeterColor(outputMeterR, outputR);
    } else {
        // Pipeline stopped - reset all displays
        inputMeterL->setValue(0);
        inputMeterR->setValue(0);
        outputMeterL->setValue(0);
        outputMeterR->setValue(0);

        latencyLabel->setText("--");
        cpuLabel->setText("--");
        memoryLabel->setText("--");
    }
}

void Dashboard::updateMeterColor(QProgressBar *meter, int value)
{
    QString color;
    if (value < 60) {
        color = "#4ade80"; // Green
    } else if (value < 80) {
        color = "#fbbf24"; // Yellow/Orange
    } else {
        color = "#ef4444"; // Red
    }

    meter->setStyleSheet(QString(
        "QProgressBar {"
        "    border: none;"
        "    border-radius: 6px;"
        "    background-color: rgba(255, 255, 255, 0.05);"
        "    text-align: center;"
        "}"
        "QProgressBar::chunk {"
        "    background-color: %1;"
        "    border-radius: 6px;"
        "}"
    ).arg(color));
}

void Dashboard::applyModernStyles()
{
    setStyleSheet(R"(
        /* Dashboard Title */
        #dashboardTitle {
            font-size: 28px;
            font-weight: bold;
            color: #ffffff;
            margin-bottom: 5px;
        }

        /* Status Label */
        #statusLabel {
            font-size: 18px;
            font-weight: 600;
            color: #94a3b8;
        }

        #statusLabel[status="running"] {
            color: #4ade80;
        }

        #statusLabel[status="stopped"] {
            color: #94a3b8;
        }

        /* Card styling */
        #card {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                      stop:0 rgba(45, 45, 60, 0.9),
                                      stop:1 rgba(35, 35, 50, 0.9));
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 20px;
        }

        /* Metric card styling */
        #metricCard {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                      stop:0 rgba(55, 55, 70, 0.9),
                                      stop:1 rgba(45, 45, 60, 0.9));
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 20px;
            min-width: 140px;
        }

        #metricTitle {
            font-size: 11px;
            font-weight: 700;
            color: #94a3b8;
            letter-spacing: 1px;
        }

        #metricValue {
            font-size: 48px;
            font-weight: bold;
            color: #42a5f5;
            margin: 10px 0;
        }

        #metricUnit {
            font-size: 12px;
            color: #64748b;
        }

        /* Section titles */
        #sectionTitle {
            font-size: 18px;
            font-weight: 600;
            color: #e2e8f0;
            margin-bottom: 10px;
        }

        /* Meter labels */
        #meterLabel {
            font-size: 11px;
            font-weight: 700;
            color: #94a3b8;
            letter-spacing: 1px;
            margin-bottom: 5px;
        }

        /* Primary button */
        #primaryButton {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                      stop:0 #3b82f6,
                                      stop:1 #2563eb);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 18px;
            font-weight: 600;
            padding: 8px;
        }

        #primaryButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                      stop:0 #2563eb,
                                      stop:1 #1d4ed8);
        }

        #primaryButton:pressed {
            background: #1e40af;
        }

        #primaryButton[running="true"] {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                      stop:0 #dc2626,
                                      stop:1 #b91c1c);
        }

        #primaryButton[running="true"]:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                      stop:0 #b91c1c,
                                      stop:1 #991b1b);
        }

        /* Progress bars - default green */
        QProgressBar {
            border: none;
            border-radius: 6px;
            background-color: rgba(255, 255, 255, 0.05);
            text-align: center;
        }

        QProgressBar::chunk {
            background-color: #4ade80;
            border-radius: 6px;
        }
    )");
}
