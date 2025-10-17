#include "ui/SettingsView.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QLabel>
#include <QMessageBox>
#include <QFrame>
#include <QScrollArea>

SettingsView::SettingsView(QWidget *parent)
    : QWidget(parent)
{
    setupUI();
    applyModernStyles();
    loadSettings();
}

SettingsView::~SettingsView()
{
}

void SettingsView::setupUI()
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
    QLabel *titleLabel = new QLabel("Settings", this);
    titleLabel->setObjectName("pageTitle");
    mainLayout->addWidget(titleLabel);

    QLabel *subtitleLabel = new QLabel("Configure application preferences and performance options", this);
    subtitleLabel->setObjectName("pageSubtitle");
    mainLayout->addWidget(subtitleLabel);

    mainLayout->addSpacing(10);

    // === GENERAL SETTINGS CARD ===
    QWidget *generalCard = createSettingsCard();
    QVBoxLayout *generalLayout = new QVBoxLayout(generalCard);

    QLabel *generalTitle = new QLabel("General", this);
    generalTitle->setObjectName("sectionTitle");
    generalLayout->addWidget(generalTitle);

    generalLayout->addSpacing(5);

    autoStartCheck = new QCheckBox("Start with Windows", this);
    autoStartCheck->setObjectName("settingsCheckbox");
    generalLayout->addWidget(autoStartCheck);

    minimizeToTrayCheck = new QCheckBox("Minimize to system tray", this);
    minimizeToTrayCheck->setObjectName("settingsCheckbox");
    generalLayout->addWidget(minimizeToTrayCheck);

    mainLayout->addWidget(generalCard);

    // === PERFORMANCE SETTINGS CARD ===
    QWidget *perfCard = createSettingsCard();
    QVBoxLayout *perfLayout = new QVBoxLayout(perfCard);

    QLabel *perfTitle = new QLabel("Performance", this);
    perfTitle->setObjectName("sectionTitle");
    perfLayout->addWidget(perfTitle);

    perfLayout->addSpacing(5);

    hardwareAccelCheck = new QCheckBox("Enable hardware acceleration (AES-NI)", this);
    hardwareAccelCheck->setObjectName("settingsCheckbox");
    hardwareAccelCheck->setChecked(true);
    perfLayout->addWidget(hardwareAccelCheck);

    perfLayout->addSpacing(10);

    QHBoxLayout *latencyLayout = new QHBoxLayout();
    QLabel *latencyLabel = new QLabel("Target Latency (ms):", this);
    latencyLabel->setObjectName("settingLabel");
    maxLatencySpin = new QSpinBox(this);
    maxLatencySpin->setObjectName("settingSpin");
    maxLatencySpin->setRange(1, 50);
    maxLatencySpin->setValue(5);
    maxLatencySpin->setMinimumWidth(100);
    latencyLayout->addWidget(latencyLabel);
    latencyLayout->addWidget(maxLatencySpin);
    latencyLayout->addStretch();
    perfLayout->addLayout(latencyLayout);

    QHBoxLayout *priorityLayout = new QHBoxLayout();
    QLabel *priorityLabel = new QLabel("CPU Priority Level:", this);
    priorityLabel->setObjectName("settingLabel");
    cpuPrioritySpin = new QSpinBox(this);
    cpuPrioritySpin->setObjectName("settingSpin");
    cpuPrioritySpin->setRange(0, 10);
    cpuPrioritySpin->setValue(8);
    cpuPrioritySpin->setMinimumWidth(100);
    priorityLayout->addWidget(priorityLabel);
    priorityLayout->addWidget(cpuPrioritySpin);
    priorityLayout->addStretch();
    perfLayout->addLayout(priorityLayout);

    mainLayout->addWidget(perfCard);

    // === SYSTEM INFO CARD ===
    QWidget *infoCard = createSettingsCard();
    QVBoxLayout *infoLayout = new QVBoxLayout(infoCard);

    QLabel *infoTitle = new QLabel("System Information", this);
    infoTitle->setObjectName("sectionTitle");
    infoLayout->addWidget(infoTitle);

    infoLayout->addSpacing(5);

    QLabel *versionLabel = new QLabel("Version: 1.0.0", this);
    versionLabel->setObjectName("infoLabel");
    infoLayout->addWidget(versionLabel);

    QLabel *platformLabel = new QLabel("Platform: Linux / Windows 10/11", this);
    platformLabel->setObjectName("infoLabel");
    infoLayout->addWidget(platformLabel);

    QLabel *qtLabel = new QLabel("Qt Version: 6.x", this);
    qtLabel->setObjectName("infoLabel");
    infoLayout->addWidget(qtLabel);

    QLabel *opensslLabel = new QLabel("OpenSSL: 3.x", this);
    opensslLabel->setObjectName("infoLabel");
    infoLayout->addWidget(opensslLabel);

    mainLayout->addWidget(infoCard);

    // === BUTTONS ===
    QWidget *buttonCard = createSettingsCard();
    QHBoxLayout *buttonLayout = new QHBoxLayout(buttonCard);

    saveButton = new QPushButton("💾  Save Settings", this);
    saveButton->setObjectName("primaryButton");
    saveButton->setMinimumHeight(50);
    saveButton->setMinimumWidth(200);
    saveButton->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    connect(saveButton, &QPushButton::clicked, this, &SettingsView::onSaveSettings);
    buttonLayout->addWidget(saveButton);

    resetButton = new QPushButton("↻  Reset to Defaults", this);
    resetButton->setObjectName("secondaryButton");
    resetButton->setMinimumHeight(50);
    resetButton->setMinimumWidth(200);
    resetButton->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    connect(resetButton, &QPushButton::clicked, this, &SettingsView::onResetSettings);
    buttonLayout->addWidget(resetButton);

    buttonLayout->addStretch();

    mainLayout->addWidget(buttonCard);

    mainLayout->addStretch();
}

QWidget* SettingsView::createSettingsCard()
{
    QWidget *card = new QWidget(this);
    card->setObjectName("settingsCard");
    return card;
}

void SettingsView::applyModernStyles()
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

        #settingsCard {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                      stop:0 rgba(45, 45, 60, 0.9),
                                      stop:1 rgba(35, 35, 50, 0.9));
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 24px;
        }

        #sectionTitle {
            font-size: 18px;
            font-weight: 600;
            color: #e2e8f0;
            margin-bottom: 10px;
        }

        #settingsCheckbox {
            font-size: 14px;
            color: #cbd5e1;
            padding: 6px;
        }

        #settingsCheckbox::indicator {
            width: 20px;
            height: 20px;
            border-radius: 4px;
            border: 2px solid #475569;
            background-color: #1e293b;
        }

        #settingsCheckbox::indicator:hover {
            border-color: #3b82f6;
        }

        #settingsCheckbox::indicator:checked {
            background-color: #3b82f6;
            border-color: #3b82f6;
            image: none;
        }

        #settingLabel {
            font-size: 14px;
            color: #cbd5e1;
        }

        #settingSpin {
            background-color: #1e293b;
            border: 1px solid #475569;
            border-radius: 6px;
            padding: 8px;
            color: #f1f5f9;
            font-size: 14px;
        }

        #settingSpin:hover {
            border-color: #3b82f6;
        }

        #settingSpin:focus {
            border-color: #3b82f6;
        }

        #infoLabel {
            font-size: 14px;
            color: #94a3b8;
            padding: 2px 0;
        }

        #primaryButton {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                      stop:0 #3b82f6,
                                      stop:1 #2563eb);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 15px;
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

        #secondaryButton {
            background-color: #475569;
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 15px;
            font-weight: 600;
            padding: 8px;
        }

        #secondaryButton:hover {
            background-color: #64748b;
        }

        #secondaryButton:pressed {
            background-color: #334155;
        }
    )");
}

void SettingsView::loadSettings()
{
    // Placeholder - will load from config file
    autoStartCheck->setChecked(false);
    minimizeToTrayCheck->setChecked(true);
    hardwareAccelCheck->setChecked(true);
    maxLatencySpin->setValue(5);
    cpuPrioritySpin->setValue(8);
}

void SettingsView::onSaveSettings()
{
    // Placeholder - will save to config file
    QMessageBox::information(this, "Settings", "Settings saved successfully!");
}

void SettingsView::onResetSettings()
{
    autoStartCheck->setChecked(false);
    minimizeToTrayCheck->setChecked(true);
    hardwareAccelCheck->setChecked(true);
    maxLatencySpin->setValue(5);
    cpuPrioritySpin->setValue(8);

    QMessageBox::information(this, "Settings", "Settings reset to defaults!");
}
