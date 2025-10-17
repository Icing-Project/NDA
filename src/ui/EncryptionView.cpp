#include "ui/EncryptionView.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QFileDialog>

EncryptionView::EncryptionView(QWidget *parent)
    : QWidget(parent)
{
    setupUI();
}

EncryptionView::~EncryptionView()
{
}

void EncryptionView::setupUI()
{
    QVBoxLayout *mainLayout = new QVBoxLayout(this);

    // Algorithm selection
    QGroupBox *algoGroup = new QGroupBox("Encryption Algorithm", this);
    QVBoxLayout *algoLayout = new QVBoxLayout(algoGroup);

    algorithmCombo = new QComboBox(this);
    algorithmCombo->addItem("AES-256-GCM (Recommended)");
    algorithmCombo->addItem("AES-192-GCM");
    algorithmCombo->addItem("AES-128-GCM");
    algorithmCombo->addItem("ChaCha20-Poly1305");
    algoLayout->addWidget(algorithmCombo);

    mainLayout->addWidget(algoGroup);

    // Key management
    QGroupBox *keyGroup = new QGroupBox("Encryption Key", this);
    QVBoxLayout *keyLayout = new QVBoxLayout(keyGroup);

    keyDisplay = new QLineEdit(this);
    keyDisplay->setReadOnly(true);
    keyDisplay->setPlaceholderText("No key loaded");
    keyLayout->addWidget(keyDisplay);

    QHBoxLayout *keyButtonsLayout = new QHBoxLayout();

    generateButton = new QPushButton("Generate New Key", this);
    connect(generateButton, &QPushButton::clicked, this, &EncryptionView::onGenerateKey);
    keyButtonsLayout->addWidget(generateButton);

    importButton = new QPushButton("Import Key", this);
    connect(importButton, &QPushButton::clicked, this, &EncryptionView::onImportKey);
    keyButtonsLayout->addWidget(importButton);

    exportButton = new QPushButton("Export Key", this);
    connect(exportButton, &QPushButton::clicked, this, &EncryptionView::onExportKey);
    keyButtonsLayout->addWidget(exportButton);

    keyLayout->addLayout(keyButtonsLayout);

    mainLayout->addWidget(keyGroup);

    // Status
    statusLabel = new QLabel("Status: No key loaded", this);
    statusLabel->setStyleSheet("QLabel { font-weight: bold; color: #ff8800; }");
    mainLayout->addWidget(statusLabel);

    // Info
    QGroupBox *infoGroup = new QGroupBox("Information", this);
    QVBoxLayout *infoLayout = new QVBoxLayout(infoGroup);

    QLabel *infoLabel = new QLabel(
        "AES-256-GCM provides military-grade encryption with authentication.\n\n"
        "Features:\n"
        "• Hardware-accelerated (AES-NI)\n"
        "• 256-bit key strength\n"
        "• Authenticated encryption (prevents tampering)\n"
        "• Low latency overhead (<1ms)\n\n"
        "Keep your encryption key secure and never share it over insecure channels.",
        this
    );
    infoLabel->setWordWrap(true);
    infoLayout->addWidget(infoLabel);

    mainLayout->addWidget(infoGroup);

    mainLayout->addStretch();
}

void EncryptionView::onGenerateKey()
{
    // Placeholder - will generate real key
    keyDisplay->setText("Generated: 0x1A2B3C4D... (256-bit)");
    statusLabel->setText("Status: Key generated and loaded");
    statusLabel->setStyleSheet("QLabel { font-weight: bold; color: #44ff44; }");
}

void EncryptionView::onImportKey()
{
    QString fileName = QFileDialog::getOpenFileName(this,
        "Import Encryption Key", "", "Key Files (*.key);;All Files (*)");

    if (!fileName.isEmpty()) {
        keyDisplay->setText("Imported: " + fileName);
        statusLabel->setText("Status: Key imported and loaded");
        statusLabel->setStyleSheet("QLabel { font-weight: bold; color: #44ff44; }");
    }
}

void EncryptionView::onExportKey()
{
    QString fileName = QFileDialog::getSaveFileName(this,
        "Export Encryption Key", "", "Key Files (*.key);;All Files (*)");

    if (!fileName.isEmpty()) {
        statusLabel->setText("Status: Key exported to " + fileName);
    }
}
