#include "ui/MainWindow.h"
#include "ui/UnifiedPipelineView.h"
#include "ui/ImportKeysDialog.h"
#include "ui/ExportKeysDialog.h"
#include "plugins/PluginPaths.h"
#include "crypto/CryptoManager.h"
#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QClipboard>
#include <QCoreApplication>
#include <QDebug>
#include <QDialog>
#include <QDir>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QPushButton>
#include <QStatusBar>
#include <QVBoxLayout>
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

    // Crypto menu
    QMenu *cryptoMenu = menuBar()->addMenu("&Crypto");

    QAction *generateAESAction = new QAction("Generate AES-256 Key", this);
    generateAESAction->setShortcut(QKeySequence("Ctrl+G"));
    connect(generateAESAction, &QAction::triggered, this, &MainWindow::onGenerateAESKey);
    cryptoMenu->addAction(generateAESAction);

    QAction *generateX25519Action = new QAction("Generate X25519 Key Pair", this);
    connect(generateX25519Action, &QAction::triggered, this, &MainWindow::onGenerateX25519KeyPair);
    cryptoMenu->addAction(generateX25519Action);

    cryptoMenu->addSeparator();

    QAction *importKeysAction = new QAction("Import Keys...", this);
    importKeysAction->setShortcut(QKeySequence("Ctrl+I"));
    connect(importKeysAction, &QAction::triggered, this, &MainWindow::onImportKeys);
    cryptoMenu->addAction(importKeysAction);

    QAction *exportKeysAction = new QAction("Export Keys...", this);
    exportKeysAction->setShortcut(QKeySequence("Ctrl+E"));
    connect(exportKeysAction, &QAction::triggered, this, &MainWindow::onExportKeys);
    cryptoMenu->addAction(exportKeysAction);

    cryptoMenu->addSeparator();

    QAction *deriveSharedKeyAction = new QAction("Derive Shared Key (ECDH)", this);
    connect(deriveSharedKeyAction, &QAction::triggered, this, &MainWindow::onDeriveSharedKey);
    cryptoMenu->addAction(deriveSharedKeyAction);

    cryptoMenu->addSeparator();

    QAction *clearKeysAction = new QAction("Clear All Keys", this);
    connect(clearKeysAction, &QAction::triggered, this, &MainWindow::onClearKeys);
    cryptoMenu->addAction(clearKeysAction);

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

// v2.2: Removed dead onStatusUpdate slot

void MainWindow::autoLoadPlugins()
{
    if (!pluginManager_) return;

#ifndef NDA_ENABLE_PYTHON
    qDebug() << "[MainWindow] Python plugin support disabled in this build";
#endif

    int loadedCount = 0;

#ifdef NDA_ENABLE_PYTHON
    // Load Python plugins using centralized path resolution
    auto pythonPaths = nda::PluginPaths::getPythonPluginSearchPaths();
    for (const auto& dir : pythonPaths) {
        if (QDir(dir).exists()) {
            auto pluginFiles = pluginManager_->scanPluginDirectory(dir.toStdString());
            for (const auto& path : pluginFiles) {
                if (pluginManager_->loadPlugin(path)) {
                    loadedCount++;
                }
            }
            if (!pluginFiles.empty()) {
                qDebug() << "[MainWindow] Found" << pluginFiles.size()
                         << "Python plugins in:" << dir;
                break;  // Stop after first successful directory
            }
        }
    }
#endif

    // Load C++ plugins using centralized path resolution
    auto cppPaths = nda::PluginPaths::getCppPluginSearchPaths();
    for (const auto& dir : cppPaths) {
        if (QDir(dir).exists()) {
            auto pluginFiles = pluginManager_->scanPluginDirectory(dir.toStdString());
            for (const auto& path : pluginFiles) {
                if (pluginManager_->loadPlugin(path)) {
                    loadedCount++;
                }
            }
            if (!pluginFiles.empty()) {
                qDebug() << "[MainWindow] Found" << pluginFiles.size()
                         << "C++ plugins in:" << dir;
                break;  // Stop after first successful directory
            }
        }
    }

    qDebug() << "[MainWindow] Auto-loaded" << loadedCount << "total plugins";

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

// ========================
// Crypto Menu Slot Implementations
// ========================

void MainWindow::onGenerateAESKey()
{
    std::string hexKey = nda::CryptoManager::instance().generateAES256Key();
    if (hexKey.empty()) {
        QMessageBox::critical(this, "Key Generation Error",
            "Failed to generate AES-256 key:\n" +
            QString::fromStdString(nda::CryptoManager::instance().getLastError()));
        return;
    }

    // Create custom dialog
    QDialog dialog(this);
    dialog.setWindowTitle("AES-256 Key Generated");
    dialog.setMinimumWidth(600);

    QVBoxLayout* layout = new QVBoxLayout(&dialog);

    QLabel* infoLabel = new QLabel(
        "Store this key securely. Share it with the decryption side via secure channel.\n"
        "This key can be used with AES-256-GCM encryptor/decryptor plugins.");
    infoLabel->setWordWrap(true);
    layout->addWidget(infoLabel);

    // Key display (read-only, monospace)
    QLineEdit* keyEdit = new QLineEdit(QString::fromStdString(hexKey));
    keyEdit->setReadOnly(true);
    keyEdit->setFont(QFont("Courier New", 10));
    keyEdit->selectAll();
    layout->addWidget(keyEdit);

    // Buttons
    QHBoxLayout* buttonLayout = new QHBoxLayout();

    QPushButton* copyBtn = new QPushButton("Copy to Clipboard");
    connect(copyBtn, &QPushButton::clicked, [hexKey]() {
        QClipboard* clipboard = QApplication::clipboard();
        clipboard->setText(QString::fromStdString(hexKey));
    });
    buttonLayout->addWidget(copyBtn);

    QPushButton* applyBtn = new QPushButton("Apply to Selected Plugin");
    connect(applyBtn, &QPushButton::clicked, [this, hexKey, &dialog]() {
        if (applyKeyToSelectedPlugin("aes_256_key", hexKey)) {
            QMessageBox::information(this, "Success", "Key applied to plugin(s)");
            dialog.accept();
        }
    });
    buttonLayout->addWidget(applyBtn);

    QPushButton* closeBtn = new QPushButton("Close");
    connect(closeBtn, &QPushButton::clicked, &dialog, &QDialog::accept);
    buttonLayout->addWidget(closeBtn);

    layout->addLayout(buttonLayout);

    dialog.exec();
}

void MainWindow::onGenerateX25519KeyPair()
{
    if (!nda::CryptoManager::instance().generateX25519KeyPair()) {
        QMessageBox::critical(this, "Key Generation Error",
            "Failed to generate X25519 key pair:\n" +
            QString::fromStdString(nda::CryptoManager::instance().getLastError()));
        return;
    }

    std::string publicKeyHex = nda::CryptoManager::instance().exportX25519PublicKey();
    std::string privateKeyHex = nda::CryptoManager::instance().exportX25519PrivateKey();

    QDialog dialog(this);
    dialog.setWindowTitle("X25519 Key Pair Generated");
    dialog.setMinimumWidth(650);

    QVBoxLayout* layout = new QVBoxLayout(&dialog);

    // Public key section (for sharing)
    QGroupBox* publicGroup = new QGroupBox("Public Key (Share with peer)");
    QVBoxLayout* publicLayout = new QVBoxLayout(publicGroup);

    QLabel* publicInfo = new QLabel("Send this public key to your communication peer:");
    publicInfo->setWordWrap(true);
    publicLayout->addWidget(publicInfo);

    QLineEdit* publicKeyEdit = new QLineEdit(QString::fromStdString(publicKeyHex));
    publicKeyEdit->setReadOnly(true);
    publicKeyEdit->setFont(QFont("Courier New", 9));
    publicLayout->addWidget(publicKeyEdit);

    QPushButton* copyPublicBtn = new QPushButton("Copy Public Key");
    connect(copyPublicBtn, &QPushButton::clicked, [publicKeyHex]() {
        QApplication::clipboard()->setText(QString::fromStdString(publicKeyHex));
    });
    publicLayout->addWidget(copyPublicBtn);

    layout->addWidget(publicGroup);

    // Private key section (secure storage warning)
    QGroupBox* privateGroup = new QGroupBox("Private Key (Keep Secret!)");
    QVBoxLayout* privateLayout = new QVBoxLayout(privateGroup);

    QLabel* warningLabel = new QLabel("Never share your private key!");
    warningLabel->setStyleSheet("color: #ff6b6b; font-weight: bold;");
    privateLayout->addWidget(warningLabel);

    QLineEdit* privateKeyEdit = new QLineEdit(QString::fromStdString(privateKeyHex));
    privateKeyEdit->setReadOnly(true);
    privateKeyEdit->setFont(QFont("Courier New", 9));
    privateKeyEdit->setEchoMode(QLineEdit::Password);  // Hidden by default
    privateLayout->addWidget(privateKeyEdit);

    QCheckBox* showPrivateCheck = new QCheckBox("Show private key");
    connect(showPrivateCheck, &QCheckBox::toggled, [privateKeyEdit](bool checked) {
        privateKeyEdit->setEchoMode(checked ? QLineEdit::Normal : QLineEdit::Password);
    });
    privateLayout->addWidget(showPrivateCheck);

    layout->addWidget(privateGroup);

    // Close button
    QPushButton* closeBtn = new QPushButton("Close");
    connect(closeBtn, &QPushButton::clicked, &dialog, &QDialog::accept);
    layout->addWidget(closeBtn);

    dialog.exec();
}

void MainWindow::onImportKeys()
{
    ImportKeysDialog dialog(this);
    dialog.exec();
}

void MainWindow::onExportKeys()
{
    ExportKeysDialog dialog(this);
    dialog.exec();
}

void MainWindow::onDeriveSharedKey()
{
    if (!nda::CryptoManager::instance().hasX25519KeyPair()) {
        QMessageBox::warning(this, "Missing Keys",
            "No X25519 key pair loaded.\n\n"
            "Generate or import a key pair first using:\n"
            "Crypto -> Generate X25519 Key Pair");
        return;
    }

    if (!nda::CryptoManager::instance().hasX25519PeerPublicKey()) {
        QMessageBox::warning(this, "Missing Peer Key",
            "No peer public key loaded.\n\n"
            "Import your peer's public key first using:\n"
            "Crypto -> Import Keys");
        return;
    }

    if (nda::CryptoManager::instance().deriveSharedAES256Key()) {
        std::string derivedKey = nda::CryptoManager::instance().exportAES256Key();

        QMessageBox msgBox(this);
        msgBox.setWindowTitle("Key Derivation Success");
        msgBox.setIcon(QMessageBox::Information);
        msgBox.setText("Shared AES-256 key derived successfully via ECDH!");
        msgBox.setInformativeText(
            "The derived key is now loaded and can be applied to encryptor/decryptor plugins.\n\n"
            "Derived Key (first 32 chars): " + QString::fromStdString(derivedKey.substr(0, 32)) + "...\n\n"
            "Use 'Generate AES-256 Key' dialog or 'Apply to Plugin' to use this key.");
        msgBox.setStandardButtons(QMessageBox::Ok);
        msgBox.exec();
    } else {
        QMessageBox::critical(this, "Key Derivation Error",
            "Failed to derive shared key:\n" +
            QString::fromStdString(nda::CryptoManager::instance().getLastError()));
    }
}

void MainWindow::onClearKeys()
{
    QMessageBox::StandardButton reply = QMessageBox::question(this,
        "Clear All Keys",
        "This will clear all cryptographic keys from memory:\n"
        "- AES-256 symmetric key\n"
        "- X25519 key pair\n"
        "- Peer's X25519 public key\n\n"
        "Are you sure?",
        QMessageBox::Yes | QMessageBox::No);

    if (reply == QMessageBox::Yes) {
        nda::CryptoManager::instance().clearAES256Key();
        nda::CryptoManager::instance().clearX25519Keys();
        QMessageBox::information(this, "Keys Cleared",
            "All cryptographic keys have been securely cleared from memory.");
    }
}

bool MainWindow::applyKeyToSelectedPlugin(const std::string& paramName, const std::string& hexValue)
{
    auto txProcessor = unifiedView_->getTXProcessor();
    auto rxProcessor = unifiedView_->getRXProcessor();

    if (!txProcessor && !rxProcessor) {
        QMessageBox::warning(this, "No Plugin Selected",
            "No processor plugins are currently selected.\n\n"
            "Please select an encryptor or decryptor plugin in the TX or RX pipeline first.");
        return false;
    }

    bool appliedToTX = false;
    bool appliedToRX = false;
    QString txPluginName;
    QString rxPluginName;

    // Apply to TX processor if present
    if (txProcessor) {
        txPluginName = QString::fromStdString(txProcessor->getInfo().name);
        txProcessor->setParameter(paramName, hexValue);
        appliedToTX = true;
    }

    // Apply to RX processor if present
    if (rxProcessor) {
        rxPluginName = QString::fromStdString(rxProcessor->getInfo().name);
        rxProcessor->setParameter(paramName, hexValue);
        appliedToRX = true;
    }

    // Show result message
    if (appliedToTX && appliedToRX) {
        QMessageBox::information(this, "Key Applied",
            "Successfully applied key to both pipelines:\n\n"
            "TX: " + txPluginName + "\n"
            "RX: " + rxPluginName + "\n\n"
            "Parameter: " + QString::fromStdString(paramName));
        return true;
    } else if (appliedToTX) {
        QMessageBox::information(this, "Key Applied",
            "Successfully applied key to TX pipeline:\n\n"
            "TX: " + txPluginName + "\n\n"
            "Parameter: " + QString::fromStdString(paramName));
        return true;
    } else if (appliedToRX) {
        QMessageBox::information(this, "Key Applied",
            "Successfully applied key to RX pipeline:\n\n"
            "RX: " + rxPluginName + "\n\n"
            "Parameter: " + QString::fromStdString(paramName));
        return true;
    } else {
        QMessageBox::warning(this, "Parameter Not Supported",
            "The selected plugin(s) do not support the parameter: " +
            QString::fromStdString(paramName) + "\n\n"
            "This key can still be manually copied and pasted into compatible plugins.");
        return false;
    }
}
