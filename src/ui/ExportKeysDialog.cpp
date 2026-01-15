#include "ui/ExportKeysDialog.h"
#include "crypto/CryptoManager.h"
#include <QApplication>
#include <QClipboard>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QVBoxLayout>
#include <QFont>

ExportKeysDialog::ExportKeysDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle("Export Keys");
    setMinimumWidth(750);
    setupUI();
}

void ExportKeysDialog::setupUI()
{
    QVBoxLayout* mainLayout = new QVBoxLayout(this);

    // Info label
    QLabel* infoLabel = new QLabel(
        "All currently loaded cryptographic keys are shown below.\n"
        "Use the copy buttons to copy keys to the clipboard for sharing or backup.");
    infoLabel->setWordWrap(true);
    mainLayout->addWidget(infoLabel);

    mainLayout->addSpacing(10);

    // Get keys from CryptoManager
    nda::CryptoManager& crypto = nda::CryptoManager::instance();

    // AES-256 Symmetric Key
    QString aes256Key = QString::fromStdString(crypto.exportAES256Key());
    QString aes256Status = crypto.hasAES256Key() ? "Loaded" : "Not Set";
    addKeyRow(mainLayout, QString("AES-256 Symmetric Key"), aes256Key, aes256Status, false);

    mainLayout->addSpacing(5);

    // X25519 Private Key
    QString x25519PrivateKey = QString::fromStdString(crypto.exportX25519PrivateKey());
    QString x25519PrivateStatus = crypto.hasX25519KeyPair() ? "Loaded" : "Not Set";
    addKeyRow(mainLayout, QString("X25519 Private Key (Keep Secret!)"), x25519PrivateKey, x25519PrivateStatus, true);

    mainLayout->addSpacing(5);

    // X25519 Public Key
    QString x25519PublicKey = QString::fromStdString(crypto.exportX25519PublicKey());
    QString x25519PublicStatus = crypto.hasX25519KeyPair() ? "Loaded" : "Not Set";
    addKeyRow(mainLayout, QString("X25519 Public Key (For Sharing)"), x25519PublicKey, x25519PublicStatus, false);

    mainLayout->addSpacing(5);

    // X25519 Peer Public Key
    QString x25519PeerPublicKey = QString::fromStdString(crypto.exportX25519PeerPublicKey());
    QString x25519PeerPublicStatus = crypto.hasX25519PeerPublicKey() ? "Loaded" : "Not Set";
    addKeyRow(mainLayout, QString("X25519 Peer Public Key"), x25519PeerPublicKey, x25519PeerPublicStatus, false);

    mainLayout->addSpacing(10);

    // Close button
    QHBoxLayout* buttonLayout = new QHBoxLayout();
    buttonLayout->addStretch();

    QPushButton* closeButton = new QPushButton("Close");
    connect(closeButton, &QPushButton::clicked, this, &QDialog::accept);
    buttonLayout->addWidget(closeButton);

    mainLayout->addLayout(buttonLayout);
}

void ExportKeysDialog::addKeyRow(QVBoxLayout* layout, const QString& label,
                                  const QString& key, const QString& status, bool isPrivate)
{
    QGroupBox* groupBox = new QGroupBox(label);
    QVBoxLayout* groupLayout = new QVBoxLayout(groupBox);

    // Status label
    QLabel* statusLabel = new QLabel("Status: " + status);
    if (status == "Loaded") {
        statusLabel->setStyleSheet("color: #2ecc71; font-weight: bold;");
    } else {
        statusLabel->setStyleSheet("color: #95a5a6;");
    }
    groupLayout->addWidget(statusLabel);

    if (!key.isEmpty()) {
        // Key display (read-only, monospace)
        QLineEdit* keyEdit = new QLineEdit(key);
        keyEdit->setReadOnly(true);
        keyEdit->setFont(QFont("Courier New", 9));

        if (isPrivate) {
            keyEdit->setEchoMode(QLineEdit::Password);  // Hidden by default for private keys

            // Add warning for private keys
            QLabel* warningLabel = new QLabel("⚠️ Never share your private key!");
            warningLabel->setStyleSheet("color: #ff6b6b; font-weight: bold;");
            groupLayout->addWidget(warningLabel);
        }

        groupLayout->addWidget(keyEdit);

        // Button layout
        QHBoxLayout* buttonLayout = new QHBoxLayout();

        // Copy button
        QPushButton* copyButton = new QPushButton("Copy to Clipboard");
        connect(copyButton, &QPushButton::clicked, [key]() {
            QApplication::clipboard()->setText(key);
        });
        buttonLayout->addWidget(copyButton);

        // Show/hide button for private keys
        if (isPrivate) {
            QPushButton* toggleButton = new QPushButton("Show");
            connect(toggleButton, &QPushButton::clicked, [keyEdit, toggleButton]() {
                if (keyEdit->echoMode() == QLineEdit::Password) {
                    keyEdit->setEchoMode(QLineEdit::Normal);
                    toggleButton->setText("Hide");
                } else {
                    keyEdit->setEchoMode(QLineEdit::Password);
                    toggleButton->setText("Show");
                }
            });
            buttonLayout->addWidget(toggleButton);
        }

        buttonLayout->addStretch();
        groupLayout->addLayout(buttonLayout);
    } else {
        QLabel* notSetLabel = new QLabel("No key loaded");
        notSetLabel->setStyleSheet("color: #95a5a6; font-style: italic;");
        groupLayout->addWidget(notSetLabel);
    }

    layout->addWidget(groupBox);
}
