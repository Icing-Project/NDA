#include "ui/ImportKeysDialog.h"
#include "crypto/CryptoManager.h"
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QTabWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QRegularExpression>
#include <QFont>

ImportKeysDialog::ImportKeysDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle("Import Keys");
    setMinimumWidth(700);
    setupUI();
}

void ImportKeysDialog::setupUI()
{
    QVBoxLayout* mainLayout = new QVBoxLayout(this);

    // Info label
    QLabel* infoLabel = new QLabel(
        "Import cryptographic keys by pasting hexadecimal strings below.\n"
        "All keys must be exactly 64 hexadecimal characters (0-9, a-f, A-F).");
    infoLabel->setWordWrap(true);
    mainLayout->addWidget(infoLabel);

    // Tab widget for different key types
    tabWidget_ = new QTabWidget(this);

    // === Tab 1: AES-256 Symmetric Key ===
    QWidget* aes256Tab = new QWidget();
    QVBoxLayout* aes256Layout = new QVBoxLayout(aes256Tab);

    QLabel* aes256Info = new QLabel(
        "AES-256 symmetric key for encryption/decryption.\n"
        "This key must be shared securely with your communication peer.");
    aes256Info->setWordWrap(true);
    aes256Layout->addWidget(aes256Info);

    QLabel* aes256Label = new QLabel("Key (64 hex characters):");
    aes256Layout->addWidget(aes256Label);

    aes256Input_ = new QLineEdit();
    aes256Input_->setFont(QFont("Courier New", 10));
    aes256Input_->setPlaceholderText("Paste 64-character hexadecimal key here...");
    connect(aes256Input_, &QLineEdit::textChanged, this, &ImportKeysDialog::onAES256TextChanged);
    aes256Layout->addWidget(aes256Input_);

    aes256ValidationLabel_ = new QLabel();
    aes256ValidationLabel_->setWordWrap(true);
    aes256Layout->addWidget(aes256ValidationLabel_);

    aes256Layout->addStretch();
    tabWidget_->addTab(aes256Tab, "AES-256 Symmetric Key");

    // === Tab 2: X25519 Private Key ===
    QWidget* x25519PrivateTab = new QWidget();
    QVBoxLayout* x25519PrivateLayout = new QVBoxLayout(x25519PrivateTab);

    QLabel* x25519PrivateInfo = new QLabel(
        "Your X25519 private key for ECDH key exchange.\n"
        "WARNING: Keep this key secret! Never share your private key.");
    x25519PrivateInfo->setWordWrap(true);
    x25519PrivateInfo->setStyleSheet("color: #ff6b6b; font-weight: bold;");
    x25519PrivateLayout->addWidget(x25519PrivateInfo);

    QLabel* x25519PrivateLabel = new QLabel("Private Key (64 hex characters):");
    x25519PrivateLayout->addWidget(x25519PrivateLabel);

    x25519PrivateInput_ = new QLineEdit();
    x25519PrivateInput_->setFont(QFont("Courier New", 10));
    x25519PrivateInput_->setPlaceholderText("Paste 64-character hexadecimal private key here...");
    x25519PrivateInput_->setEchoMode(QLineEdit::Password);  // Hidden by default
    connect(x25519PrivateInput_, &QLineEdit::textChanged, this, &ImportKeysDialog::onX25519PrivateTextChanged);
    x25519PrivateLayout->addWidget(x25519PrivateInput_);

    x25519PrivateValidationLabel_ = new QLabel();
    x25519PrivateValidationLabel_->setWordWrap(true);
    x25519PrivateLayout->addWidget(x25519PrivateValidationLabel_);

    x25519PrivateLayout->addStretch();
    tabWidget_->addTab(x25519PrivateTab, "X25519 Private Key");

    // === Tab 3: X25519 Peer Public Key ===
    QWidget* x25519PeerPublicTab = new QWidget();
    QVBoxLayout* x25519PeerPublicLayout = new QVBoxLayout(x25519PeerPublicTab);

    QLabel* x25519PeerPublicInfo = new QLabel(
        "Your peer's X25519 public key for ECDH key exchange.\n"
        "Your peer should send you their public key via a secure channel.");
    x25519PeerPublicInfo->setWordWrap(true);
    x25519PeerPublicLayout->addWidget(x25519PeerPublicInfo);

    QLabel* x25519PeerPublicLabel = new QLabel("Peer Public Key (64 hex characters):");
    x25519PeerPublicLayout->addWidget(x25519PeerPublicLabel);

    x25519PeerPublicInput_ = new QLineEdit();
    x25519PeerPublicInput_->setFont(QFont("Courier New", 10));
    x25519PeerPublicInput_->setPlaceholderText("Paste 64-character hexadecimal peer public key here...");
    connect(x25519PeerPublicInput_, &QLineEdit::textChanged, this, &ImportKeysDialog::onX25519PeerPublicTextChanged);
    x25519PeerPublicLayout->addWidget(x25519PeerPublicInput_);

    x25519PeerPublicValidationLabel_ = new QLabel();
    x25519PeerPublicValidationLabel_->setWordWrap(true);
    x25519PeerPublicLayout->addWidget(x25519PeerPublicValidationLabel_);

    x25519PeerPublicLayout->addStretch();
    tabWidget_->addTab(x25519PeerPublicTab, "X25519 Peer Public Key");

    mainLayout->addWidget(tabWidget_);

    // Buttons
    QHBoxLayout* buttonLayout = new QHBoxLayout();
    buttonLayout->addStretch();

    importButton_ = new QPushButton("Import");
    importButton_->setEnabled(false);  // Disabled until valid input
    connect(importButton_, &QPushButton::clicked, this, &ImportKeysDialog::onImportClicked);
    buttonLayout->addWidget(importButton_);

    cancelButton_ = new QPushButton("Cancel");
    connect(cancelButton_, &QPushButton::clicked, this, &QDialog::reject);
    buttonLayout->addWidget(cancelButton_);

    mainLayout->addLayout(buttonLayout);
}

void ImportKeysDialog::onAES256TextChanged(const QString& text)
{
    bool isValid = validateHexString(text, 64);
    updateValidationLabel(aes256ValidationLabel_, isValid, text.isEmpty());

    // Enable import button if any field has valid input
    importButton_->setEnabled(isValid ||
                               validateHexString(x25519PrivateInput_->text(), 64) ||
                               validateHexString(x25519PeerPublicInput_->text(), 64));
}

void ImportKeysDialog::onX25519PrivateTextChanged(const QString& text)
{
    bool isValid = validateHexString(text, 64);
    updateValidationLabel(x25519PrivateValidationLabel_, isValid, text.isEmpty());

    importButton_->setEnabled(isValid ||
                               validateHexString(aes256Input_->text(), 64) ||
                               validateHexString(x25519PeerPublicInput_->text(), 64));
}

void ImportKeysDialog::onX25519PeerPublicTextChanged(const QString& text)
{
    bool isValid = validateHexString(text, 64);
    updateValidationLabel(x25519PeerPublicValidationLabel_, isValid, text.isEmpty());

    importButton_->setEnabled(isValid ||
                               validateHexString(aes256Input_->text(), 64) ||
                               validateHexString(x25519PrivateInput_->text(), 64));
}

void ImportKeysDialog::onImportClicked()
{
    int successCount = 0;
    QString successMessage;
    QString errorMessage;

    // Try to import AES-256 key
    QString aes256Text = aes256Input_->text().trimmed();
    if (!aes256Text.isEmpty()) {
        if (validateHexString(aes256Text, 64)) {
            if (nda::CryptoManager::instance().importAES256Key(aes256Text.toStdString())) {
                successCount++;
                successMessage += "- AES-256 symmetric key\n";
            } else {
                errorMessage += "- AES-256 key: " +
                    QString::fromStdString(nda::CryptoManager::instance().getLastError()) + "\n";
            }
        }
    }

    // Try to import X25519 private key
    QString x25519PrivateText = x25519PrivateInput_->text().trimmed();
    if (!x25519PrivateText.isEmpty()) {
        if (validateHexString(x25519PrivateText, 64)) {
            if (nda::CryptoManager::instance().importX25519PrivateKey(x25519PrivateText.toStdString())) {
                successCount++;
                successMessage += "- X25519 private key (keypair)\n";
            } else {
                errorMessage += "- X25519 private key: " +
                    QString::fromStdString(nda::CryptoManager::instance().getLastError()) + "\n";
            }
        }
    }

    // Try to import X25519 peer public key
    QString x25519PeerPublicText = x25519PeerPublicInput_->text().trimmed();
    if (!x25519PeerPublicText.isEmpty()) {
        if (validateHexString(x25519PeerPublicText, 64)) {
            if (nda::CryptoManager::instance().importX25519PeerPublicKey(x25519PeerPublicText.toStdString())) {
                successCount++;
                successMessage += "- X25519 peer public key\n";
            } else {
                errorMessage += "- X25519 peer public key: " +
                    QString::fromStdString(nda::CryptoManager::instance().getLastError()) + "\n";
            }
        }
    }

    // Show results
    if (successCount > 0 && errorMessage.isEmpty()) {
        QMessageBox::information(this, "Import Successful",
            "Successfully imported " + QString::number(successCount) + " key(s):\n\n" + successMessage);
        accept();  // Close dialog
    } else if (successCount > 0 && !errorMessage.isEmpty()) {
        QMessageBox::warning(this, "Partial Import",
            "Imported " + QString::number(successCount) + " key(s):\n" + successMessage +
            "\n\nFailed to import:\n" + errorMessage);
        accept();  // Close dialog (partial success)
    } else if (!errorMessage.isEmpty()) {
        QMessageBox::critical(this, "Import Failed",
            "Failed to import keys:\n\n" + errorMessage);
    } else {
        QMessageBox::warning(this, "No Keys to Import",
            "Please enter at least one valid key to import.");
    }
}

bool ImportKeysDialog::validateHexString(const QString& hex, int expectedLength)
{
    if (hex.length() != expectedLength) {
        return false;
    }

    // Check all characters are valid hex digits
    static QRegularExpression hexRegex("^[0-9a-fA-F]+$");
    return hexRegex.match(hex).hasMatch();
}

void ImportKeysDialog::updateValidationLabel(QLabel* label, bool isValid, bool isEmpty)
{
    if (isEmpty) {
        label->clear();
        label->setStyleSheet("");
    } else if (isValid) {
        label->setText("✓ Valid (64 hex characters)");
        label->setStyleSheet("color: #2ecc71; font-weight: bold;");
    } else {
        label->setText("✗ Invalid (must be exactly 64 hexadecimal characters)");
        label->setStyleSheet("color: #e74c3c; font-weight: bold;");
    }
}
