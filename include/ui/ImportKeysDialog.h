#ifndef IMPORTKEYSDIALOG_H
#define IMPORTKEYSDIALOG_H

#include <QDialog>

class QLineEdit;
class QLabel;
class QTabWidget;
class QPushButton;

/**
 * @brief Dialog for importing cryptographic keys
 *
 * Provides a tabbed interface for importing:
 * - AES-256 symmetric keys (64 hex characters)
 * - X25519 private keys (64 hex characters)
 * - X25519 peer public keys (64 hex characters)
 *
 * Features real-time validation with visual feedback.
 */
class ImportKeysDialog : public QDialog
{
    Q_OBJECT

public:
    explicit ImportKeysDialog(QWidget *parent = nullptr);
    ~ImportKeysDialog() override = default;

private slots:
    void onAES256TextChanged(const QString& text);
    void onX25519PrivateTextChanged(const QString& text);
    void onX25519PeerPublicTextChanged(const QString& text);
    void onImportClicked();

private:
    void setupUI();
    bool validateHexString(const QString& hex, int expectedLength);
    void updateValidationLabel(QLabel* label, bool isValid, bool isEmpty);

    // Tab widgets
    QTabWidget* tabWidget_;

    // AES-256 tab
    QLineEdit* aes256Input_;
    QLabel* aes256ValidationLabel_;

    // X25519 private key tab
    QLineEdit* x25519PrivateInput_;
    QLabel* x25519PrivateValidationLabel_;

    // X25519 peer public key tab
    QLineEdit* x25519PeerPublicInput_;
    QLabel* x25519PeerPublicValidationLabel_;

    // Buttons
    QPushButton* importButton_;
    QPushButton* cancelButton_;
};

#endif // IMPORTKEYSDIALOG_H
