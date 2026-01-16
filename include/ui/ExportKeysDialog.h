#ifndef EXPORTKEYSDIALOG_H
#define EXPORTKEYSDIALOG_H

#include <QDialog>

class QLineEdit;
class QLabel;
class QPushButton;
class QVBoxLayout;

/**
 * @brief Dialog for exporting cryptographic keys
 *
 * Displays all loaded keys with status indicators:
 * - AES-256 symmetric key
 * - X25519 private key
 * - X25519 public key (for sharing)
 * - X25519 peer public key
 *
 * Provides copy-to-clipboard buttons for each key.
 */
class ExportKeysDialog : public QDialog
{
    Q_OBJECT

public:
    explicit ExportKeysDialog(QWidget *parent = nullptr);
    ~ExportKeysDialog() override = default;

private:
    void setupUI();
    void addKeyRow(QVBoxLayout* layout, const QString& label, const QString& key,
                   const QString& status, bool isPrivate = false);
};

#endif // EXPORTKEYSDIALOG_H
